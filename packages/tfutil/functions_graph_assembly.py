import os
import sys
import pdb
import glob
import json
import copy
import warnings
import numpy as np

os.environ['TF_CUDNN_USE_AUTOTUNE']='0'
import tensorflow as tf

import functions_brain_network


# List of ops to locate on controller device when running in Muli-GPU Mode
PS_OPS = [
    'Variable',
    'VariableV2',
    'AutoReloadVariable',
    'MutableHashTable',
    'MutableHashTableOfTensors',
    'MutableDenseHashTable'
]


def build_tfrecords_iterator(
        tfrecords_regex,
        feature_parsing_dict={},
        iterator_type='one-shot',
        num_epochs=1,
        batch_size=8,
        n_prefetch=10,
        buffer=1000,
        shuffle_flag=True,
        dataset_filter_params={},
        **kwargs):
    '''
    Builds tensorflow iterator for feeding graph with data from tfrecords.
    
    Args
    ----
    tfrecords_regex (str): regular expression capturing all tfrecords to include in dataset
    feature_parsing_dict (dict): keys are tfrecords feature keys, values are dictionaries with 'dtype' and 'shape' keys
    iterator_type (str): must be either 'one-shot' (for training) or 'initializable' (for validation)
    num_epochs (int): number of times to repeat dataset
    batch_size (int): number of examples per batch
    n_prefetch (int): argument for dataset.prefetch (max number of elements to buffer when prefetching)
    buffer (int): argument for dataset.shuffle (size of shuffle buffer)
    shuffle_flag (bool): if True, dataset will be shuffled
    dataset_filter_params (dict): parameters for filtering out examples from the tfrecords
        keys are tfrecords feature keys (paths to data in the tfrecords)
        values are filter constraints (conditions for including examples)
    
    Returns
    -------
    iterator (tf iterator object): iterator whose `get_next()` method returns `input_tensor_dict`
    dataset (tf dataset object): dataset object used to construct the iterator
    iterator_saveable_object (tf saveable object): saveable object for saving the iterator state
    '''
    ### Helper dictionary to map strings to tf.dtype objects (which are not easily saved in JSON)
    string_to_dtype = {
        'tf.float32': tf.float32,
        'tf.float64': tf.float64,
        'tf.int32': tf.int32,
        'tf.int64': tf.int64,
        'tf.string': tf.string,
    }
    
    ### Set up feature_dict to use for parsing tfrecords
    feature_dict = {}
    for path in sorted(feature_parsing_dict.keys()):
        path_dtype = string_to_dtype[feature_parsing_dict[path]['dtype']]
        path_shape = feature_parsing_dict[path].get('shape', ())
        if len(path_shape) > 0: path_dtype = tf.string
        feature_dict[path] = tf.FixedLenFeature([], path_dtype)
    
    ### Define the tfrecords parsing function
    def parse_tfrecord_example(record):
        ''' Parsing function returns dictionary of tensors with tfrecords paths as keys '''
        # Parse the record read by the reader
        parsed_features = tf.parse_single_example(record, features=feature_dict)
        # Decode features and return as a dictionary of tensors
        input_tensor_dict = {}
        for path in sorted(feature_parsing_dict.keys()):
            path_dtype = string_to_dtype[feature_parsing_dict[path]['dtype']]
            path_shape = feature_parsing_dict[path].get('shape', ())
            if len(path_shape) > 0: # Array-like features are read-in as bytes and must be decoded
                decoded_bytes_feature = tf.decode_raw(parsed_features[path], path_dtype)
                if decoded_bytes_feature.dtype == tf.float64:
                    # This will cast tf.float64 inputs to tf.float32, since many tf ops do not support tf.float64.
                    # If we want control over this (i.e. make the network run using tf.float16, we should either
                    # change the tfrecords files or add a cast operation after calling the iterator).
                    decoded_bytes_feature = tf.cast(decoded_bytes_feature, tf.float32)
                input_tensor_dict[path] = tf.reshape(decoded_bytes_feature, path_shape)
            else:
                input_tensor_dict[path] = parsed_features[path]
        return input_tensor_dict
    
    ### Create tensorflow dataset by parsing examples from tfrecords
    input_data_filenames = sorted(glob.glob(tfrecords_regex))
    print('### Files found: {}'.format(len(input_data_filenames)))
    print(input_data_filenames[0],'\n...\n', input_data_filenames[-1])
    dataset = tf.data.Dataset.list_files(input_data_filenames)
    dataset = dataset.apply(tf.data.experimental.parallel_interleave(lambda x:tf.data.TFRecordDataset(x,
                            compression_type="GZIP").map(parse_tfrecord_example, num_parallel_calls=1),
                            cycle_length=10, block_length=16))
    
    ### Apply filter(s) to dataset if `dataset_filter_params` is specified
    for key in dataset_filter_params.keys():
        constraint = dataset_filter_params[key]
        if isinstance(constraint, (list, tuple)) and len(constraint) == 2:
            filter_fn = lambda x: tf.math.logical_and(x[key] >= constraint[0], x[key] <= constraint[1])
        elif isinstance(constraint, (int, float, bool)):
            filter_fn = lambda x: x[key] == constraint
        else:
            # TODO: implement other constraints; perhaps move filtering to another function
            raise ValueError("Unsupported constraint: `dataset_filter_params[{}]`".format(key))
        dataset = dataset.filter(filter_fn)
    
    ### Shuffle, repeat, batch, and prefetch dataset
    if shuffle_flag: dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(buffer, num_epochs))
    else: dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(n_prefetch)
    
    ### Return dataset iterator
    if iterator_type == 'one-shot':
        iterator = dataset.make_one_shot_iterator()
    elif iterator_type == 'initializable':
        iterator = dataset.make_initializable_iterator()
    else:
        raise ValueError('iterator type not supported: use one-shot or initializeable')
    iterator_saveable_object = tf.data.experimental.make_saveable_from_iterator(iterator)
    tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, iterator_saveable_object)
    return iterator, dataset, iterator_saveable_object


def build_saver(
        sess,
        var_list,
        output_location,
        restore_model_path=None,
        ckpt_prefix_name='model.ckpt',
        attempt_load=True):
    '''
    This function creates a saver object and attempts to load a checkpoint.
    If a restore_model_path is not specified, attempt to load latest checkpoint.
    If no checkpoint is found, no checkpoint is loaded.
    
    Args
    ----
    sess (tensorflow session object)
    var_list (list): list of variables for tensorflow saver object
    output_location (str): directory in which to save new model checkpoints
    restore_model_path (str): used to load a specific model checkpoint (full path)
    ckpt_prefix_name (str): prefix for saving new model checkpoints
    attempt_load (bool): if set to False, this function will not load a checkpoint
    
    Returns
    -------
    saver (tensorflow saver object): saver for specified var_list
    out_ckpt_loc (string): full path for saving future checkpoints
    ckpt_num (int): loaded ckpt_num (0 if no checkpoint is loaded)
    '''
    saver = tf.train.Saver(var_list=var_list, max_to_keep=0)
    if restore_model_path is not None:
        ckpt_num = int(restore_model_path.split('-')[-1].split('.')[0])
        if attempt_load:
            print('### Loading variables from specified checkpoint: {}'.format(restore_model_path))
            saver.restore(sess, restore_model_path)
        else: warnings.warn("CAUTION: ckpt_num={}, but no checkpoint was loaded".format(ckpt_num))
    else:
        saved_checkpoints = sorted(glob.glob(os.path.join(output_location, ckpt_prefix_name) + '*.index'))
        saved_checkpoint_numbers = []
        saved_checkpoint_numbers = [int(s.split('-')[-1].split('.')[0]) for s in saved_checkpoints]
        if len(saved_checkpoint_numbers) > 0: 
            ckpt_num = np.max(saved_checkpoint_numbers)
            restore_model_path = os.path.join(output_location, ckpt_prefix_name + '-{}'.format(ckpt_num))
            if attempt_load:
                print('### Loading variables from latest checkpoint: {}'.format(restore_model_path))
                saver.restore(sess, restore_model_path)
            else: warnings.warn("CAUTION: ckpt_num={}, but no checkpoint was loaded".format(ckpt_num))
        else:
            print('### No previous checkpoint found, restarting training : {}'.format(output_location))            
            ckpt_num = 0
    out_ckpt_loc = os.path.join(output_location, ckpt_prefix_name)
    return saver, out_ckpt_loc, ckpt_num


def build_optimizer(
        combined_batch_size,
        learning_rate=1e-4,
        optm_type='adam',
        learning_rate_decay_type='fixed',
        decay_rate=0.96,
        num_decay_samples=697272,
        adam_epsilon=1e-4):
    """
    Builds the optimizer for training and creates the global_step tensor.
 
    Args
    ----
    combined_batch_size (int) : The batch size used for training, totaled 
        across all GPUs
    learning_rate (float) : The initial learning rate for the optimizer.
    optm_type (string) : The type of optimizer used for for training.
    learning_rate_decay_type (string) : The type of learning rate decay.
    decay_rate (float) : The amount of decay for each step.
    num_decay_samples (int) : The number of samples (exemplars) before
        the learning rate decays. Default is 3x the number of samples 
        in jsinv3.
    
    Returns
    -------
    optm (tensorflow optimizer) : Optimization object for training. 
    global_step (tensor) : Global step variable to use for training. 
    
    Raises
    ------
    ValueError: The optimization type specified is not implemented. 

    """
    global_step = tf.Variable(0, trainable=False, name='global_step')
    decay_steps=np.ceil(num_decay_samples/combined_batch_size)
    learning_rate_tensor=make_learning_rate_tensor(learning_rate,
                                                   global_step,
                                                   learning_rate_decay_type,
                                                   decay_steps,
                                                   decay_rate)
    if optm_type=='adam':
        optm=tf.train.AdamOptimizer(learning_rate_tensor,
                                    epsilon=adam_epsilon)
    else:
        raise ValueError('Specified optm_type %s is not implemented.' 
                         % optm_type)

    return optm, global_step


def make_learning_rate_tensor(learning_rate, 
                              global_step, 
                              learning_rate_decay_type,
                              decay_steps, 
                              decay_rate):
    """
    Creates a tensor that contains the (possibly changing) learning rate.

    Args
    ----
    learning_rate (float) : The initial learning rate for the optimizer.
    global_step (tensor) : The global step during training.
    learning_rate_decay_type (string) : The type of learning rate decay.
    decay_steps (int) : The number of steps between each decay.
    decay_rate (float) : The amount of decay for each step.
    
    Returns
    -------
    learning_rate (tensor) : The (possibly changing) learning rate to use 
        in the optimizer

    Raises
    ------
    ValueError: if the specified learning_rate_decay_type is not implemented
    """
    if learning_rate_decay_type == 'exponential':
        return tf.train.exponential_decay(learning_rate,
                                          global_step,
                                          decay_steps,
                                          decay_rate,
                                          staircase=True,
                                          name='exponential_decay_learning_rate')
    elif learning_rate_decay_type == 'fixed':
        return tf.constant(learning_rate, name='fixed_learning_rate')
    else:
        raise ValueError('Specified learning_rate_decay_type %s is not '
                         'implemented.' % learning_rate_decay_type)


def runtime_print_metrics(batch_out_dict, batch_labels_dict, batch_loss_dict):
    """
    This function builds the print statements at metrics to use when training
    
    Args
    ----
    batch_out_dict (dict of tensor) : logits output by the graph
    batch_labels_dict (dict of tensor) : labels for the input data
    batch_loss_dict (dict of losses) : contains the loss for each task
    
    Returns
    -------
    disp_str (string) : contains the string with formatting for printing 
    print_metric_list (list) : list of tensors that are used in disp_str
    """
    print_metric_list = []
    disp_str = ' ### batch_num: {:08}, total_examples:{:010}, time:{:.2f}, batch_loss: {:.4f} '

    for batch_labels_key in sorted(batch_labels_dict.keys()):
        batch_labels = batch_labels_dict[batch_labels_key]
        batch_out = batch_out_dict[batch_labels_key]

        if not len(batch_labels.shape) > 1:
            disp_str += '\n ### ' + batch_labels_key + ': task_loss {:.4f}, batch_acc: {:.2f}'
            labels_true = tf.cast(batch_labels, tf.int64)
            labels_pred = tf.argmax(batch_out, axis=1)
            correct_pred = tf.equal(labels_pred, labels_true)
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            print_metric_list.append(batch_loss_dict[batch_labels_key])
            print_metric_list.append(accuracy)
        else:  
            disp_str += '\n ### ' + batch_labels_key + ': task_loss {:.4f}, batch_r2 {:.2f}, batch_mean_abs {:2f}, batch_mean_exact_equal {:2f}'
            probs_out = tf.sigmoid(batch_out)
            labels_true = tf.cast(batch_labels, tf.float32) 
            mean_absolute_value = tf.reduce_mean(tf.abs(tf.subtract(labels_true, probs_out)))
            mean_true, var_true = tf.nn.moments(labels_true, axes=[0, 1])
            mean_pred, var_pred = tf.nn.moments(probs_out, axes=[0, 1])
            r2_numerator = tf.reduce_mean(tf.multiply(tf.subtract(labels_true, mean_true), tf.subtract(probs_out, mean_pred)))
            r2_denominator = tf.multiply(tf.sqrt(var_true), tf.sqrt(var_pred))
            r2 = tf.divide(r2_numerator, r2_denominator)
            exactly_equal = tf.reduce_all(tf.equal(tf.round(probs_out), labels_true), axis=[1])
            average_equal_batch = tf.reduce_mean(tf.cast(exactly_equal, tf.float32), axis=[0])
            print_metric_list.append(batch_loss_dict[batch_labels_key])
            print_metric_list.append(r2)
            print_metric_list.append(mean_absolute_value)
            print_metric_list.append(average_equal_batch)
        print('PRINT METRIC LIST')
        print(print_metric_list)
    
    # For any losses that are not label related, print only the loss
    for loss_key in sorted(batch_loss_dict.keys()):
        if loss_key not in batch_labels_dict.keys():
            disp_str += '\n ### ' + loss_key + ': loss {:.4f}'
            print_metric_list.append(batch_loss_dict[loss_key])
    
    return disp_str, print_metric_list


def build_brain_graph(
        batch_subbands,
        N_CLASSES_DICT,
        config,
        batchnorm_flag=True,
        dropout_flag=True,
        save_pckl_path=None,
        save_arch_path=None,
        trainable=True,
        only_include_layers=None,
        var_scope_name='brain_network',
        var_scope_reuse=False,
        **kwargs):
    '''
    This function builds the graph for the brain network.
    
    Args
    ----
    batch_subbands (tensor): input peripheral representation
    N_CLASSES_DICT (dict): dictionary containing the number of output classes, keys are the task paths
    config (dict or string) : dictionary containing network config, or path to json with same info 
    batchnorm_flag (boolean): if True, batch norm moving averages will update (training mode)
    dropout_flag (boolean): if True, dropout will occur
    save_pckl_path (string) : path where a pickle should be saved containing partial funcs to rebuild graph
    save_arch_path (string) : path where a .json should be saved containing the graph architecture
    trainable (boolean): if True, network parameters are trainable
    only_include_layers (set) : if not None, stops building the graph when all of the keys included in this set are constructed
    var_scope_name (str): variable scope name for brain network graph
    var_scope_reuse (str): sets the reuse flag within the variable scope containing the brain network, used to share 
        variables across multiple brain networks that are constructed (ie for multi tower models, or when building 
        the reconstruction loss)
    
    Returns
    -------
    batch_out_dict (dict of tensor) : logits output by the graph, keys are the task paths
    '''
    use_functions_brain_network = True
    if (type(config) is str) and (os.path.isfile(config)):
        print('Loading brain network config from %s'%config)
        with open(config) as json_data:
            dict_config = json.load(json_data)
    elif type(config) is dict:
        print('Using dictionary as brain network config')
        dict_config = config
    elif (type(config) is str) and (os.path.isdir(config)):
        use_functions_brain_network = False
    else:
        raise ValueError("Unrecognized format for brain network config.") 
    
    with tf.variable_scope(var_scope_name, reuse=var_scope_reuse):
        if use_functions_brain_network:
            final_layer, brain_network = functions_brain_network.make_brain_net(
                batch_subbands,
                N_CLASSES_DICT,  
                dict_config,
                trainable=trainable,
                batchnorm_flag=batchnorm_flag, 
                dropout_flag=dropout_flag,
                save_pckl_path=save_pckl_path, 
                save_arch_path=save_arch_path,
                only_include_layers=only_include_layers)
        else:
            raise ValueError("Unrecognized format for brain network config.")

        # Make sure that the final output layer is in a dictionary (necessary to keep track of the task parameters)
        if len(list(N_CLASSES_DICT.keys()))==1 and type(final_layer) is not dict:
            batch_out_dict = {list(N_CLASSES_DICT.keys())[0]: final_layer}
        else:
            batch_out_dict = final_layer

    return batch_out_dict, brain_network


def build_task_loss_graph(batch_logits_dict, batch_labels_dict, TASK_LOSS_PARAMS={}):
    '''
    This function builds the graph for the task loss functions.
    
    Args
    ----
    batch_logits_dict (dict): dictionary of task-specific logits (keys are task names, fields are tensors)
    batch_labels_dict (dict): dictionary of task-specific labels (keys are task names, fields are tensors)
    TASK_LOSS_PARAMS (dict): dictionary of task-specific loss function parameters (keys are task names, fields are dicts)
        TASK_LOSS_PARAMS['stimuli/word_int'] = {
            'activation_type': None,
            'loss_type': 'sparse_softmax_cross_entropy_with_logits',
            'apply_adaptive_uncertainty': False,
            'weight': 1.0,
        }
        If a task key does not appear in TASK_LOSS_PARAMS, loss functions will default to either
        one-hot or multi-hot classification.
    
    Returns
    -------
    batch_loss (tensor): tensor with shape [] (total loss)
    batch_loss_dict (dict): dictionary of tensors corresponding to task-specific losses
    '''
    if TASK_LOSS_PARAMS is None: TASK_LOSS_PARAMS = {}
    # Supported activation functions (used for loss functions that are not fused with activations)
    activation_functions = {
        None: None,
        'identity': tf.identity,
        'linear': tf.identity,
        'relu': tf.nn.relu,
        'sigmoid': tf.math.sigmoid,
        'softmax': tf.nn.softmax,
    }
    # Supported loss functions ("a" refers to logits or activations, "b" refers to labels)
    loss_functions = {
        'sigmoid_cross_entropy_with_logits': lambda a, b : tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=a, labels=tf.cast(b, a.dtype))),
        'sparse_softmax_cross_entropy_with_logits': lambda a, b: tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=a, labels=b)),
        'l2': lambda a, b : tf.reduce_mean(tf.squared_difference(a, b)),
        'l1': lambda a, b : tf.reduce_mean(tf.math.abs(tf.math.subtract(a, b))),
        'tf.losses.mean_squared_error': lambda a, b : tf.losses.mean_squared_error(b, a),
        'tf.losses.huber_loss': lambda a, b : tf.losses.huber_loss(b, a),
    }
    # Initialize values to return (loss dict and total loss)
    batch_loss_dict = {}
    batch_loss = 0
    for task_key in batch_labels_dict.keys():
        # Get the loss function parameters for each task
        current_task_loss_params = TASK_LOSS_PARAMS.get(task_key, {})
        task_weight = current_task_loss_params.get('weight', 1.0)
        task_loss_type = current_task_loss_params.get('loss_type', None)
        task_activation_type = current_task_loss_params.get('activation_type', None)
        if task_loss_type is None:
            # If loss_type is not specified, default to one-hot or multi-hot classification depending on label shape
            if len(batch_labels_dict[task_key].shape) > 1: task_loss_type = 'sigmoid_cross_entropy_with_logits'
            else: task_loss_type = 'sparse_softmax_cross_entropy_with_logits'
            print('`loss_type` not specified for task `{}` --> defaulting to `{}`'.format(task_key, task_loss_type))
        # Check the specified loss type and activation type
        assert task_loss_type in loss_functions, "`loss_type={}` is not a recognized task loss".format(task_loss_type)
        assert task_activation_type in activation_functions, "activation `{}` not recognized".format(task_activation_type)
        if ('with_logits' in task_loss_type) and (task_activation_type in ['sigmoid', 'softmax']):
            # If activation is fused with loss function, raise error if sigmoid or softmax activation is specified
            err_msg = 'invalid activation function / loss function combination: `{}` and `{}`'
            raise ValueError(err_msg.format(task_activation_type, task_loss_type))
        # Compute activations (may simply be the logits) for current task
        task_activation_function = activation_functions[task_activation_type]
        if task_activation_function is None: task_activations_or_logits = batch_logits_dict[task_key]
        else: task_activations_or_logits = task_activation_function(batch_logits_dict[task_key])
        if len(task_activations_or_logits.shape) > len(batch_labels_dict[task_key].shape):
            # This handles regression case where logits have shape [batch, 1] and labels have shape [batch]
            if task_activations_or_logits.shape[-1] == 1:
                task_activations_or_logits = tf.squeeze(task_activations_or_logits, axis=-1)
        # Compute loss for current task
        task_loss_function = loss_functions[task_loss_type]
        batch_loss_dict[task_key] = task_loss_function(task_activations_or_logits, batch_labels_dict[task_key])
        
        # If specified, apply adaptive uncertainty weighting to the current task
        if current_task_loss_params.get('apply_adaptive_uncertainty', False):
            raise NotImplementedError("adaptive uncertainty weighting is not implemented")
        
        batch_loss += task_weight * batch_loss_dict[task_key]
    return batch_loss, batch_loss_dict


def training_model(
        batch_input_signal,
        signal_rate,
        batch_labels_dict,
        N_CLASSES_DICT,
        TASK_LOSS_PARAMS=None, 
        FRONTEND_PARAMS={},
        COCH_PARAMS={},
        BRAIN_PARAMS={},
        NORMAL_HEARING_PARAMS={},
        normal_hearing_batch_input_signal=None):
    '''
    This function constructs the whole graph for training
    (designed to be called by `create_parallel_optimization`)
    
    Args
    ----
    batch_input_signal (tensor): input signal (e.g. waveform, pre-computed subbands, RGB image)
    signal_rate (int): sampling rate of input signal (Hz)
    batch_labels_dict (dict): label dictionary corresponding to each task for batch_input_signal
    N_CLASSES_DICT (dict): dictionary containing the number of output classes, keys are the task paths
    TASK_LOSS_PARAMS (dict): dictionary containing the weights for each task, keys are the task paths
    FRONTEND_PARAMS (dict): passed to frontend_model builder if non-empty
    COCH_PARAMS (dict): passed to peripheral model builder if non-empty
    BRAIN_PARAMS (dict): passed to brain network builder
    NORMAL_HEARING_PARAMS (dict): contains parameters for the "normal" hearing network, if matching on layer activations
    normal_hearing_batch_input_signal (tensor): input signal to the "normal" hearing network, if matching 
      on layer activations.
    
    Returns
    -------
    batch_loss (tensor): loss
    batch_loss_dict (dict of tensor): loss for each task, keys are the task paths
    batch_out_dict (dict of tensor): logits output by the graph, keys are the task paths
    batch_labels_dict (dict): label dictionary corresponding to each task for batch_input_signal
    batch_audio_dict (dict): dictionary containing input audio and frontend-modified audio tensors (if applicable)
    coch_container (dict): dictionary of peripheral model tensors and functions/parameters (if applicable)
    brain_container (dict): dictionary of brain network tensors (if applicable)
    '''
    # Build frontend graph if FRONTEND_PARAMS is not an empty dictionary
    batch_audio_dict = {'input_audio': batch_input_signal}
    if NORMAL_HEARING_PARAMS:
        raise NotImplementedError("NORMAL_HEARING_PARAMS is not implemented")
    
    if FRONTEND_PARAMS:
        raise NotImplementedError("FRONTEND_PARAMS is not implemented")
    
    # Build peripheral model graph if COCH_PARAMS is not an empty dictionary
    if COCH_PARAMS:
        raise NotImplementedError("COCH_PARAMS is not implemented")
    else:
        # If COCH_PARAMS is empty, pass input directly to brain network
        assert FRONTEND_PARAMS == {}, "frontend graph is not supported when COCH_PARAMS={}"
        batch_audio_dict = {} # No audio to store when input signal is not a waveform
        # NOTE: 2-dimensional batch_input_signal is assumed to be [batch, time]
        if len(batch_input_signal.shape) == 2:
            # If batch_input_signal has shape [batch, time], reshape to [batch, freq=1, time, channels=1]
            batch_input_signal = batch_input_signal[:, tf.newaxis, :, tf.newaxis]
        assert len(batch_input_signal.shape) > 2, "batch_input_signal must have rank > 2 when COCH_PARAMS={}"
        print('using batch_input_signal as batch_subbands (COCH_PARAMS={}):', batch_input_signal.shape)
        batch_subbands = batch_input_signal
        coch_container = {}
    
    # Ensure that peripheral representation has expected dimensions [batch, height, width, channels]
    while not len(batch_subbands.shape) == 4:
        if len(batch_subbands.shape) < 4:
            # This will add channel dimension if batch_subbands does not already have
            batch_subbands = tf.expand_dims(batch_subbands, axis=-1)
        else:
            # This will remove extra dimensions if batch_subbands has them (throws error if extra dimension is not 1)
            assert batch_subbands.shape[-1] == 1, "Failed to shape peripheral representation in `fga.training_model`"
            batch_subbands = tf.squeeze(batch_subbands, axis=-1)
    print('using batch_subbands with final shape:', batch_subbands.shape)
    
    # Build brain network graph and task losses if BRAIN_PARAMS is not an empty dictionary
    if BRAIN_PARAMS:
        batch_out_dict, brain_container = build_brain_graph(batch_subbands,
                                                            N_CLASSES_DICT,
                                                            **BRAIN_PARAMS,
                                                            var_scope_reuse=False)
        batch_loss, batch_loss_dict = build_task_loss_graph(batch_out_dict,
                                                            batch_labels_dict,
                                                            TASK_LOSS_PARAMS=TASK_LOSS_PARAMS)
    else:
        batch_out_dict = {}
        brain_container = {}
        batch_loss = 0 # Initialize batch_loss to zero (other losses may be added)
        batch_loss_dict = {} # Initialize batch_loss_dict (other losses may be appended)
        
    return batch_loss, batch_loss_dict, batch_out_dict, batch_labels_dict, batch_audio_dict, coch_container, brain_container


def create_parallel_graph(
        training_model,
        input_tensor_dict,
        input_signal_key,
        signal_rate,
        N_CLASSES_DICT,
        TASK_LOSS_PARAMS, 
        FRONTEND_PARAMS,
        COCH_PARAMS,
        BRAIN_PARAMS,
        NORMAL_HEARING_PARAMS={},
        controller="/cpu:0",
        gpu_list=None,
        compute_gradients=False,
        optimizer=None):
    '''
    This function builds the specified `training_model` on multiple GPU towers and
    returns the outputs as dictionaries containing list of tensors. This function
    has been separated from `create_parallel_optimization` so that it can be used
    more easily during either training or evaluation. The returned values are lists
    or dictionaries containing lists with length equal to the number of towers. The
    gradients are only included in the graph if the `compute_gradients` flag is True.
    
    Args
    ----
    training_model (function): builds the whole model graph (defined above)
    input_tensor_dict (dict): dictionary of tensors that contains output of iterator.get_next()
        **NOTE: `input_tensor_dict` is modified in-place: tensors are split into list of tensors**
    input_signal_key (str): key in `input_tensor_dict` that points to the input signal
    signal_rate (int): sampling rate of input signal (Hz)
    N_CLASSES_DICT (dict): dictionary containing the number of output classes, keys are the task paths
    TASK_LOSS_PARAMS (dict): dictionary containing the loss parameters for each task, keys are the task paths
    FRONTEND_PARAMS (dict): passed to training_model
    COCH_PARAMS (dict): passed to training_model
    BRAIN_PARAMS (dict): passed to training_model
    NORMAL_HEARING_PARAMS (dict): contains parameters for the "normal" hearing network, if matching on layer activations
    controller (str): specifies device for averaging/applying the gradients and concatenating tower outputs
    gpu_list (list): if not None, `gpu_list` should contain list of device IDs on which to build graph
    compute_gradients (bool): flag to include gradients in the graph (only set to True for training)
    optimizer (tensorflow optimizer object): used to compute gradients if compute_gradients==True
    
    Returns
    -------
    tower_grads (list): list of gradient tensors across all towers (empty list if compute_gradients==False)
    superbatch_loss (list): list of tensors containing the summed loss across all tasks
    superbatch_loss_dict (dict): dict of lists of tensors (loss for each task and tower)
    superbatch_out_dict (dict): dict of lists of tensors (logits for each task and tower)
    superbatch_labels_dict (dict) : dict of lists of tensors (labels for each task and tower)
    superbatch_audio_dict (dict): dictionary containing input audio and frontend-modified audio tensors (if applicable)
    controls_dict (dict): dictionary containing tensors/functions/features that need to be accessed elsewhere
    '''
    # Acquire list of available devices
    if gpu_list is None:
        gpu_list = get_available_gpus() # function defined below returns list of device IDs
        gpu_list = [x.replace('device:', '') for x in gpu_list] # tf.device() does not recognize '/device:GPU:0' format
    
    # If BRAIN_PARAMS is specified, convert dropout and batchnorm flags to placeholders that can be controlled
    controls_dict = {}
    with tf.device(controller):
        if BRAIN_PARAMS:
            batchnorm_flag = BRAIN_PARAMS.get('batchnorm_flag', True)
            dropout_flag = BRAIN_PARAMS.get('dropout_flag', True)
            BRAIN_PARAMS['batchnorm_flag'] = tf.placeholder(tf.bool, (), 'batchnorm_flag')
            BRAIN_PARAMS['dropout_flag'] = tf.placeholder(tf.bool, (), 'dropout_flag')
            controls_dict['batchnorm_flag'] = batchnorm_flag
            controls_dict['batchnorm_flag_placeholder'] = BRAIN_PARAMS['batchnorm_flag'],
            controls_dict['dropout_flag'] = dropout_flag
            controls_dict['dropout_flag_placeholder'] = BRAIN_PARAMS['dropout_flag']
    
    # Split the input_tensor_dict values into lists of tensors for the individual GPUs
    for key in input_tensor_dict.keys():
        input_tensor_dict[key] = tf.split(input_tensor_dict[key], len(gpu_list))
    
    # Initialize outputs (dictionaries will contain lists of tensors from each GPU)
    tower_grads = [] # List to compile gradients from all GPU towers (if compute_gradients == True)
    superbatch_loss = []
    superbatch_loss_dict, superbatch_out_dict, superbatch_labels_dict, superbatch_audio_dict = {}, {}, {}, {}
    for key in N_CLASSES_DICT.keys():
        superbatch_loss_dict[key], superbatch_out_dict[key], superbatch_labels_dict[key] = [], [], []

    # Initialize losses for the regularization matching layers
    for layer in NORMAL_HEARING_PARAMS.get('coch_layers', {}):
        superbatch_loss_dict['coch_%s_activation_loss'%layer] = []
    for layer in NORMAL_HEARING_PARAMS.get('brain_layers', {}):
        superbatch_loss_dict['brain_%s_activation_loss'%layer] = [] 
    # Get current variable scope so we can reuse variables between GPUs
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE) as outer_scope:
        # Iterate over all of the available GPUs
        for GPU_idx, device_id in enumerate(gpu_list):
            # Use the assign_to_device function to ensure that variables are created on the controller.
            with tf.device(assign_to_device(device_id, controller)), tf.name_scope('tower_{}'.format(GPU_idx)):
                # Grab input signal and task labels for the current GPU
                batch_input_signal = input_tensor_dict[input_signal_key][GPU_idx]
                # If there is a network for regularization, get the input to that network
                if NORMAL_HEARING_PARAMS:
                    normal_input_signal_key = NORMAL_HEARING_PARAMS.get('normal_input_signal_key', None)
                    # If a key is not specified, use the input_signal_key
                    if normal_input_signal_key is None: 
                        normal_input_signal_key = input_signal_key
                    normal_hearing_batch_input_signal = input_tensor_dict[normal_input_signal_key][GPU_idx]
                else:
                    normal_hearing_batch_input_signal = None
                batch_labels_dict = {}
                for key in N_CLASSES_DICT.keys():
                    batch_labels_dict[key] = input_tensor_dict[key][GPU_idx]
                # Build the model on each GPU
                batch_loss, batch_loss_dict, batch_out_dict, batch_labels_dict, batch_audio_dict, coch_container, brain_container = training_model(
                    batch_input_signal, signal_rate, batch_labels_dict, N_CLASSES_DICT, TASK_LOSS_PARAMS, 
                    FRONTEND_PARAMS=FRONTEND_PARAMS, COCH_PARAMS=COCH_PARAMS, BRAIN_PARAMS=BRAIN_PARAMS,
                    NORMAL_HEARING_PARAMS=NORMAL_HEARING_PARAMS, 
                    normal_hearing_batch_input_signal=normal_hearing_batch_input_signal)
                # Store training model tensors/functions/features that need to be acccessed in controls_dict
                if 'load_coch_vars' in coch_container.keys():
                    controls_dict['tower_{}/load_coch_vars'.format(GPU_idx)] = coch_container['load_coch_vars']
                # Compile the output tensors in the superbatch dictionaries to return
                superbatch_loss.append(batch_loss)
                for key in batch_loss_dict.keys():
                    superbatch_loss_dict[key].append(batch_loss_dict[key])
                    # if key is associated with a task, append labels and logits for concatenation
                    if key in batch_out_dict.keys():
                        superbatch_out_dict[key].append(batch_out_dict[key])
                        superbatch_labels_dict[key].append(batch_labels_dict[key])
                for key in batch_audio_dict.keys():
                    if key in superbatch_audio_dict.keys(): superbatch_audio_dict[key].append(batch_audio_dict[key])
                    else: superbatch_audio_dict[key] = [batch_audio_dict[key]]
                # If compute_gradients flag is True, include the gradients for each tower in the graph
                if compute_gradients:
                    with tf.name_scope('compute_gradients'):
                        trainable_vars = []
                        if BRAIN_PARAMS.get('trainable', False):
                            trainable_vars += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='brain_network')
                            print('### Including brain_network variables as trainable')
                            print('### len(trainable_vars) = {}'.format(len(trainable_vars)))
                        if FRONTEND_PARAMS.get('trainable', False):
                            trainable_vars += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='frontend_model')
                            print('### Including frontend_model variables as trainable')
                            print('### len(trainable_vars) = {}'.format(len(trainable_vars)))
                        grads = optimizer.compute_gradients(superbatch_loss[GPU_idx], var_list=trainable_vars)
                        tower_grads.append(grads)
    
    return tower_grads, superbatch_loss, superbatch_loss_dict, superbatch_out_dict, superbatch_labels_dict, superbatch_audio_dict, controls_dict


def create_parallel_optimization(
        training_model,
        optimizer,
        input_tensor_dict,
        input_signal_key,
        signal_rate,
        global_step,
        N_CLASSES_DICT,
        TASK_LOSS_PARAMS,
        FRONTEND_PARAMS,
        COCH_PARAMS,
        BRAIN_PARAMS,
        NORMAL_HEARING_PARAMS={},
        controller="/cpu:0",
        gpu_list=None):
    '''    
    This function builds the multi-tower model and creates the optimization operation.
    Gradients computed on each of the towers are combined and applied on the `controller`.
        
    Args
    ----
    training_model (function): defined above
    optimizer (tensorflow optimizer object): i.e. tf.train.AdamOptimizer
    input_tensor_dict (dict): dictionary of tensors that contains output of iterator.get_next()
    input_signal_key (str): key in `input_tensor_dict` that points to the input signal
    signal_rate (int): sampling rate of input signal (Hz)
    global_step (tensor): the global step for the gradient accumulation.
    N_CLASSES_DICT (dict): dictionary containing the number of output classes, keys are the task paths
    TASK_LOSS_PARAMS (dict): dictionary containing the loss parameters for each task, keys are the task paths
    FRONTEND_PARAMS (dict): passed to training_model
    COCH_PARAMS (dict): passed to training_model
    BRAIN_PARAMS (dict): passed to training_model
    NORMAL_HEARING_PARAMS (dict): contains parameters for the "normal" hearing network, if matching on layer activations
    controller (str): specifies device for averaging/applying the gradients and concatenating tower outputs
    gpu_list (list): if not None, `gpu_list` should contain list of device IDs on which to build graph
    
    Returns
    -------
    apply_gradient_op
    avg_loss (tensor): loss averaged across all tasks and GPU towers
    task_loss_dict (dict of tensors): keys are task paths, values are loss averaged across GPU towers
    task_logits_dict (dict of tensors): keys are task paths, values are logits concatenated across GPU towers
    task_labels_dict (dict of tensors): keys are task paths, values are labels concatenated across GPU towers
    controls_dict (dict): dictionary containing tensors/functions/features that need to be accessed elsewhere
    '''
    # Use the `create_parallel_graph` function to build the multi-tower model and get the list of tower gradients
    # NOTE: the superbatch_audio_dict returned by `create_parallel_graph` is not used for training
    tower_grads, superbatch_loss, superbatch_loss_dict, superbatch_out_dict, superbatch_labels_dict, _, controls_dict = create_parallel_graph(
        training_model, input_tensor_dict, input_signal_key, signal_rate, N_CLASSES_DICT, TASK_LOSS_PARAMS, 
        FRONTEND_PARAMS, COCH_PARAMS, BRAIN_PARAMS, NORMAL_HEARING_PARAMS=NORMAL_HEARING_PARAMS, 
        controller=controller, gpu_list=gpu_list, compute_gradients=True, optimizer=optimizer)
    
    # Initialize dictionaries to collect outputs merged across GPU towers
    task_loss_dict = {} # Will store loss averaged across GPU towers for each task
    task_logits_dict = {} # Will store logits concatenated across GPU towers for each task
    task_labels_dict = {} # Will store labels concatenated across GPU towers for each task
    
    # Apply gradients and concatenate logits/labels on the controlling device
    with tf.name_scope('apply_gradients'), tf.device(controller):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # Collect update ops for all batch normalizations
        with tf.control_dependencies(update_ops): # Ensure batch norm update ops are called once per train step
            gradients = average_gradients(tower_grads)
            apply_gradient_op = optimizer.apply_gradients(gradients, global_step)
            avg_loss = tf.reduce_mean(superbatch_loss) # Mean loss across all tasks and GPU towers
            for task_key in superbatch_loss_dict.keys():
                task_loss_dict[task_key] = tf.reduce_mean(superbatch_loss_dict[task_key])
                if task_key in superbatch_out_dict.keys():
                    task_logits_dict[task_key] = tf.concat(superbatch_out_dict[task_key], 0)
                    task_labels_dict[task_key] = tf.concat(superbatch_labels_dict[task_key], 0)
    
    return apply_gradient_op, avg_loss, task_loss_dict, task_logits_dict, task_labels_dict, controls_dict


def assign_to_device(device, ps_device):
    def _assign(op):
        node_def = op if isinstance(op, tf.NodeDef) else op.node_def
        if node_def.op in PS_OPS: return ps_device
        else: return device
    return _assign


def get_available_gpus():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = [g for g, _ in grad_and_vars]
        grad = tf.reduce_mean(grads, 0)
        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads
