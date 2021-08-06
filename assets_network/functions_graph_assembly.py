import os
import sys
sys.path.append('/code_location/WaveNet-Enhancement')
sys.path.append('/code_location/Wave-U-Net')
os.environ['TF_CUDNN_USE_AUTOTUNE']='0'
import numpy as np
import tensorflow as tf
import glob
import functions_preprocessing
import functions_cochlear_model # TODO: currently contains function `wavenet_logits_to_waveform` (should probably be moved)
import functions_brain_network
import bawn_with_arg_inputs # TODO: move frontend-specific imports and functions to new file (functions_frontend_model.py)
from unet_models import UnetAudioSeparator_ibm
import json
import pdb
import warnings
import copy

# List of ops to locate on controller device when running in Muli-GPU Mode
PS_OPS = ['Variable', 'VariableV2', 'AutoReloadVariable', 'MutableHashTable',
          'MutableHashTableOfTensors', 'MutableDenseHashTable']


# Used for the memory checkpointing
def add_gradient_checkpoints(brain_net, grad_checkpoint_layer_names=['pool']):
    """
    Chooses which tensors to add the gradient checkpoints to. 
    Names here correspond to the names within brain net
    Args
    ----
    brain_net (dict) : contains the tensors constructing the tensorflow graph
    grad_checkpoint_layer_names (list) : contains names to search nets to add to collection. 
    """
    if grad_checkpoint_layer_names is not None:
        check_ops = [brain_net[op] for op in brain_net.keys() if any([grad_op_name in op for grad_op_name in grad_checkpoint_layer_names])]
        print('ADDING OPS TO CHECKPOINTS')
        print(check_ops)
        for op in check_ops:
            tf.add_to_collection('checkpoints', op)


def build_hdf5_iterator():
    '''
    Building iterators that read data from hdf5 datasets will soon be added.
    '''
    pass


def build_tfrecords_iterator(tfrecords_regex, feature_parsing_dict={},
                             iterator_type='one-shot', num_epochs=1, batch_size=8,
                             n_prefetch=10, buffer=1000, shuffle_flag=True,
                             dataset_filter_params={}, dataset_preprocess_params={},
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
    dataset_preprocess_params (dict): kwargs for `build_dataset_preprocess_graph`
    
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
    
    ### Apply preprocessing to dataset if `dataset_preprocess_params` is specified
    if dataset_preprocess_params:
        preprocess_fn = lambda x : functions_preprocessing.build_dataset_preprocess_graph(
            x, **dataset_preprocess_params)
        dataset = dataset.map(preprocess_fn)
    
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


def build_saver(sess, var_list, output_location, restore_model_path=None,
                ckpt_prefix_name='model.ckpt', attempt_load=True):
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


def build_optimizer(combined_batch_size, learning_rate=1e-4, optm_type='adam',
                    learning_rate_decay_type='fixed', decay_rate=0.96,
                    num_decay_samples=697272, adam_epsilon=1e-4):
    """
    Builds the optimizer for training and creates the global_step tensor.
 
    Args:
        combined_batch_size (int) : The batch size used for training, totaled 
            across all GPUs
        learning_rate (float) : The initial learning rate for the optimizer.
        optm_type (string) : The type of optimizer used for for training.
        learning_rate_decay_type (string) : The type of learning rate decay.
        decay_rate (float) : The amount of decay for each step.
        num_decay_samples (int) : The number of samples (exemplars) before
            the learning rate decays. Default is 3x the number of samples 
            in jsinv3.
    Returns: 
        optm (tensorflow optimizer) : Optimization object for training. 
        global_step (tensor) : Global step variable to use for training. 
    Raises:
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

    Args:
        learning_rate (float) : The initial learning rate for the optimizer.
        global_step (tensor) : The global step during training.
        learning_rate_decay_type (string) : The type of learning rate decay.
        decay_steps (int) : The number of steps between each decay.
        decay_rate (float) : The amount of decay for each step.
    Returns:
        learning_rate (tensor) : The (possibly changing) learning rate to use 
            in the optimizer

    Raises:
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
    Args: 
        batch_out_dict (dict of tensor) : logits output by the graph
        batch_labels_dict (dict of tensor) : labels for the input data
        batch_loss_dict (dict of losses) : contains the loss for each task
    Returns:
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
            # TODO: take median r2 value across full batch (rather than take r2 of full batch)
            mean_true, var_true = tf.nn.moments(labels_true, axes=[0, 1])
            mean_pred, var_pred = tf.nn.moments(probs_out, axes=[0, 1])
            r2_numerator = tf.reduce_mean(tf.multiply(tf.subtract(labels_true, mean_true), tf.subtract(probs_out, mean_pred)))
            r2_denominator = tf.multiply(tf.sqrt(var_true), tf.sqrt(var_pred))
            r2 = tf.divide(r2_numerator, r2_denominator)
            # Exactly correct
            exactly_equal = tf.reduce_all(tf.equal(tf.round(probs_out), labels_true), axis=[1])
            average_equal_batch = tf.reduce_mean(tf.cast(exactly_equal, tf.float32), axis=[0])
#             average_equal_batch = tf.print(average_equal_batch, [probs_out, labels_true])
            # TODO: implement in top n
#             print_metric_list = [r2, mean_absolute_value, average_equal_batch] 
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


def build_wavenet_graph(batch_waveform, signal_rate=20000, WAVENET_PARAMS={}):
    '''
    This function builds the graph for the wavenet.
    Args:
        batch_waveform (tensor): input signal waveform
        signal_rate (int): sampling rate of input signal (Hz)
        WAVENET_PARAMS (dict): wavenet configuration parameters
    Returns:
        batch_wavenet_waveform (tensor): wavenet output waveform
    '''
    padding = WAVENET_PARAMS.get('padding', [[0,0], [4093,4093]])
    batch_waveform_zero_padded = tf.pad(batch_waveform, padding)
    logits = bawn_with_arg_inputs.model_simple(batch_waveform_zero_padded, **WAVENET_PARAMS)
    batch_wavenet_waveform = functions_cochlear_model.wavenet_logits_to_waveform(logits, signal_rate=signal_rate)
    return batch_wavenet_waveform


def build_unet_graph(batch_waveform, signal_rate=20000, UNET_PARAMS={}):
    '''
    This function builds the graph for the U-net.

    Args:
        batch_waveform (tensor): input signal waveform
        signal_rate (int): sampling rate of input signal (Hz)
        UNET_PARAMS (dict): U-net configuration parameters
        var_scope_name (str): variable scope name for peripheral model graph
    Returns:
        batch_unet_waveform_squeezed (tensor): U-net output waveform
    '''
    padding = UNET_PARAMS.get('padding', [[0,0], [480,480]])
    batch_waveform_zero_padded = tf.pad(batch_waveform, padding)
    batch_waveform_expanded = tf.expand_dims(batch_waveform_zero_padded,axis=2)
    seperator_class = UnetAudioSeparator_ibm.UnetAudioSeparator(UNET_PARAMS)
    batch_unet_dict = seperator_class.get_output(batch_waveform_expanded,
                                                 training=True,
                                                 return_spectrogram=False,
                                                 reuse=tf.AUTO_REUSE)
    batch_unet_waveform = batch_unet_dict["enhancement"]
    batch_unet_waveform_sliced = batch_unet_waveform[:, padding[1][0]:-padding[1][1],:]
    batch_unet_waveform_squeezed = tf.squeeze(batch_unet_waveform_sliced, axis=2)
    return batch_unet_waveform_squeezed


def build_frontend_graph(batch_waveform, signal_rate=20000, FRONTEND_PARAMS={},
                         var_scope_name='frontend_model'):
    '''
    This function builds the graph for the frontend audio transformation.
    Exactly one frontend_model must be specified (WAVENET or UNET, as of 2019SEP24).
    
    Args:
        batch_waveform (tensor): input signal waveform
        signal_rate (int): sampling rate of input signal (Hz)
        FRONTEND_PARAMS (dict): contains frontend configuration parameters
            The type of frontend model to build is determined by the existence
            of either "WAVENET_PARAMS" or "UNET_PARAMS" in FRONTEND_PARAMS.keys()
        var_scope_name (str): variable scope name for peripheral model graph
    Returns:
        batch_frontend_waveform (tensor): frontend-transformed waveform
    '''
    # TODO: move build_wavenet_graph, build_unet_graph, and future frontends to separate file
    frontend_checklist = [
        "WAVENET_PARAMS" in FRONTEND_PARAMS.keys(),
        "UNET_PARAMS" in FRONTEND_PARAMS.keys(),
    ]
    if not sum(frontend_checklist) == 1:
        raise ValueError(("Exactly one frontend_model must be specified. As specified, "
                          "FRONTEND_PARAMS contains params for {} models:\n {}".format(
                              sum(frontend_checklist), FRONTEND_PARAMS)))
    with tf.variable_scope(var_scope_name):
        if "WAVENET_PARAMS" in FRONTEND_PARAMS.keys():
            batch_frontend_waveform = build_wavenet_graph(batch_waveform, signal_rate=signal_rate,
                                                          WAVENET_PARAMS=FRONTEND_PARAMS["WAVENET_PARAMS"])
        elif "UNET_PARAMS" in FRONTEND_PARAMS.keys():
            batch_frontend_waveform = build_unet_graph(batch_waveform, signal_rate=signal_rate,
                                                       UNET_PARAMS=FRONTEND_PARAMS["UNET_PARAMS"])
        else:
            raise NotImplementedError("Requested frontend_model not implemented")
    return batch_frontend_waveform


def build_coch_graph(batch_input_signal, signal_rate=20000, COCH_PARAMS={},
                     var_scope_name='peripheral_model'):
    '''
    This wrapper function builds the graph for the peripheral model.
    Args:
        batch_input_signal (tensor): input signal for peripheral model
        signal_rate (int): sampling rate of input signal (Hz)
        COCH_PARAMS (dict): peripheral model configuration parameters
        var_scope_name (str): variable scope name for peripheral model graph
    Returns:
        batch_subbands (tensor): output of peripheral model
        coch_container (dict): dictionary of peripheral model tensors
    '''
    with tf.variable_scope(var_scope_name):
        batch_subbands, coch_container = functions_cochlear_model.peripheral_model_graph(batch_input_signal, 
                                                                                         signal_rate=signal_rate,
                                                                                         **COCH_PARAMS)
    return batch_subbands, coch_container


def build_brain_graph(batch_subbands, N_CLASSES_DICT, config,
                      batchnorm_flag=True, dropout_flag=True,
                      save_pckl_path=None, save_arch_path=None,
                      grad_checkpoint_layer_names=['pool'], trainable=True,
                      only_include_layers=None, var_scope_name='brain_network',
                      var_scope_reuse=False, **kwargs):
    '''
    This function builds the graph for the brain network.
    
    Args:
        batch_subbands (tensor): input peripheral representation
        N_CLASSES_DICT (dict): dictionary containing the number of output classes, keys are the task paths
        config (dict or string) : dictionary containing network config, or path to json with same info 
        batchnorm_flag (boolean): if True, batch norm moving averages will update (training mode)
        dropout_flag (boolean): if True, dropout will occur
        save_pckl_path (string) : path where a pickle should be saved containing partial funcs to rebuild graph
        save_arch_path (string) : path where a .json should be saved containing the graph architecture
        grad_checkpoint_layer_names (list) : contains names to search nets to add to collection.
        trainable (boolean): if True, network parameters are trainable
        only_include_layers (set) : if not None, stops building the graph when all of the keys included in this set are constructed
        var_scope_name (str): variable scope name for brain network graph
        var_scope_reuse (str): sets the reuse flag within the variable scope containing the brain network, used to share 
            variables across multiple brain networks that are constructed (ie for multi tower models, or when building 
            the reconstruction loss)
    Returns:
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
        elif 'SpeechDenoisingWithDeepFeatureLosses' in config:
            sys.path.append(config)
            import model
            loss_list = model.lossnet(
                batch_subbands,
                reuse=var_scope_reuse,
                n_layers=14,
                norm_type="SBN",
                base_channels=32,
                blk_channels=5)
            brain_network = {}
            for itr0, loss in enumerate(loss_list):
                brain_network['layer_{}'.format(itr0)] = loss
#             final_layer = {'layer_{}'.format(itr0): loss}
            final_layer = {}
        else:
            raise ValueError("Unrecognized format for brain network config.")

        # Make sure that the final output layer is in a dictionary (necessary to keep track of the task parameters)
        if len(list(N_CLASSES_DICT.keys()))==1 and type(final_layer) is not dict:
            batch_out_dict = {list(N_CLASSES_DICT.keys())[0]: final_layer}
        else:
            batch_out_dict = final_layer

    # Add the gradient checkpointing to the graph. 
    add_gradient_checkpoints(brain_network, grad_checkpoint_layer_names=grad_checkpoint_layer_names)

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
            if not task_weight == 1.0:
                warnings.warn("applying adaptive uncertainty AND manually specified task loss weight")
            # This function will modify batch_loss_dict[task_key] in-place
            warnings.warn(("adaptive uncertainty is still in development, set "
                           "`apply_adaptive_uncertainty=True` at your own risk"))
            batch_loss_dict = apply_adaptive_uncertainty_loss(batch_loss_dict, batch_labels_dict,
                                                              task_key, task_loss_type, **current_task_loss_params)
        
        batch_loss += task_weight * batch_loss_dict[task_key]
    return batch_loss, batch_loss_dict


def training_model(batch_input_signal, signal_rate, batch_labels_dict, N_CLASSES_DICT, TASK_LOSS_PARAMS=None, 
                   FRONTEND_PARAMS={}, COCH_PARAMS={}, BRAIN_PARAMS={}, NORMAL_HEARING_PARAMS={},
                   normal_hearing_batch_input_signal=None):
    '''
    This function constructs the whole graph for training
    (designed to be called by `create_parallel_optimization`)
    Args:
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
    Returns:
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
        batch_audio_dict['normal_input_audio'] = normal_hearing_batch_input_signal
    
    if FRONTEND_PARAMS:
        batch_input_signal = build_frontend_graph(batch_input_signal, signal_rate=signal_rate,
                                                  FRONTEND_PARAMS=FRONTEND_PARAMS)
        batch_audio_dict['frontend_audio'] = batch_input_signal
    
    # Build peripheral model graph if COCH_PARAMS is not an empty dictionary
    if COCH_PARAMS:
        batch_subbands, coch_container = build_coch_graph(batch_input_signal, signal_rate=signal_rate, COCH_PARAMS=COCH_PARAMS)
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
    
    # Include a component of the loss that matches layer activations
    if NORMAL_HEARING_PARAMS:
        trainable_brain_msg = "Regularizing with activations is not supported for trainable brain networks"
        if BRAIN_PARAMS and BRAIN_PARAMS.get('trainable', True):
            raise ValueError(trainable_brain_msg)
        NORMAL_HEARING_PARAMS['COCH_PARAMS'] = copy.deepcopy(COCH_PARAMS)
        coch_key_msg = 'Specifying a key in NORMAL_HEARING_PARAMS[COCH_DIFF] that does not exist in the main COCH_PARAMS.'
        if not set(NORMAL_HEARING_PARAMS['COCH_DIFF'].keys()).issubset(set(COCH_PARAMS.keys())):
            raise ValueError(coch_key_msg)
        NORMAL_HEARING_PARAMS['COCH_PARAMS'].update(NORMAL_HEARING_PARAMS['COCH_DIFF'])
        batch_subbands_normal, coch_container_normal = build_coch_graph(batch_audio_dict['normal_input_audio'],
                                                                        COCH_PARAMS=NORMAL_HEARING_PARAMS['COCH_PARAMS'],
                                                                        signal_rate=signal_rate)
        # If we want to match on any brain activations, build the brain graph
        if NORMAL_HEARING_PARAMS.get('multibrain_layers', {}):
            # TODO(jfeather): make this more natural? However it does match how we are currently configuring the multi brain nets
            raise ValueError('When in single brain mode, must specify brain_layers rather than '
                             'multibrain_layers in NORMAL_HEARING_PARAMS')
        if 'brain_layers' in NORMAL_HEARING_PARAMS:
            brain_params_missing_msg = ("When regularizing with activations,"
                                        " you must specify a brain network"
                                        " via the BRAIN_PARAMS dictionary")
            if not BRAIN_PARAMS:
                raise ValueError(brain_params_missing_msg)
            # We keep the configs for the brain network, but don't save the architecture 
            NORMAL_BRAIN_PARAMS = {}
            NORMAL_BRAIN_PARAMS['config'] = BRAIN_PARAMS['config']
            NORMAL_BRAIN_PARAMS['trainable'] = False
            NORMAL_BRAIN_PARAMS['save_pckl_path'] = None
            NORMAL_BRAIN_PARAMS['save_arch_path'] = None             
            NORMAL_BRAIN_PARAMS['batchnorm_flag'] = BRAIN_PARAMS['batchnorm_flag']
            NORMAL_BRAIN_PARAMS['dropout_flag'] = BRAIN_PARAMS['dropout_flag']
            normal_out_dict, brain_container_normal = build_brain_graph(batch_subbands_normal, N_CLASSES_DICT,
                **NORMAL_BRAIN_PARAMS, grad_checkpoint_layer_names=NORMAL_HEARING_PARAMS['brain_layers'].keys(),
                only_include_layers=set(NORMAL_HEARING_PARAMS['brain_layers'].keys()), var_scope_reuse=True)
        else: 
            brain_container_normal = {}
        
        match_layer_losses = apply_layer_matching_loss(coch_container, coch_container_normal, brain_container,
                                                       brain_container_normal, 
                                                       brain_layers=NORMAL_HEARING_PARAMS.get('brain_layers', None),
                                                       coch_layers=NORMAL_HEARING_PARAMS.get('coch_layers', None),
                                                      )
        
        # Include the weighted loss for each of the matched layers
        if 'coch_layers' in NORMAL_HEARING_PARAMS:
            for layer, layer_values in NORMAL_HEARING_PARAMS['coch_layers'].items():
                batch_loss_dict['coch_%s_activation_loss'%layer] = tf.reduce_mean(match_layer_losses['coch_layers'][layer])
                batch_loss += batch_loss_dict['coch_%s_activation_loss'%layer] * layer_values['weight']
        if 'brain_layers' in NORMAL_HEARING_PARAMS:
            for layer, layer_values in NORMAL_HEARING_PARAMS['brain_layers'].items():
                batch_loss_dict['brain_%s_activation_loss'%layer] = tf.reduce_mean(match_layer_losses['brain_layers'][layer])
                batch_loss += batch_loss_dict['brain_%s_activation_loss'%layer] * layer_values['weight']
        
    return batch_loss, batch_loss_dict, batch_out_dict, batch_labels_dict, batch_audio_dict, coch_container, brain_container


def create_parallel_graph(training_model, input_tensor_dict, input_signal_key, signal_rate, N_CLASSES_DICT, TASK_LOSS_PARAMS, 
                          FRONTEND_PARAMS, COCH_PARAMS, BRAIN_PARAMS, NORMAL_HEARING_PARAMS={}, controller="/cpu:0",
                          gpu_list=None, compute_gradients=False, optimizer=None):
    '''
    This function builds the specified `training_model` on multiple GPU towers and
    returns the outputs as dictionaries containing list of tensors. This function
    has been separated from `create_parallel_optimization` so that it can be used
    more easily during either training or evaluation. The returned values are lists
    or dictionaries containing lists with length equal to the number of towers. The
    gradients are only included in the graph if the `compute_gradients` flag is True.
    
    Args:
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
    Returns:
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


def create_parallel_optimization(training_model, optimizer, input_tensor_dict, input_signal_key, signal_rate,
                                 global_step, N_CLASSES_DICT, TASK_LOSS_PARAMS, FRONTEND_PARAMS, COCH_PARAMS,
                                 BRAIN_PARAMS, NORMAL_HEARING_PARAMS={}, controller="/cpu:0", gpu_list=None):
    '''    
    This function builds the multi-tower model and creates the optimization operation.
    Gradients computed on each of the towers are combined and applied on the `controller`.
        
    Args:
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
    Returns:
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
    
    # TODO?
    # It might be unnecessarily inefficient to concatenate `superbatch_out_dict` and `superbatch_labels_dict`
    # on the `controller` (It's being done here so we can use `task_logits_dict` and `task_labels_dict` as inputs to
    # `runtime_print_metrics`). We could instead compute metrics on the separate towers and merge the outputs.
    # This might save time when running the intermittent validation. To do this here, simply return
    # `superbatch_out_dict` and `superbatch_labels_dict` instead of `task_logits_dict` and `task_labels_dict`.
    return apply_gradient_op, avg_loss, task_loss_dict, task_logits_dict, task_labels_dict, controls_dict


def build_device_to_graph_map(frontend_device_id_list, MULTIBRAIN_PARAMS, available_device_id_list):
    '''
    This helper function creates a dictionary that maps from available devices (keys)
    to graph components (values). A very simple `device_to_graph_map` might resemble:
    {
        0: ['FRONTEND_0', 'COCH_0', 'brain2'],
        1: ['COCH_0', 'brain0', 'brain1'],
    }
    NOTE: 'COCH_#' receives its input from 'FRONTEND_#' on device #.
    
    Args
    ----
    frontend_device_id_list (list of ints): list of devices on which to build frontend graphs
    MULTIBRAIN_PARAMS (dict): parameters for the brain networks, including the devices for each network
    available_device_id_list (list of ints): list of all available GPUs
    
    Returns
    -------
    device_to_graph_map (dict): maps device indexes to lists of graph components
        keys are GPU device IDs (ints)
        values are lists of graph components (lists of strings)
    '''
    device_to_graph_map = {}
    for device_id in available_device_id_list: device_to_graph_map[device_id] = []
    # Add frontend graph(s) to the device map
    for device_id in frontend_device_id_list:
        device_to_graph_map[device_id].append('FRONTEND_{}'.format(device_id))
    # Add brain graph(s) to the device map
    for brain_key in MULTIBRAIN_PARAMS.keys():
        device_id = MULTIBRAIN_PARAMS[brain_key]['device']
        assert isinstance(device_id, int), "MULTIBRAIN device must be an int"
        device_to_graph_map[device_id].append(brain_key)
        # Add cochlear model graph to each device containing a brain graph
        msg = ("device_frontend must be specified for each brain net "
               "if more than one frontend graph is created")
        assert 'device_frontend' in MULTIBRAIN_PARAMS[brain_key] or len(frontend_device_id_list) == 1, msg
        frontend_target_device_id = MULTIBRAIN_PARAMS[brain_key].get('device_frontend', frontend_device_id_list[0])
        coch_with_frontend_device_key = 'COCH_{}'.format(frontend_target_device_id)
        if coch_with_frontend_device_key not in device_to_graph_map[device_id]:
            device_to_graph_map[device_id].append(coch_with_frontend_device_key)
    # Check no device contains duplicate graph components. There must be no more than
    # 1 frontend model and 1 peripheral model per device. Each device's brain network(s)
    # must receive input from a peripheral model on the same device.
    for device_id in device_to_graph_map.keys():
        current_device_graphs = device_to_graph_map[device_id]
        if not len(set(current_device_graphs)) == len(current_device_graphs):
            raise ValueError("INVALID GRAPH: duplicate components on device {}".format(device_id))
        current_device_coch_graphs = [coch for coch in current_device_graphs if 'COCH' in coch]
        msg = ("INVALID GRAPH: requested two cochlear models `{}` by specifying "
               "different frontend devices for brain networks on the same GPU")
        if len(current_device_coch_graphs) > 1:
            raise ValueError(msg.format(device_id))
    return device_to_graph_map


def create_multibrain_parallel_graph(input_tensor_dict, input_signal_key,
                                     signal_rate, N_CLASSES_DICT, TASK_LOSS_PARAMS,
                                     FRONTEND_PARAMS, COCH_PARAMS, BRAIN_PARAMS,
                                     NORMAL_HEARING_PARAMS={}, controller='/cpu:0', 
                                     compute_gradients=False, optimizer=None):
    '''
    Builds the "multibrain" graph, which is designed to train a single frontend model
    using multiple different brain networks at once.
    
    Args
    ----
    input_tensor_dict (dict): dictionary of tensors that contains output of iterator.get_next()
    input_signal_key (str): key in `input_tensor_dict` that points to the input signal
    signal_rate (int): sampling rate of input signal (Hz)
    N_CLASSES_DICT (dict): dictionary containing the number of output classes, keys are the task paths
    TASK_LOSS_PARAMS (dict): dictionary containing the loss parameters for each task, keys are the task paths
    FRONTEND_PARAMS (dict): contains frontend configuration parameters and determines which frontend to build
    COCH_PARAMS (dict): peripheral model configuration parameters
    BRAIN_PARAMS (dict): brain network configuration parameters (must include 'MULTIBRAIN_PARAMS')
    NORMAL_HEARING_PARAMS (dict): contains parameters for the "normal" hearing network, if matching on layer activations
    controller (str): specifies device for averaging/applying the gradients
    compute_gradients (bool): flag to include gradients in the graph (only set to True for training)
    optimizer (tensorflow optimizer object): used to compute gradients if compute_gradients==True
    
    Returns
    -------
    brain_net_grads (list): list of gradient tensors across all brain networks (empty if compute_gradients==False)
    device_to_tensor_map (dict): nested dictionary that maps devices --> graph components --> tensors
        example_device_to_tensor_map = {
            0: {
                'FRONTEND_0': {'waveform_input': tensor, 'waveform_output': tensor},
                'COCH_0': {'subbands': tensor, 'container': ...},
                'brain2': {'task_logits_dict': ..., 'container': ..., etc.},
            },
            1: {
                'COCH_0': {'subbands': tensor, 'container': ...},
                'brain0': {'task_logits_dict': ..., 'container': ..., etc.},
                'brain1': {'task_logits_dict': ..., 'container': ..., etc.},
            },
        }
        NOTE: `...` indicates further nesting (e.g. dictionary of task-specific tensors)
    device_to_graph_map (dict): maps device indexes to lists of graph components
    '''
    assert FRONTEND_PARAMS, "FRONTEND_PARAMS must be non-empty"
    assert BRAIN_PARAMS['MULTIBRAIN_PARAMS'], "MULTIBRAIN_PARAMS must exist and be non-empty"
    MULTIBRAIN_PARAMS = BRAIN_PARAMS['MULTIBRAIN_PARAMS']
    available_device_id_list = [int(x.replace('/device:GPU:', '')) for x in get_available_gpus()]
    
    # Build a dictionary to map device IDs to graph parts
    frontend_device_id_list = FRONTEND_PARAMS.get('device', [0])
    assert isinstance(frontend_device_id_list, list), "FRONTEND_PARAMS['device'] must be a list"
    device_to_graph_map = build_device_to_graph_map(frontend_device_id_list,
                                                    MULTIBRAIN_PARAMS=MULTIBRAIN_PARAMS,
                                                    available_device_id_list=available_device_id_list)
    
    # Initialize a dictionary to map devices to tensors
    device_to_tensor_map = {}
    for device_key in available_device_id_list: device_to_tensor_map[device_key] = {}
    
    # Parse input waveform and labels from input_tensor_dict 
    batch_input_waveform = input_tensor_dict[input_signal_key]
    batch_labels_dict = {}
    for key in N_CLASSES_DICT.keys():
        batch_labels_dict[key] = input_tensor_dict[key]

    # Update the NORMAL_HEARING parameters
    if NORMAL_HEARING_PARAMS:
        NORMAL_HEARING_PARAMS['COCH_PARAMS'] = copy.deepcopy(COCH_PARAMS)
        coch_key_message = 'Specifying a key in NORMAL_HEARING_PARAMS[COCH_DIFF] that does not exist in the main COCH_PARAMS.'
        if not set(NORMAL_HEARING_PARAMS['COCH_DIFF'].keys()).issubset(set(COCH_PARAMS.keys())):
            raise ValueError(coch_key_message)
        NORMAL_HEARING_PARAMS['COCH_PARAMS'].update(NORMAL_HEARING_PARAMS['COCH_DIFF'])
        # If there is a network for regularization, get the input to that network
        normal_input_signal_key = NORMAL_HEARING_PARAMS.get('normal_input_signal_key', None)
        # If a key is not specified, use the input_signal_key
        if normal_input_signal_key is None:
            normal_input_signal_key = input_signal_key
        normal_hearing_batch_input_waveform = input_tensor_dict[normal_input_signal_key]
    else:
        normal_hearing_batch_input_waveform = None
    
    # Build the FRONTEND graph(s)
    frontend_var_scope = 'frontend_model' # frontend_var_scope is shared across devices
    for device_id in device_to_graph_map.keys():
        device = '/GPU:{}'.format(device_id)
        frontend_with_device_key = 'FRONTEND_{}'.format(device_id)
        if frontend_with_device_key in device_to_graph_map[device_id]:
            # Get current variable scope so we can reuse FRONTEND variables between GPUs
            with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE) as outer_scope:
                # Use the assign_to_device function to ensure that variables are created on the controller.
                with tf.device(assign_to_device(device, controller)):
                    # Build FRONTEND graph and store tensors in device_to_tensor_map
                    batch_frontend_waveform = build_frontend_graph(batch_input_waveform,
                                                                   signal_rate=signal_rate,
                                                                   FRONTEND_PARAMS=FRONTEND_PARAMS,
                                                                   var_scope_name=frontend_var_scope)
                    device_to_tensor_map[device_id][frontend_with_device_key] = {
                        'waveform_input': batch_input_waveform,
                        'waveform_output': batch_frontend_waveform,
                        'normal_hearing_waveform_input': normal_hearing_batch_input_waveform
                    }

    # Build the COCHLEAR MODEL graph(s)
    for device_id in device_to_graph_map.keys():
        device = '/GPU:{}'.format(device_id)
        if 'COCH' in '\t'.join(device_to_graph_map[device_id]):
            # Use device_to_graph_map[device] to figure out which FRONTEND output to use as input for cochlea.
            # This is just a simple way to assign cochlear models to frontends
            # and can certainly be extended.
            device_coch_list = [val for val in device_to_graph_map[device_id] if 'COCH' in val]
            msg = "INTERNAL ERROR: duplicate peripheral models on device {}: {}"
            assert len(device_coch_list) == 1, msg.format(device_id, device_coch_list)
            coch_with_frontend_device_key = device_coch_list[0]
            frontend_idx = int(coch_with_frontend_device_key.split('_')[1])
            frontend_with_device_key = 'FRONTEND_{}'.format(frontend_idx)
            waveform_tensor = device_to_tensor_map[frontend_idx][frontend_with_device_key]['waveform_output']
            # Set variable scope and device
            coch_var_scope = 'peripheral_model_device_{}'.format(device_id)
            with tf.device(assign_to_device(device, controller)):
                # Build cochlear model graph using correct FRONTEND output
                batch_subbands, coch_container = build_coch_graph(waveform_tensor,
                                                                  signal_rate=signal_rate,
                                                                  COCH_PARAMS=COCH_PARAMS,
                                                                  var_scope_name=coch_var_scope)
                if len(batch_subbands.shape) == 2:
                    print('expanding dims of batch_subbands from [batch, time] to [batch, 1, time, 1]')
                    batch_subbands = tf.expand_dims(batch_subbands, 1)
                    batch_subbands = tf.expand_dims(batch_subbands, 3)
                print('using batch_subbands with shape:', batch_subbands.shape)
                msg = "batch_subbands must have shape: [batch, height, width, channels]"
                assert len(batch_subbands.shape) == 4, msg
                # If NORMAL_HEARING_PARAMS is specified, add a normal ear to the device. 
                # TODO(jfeather): Currently, a normal hearing ear is built for each device where there is a 
                # training cochleagram built. This is unnecessary computation and instead a normal cochleagram 
                # could just be built on the devices where it is needed (ie one for a cochlealeagram loss and on 
                # other devices only if there are brain layers specified for regularization)
                if NORMAL_HEARING_PARAMS:
                    waveform_input_tensor = device_to_tensor_map[frontend_idx][frontend_with_device_key]['normal_hearing_waveform_input']
                    batch_subbands_normal, coch_container_normal = build_coch_graph(
                        waveform_input_tensor, 
                        COCH_PARAMS=NORMAL_HEARING_PARAMS['COCH_PARAMS'], 
                        signal_rate=signal_rate)
                    if len(batch_subbands_normal.shape) == 2:
                        print('expanding dims of batch_subbands_normal from [batch, time] to [batch, 1, time, 1]')
                        batch_subbands_normal = tf.expand_dims(batch_subbands_normal, 1)
                        batch_subbands_normal = tf.expand_dims(batch_subbands_normal, 3)
                    # The cochlear loss will only be calculated and applied once. BUILD_COCH_LOSS stays true until the cochlear loss is built on the first device that has a cochlea. 
                    BUILD_COCH_LOSS=True
                else:
                    batch_subbands_normal = None
                    coch_container_normal = None
            # Store outputs of cochlear model in device_to_tensor_map
            device_to_tensor_map[device_id][coch_with_frontend_device_key] = {
                'subbands': batch_subbands,
                'container': coch_container,
                'subbands_normal': batch_subbands_normal,
                'container_normal': coch_container_normal
            }

    # Build the BRAIN graph(s)
    brain_net_grads = {} # Dictionary to compile gradients from BRAIN graphs (if compute_gradients == True)
    for device_id in device_to_graph_map.keys():
        device = '/GPU:{}'.format(device_id)
        # All BRAIN graphs on the same device share the same input
        device_coch_list = [val for val in device_to_graph_map[device_id] if 'COCH' in val]
        coch_with_frontend_device_key = device_coch_list[0]
        batch_subbands_tensor = device_to_tensor_map[device_id][coch_with_frontend_device_key]['subbands']
        # Iterate over all BRAIN graphs on the current device and build them using the same subbands
        current_device_brain_key_list = set(device_to_graph_map[device_id]).intersection(
                                        BRAIN_PARAMS['MULTIBRAIN_PARAMS'].keys())
        for brain_key in current_device_brain_key_list:
            # Set variable scope and device
            brain_var_scope = '{}_device_{}'.format(brain_key, device_id)
            with tf.device(assign_to_device(device, controller)):
                # Specify CURRENT_BRAIN_PARAMS and build the graph
                CURRENT_BRAIN_PARAMS = {}
                for key in set(BRAIN_PARAMS.keys()).difference(['MULTIBRAIN_PARAMS']):
                    # Non-architecture-specific parameters are copied from BRAIN_PARAMS
                    CURRENT_BRAIN_PARAMS[key] = BRAIN_PARAMS[key]
                for key in ['config', 'save_arch_path', 'save_pckl_path']:
                    # Architecture-specific parameters live in BRAIN_PARAMS['MULTIBRAIN_PARAMS']
                    CURRENT_BRAIN_PARAMS[key] = MULTIBRAIN_PARAMS[brain_key].get(key, None)
                msg = "BRAIN_PARAMS['trainable'] must be set to False in multibrain mode"
                assert CURRENT_BRAIN_PARAMS.get('trainable', True) == False, msg
                network_task_list = MULTIBRAIN_PARAMS[brain_key].get('task_key_list', [])
                n_classes_dict_subset = {k:N_CLASSES_DICT[k] for k in
                                         set(N_CLASSES_DICT).intersection(network_task_list)}
                batch_out_dict, brain_container = build_brain_graph(batch_subbands_tensor,
                                                                    n_classes_dict_subset,
                                                                    **CURRENT_BRAIN_PARAMS,
                                                                    var_scope_name=brain_var_scope,
                                                                    var_scope_reuse=tf.AUTO_REUSE)
                # Build the TASK LOSSES
                batch_labels_dict_subset = {k:batch_labels_dict[k] for k in
                                            set(batch_labels_dict).intersection(network_task_list)}
                batch_loss, batch_loss_dict = build_task_loss_graph(batch_out_dict,
                                                                    batch_labels_dict_subset,
                                                                    TASK_LOSS_PARAMS)
                # Build layer regularization losses
                # TODO(jfeather): make helper function for the normal hearing construction
                if NORMAL_HEARING_PARAMS:
                    assert CURRENT_BRAIN_PARAMS['trainable']==False, 'Regularizing with activations is only supported for a fixed brain network'
                    # Only build the cochleagram if we haven't constructed the loss yet, or if we need to calculate a brain loss
                    if BUILD_COCH_LOSS or NORMAL_HEARING_PARAMS.get('multibrain_layers', {}).get(brain_key):
                        batch_subbands_tensor_normal = device_to_tensor_map[device_id][coch_with_frontend_device_key]['subbands_normal']
                        coch_container_normal = device_to_tensor_map[device_id][coch_with_frontend_device_key]['container_normal']
                        coch_container = device_to_tensor_map[device_id][coch_with_frontend_device_key]['container']
                    # If we want to match on any brain activations, build the brain graph
                    if NORMAL_HEARING_PARAMS.get('brain_layers', {}):
                        raise ValueError('When in multi brain mode, must specify multibrain_layers rather than '
                                         'brain_layers in NORMAL_HEARING_PARAMS')
                    if 'multibrain_layers' in NORMAL_HEARING_PARAMS:
                        # We keep the configs for the brain network, but don't save the architecture
                        NORMAL_BRAIN_PARAMS = {}
                        NORMAL_BRAIN_PARAMS['config'] = CURRENT_BRAIN_PARAMS['config']
                        NORMAL_BRAIN_PARAMS['trainable'] = False
                        NORMAL_BRAIN_PARAMS['save_pckl_path'] = None
                        NORMAL_BRAIN_PARAMS['save_arch_path'] = None
                        NORMAL_BRAIN_PARAMS['batchnorm_flag'] = CURRENT_BRAIN_PARAMS['batchnorm_flag']
                        NORMAL_BRAIN_PARAMS['dropout_flag'] = CURRENT_BRAIN_PARAMS['dropout_flag']
                        normal_out_dict, brain_container_normal = build_brain_graph(batch_subbands_normal, 
                            N_CLASSES_DICT, **NORMAL_BRAIN_PARAMS, 
                            grad_checkpoint_layer_names=NORMAL_HEARING_PARAMS['multibrain_layers'][brain_key].keys(),
                            only_include_layers=set(NORMAL_HEARING_PARAMS['multibrain_layers'][brain_key].keys()),
                            var_scope_name=brain_var_scope, var_scope_reuse=True)
                    else:
                        brain_container_normal = {}
                    match_layer_losses = apply_layer_matching_loss(coch_container, coch_container_normal, brain_container,
                                                                   brain_container_normal, 
                                                                   brain_layers=NORMAL_HEARING_PARAMS.get('multibrain_layers', {}).get(brain_key, None),
                                                                   coch_layers=NORMAL_HEARING_PARAMS.get('coch_layers', None), 
                                                                  )
                    # Include the weighted loss for each of the matched layers
                    if BUILD_COCH_LOSS:
                        if 'coch_layers' in NORMAL_HEARING_PARAMS:
                            for layer, layer_values in NORMAL_HEARING_PARAMS['coch_layers'].items():
                                batch_loss_dict['coch_%s_activation_loss'%layer] = tf.reduce_mean(match_layer_losses['coch_layers'][layer])
                                batch_loss += batch_loss_dict['coch_%s_activation_loss'%layer] * layer_values['weight']
                            BUILD_COCH_LOSS = False
                    if 'multibrain_layers' in NORMAL_HEARING_PARAMS:
                        for layer, layer_values in NORMAL_HEARING_PARAMS['multibrain_layers'].get(brain_key, {}).items():
                            batch_loss_dict['brain_%s_activation_loss'%layer] = tf.reduce_mean(match_layer_losses['brain_layers'][layer])
                            batch_loss += batch_loss_dict['brain_%s_activation_loss'%layer] * layer_values['weight']

            # Store brain network logits, labels, and losses in device_to_tensor_map
            device_to_tensor_map[device_id][brain_key] = {
                'task_logits_dict': batch_out_dict,
                'container': brain_container,
                'task_labels_dict': batch_labels_dict_subset,
                'task_loss': batch_loss,
                'task_loss_dict': batch_loss_dict,
            }
            # If specified, compute the gradients w.r.t FRONTEND variables for each BRAIN graph
            if compute_gradients:
                assert optimizer is not None, "optimizer is required when compute_gradients == True"
                with tf.name_scope('compute_gradients'):
                    trainable_vars = []
                    if FRONTEND_PARAMS.get('trainable', False):
                        # Find correct scope for frontend variables
                        trainable_vars += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=frontend_var_scope)
                        print('### Including frontend variables as trainable')
                        print('### len(trainable_vars) = {}'.format(len(trainable_vars)))
                    # Attempting to direct gradients to follow same setup as model placement
                    grads = optimizer.compute_gradients(batch_loss, var_list=trainable_vars,
                                                        colocate_gradients_with_ops=True)
                    brain_net_grads[brain_key] = grads
    return brain_net_grads, device_to_tensor_map, device_to_graph_map


def create_multibrain_parallel_optimization(optimizer, input_tensor_dict, input_signal_key,
                                            signal_rate, global_step, N_CLASSES_DICT,
                                            TASK_LOSS_PARAMS, FRONTEND_PARAMS, COCH_PARAMS, 
                                            BRAIN_PARAMS, NORMAL_HEARING_PARAMS={},
                                            controller="/cpu:0"):
    '''
    Builds the multibrain graph and creates the optimization operation for training a
    single frontend model using multiple brain networks at once.
    
    Args
    ----
    optimizer (tensorflow optimizer object): used to compute gradients if compute_gradients==True
    input_tensor_dict (dict): dictionary of tensors that contains output of iterator.get_next()
    input_signal_key (str): key in `input_tensor_dict` that points to the input signal
    signal_rate (int): sampling rate of input signal (Hz)
    global_step (tensor): the global step for the gradient accumulation
    N_CLASSES_DICT (dict): dictionary containing the number of output classes, keys are the task paths
    TASK_LOSS_PARAMS (dict): dictionary containing the loss parameters for each task, keys are the task paths
    FRONTEND_PARAMS (dict): contains frontend configuration parameters and determines which frontend to build
    COCH_PARAMS (dict): peripheral model configuration parameters
    BRAIN_PARAMS (dict): brain network configuration parameters (must include 'MULTIBRAIN_PARAMS')
    NORMAL_HEARING_PARAMS (dict): contains parameters for the "normal" hearing network, if matching on layer activations
    controller (str): specifies device for averaging/applying the gradients
    
    Returns
    -------
    apply_gradient_op (tensorflow op): applies the gradients from the optimizer 
    total_loss (tensor): total loss across all tasks and brain networks with weights applied
    task_loss_dict (dict of tensors): keys are combined brain_network/task paths, values are loss
    task_logits_dict (dict of tensors): keys are combined brain_network/task paths, values are logits
    task_labels_dict (dict of tensors): keys are combined brain_network/task paths, values are labels
    controls_dict (dict): dictionary containing tensors/functions/features that need to be accessed elsewhere
    '''
    # Build the MULTIBRAIN parallel graph (with compute_gradients=True)
    brain_net_grads, device_to_tensor_map, device_to_graph_map = create_multibrain_parallel_graph(
        input_tensor_dict, input_signal_key, signal_rate, N_CLASSES_DICT, TASK_LOSS_PARAMS,
        FRONTEND_PARAMS, COCH_PARAMS, BRAIN_PARAMS, NORMAL_HEARING_PARAMS=NORMAL_HEARING_PARAMS,
        controller=controller, compute_gradients=True, optimizer=optimizer)
    
    # Initialize dictionaries to collect logits, labels, and losses for all BRAIN graphs
    task_logits_dict = {} # Will store logits for each brain network and task
    task_labels_dict = {} # Will store labels for each brain network and task
    task_loss_dict = {} # Will store loss for each brain network and task
    total_loss_dict = {} # Will store the loss for each brain network
    # This block populates those dictionaries with tensors from `device_to_tensor_map`
    for device_id in device_to_tensor_map.keys():
        current_device_brain_key_list = set(device_to_graph_map[device_id]).intersection(
                                        BRAIN_PARAMS['MULTIBRAIN_PARAMS'].keys())
        for brain_key in current_device_brain_key_list:
            brain_key_task_logits_dict = device_to_tensor_map[device_id][brain_key]['task_logits_dict']
            brain_key_task_labels_dict = device_to_tensor_map[device_id][brain_key]['task_labels_dict']
            brain_key_task_loss_dict = device_to_tensor_map[device_id][brain_key]['task_loss_dict']
            brain_key_total_loss = device_to_tensor_map[device_id][brain_key]['task_loss']
            # The three dictionaries above must have identical keys (corresponding to tasks)
            msg = "INTERNAL ERROR: labels and logits keys do not match in `create_multibrain_parallel_optimization`"
            assert set(brain_key_task_logits_dict.keys()) == set(brain_key_task_labels_dict.keys()), msg
            for task_key in brain_key_task_loss_dict.keys():
                brain_device_task_key = '{}_device_{}_{}'.format(brain_key, device_id, task_key)
                # If task corresponds to a network output layer, store logits and labels
                if task_key in brain_key_task_labels_dict.keys():
                    task_logits_dict[brain_device_task_key] = brain_key_task_logits_dict[task_key]
                    task_labels_dict[brain_device_task_key] = brain_key_task_labels_dict[task_key]
                task_loss_dict[brain_device_task_key] = brain_key_task_loss_dict[task_key]
            total_loss_dict['{}_device_{}'.format(brain_key, device_id)] = brain_key_total_loss
    
    # Apply gradients and compute average loss on the controlling device
    with tf.name_scope('apply_gradients'), tf.device(controller):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # Collect update ops for all batch normalizations
        with tf.control_dependencies(update_ops): # Ensure batch norm update ops are called once per train step
            gradients = average_gradients(list(brain_net_grads.values()))
            apply_gradient_op = optimizer.apply_gradients(gradients, global_step)
            total_loss = tf.reduce_sum(list(total_loss_dict.values()))
    
    controls_dict = {}
    for device_key in sorted(device_to_tensor_map.keys()):
        for component_key in sorted(device_to_tensor_map[device_key].keys()):
            for tensor_key in sorted(device_to_tensor_map[device_key][component_key].keys()):
                if isinstance(device_to_tensor_map[device_key][component_key][tensor_key], dict):
                    if 'load_coch_vars' in device_to_tensor_map[device_key][component_key][tensor_key]:
                        lcv_fcn = device_to_tensor_map[device_key][component_key][tensor_key]['load_coch_vars']
                        lcv_key = '{}/{}/{}/{}'.format(device_key, component_key, tensor_key, 'load_coch_vars')
                        controls_dict[lcv_key] = lcv_fcn
    return apply_gradient_op, total_loss, task_loss_dict, task_logits_dict, task_labels_dict, controls_dict


def apply_layer_matching_loss(coch_container, coch_container_normal,
                              brain_container, brain_container_normal,
                              coch_layers=None, brain_layers=None):
    '''
    Calculates a loss that matches activations at a particular layer with the activations given by a separate ("normal") peripheral model
    Args:
        coch_container (dict) : dictionary containing the peripheral model for the graph that is being optimized
        coch_container_normal (dict) : dictionary containing the "normal" peripheral model used for comparison
        brain_container (dict) : dictionary containing the brain network for the graph that is being optimized
        brain_container_normal (dict) : dictionary containing the brain network with input from the "normal" peripheral model
        coch_layers (dict) : keys are the cochleagram layers that will be matched, which each have an associated 'loss_type' and 'weight'
        brain_layers (dict) : keys are the brain layers that will be matched, which each have an associated 'loss_type' and 'weight'
    
    Returns: 
        match_layer_losses (dict) : contains the losses calculated on the specfied layers
    '''
    
    def _layer_loss_function_wrapper(a, b, loss_type=None, threshold_for_mask=None, weight=None):
        '''
        Flexible layer loss function wrapper designed to be called like:
        layer_loss = _layer_loss_function_wrapper(coch_container[layer], 
                                                  coch_container_normal[layer],
                                                  **coch_layers[layer])
        '''
        del weight # Layer loss weights are applied later (outside scope of this function)
        loss_type = loss_type.lower()
        
        if loss_type == 'l2':
            return tf.reduce_sum(tf.square(a-b),
                                 axis=np.arange(1,len(a.get_shape().as_list())))
        elif loss_type == 'l1':
            return tf.reduce_sum(tf.abs(a-b),
                                 axis=np.arange(1,len(a.get_shape().as_list())))
        elif loss_type == 'time_average_l2':
            return tf.reduce_sum(tf.square(tf.reduce_mean(a, axis=2)-tf.reduce_mean(b, axis=2)),
                                 axis=np.arange(1,len(a.get_shape().as_list())-1))
        elif loss_type == 'freq_average_l2':
            return tf.reduce_sum(tf.square(tf.reduce_mean(a, axis=1)-tf.reduce_mean(b, axis=1)),
                                 axis=np.arange(1,len(a.get_shape().as_list())-1))
        elif loss_type == 'threshold_mask_l2_regularization':
            del b # For this regularization, do not use the "normal" matched value
            mask = tf.math.less(a, threshold_for_mask)
            return tf.reduce_sum(tf.square(tf.boolean_mask(a, mask)))
        else:
            msg = 'loss_type=`{}` is not recognized in `_layer_loss_function_wrapper`'
            raise ValueError(msg.format(loss_type))
    
    match_layer_losses = {}
    # Populate match_layer_losses with losses derived from cochlear model layers
    if coch_layers is not None:
        match_layer_losses['coch_layers'] = {}
        for layer in coch_layers.keys():
            match_layer_losses['coch_layers'][layer] = _layer_loss_function_wrapper(coch_container[layer], 
                                                                                    coch_container_normal[layer],
                                                                                    **coch_layers[layer])
    # Populate match_layer_losses with losses derived from brain network layers
    if brain_layers is not None:
        match_layer_losses['brain_layers'] = {}
        for layer in brain_layers.keys():
            match_layer_losses['brain_layers'][layer] = _layer_loss_function_wrapper(brain_container[layer], 
                                                                                     brain_container_normal[layer],
                                                                                     **brain_layers[layer])
    
    return match_layer_losses


def apply_adaptive_uncertainty_loss(batch_loss_dict, batch_labels_dict, task_key, task_loss_type, **kwargs):
    '''
    This function applies adaptive uncertainty weighting for multi-task learning.
    Current implementation is based on Kendall, Gal, and Cipolla (CVPR 2018), and was extended
    by @francl and @jfeather for AudioSet multi-hot classification (by treating each class as
    an independent task with shared uncertainty).
    
    Args
    ----
    batch_loss_dict (dict): dictionary of loss tensors for each task
    batch_labels_dict (dict): dictionary of labels tensors for each task
    task_key (str): key in `batch_loss_dict` that points to loss tensor for the current task
    task_loss_type (str): name of loss function used to compute `batch_loss_dict[task_key]`
    kwargs (dict): necessary to catch all of the `current_task_loss_params` not used in this function
    
    Returns
    -------
    batch_loss_dict (dict): dictionary of loss tensors for each task (`batch_loss_dict[task_key]` is modified)
    '''
    tensor_task_loss = batch_loss_dict[task_key]
    with tf.variable_scope('brain_network'):
        var_name = str(task_key).split('/')[-1] + '_adaptive_uncertainty_log_sigma'
        log_sigma = tf.get_variable(name=var_name, initializer=tf.constant(0.), trainable=True)
    if 'cross_entropy' in task_loss_type: precision = tf.math.exp(-log_sigma)
    else: precision = tf.math.divide(tf.math.exp(-log_sigma), 2.)
    # TODO: find some thing that works better for 3-task networks than this N*log_sigma approach
    if len(batch_labels_dict[task_key].shape) == 1: N = 1
    else: N = int(batch_labels_dict[task_key].shape[-1])
    batch_loss_dict[task_key] = precision * tensor_task_loss + N * log_sigma
    return batch_loss_dict


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
