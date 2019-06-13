import sys
import tensorflow as tf
from tensorflow.python.ops import math_ops
import numpy as np
import glob
import os

sys.path.append('/om2/user/msaddler/phaselocking_pitchnet/netBuilder') # Add Ray's netBuilder to path
import dill # Required to load Ray's pickled networks
from netBuilder import layerGeneratorMotif # Required to load Ray's pickled networks


def OOC_build_net_from_pickle(tensor_input, model_pkl_file, bnorm_training=True, dropout_keep_prob=0.5):
    ''' TEMPORARY UNTIL I CAN LOAD PICKLE FILES FROM INSIDE CONTAINER '''
    x = tensor_input
    num_classes=257
    is_training = bnorm_training
    keep_prob = dropout_keep_prob
    ''' Build the tensorflow graph for network '''
    
    kernel_shapes = [(3,75), (3,25), (5,9), (3,5), (3,3)]
    pooling_sizes = [(1,5), (1,2), (2,2), (2,2), (2,2)]
    batch_norm_flags = [1, 1, 1, 1, 1]
    layer_nfilts = [32, 64, 128, 256, 512]
    
    # Reshape input to use within network
    with tf.variable_scope('input') as varscope:
        y = tf.reshape(x, [-1, x.shape[1].value, x.shape[2].value, 1])
        print(varscope.name, y.shape)
    
    # Build conv-pooling-normalization layers
    for itr0 in range(len(layer_nfilts)):
        with tf.variable_scope('conv{}'.format(itr0 + 1)) as varscope:
            kernel_shape = [kernel_shapes[itr0][0], kernel_shapes[itr0][1],
                            y.shape[3].value, layer_nfilts[itr0]]
            pooling_size = [1, pooling_sizes[itr0][0], pooling_sizes[itr0][1], 1]
            W = tf.Variable(tf.truncated_normal(kernel_shape, stddev = 0.1), name = 'W')
            b = tf.Variable(tf.constant(0.1, shape = [layer_nfilts[itr0]]), name = 'b')
            y = tf.nn.conv2d(y, W, strides = [1, 1, 1, 1], padding = 'SAME', name = 'conv')
            y = tf.nn.max_pool(y, ksize = pooling_size, strides = pooling_size, padding = 'SAME', name = 'max_pool')
            y = tf.nn.relu(y + b, name = 'relu') ### Moved this after max pool (MS 2018.11.05)
            if batch_norm_flags[itr0]:
                y = tf.layers.batch_normalization(y, training = is_training, name = 'batch_norm')
            print(varscope.name, y.shape, W.name, W.shape)
            
    # Fully connected layer
    with tf.variable_scope('fc1') as varscope:
        fc1_channels = y.shape[3].value
        kernel_shape = [y.shape[1].value * y.shape[2].value * y.shape[3].value, fc1_channels]
        W = tf.Variable(tf.truncated_normal(kernel_shape, stddev = 0.1), name = 'W')
        b = tf.Variable(tf.constant(0.1, shape = [fc1_channels]), name = 'b')
        y = tf.reshape(y, [-1, y.shape[1].value * y.shape[2].value * y.shape[3].value])
        y = tf.identity(tf.matmul(y, W) + b, name = 'fc1')
        print(varscope.name, y.shape)
        
    # Dropout layer and readout layer
    with tf.variable_scope('fc2') as varscope:   
        y = tf.nn.dropout(y, keep_prob)
        kernel_shape = [y.shape[1].value, num_classes]
        W = tf.Variable(tf.truncated_normal(kernel_shape, stddev = 0.1), name = 'W')
        b = tf.Variable(tf.constant(0.1, shape = [num_classes]), name = 'b')
        y = tf.matmul(y, W) + b
        print(varscope.name, y.shape)

    return y





def get_f0_bins(f0_min=100.0, f0_max=252.0, binwidth_in_octaves=1/192):
    '''
    Get f0 bins for digitizing f0 values to log-spaced bins.
    
    Args
    ----
    f0_min (float): minimum f0 value, sets reference value for bins
    f0_max (float): maximum f0 value, determines number of bins
    binwidth_in_octaves (float): octaves above f0_min (1/192 = 1/16 semitone bins)
    
    Returns
    -------
    bins (array of float): f0 bins
    '''
    max_octave = np.log2(f0_max / f0_min)
    bins = np.arange(0, max_octave, binwidth_in_octaves)
    bins = f0_min * (np.power(2, bins))
    return bins


def f0_to_bin_labels(f0_tensor, bins):
    return math_ops._bucketize(f0_tensor, bins.tolist())


def bin_labels_to_f0(labels_tensor, bins):
    return tf.gather(bins.tolist(), labels_tensor)


def build_net_from_pickle(tensor_input, model_pkl_file, bnorm_training=True, dropout_keep_prob=0.5):
    net = dill.load(open(model_pkl_file, 'rb'))
    assert len(tensor_input.shape) == 4, 'Batch input shape must be [?, freq, time, channels]'
    tensor_logits, net_ops_list = net(tensor_input, return_net_ops_list=True,
                                     bnorm_training=bnorm_training,
                                     dropout_keep_prob=dropout_keep_prob)
    return tensor_logits


def build_input_iterator(tfrecords_regex, feature_parsing_dict={}, iterator_type='one-shot',
                         num_epochs=1, batch_size=128, n_prefetch=1, buffer=1000):
    '''
    Builds tensorflow iterator for feeding graph with data from tfrecords.
    
    Args
    ----
    tfrecords_regex (str):
    feature_parsing_dict (dict):
    iterator_type (str):
    num_epochs (int):
    batch_size (int):
    n_prefetch (int):
    buffer (int):
    
    Returns
    -------
    iterator (tf iterator object):
    dataset (tf dataset object):
    '''
    
    ### Set up feature_dict to use for parsing tfrecords
    feature_dict = {}
    for path in sorted(feature_parsing_dict.keys()):
        path_dtype = feature_parsing_dict[path]['dtype']
        path_shape = feature_parsing_dict[path].get('shape', ())
        if len(path_shape) > 0: path_dtype = tf.string
        feature_dict[path] = tf.FixedLenFeature([], path_dtype)
    
    ### Define the tfrecords parsing function
    def parse_tfrecord_example(record):
        # Parse the record read by the reader
        parsed_features = tf.parse_single_example(record, features=feature_dict)
        # Decode features and return as a dictionary of tensors
        tensor_dict = {}
        for path in sorted(feature_parsing_dict.keys()):
            path_dtype = feature_parsing_dict[path]['dtype']
            path_shape = feature_parsing_dict[path].get('shape', ())
            if len(path_shape) > 0:
                raw_input = tf.decode_raw(parsed_features[path], path_dtype)
                tensor_dict[path] = tf.reshape(raw_input, path_shape)
            else:
                tensor_dict[path] = parsed_features[path]
        return tensor_dict
    
    ### Create tensorflow dataset
    input_data_filenames = sorted(glob.glob(tfrecords_regex))
    print('### Files found: {}'.format(len(input_data_filenames)))
    print(input_data_filenames[0],'\n...\n', input_data_filenames[-1])
    dataset = tf.data.Dataset.list_files(tfrecords_regex)
    dataset = dataset.apply(tf.data.experimental.parallel_interleave(lambda x:tf.data.TFRecordDataset(x,
                            compression_type="GZIP").map(parse_tfrecord_example, num_parallel_calls=1),
                            cycle_length=10, block_length=16))
    if num_epochs > 1: dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(buffer, num_epochs))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(n_prefetch)
    
    ### Return dataset iterator
    if iterator_type == 'one-shot':
        iterator = dataset.make_one_shot_iterator()
    elif iterator_type == 'initializable':
        iterator = dataset.make_initializable_iterator()
    else:
        assert False, 'iterator type not supported: use one-shot or initializeable'
    return iterator, dataset


def build_pitchnet_graph(model_pkl_file, input_tensor_dict, f0_bin_parameters={},
                         feature_input_path='/meanrates', feature_labels_path='/f0'):
    '''
    This function assembles the network and returns a dictionary of tensors.
    
    Args
    ----
    model_pkl_file (str): filename of pickled function to build network architecture
    input_tensor_dict (dict): dictionary of tensors returned by the iterator
    f0_bin_parameters (dict): dictionary of `get_f0_bins` kwargs
    feature_input_path (str): key in input_tensor_dict that points to network input
    feature_label_path (str): key in input_tensor_dict that points to training labels
        Note: currently labels must be F0 values, which are converted to F0 bin labels
    
    Returns
    -------
    tensors (dict): dictionary of useful tensors (i.e. loss, accuracy, predictions, etc.)
    '''
    
    assert 'f0' in feature_labels_path, 'Only F0-based labels/training is currently supported'
    # TODO: number of classes should be input to net builder, bnorm_training + dropout need to be controlled by variables
    
    bins = get_f0_bins(**f0_bin_parameters)
    tensors = {}
    tensors['input'] = input_tensor_dict[feature_input_path]
    tensors['f0'] = input_tensor_dict[feature_labels_path]
    tensors['labels'] = f0_to_bin_labels(tensors['f0'], bins)
    tensors['logits'] = build_net_from_pickle(tensors['input'], model_pkl_file,
                                              bnorm_training=True, dropout_keep_prob=0.5)
    tensors['loss'] = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tensors['labels'],
                                                                                    logits=tensors['logits']))
    tensors['softmax'] = tf.nn.softmax(tensors['logits'])
    tensors['pred_labels'] = tf.cast(tf.argmax(tensors['logits'], axis=1), tensors['labels'].dtype)
    tensors['pred_f0'] = bin_labels_to_f0(tensors['pred_labels'], bins)
    tensors['correct'] = tf.equal(tensors['labels'], tensors['pred_labels'])
    tensors['accuracy'] = tf.reduce_mean(tf.cast(tensors['correct'], tensors['logits'].dtype))
    return tensors


def build_saver(sess, var_list, output_dir, ckpt_prefix='model.ckpt', ckpt_num=None):
    '''
    This function creates a saver object and attempts to load a checkpoint.
    If no restore_ckpt_num is specified, attempt to load latest checkpoint.
    If no checkpoints are found, no checkpoint is loaded.
    
    Args
    ----
    sess (tf session object)
    var_list (list): list of variables for tf saver object
    output_dir (str): directory for model checkpoints
    ckpt_prefix (str): filename for checkpoints in output_dir
    ckpt_num (int or None): checkpoint number to load (None loads latest)
    
    Returns
    -------
    saver (tf saver object): saver for specified var_list
    ckpt_fn_fmt (str): formattable ckpt filename (ends with `-{}` for ckpt_num)
    ckpt_num (int): loaded ckpt_num (0 if no checkpoint loaded)
    '''
    ckpt_fn_fmt = os.path.join(output_dir, ckpt_prefix + '-{:06}')
    saver = tf.train.Saver(var_list=var_list, max_to_keep=0)
    if not ckpt_num == None:
        print('### Loading variables from specified checkpoint: {}'.format(ckpt_fn_fmt.format(ckpt_num)))
        saver.restore(sess, ckpt_fn_fmt.format(ckpt_num))
    else:
        saved_checkpoints = sorted(glob.glob(ckpt_fn_fmt[:ckpt_fn_fmt.find('-{')]  + '*.index'))
        saved_checkpoint_numbers = [int(s.split('-')[-1].split('.')[0]) for s in saved_checkpoints]
        if len(saved_checkpoint_numbers) > 0:
            ckpt_num = np.max(saved_checkpoint_numbers)
            print('### Loading variables from latest checkpoint: {}'.format(ckpt_fn_fmt.format(ckpt_num)))
            saver.restore(sess, ckpt_fn_fmt.format(ckpt_num))
        else:
            print('### No previous checkpoint found, restarting training: {}'.format(output_dir))
            ckpt_num = 0
    return saver, ckpt_fn_fmt, ckpt_num


def run_validation(sess, valid_feed_dict, valid_init_op, tensors_to_eval={}):
    '''
    This function performs one sweep through the validation dataset.
    
    Args
    ----
    sess (tf session object): active tensorflow session
    valid_feed_dict (dict): feed_dict to pass to sess.run (switch to validation iterator)
    valid_init_op (iterator.intializer op): initializer op for the validation iterator
    tensors_to_eval (dict): dictionary of tensors to evaluate during validation
    
    Returns
    -------
    n_examples (int): keeps count of number of examples in validation epoch
    n_correct (int): keeps count of number of correct predictions
    
    #TODO: save validation metrics to a file?
    #TODO: all sanity checks indicate that batchnorm moving mean/var are not changing
    #TODO: should it be possible to enable/disable dropout?
    #TODO: figure out why validation acc << training batch acc on same data
    '''
    sess.run(valid_init_op) # Validation iterator is an initializeable iterator
    (n_examples, n_correct) = (0, 0)
    while True:
        try:
            evaluated_tensors = sess.run(tensors_to_eval, feed_dict=valid_feed_dict)
            n_examples += len(evaluated_tensors['correct'])
            n_correct += np.sum(evaluated_tensors['correct'])
        except tf.errors.OutOfRangeError:
            break
    return n_examples, n_correct


def training_routine(output_dir, ckpt_prefix='model.ckpt', ckpt_num=None,
                     model_pkl_file='/om/user/msaddler/models_pitch50ms_bez2018/arch160/net.pkl',
                     train_tfrecords_regex=None, valid_tfrecords_regex=None, feature_parsing_dict={},
                     feature_input_path='/meanrates', feature_labels_path='/f0',
                     learning_rate=1e-4, batch_size=128, num_epochs=1, save_step=3750, disp_step=100,
                     f0_bin_parameters={}, random_seed=517, **kwargs):
    
    ### Reset default graph and set random seeds
    tf.reset_default_graph()
    tf.random.set_random_seed(random_seed)
    np.random.seed(random_seed)
    
    ### Build input pipelines
    train_iterator, train_dataset = build_input_iterator(train_tfrecords_regex,
                                        feature_parsing_dict=feature_parsing_dict,
                                        iterator_type='one-shot', num_epochs=num_epochs, batch_size=batch_size)
    valid_iterator, train_dataset = build_input_iterator(valid_tfrecords_regex,
                                        feature_parsing_dict=feature_parsing_dict,
                                        iterator_type='initializable', num_epochs=1, batch_size=batch_size)
    iterator_handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(iterator_handle,
                                                   train_dataset.output_types,
                                                   train_dataset.output_shapes)
    input_tensor_dict = iterator.get_next()
    
    ### Build the model graph and optimizer
    with tf.variable_scope('pitchnet'):
        tensors = build_pitchnet_graph(model_pkl_file, input_tensor_dict,
                                       f0_bin_parameters=f0_bin_parameters,
                                       feature_input_path=feature_input_path,
                                       feature_labels_path=feature_labels_path)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    trainable_vars = tf.trainable_variables(scope='pitchnet')
    with tf.control_dependencies(update_ops + trainable_vars):
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(tensors['loss'])

    ### Start the tensorflow session, initialize graph, prepare iterator handles
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    config = tf.ConfigProto(allow_soft_placement=True, inter_op_parallelism_threads=0, intra_op_parallelism_threads=0)
    run_metadata = tf.RunMetadata()
    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    sess = tf.Session(config=config)
    sess.run(init_op)
    train_handle = sess.run(train_iterator.string_handle())
    valid_handle = sess.run(valid_iterator.string_handle())
    train_feed_dict = {iterator_handle: train_handle}
    valid_feed_dict = {iterator_handle: valid_handle}
    coord = tf.train.Coordinator()
    
    ### Build saver and attempt to restore variables from checkpoint
    pitchnet_globals = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='pitchnet')
    pitchnet_locals = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope='pitchnet')
    saver, ckpt_fn_fmt, ckpt_num = build_saver(sess, pitchnet_globals+pitchnet_locals,
                                               output_dir, ckpt_prefix=ckpt_prefix, ckpt_num=ckpt_num)
    
    ### Main routine loop
    step = ckpt_num
    display_tensors = {
        'train_op': train_op,
        'accuracy': tensors['accuracy'],
        'loss':tensors['loss'],
        'pred_labels': tensors['pred_labels'],
    }
    validation_tensors = {
        'correct': tensors['correct'],
    }
    try:
        while not coord.should_stop():
            if step % save_step == 0:
                print('### SAVING MODEL CHECKPOINT: {}'.format(ckpt_fn_fmt.format(step)))
                save_path = saver.save(sess, ckpt_fn_fmt.format(step), write_meta_graph=False)
                print('### EVALUATING MODEL ON VALIDATION SET ...')
                n_examples, n_correct = run_validation(sess, valid_feed_dict, valid_iterator.initializer,
                                                    tensors_to_eval=validation_tensors)
                print('### VALIDATION SET = {} of {} correct ({}%)'.format(n_correct, n_examples, 100*n_correct/n_examples))
                # TODO: put the validation check in a separate function and ensure batch norm is off (OK)
                # and save outputs to file
                # WHY IS VALIDATION ACCURACY HALF AS HIGH AS TRAINING ACC ON SAME DATA????
            
            if step % disp_step == 0:
                disp_dict = sess.run(display_tensors, feed_dict=train_feed_dict)
                print('# step={:06}, acc={:.3f}, loss={:.3f}'.format(step, disp_dict['accuracy'], disp_dict['loss']))            
            else: sess.run(train_op, feed_dict=train_feed_dict)
            step += 1
    except Exception as e:
        print(e)
        coord.request_stop()
    finally:
        print('### SAVING FINAL MODEL CHECKPOINT: {}'.format(ckpt_fn_fmt.format(step)))
        save_path = saver.save(sess, ckpt_fn_fmt.format(step), write_meta_graph=False)
        
    sess.close()
    tf.reset_default_graph()
    return



if __name__ == "__main__":
    train_tfrecords_regex = '/om/user/msaddler/data_pitch50ms_bez2018/NRTW-jwss_train_CF50-SR70-sp2_filt00_30-90dB_*.tfrecords'
    valid_tfrecords_regex = '/om/user/msaddler/data_pitch50ms_bez2018/NRTW-jwss_valid_CF50-SR70-sp2_filt00_30-90dB_*.tfrecords'
    
    output_dir = '/om/user/msaddler/test_model_f0-arch160-sp2-NRTW-jwss'
    feature_parsing_dict = {
        'meanrates': {'dtype': tf.float32, 'shape':[50, 500, 1]},
        'f0': {'dtype': tf.float32},
    }

    if not os.path.exists(output_dir): os.makedirs(output_dir)
    training_routine(output_dir, ckpt_prefix='model.ckpt', ckpt_num=None,
                                   model_pkl_file='/om/user/msaddler/models_pitch50ms_bez2018/arch160/net.pkl',
                                   train_tfrecords_regex=train_tfrecords_regex,
                                   valid_tfrecords_regex=valid_tfrecords_regex,
                                   feature_parsing_dict=feature_parsing_dict,
                                   feature_input_path='meanrates', feature_labels_path='f0',
                                   learning_rate=1e-4, batch_size=128, num_epochs=50, save_step=3750, disp_step=150,
                                   f0_bin_parameters={}, random_seed=517)