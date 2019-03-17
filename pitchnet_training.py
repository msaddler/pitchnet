import sys
import tensorflow as tf
import numpy as np
import glob
import os
# import dill


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


def build_net_from_pickle(batch_input, model_pkl_file='/om/user/msaddler/models_pitch50ms_bez2018/arch160/net.pkl',
                          bnorm_training=True, dropout_keep_prob=0.5):
    with open(model_pkl_file, 'rb') as file:
        net = dill.load(file)
    assert len(batch_input.shape) == 4, 'Batch input shape must be [?, freq, time, channels]'
    batch_logits, net_ops_list = net(batch_input, return_net_ops_list=True,
                                     bnorm_training=bnorm_training,
                                     dropout_keep_prob=dropout_keep_prob)
    return batch_logits


def build_input_iterator(tfrecords_regex, feature_parsing_dict={}, iterator_type='one-shot',
                         num_epochs=1, batch_size=256, n_prefetch=1, buffer=1000):
    '''
    Builds tensorflow iterator for feeding graph with data from tfrecords.
    
    Args
    ----
    tfrecords_regex (string):
    feature_parsing_dict (dict):
    iterator_type (string):
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
    if num_epochs > 1: dataset = dataset.apply(tf.experimental.data.shuffle_and_repeat(buffer, num_epochs))
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




def train(model_fn, train_data_fn_list, label_path, meanrates_path, num_classes, hyperparameters = {}):
    
    # Training hyperparameters
    learning_rate = hyperparameters.get('learning_rate', 1e-4)
    batch_size = hyperparameters.get('batch_size', 128)
    num_epochs = hyperparameters.get('num_epochs', 1)
    batches_per_epoch = hyperparameters.get('batches_per_epoch', int(480000 / batch_size))

    # Reset default tensorflow graph
    tf.reset_default_graph()
    
    # Assemble and batch training dataset from list of .tfrecords filenames
    print('TRAINING DATA FN LIST:', len(train_data_fn_list))
    print(train_data_fn_list[0])
    print('...')
    print(train_data_fn_list[-1])
    batch_meanrates, batch_labels = assemble_input_pipeline(train_data_fn_list, label_path, meanrates_path,
                                                            batch_size = batch_size, num_epochs = num_epochs)
    
    # Assemble model graph, loss function, and optimizer
    y = build_model(batch_meanrates, num_classes, is_training = True) # Build tf graph
    loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = batch_labels, logits = y)) # Loss function
    pred = tf.cast(tf.argmax(y, axis=1), tf.int64)
    acc = tf.reduce_mean(tf.cast(tf.equal(pred, tf.cast(batch_labels, tf.int64)), tf.float32))
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # <-- required to use tf.layers.batch_normalization
    with tf.control_dependencies(extra_update_ops): # <-- required to use tf.layers.batch_normalization
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss) # Train step
        
    # Training routine
    saver = tf.train.Saver(max_to_keep = 0) # Initialize saver
    disp_step = 50 # Number of batches after which to display batch loss / training acc
    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        if tf.train.checkpoint_exists(model_fn + '*'):
            saved_checkpoints = sorted(glob.glob(model_fn + '*.index'))
            saved_checkpoint_numbers = []
            for s in saved_checkpoints:
                s = s.replace(model_fn + '-', '')
                s = s.replace('.index', '')
                saved_checkpoint_numbers.append(int(s))
            current_epoch = max(saved_checkpoint_numbers)
            assert(current_epoch < num_epochs)
            restore_fn = model_fn + '-{}'.format(current_epoch)
            saver.restore(sess, restore_fn)
            print('... [LOADING]: {}'.format(restore_fn))
        else:
            print('... NO saved checkpoint found: {}'.format(model_fn))
            current_epoch = 0
            
        batch_count = current_epoch * batches_per_epoch
        while not coord.should_stop(): 
            if batch_count % batches_per_epoch == 0: # SAVE MODEL CHECKPOINT
                current_epoch = int(np.floor(batch_count / batches_per_epoch))
                save_path = saver.save(sess, model_fn, global_step = current_epoch)
                print('... [SAVED]: {}'.format(save_path))
            try: # EXECUTE TRAINING STEP
                if batch_count % disp_step == 0:
                    [tmp_loss, tmp_pred, tmp_acc, tmp_labels, _] = sess.run([loss, pred, acc, batch_labels, train_step])
                    tmp = np.abs(tmp_pred - tmp_labels)
                    print('  | step: {:^8} | loss: {:^8.1f} | acc: {:^8.4f}'.format(batch_count, tmp_loss, tmp_acc))
                    print('__| ({},{},{},{}) of {} within (0,1,5,10) classes of true class'.format(
                        np.sum(tmp <= 0), np.sum(tmp <= 1), np.sum(tmp <= 5), np.sum(tmp <= 10), batch_size))
                    print('...', tmp_pred[0:16])
                else:
                    sess.run(train_step)
                batch_count += 1
            except tf.errors.OutOfRangeError:
                coord.request_stop()
                
        coord.request_stop()
        coord.join(threads)
        current_epoch = int(np.floor(batch_count / batches_per_epoch))
        save_path = saver.save(sess, model_fn, global_step = current_epoch)
        print('... [SAVED]: {}'.format(save_path))
        sess.close()


if __name__ == "__main__":
    # SET UP SCRIPT TO RUN FROM COMMAND LINE

    if not len(sys.argv) == 2:
        print('COMMAND LINE USAGE: run <script_name.py> <job_id>')
        assert(False)

    # Filenames for saving tensorflow models (.ckpt files)
    list_model_fn = [
        # 0-2 Species 2
        '/om/user/msaddler/models_pitch50ms_bez2018/arch160/NRTW-jwss_train_CF50-SR70-sp2_filt00_30-90dB_meanrates0.ckpt',
        '/om/user/msaddler/models_pitch50ms_bez2018/arch160/NRTW-jwss_train_CF50-SR70-sp2_filt00_30-90dB_meanrates1.ckpt',
        '/om/user/msaddler/models_pitch50ms_bez2018/arch160/NRTW-jwss_train_CF50-SR70-sp2_filt00_30-90dB_meanrates2.ckpt',
        # 3-5 Species 1
        '/om/user/msaddler/models_pitch50ms_bez2018/arch160/NRTW-jwss_train_CF50-SR70-sp1_filt00_30-90dB_meanrates0.ckpt',
        '/om/user/msaddler/models_pitch50ms_bez2018/arch160/NRTW-jwss_train_CF50-SR70-sp1_filt00_30-90dB_meanrates1.ckpt',
        '/om/user/msaddler/models_pitch50ms_bez2018/arch160/NRTW-jwss_train_CF50-SR70-sp1_filt00_30-90dB_meanrates2.ckpt',
        # 6-8 Species 2, cohc00
        '/om/user/msaddler/models_pitch50ms_bez2018/arch160/NRTW-jwss_train_CF50-SR70-sp2-cohc00_filt00_30-90dB_meanrates0.ckpt',
        '/om/user/msaddler/models_pitch50ms_bez2018/arch160/NRTW-jwss_train_CF50-SR70-sp2-cohc00_filt00_30-90dB_meanrates1.ckpt',
        '/om/user/msaddler/models_pitch50ms_bez2018/arch160/NRTW-jwss_train_CF50-SR70-sp2-cohc00_filt00_30-90dB_meanrates2.ckpt',
    ]

    # Filenames for hdf5 training datasets
    list_data_regex = [
        # 0-2 Species 2
        '/om/user/msaddler/data_pitch50ms_bez2018/NRTW-jwss_train_CF50-SR70-sp2_filt00_30-90dB_*-*.tfrecords',
        '/om/user/msaddler/data_pitch50ms_bez2018/NRTW-jwss_train_CF50-SR70-sp2_filt00_30-90dB_*-*.tfrecords',
        '/om/user/msaddler/data_pitch50ms_bez2018/NRTW-jwss_train_CF50-SR70-sp2_filt00_30-90dB_*-*.tfrecords',
        # 3-5 Species 1
        '/om/user/msaddler/data_pitch50ms_bez2018/NRTW-jwss_train_CF50-SR70-sp1_filt00_30-90dB_*-*.tfrecords',
        '/om/user/msaddler/data_pitch50ms_bez2018/NRTW-jwss_train_CF50-SR70-sp1_filt00_30-90dB_*-*.tfrecords',
        '/om/user/msaddler/data_pitch50ms_bez2018/NRTW-jwss_train_CF50-SR70-sp1_filt00_30-90dB_*-*.tfrecords',
        # 6-8 Species 2, cohc00
        '/om/user/msaddler/data_pitch50ms_bez2018/NRTW-jwss_train_CF50-SR70-sp2-cohc00_filt00_30-90dB_*-*.tfrecords',
        '/om/user/msaddler/data_pitch50ms_bez2018/NRTW-jwss_train_CF50-SR70-sp2-cohc00_filt00_30-90dB_*-*.tfrecords',
        '/om/user/msaddler/data_pitch50ms_bez2018/NRTW-jwss_train_CF50-SR70-sp2-cohc00_filt00_30-90dB_*-*.tfrecords',
    ]

    # Use job_id from command line argument to select filenames, dataset, and inputs
    job_id = int(sys.argv[1])
    model_fn = list_model_fn[job_id]
    data_regex = list_data_regex[job_id]
    train_data_fn_list = sorted(glob.glob(data_regex))

    label_path = 'labels'
    num_classes = 257
    meanrates_path = 'meanrates'

    hyperparameters = {'learning_rate':1e-4, 'batch_size':128, 'num_epochs':40}

    print('### [START] model_fn:', model_fn)
    print('### [START] data_regex:', data_regex)
    print('### [START] model_input_path:', meanrates_path)
    print('### [START] labels_path:', label_path, num_classes)
    train(model_fn, train_data_fn_list, label_path, meanrates_path, num_classes,
          hyperparameters = hyperparameters)
    print('### [END] model_fn:', model_fn)