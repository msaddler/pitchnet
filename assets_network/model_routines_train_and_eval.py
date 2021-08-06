import os
import sys
import pdb
import glob
import json
import time
import re
import warnings
import numpy as np

os.environ['TF_CUDNN_USE_AUTOTUNE']='0'
import tensorflow as tf

import functions_graph_assembly as fga
import functions_evaluation


def run_training_routine(
        train_regex,
        num_epochs=1,
        batch_size=8,
        display_step=5,
        save_step=10000,
        output_directory="/saved_models/example_arch",
        brain_net_ckpt_to_load=None,
        frontend_ckpt_to_load=None,
        controller="/cpu:0",
        iterator_device="/cpu:0",
        max_runtime=None,
        random_seed=517,
        signal_rate=20000,
        TASK_LOSS_PARAMS={},
        N_CLASSES_DICT={},
        ITERATOR_PARAMS={},
        FRONTEND_PARAMS={},
        COCH_PARAMS={},
        BRAIN_PARAMS={},
        NORMAL_HEARING_PARAMS={},
        OPTM_PARAMS={},
        valid_regex=None,
        valid_step=10000, 
        valid_display_step=100,
        early_stopping_metrics=None, 
        early_stopping_baselines=None,
        load_iterator=False,
        save_iterator=True, 
        **kwargs):
    '''
    This function runs the multi-tower training routine (brain network)
    
    Args
    ----
    train_regex (str): regex that globs .tfrecords files for training dataset
    num_epochs (int): number of times to repeat the dataset
    batch_size (int): number of examples per batch per GPU
    display_step (int): print out training info every display_step steps
    save_step (int): checkpoint trainable variables every save_step steps
    output_directory (str): location to save the new checkpoints model parameters
    brain_net_ckpt_to_load (str): path to brain_network .ckpt-# to load, None starts training from most recent checkpoint
    frontend_ckpt_to_load (str): path to frontend .ckpt-# to load, None starts training from most recent checkpoint
    controller (str): device to save the variables and consolidate gradients (GPU is more efficient on summit)
    iterator_device (str): device that hosts the input iterator (to use tf.Dataset API, must be a CPU)
    max_runtime (int): maximum time (in seconds) to run training before final checkpoint (no limit if set to None or 0)
    random_seed (int): random seed to set tensorflow and numpy
    signal_rate (int): sampling rate of input signal (Hz)
    TASK_LOSS_PARAMS (dict): dictionary containing the loss parameters for each task, keys are the task paths
    N_CLASSES_DICT (dict): dictionary specifying number of output classes for each task
    ITERATOR_PARAMS (dict): parameters for building the input data iterator
    FRONTEND_PARAMS (dict): parameters for building the frontend model graph
    COCH_PARAMS (dict): parameters for building the cochlear model
    BRAIN_PARAMS (dict): parameters for building the brain network
    NORMAL_HEARING_PARAMS (dict): contains parameters for the "normal" hearing network, if matching on layer activations
    valid_regex (str): regex that globs .tfrecords files for validation dataset (if None, no validation)
    valid_step (int): number of training steps after which to run validation procedure (if <= 0, no validation)
    valid_display (int): print out validation procedure info every valid_display_step steps
    early_stopping_metrics (dict): metric name and minimum delta pairs for early stopping (see functions_evaluation.py)
    early_stopping_baselines (dict): baseline values for the early stopping metrics to reach (see functions_evaluation.py)
    load_iterator (bool): set to False to prevent training routine from loading iterator checkpoint
    save_iterator (bool): set to False to prevent training routine from building iterator saver (cant save or load iterator)
    '''
    ### RESET DEFAULT GRAPH AND SET RANDOM SEEDS ###
    tf.reset_default_graph()
    tf.random.set_random_seed(random_seed)
    np.random.seed(random_seed)

    num_towers = len(fga.get_available_gpus()) # Used to scale the batch_size for iterator
    
    if BRAIN_PARAMS.get('MULTIBRAIN_PARAMS', {}):
        raise NotImplementedError("MULTIBRAIN_PARAMS is not implemented")    
    if FRONTEND_PARAMS:
        raise NotImplementedError("FRONTEND_PARAMS is not implemented")

    ### DECIDE WHETHER OR NOT TO RUN INTERMITTENT VALIDATION ###
    valid_flag = (valid_regex is not None) and (valid_step > 0)

    ### BUILD THE INPUT PIPELINE ###
    print('\n\n\n >>> building input iterator(s) <<< \n\n\n', flush=True)
    with tf.device(iterator_device):
        # Build the one-shot training iterator
        train_iter, train_dset, train_iter_save_obj = fga.build_tfrecords_iterator(
            train_regex,
            num_epochs=num_epochs,
            batch_size=batch_size*num_towers,
            iterator_type='one-shot',
            **ITERATOR_PARAMS)
        if valid_flag:
            # Build the initializable validation iterator
            valid_iter, valid_dset, valid_iter_save_obj = fga.build_tfrecords_iterator(
                valid_regex,
                num_epochs=1,
                shuffle_flag=False,
                batch_size=batch_size*num_towers,
                iterator_type='initializable',
                **ITERATOR_PARAMS)

        # Define the iterator_handle placeholder
        iterator_handle = tf.placeholder(tf.string, shape=[])
        train_iter_handle = train_iter.string_handle()
        if valid_flag: valid_iter_handle = valid_iter.string_handle()

        # Build a feedable iterator that takes in an iterator_handle placeholder 
        iterator = tf.data.Iterator.from_string_handle(iterator_handle,
                                                       train_iter.output_types,
                                                       train_iter.output_shapes)
        
        # Input tensor dict containing batch_size * num_towers examples:
        input_tensor_dict = iterator.get_next()

    ### BUILD GRAPH + OPTIMIZER ###
    print('\n\n\n >>> building graph and parallel optimizer <<< \n\n\n', flush=True)    

    # Define the tensorflow optimizer
    optimizer, global_step = fga.build_optimizer(batch_size*num_towers, **OPTM_PARAMS)
 
    # Get path for the model input signal in input_tensor_dict
    feature_signal_path = ITERATOR_PARAMS.get('feature_signal_path', 'stimuli/signal')
    
    # Build the parallel optimizer
    update_grads, batch_loss, batch_loss_dict, batch_out_dict, batch_labels_dict, controls_dict = fga.create_parallel_optimization(
        fga.training_model,
        optimizer,
        input_tensor_dict,
        feature_signal_path,
        signal_rate,
        global_step,
        N_CLASSES_DICT,
        TASK_LOSS_PARAMS,
        FRONTEND_PARAMS,
        COCH_PARAMS,
        BRAIN_PARAMS,
        NORMAL_HEARING_PARAMS=NORMAL_HEARING_PARAMS,
        controller=controller)
    
    # Build evaluation metrics if running intermittent validation
    if valid_flag:
        with tf.name_scope('validation'):
            all_evaluation_measures_dict = functions_evaluation.make_task_evaluation_metrics(
                batch_out_dict, batch_labels_dict, batch_loss_dict,
                batch_audio_dict={}, TASK_LOSS_PARAMS=TASK_LOSS_PARAMS)
            valid_vars = [v for v in tf.local_variables() + tf.global_variables() if 'validation/' in v.name]
            valid_vars_initializer = tf.initializers.variables(valid_vars)
            if early_stopping_metrics:
                validation_metric_cutoffs = functions_evaluation.EarlyStopping(
                    list(early_stopping_metrics.keys()),
                    min_delta=early_stopping_metrics,
                    patience=2,
                    baseline=early_stopping_baselines)
        # Setup valid_save_dict for saving validation metrics to a JSON file in the output directory
        valid_metric_filename = os.path.join(output_directory, 'validation_metrics.json')
        if os.path.isfile(valid_metric_filename):
            # If file exists, load the valid_save_dict and continue adding values
            print('loading existing valid save file: {}'.format(valid_metric_filename))
            with open(valid_metric_filename, 'r') as vsf: valid_save_dict = json.load(vsf)
        else:
            # If no file exists, initialize valid_save_dict with empty lists
            valid_save_dict = {'output_directory':output_directory, 'step':[]}
            for key in all_evaluation_measures_dict.keys():
                if (len(all_evaluation_measures_dict[key].shape) == 0) and ('update_op' not in key):
                    valid_save_dict[key] = []

    ### START SESSION AND INITIALIZE GRAPH ###
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    config = tf.ConfigProto(allow_soft_placement=True,
                            inter_op_parallelism_threads=0,
                            intra_op_parallelism_threads=0)
    run_metadata = tf.RunMetadata()
    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    sess = tf.Session(config=config)
    sess.run(init_op)

    ### FEEDABLE ITERATOR HANDLING ###
    train_feed_dict = {iterator_handle: sess.run(train_iter_handle)} # train_iter_handle.device returns `controller`
    if 'batchnorm_flag_placeholder' in controls_dict.keys():
        train_feed_dict[controls_dict['batchnorm_flag_placeholder']] = controls_dict['batchnorm_flag']
    if 'dropout_flag_placeholder' in controls_dict.keys():
        train_feed_dict[controls_dict['dropout_flag_placeholder']] = controls_dict['dropout_flag']
    
    if valid_flag:
        valid_feed_dict = {iterator_handle: sess.run(valid_iter_handle)}
        if 'batchnorm_flag_placeholder' in controls_dict.keys():
            valid_feed_dict[controls_dict['batchnorm_flag_placeholder']] = False
        if 'dropout_flag_placeholder' in controls_dict.keys():
            valid_feed_dict[controls_dict['dropout_flag_placeholder']] = False   

    ### BUILD SAVERS AND ATTEMPT TO LOAD PREVIOUS CHECKPOINTS ###
    print('\n\n\n >>> building savers and attempting to load vars <<< \n\n\n', flush=True)
    
    # If `load_coch_vars` function(s) exist(s) in the controls_dict, call within active tf session
    for control_key in sorted(controls_dict.keys()):
        if 'load_coch_vars' in control_key:
            controls_dict[control_key](session=sess)

    # Build saver for the training iterator (attempts load if checkpoint exists)
    iterator_ckpt = 0
    if save_iterator:
        saver_iterator, out_ckpt_loc_iterator, iterator_ckpt = fga.build_saver(sess, [train_iter_save_obj],
                                                                               output_directory, restore_model_path=None,
                                                                               ckpt_prefix_name='iterator.ckpt',
                                                                               attempt_load=load_iterator)
        iter_load_warning = ('Loading an iterator checkpoint also loads the checkpointed iterator batch size '
                             '(equal to batch_size * num_towers). Thus if you have changed the '
                             'number of GPUs, you should not be loading the same iterator.')
        warnings.warn(iter_load_warning)

    # Build saver for the brain_network(s) (attempt load if checkpoints exist)
    trainable_brain = BRAIN_PARAMS.get('trainable', False)
    if BRAIN_PARAMS:
        brain_var_scope = 'brain_network'
        # Build saver for single brain network (capable of saving/loading)
        brain_ckpt_prefix_name = BRAIN_PARAMS.get('save_ckpt_path', 'brain_model.ckpt')
        brain_globals = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=brain_var_scope)
        brain_locals = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope=brain_var_scope)
        brain_variables =  brain_globals + brain_locals
        if trainable_brain:
            # Since brain network is trainable, checkpoints must be loaded/saved in output_directory
            saver_brain_net, out_ckpt_loc_brain_net, brain_net_ckpt = fga.build_saver(
                sess, brain_variables, output_directory, restore_model_path=brain_net_ckpt_to_load,
                ckpt_prefix_name=brain_ckpt_prefix_name)
            # Reset the checkpoint value to 0 so that the saver will be synced with the iterator
            if not brain_net_ckpt_to_load:
                brain_net_ckpt_offset = brain_net_ckpt - iterator_ckpt
            else:
                brain_net_ckpt_offset = brain_net_ckpt
            print('brain_net_ckpt:', brain_net_ckpt)
            print('brain_net_ckpt_offset:', brain_net_ckpt_offset)
        else:
            # Since brain network is not trainable, it can be loaded from outside output_directory
            if os.path.basename(brain_ckpt_prefix_name) == brain_ckpt_prefix_name:
                brain_net_dir = output_directory
            else:
                brain_net_dir = os.path.dirname(brain_ckpt_prefix_name)
            # Since brain network is not trainable, values returned by fga.build_saver() are never used
            _saver_brain_net, _out_ckpt_loc_brain_net, _brain_net_ckpt = fga.build_saver(
                sess, brain_variables, brain_net_dir,
                restore_model_path=brain_net_ckpt_to_load,
                ckpt_prefix_name=os.path.basename(brain_ckpt_prefix_name))
    
    ### MAIN TRAINING LOOP ###
    print('\n\n\n >>> begin training routine <<< \n\n\n', flush=True)
    step = iterator_ckpt # set the step value to the iterator ckpt number
    # Assign the loaded iterator step to the global step used for the optimization
    assign_global_step = global_step.assign(step)
    sess.run(assign_global_step) 
    errors_count = 0
    disp_str, print_metric_list = fga.runtime_print_metrics(batch_out_dict, batch_labels_dict, batch_loss_dict)
    start_time = time.time()
    all_print_out = sess.run([batch_loss] + print_metric_list, feed_dict=train_feed_dict)
    print(disp_str.format(step,step*batch_size*num_towers,(time.time()-start_time),*all_print_out), flush=True)

    try:
        while True: # Training procedure will run until this infinite loop is broken

            # ====== Save step (gradients should not be updated) ======
            if (step % save_step == 0):
                # Checkpoint the iterator
                if save_iterator:
                    print("### Checkpointing iterator (step {})...".format(step), flush=True)
                    saver_iterator.save(sess, out_ckpt_loc_iterator, global_step=step, write_meta_graph=False)
                # Checkpoint brain_network (if path is specified and brain_net is trainable)
                if trainable_brain:
                    print("### Checkpointing brain_network (step {})...".format(step+brain_net_ckpt_offset), flush=True)
                    saver_brain_net.save(sess, out_ckpt_loc_brain_net, global_step=step+brain_net_ckpt_offset,
                                         write_meta_graph=False)

            # ====== Display training step (gradients should be updated) ======
            if step % display_step == 0:
                all_print_out = sess.run([batch_loss] + print_metric_list + [update_grads], feed_dict=train_feed_dict)
                print(disp_str.format(step, step*batch_size*num_towers, (time.time()-start_time), *all_print_out[:-1]),
                      flush=True) # don't print the grad
                if sig_handler.should_exit(): raise SystemExit("Approaching job scheduler limit!")

            # ====== No-display training step (gradients should be updated) ======
            else: 
                sess.run(update_grads, feed_dict=train_feed_dict) # Training step

            # ====== Validation step (gradients should not be updated) ======
            if (valid_flag) and (step % valid_step == 0) and (step != 0):
                print('\n\n ====== START OF VALIDATION CHECK ====== \n\n')
                # TODO: Compute output metrics on the different GPUs, rather than on controller?
                # TODO: Currently, network logits are being concatenated on the CPU before computing eval metrics
                vstep = 0
                sess.run([valid_iter.initializer, valid_vars_initializer]) # Re-initialize valid iterator and metrics
                while True: # Validation procedure will run until this infinite loop is broken
                    try:
                        # TODO: clean up this code and decide what the minimal set of tensors to evaluate are
                        evaluated_batch = sess.run(all_evaluation_measures_dict, feed_dict=valid_feed_dict)
                        if (valid_display_step is not None) and (vstep % valid_display_step == 0):
                            print('=== validation step {}'.format(vstep))
                            for batch_labels_key in batch_labels_dict.keys():
                                for name in list(all_evaluation_measures_dict.keys()):
                                    if ((batch_labels_key in name) and ('_update_op' not in name)
                                         and (':probs_out' not in name) and ('_audio' not in name)):
                                        print('    {}: {}'.format(name, evaluated_batch[name]))

                    except (tf.errors.OutOfRangeError, tf.errors.InvalidArgumentError) as e:
                        # NOTE: InvalidArgumentError occurs when trying to unevenly split the last batch across GPUs
                        print('=== end of validation check: saving/printing summary metrics')
                        if trainable_brain:
                            valid_save_dict['step'].append(int(step + brain_net_ckpt_offset))
                        else:
                            valid_save_dict['step'].append(int(step))
                        for key in set(valid_save_dict.keys()).intersection(evaluated_batch.keys()):
                            valid_save_dict[key].append(float(evaluated_batch[key]))
                        with open(valid_metric_filename, 'w') as vsf: json.dump(valid_save_dict, vsf)
                        print('Step {} validation metrics were appended to {}'.format(step, valid_metric_filename))
                        print_str = '| step={:06} | '
                        print_str_key_list = sorted(set(valid_save_dict.keys()).intersection(evaluated_batch.keys()))
                        for key in print_str_key_list: print_str += key + '={:04f} | '
                        for itrV in range(len(valid_save_dict['step'])):
                            print_str_vals = [valid_save_dict['step'][itrV]]
                            for key in print_str_key_list: print_str_vals.append(valid_save_dict[key][itrV])
                            print(print_str.format(*print_str_vals))
                        break
                    vstep += 1
                if early_stopping_metrics:
                    monitored_validation_metrics = {k:evaluated_batch[k] 
                                                    for k in
                                                    early_stopping_metrics.keys()}
                    stop_training, triggering_metric = validation_metric_cutoffs.early_stopping_check(monitored_validation_metrics)
                    if stop_training:
                        print('Validation Metric: {} triggered early stopping. Training aborted.'.format(triggering_metric))
                        errors_count += 1
                        break
                print('\n\n ====== END OF VALIDATION CHECK ====== \n\n')

            # If the time exceeds the max runtime, save an epoch and exit. 
            if max_runtime and (time.time()-start_time > max_runtime): 
                print('Time Exceeded Max Runtime of {} seconds'.format(max_runtime))
                errors_count += 1
                break
            
            # Update the step count
            step += 1
            
    except tf.errors.OutOfRangeError:
        print("Out of Range Error. Optimization Finished.", flush=True)
    except tf.errors.DataLossError as e:
        print("Corrupted file found!", flush=True)
        return False
    except tf.errors.ResourceExhaustedError as e:
        print("Out of memory error.", flush=True)
        with open(os.path.join(output_directory, "OOM_error.stderr"), "w+") as f:
            f.write("Arch caused OOM error")
        return False
    except SystemExit as e:
        print(e)
        return False
    finally:
        if save_iterator:
            print("### Checkpointing iterator (step {})...".format(step), flush=True)
            saver_iterator.save(sess, out_ckpt_loc_iterator, global_step=step, write_meta_graph=False)
        if trainable_brain: 
            print("### Checkpointing brain_network (step {})...".format(step+brain_net_ckpt_offset), flush=True)
            saver_brain_net.save(sess, out_ckpt_loc_brain_net, global_step=step+brain_net_ckpt_offset, write_meta_graph=False)
        print('Step: {} | Error count: {}'.format(step, errors_count), flush=True)
        print("Training stopped.", flush=True)
    sess.close()
    tf.reset_default_graph()
    return True


def run_eval_routine(
        tfrecords_regex,
        batch_size=8,
        display_step=5,
        output_directory="/saved_models/example_arch",
        brain_net_ckpt_to_load=None,
        frontend_ckpt_to_load=None,
        controller="/cpu:0", 
        random_seed=517,
        signal_rate=20000,
        N_CLASSES_DICT={},
        TASK_LOSS_PARAMS=None, 
        ITERATOR_PARAMS={},
        FRONTEND_PARAMS={},
        COCH_PARAMS={},
        BRAIN_PARAMS={},
        maindata_keyparts=[":labels_true", ":labels_pred", ":correct_pred"],
        metadata_keys=[],
        **kwargs):
    '''
    This function runs the evaluation routine (brain_network)
    
    Args
    ----
    tfrecords_regex (str): regex that globs .tfrecords files for evaluation dataset
    batch_size (int): number of examples per batch per GPU
    display_step (int): print out evaluation info every display_step steps
    output_directory (str): location to load model checkpoints from
    brain_net_ckpt_to_load (str): path to brain_network .ckpt-# to load, None checks for most recent checkpoint
    frontend_ckpt_to_load (str): path to frontend .ckpt-# to load, None checks for most recent checkpoint
    controller (str): specify device to host input pipeline and merge towers
    random_seed (int): random seed to set tensorflow and numpy
    signal_rate (int): sampling rate of input signal (Hz)
    N_CLASSES_DICT (dict): dictionary specifying number of output classes for each task
    TASK_LOSS_PARAMS (dict): dictionary containing the loss parameters for each task, keys are the task paths
    ITERATOR_PARAMS (dict): parameters for building the input data iterator
    FRONTEND_PARAMS (dict): parameters for building the frontend_model graph
    COCH_PARAMS (dict): parameters for building the cochlear model
    BRAIN_PARAMS (dict): parameters for building the brain network
    maindata_keyparts (list): list of substrings to indicate which evaluation tensors to include in output_file
    metadata_keys (list): list of strings to indicate which input tensors to include in output_dict
        
    Returns
    -------
    output_dict (dict): dictionary of labels and model predictions
    '''
    ### RESET DEFAULT GRAPH AND SET RANDOM SEEDS ###
    tf.reset_default_graph()
    tf.random.set_random_seed(random_seed)
    np.random.seed(random_seed)
    
    num_towers = len(fga.get_available_gpus()) # Used to scale the batch_size for iterator
    
    if BRAIN_PARAMS.get('MULTIBRAIN_PARAMS', {}):
        raise NotImplementedError("MULTIBRAIN_PARAMS is not implemented")    
    if FRONTEND_PARAMS:
        raise NotImplementedError("FRONTEND_PARAMS is not implemented")

    ### BUILD THE INPUT PIPELINE ###
    print('\n\n\n >>> building input iterator <<< \n\n\n')
    with tf.device(controller):
        # Build the one-shot iterator
        iterator, dataset, iterator_save_obj = fga.build_tfrecords_iterator(
            tfrecords_regex,
            num_epochs=1,
            shuffle_flag=False,
            batch_size=batch_size*num_towers,
            iterator_type='one-shot',
            **ITERATOR_PARAMS)
        # Input tensor dict containing batch_size * num_towers examples:
        input_tensor_dict = iterator.get_next()
    
    ### BUILD THE GRAPH ###
    print('\n\n\n >>> building graph and output tensors <<< \n\n\n')
    feature_signal_path = ITERATOR_PARAMS.get('feature_signal_path', 'stimuli/signal')
    tower_grads, batch_loss, batch_loss_dict, batch_out_dict, batch_labels_dict, batch_audio_dict, controls_dict = fga.create_parallel_graph(
        fga.training_model, input_tensor_dict, feature_signal_path, signal_rate, N_CLASSES_DICT, TASK_LOSS_PARAMS, 
        FRONTEND_PARAMS, COCH_PARAMS, BRAIN_PARAMS, controller=controller, compute_gradients=False)
    # Combine all of the parallel outputs of the multi-tower graph
    with tf.device(controller):
        for task_key in N_CLASSES_DICT.keys():
            batch_loss_dict[task_key] = tf.reduce_mean(batch_loss_dict[task_key])
            batch_out_dict[task_key] = tf.concat(batch_out_dict[task_key], 0)
            batch_labels_dict[task_key] = tf.concat(batch_labels_dict[task_key], 0)
        for audio_key in batch_audio_dict.keys():
            batch_audio_dict[audio_key] = tf.concat(batch_audio_dict[audio_key], 0)
    batch_audio_dict = {} # batch_audio_dict should be empty if not saving audio files
    all_evaluation_measures_dict = functions_evaluation.make_task_evaluation_metrics(
        batch_out_dict, batch_labels_dict, batch_loss_dict,
        batch_audio_dict=batch_audio_dict, TASK_LOSS_PARAMS=TASK_LOSS_PARAMS)
    
    ### INITIALIZE GRAPH ###
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    config = tf.ConfigProto(allow_soft_placement=True,
                            inter_op_parallelism_threads=0,
                            intra_op_parallelism_threads=0)
    sess = tf.Session(config=config)
    sess.run(init_op)
    
    ### BUILD SAVERS AND ATTEMPT TO LOAD PREVIOUS CHECKPOINTS ###
    # If `load_coch_vars` function(s) exist(s) in the controls_dict, call within active tf session
    for control_key in sorted(controls_dict.keys()):
        if 'load_coch_vars' in control_key:
            controls_dict[control_key](session=sess)
    frontend_ckpt, brain_net_ckpt = None, None         
    print('\n\n\n >>> building savers and attempting to load vars <<< \n\n\n')
    if BRAIN_PARAMS: # If BRAIN_PARAMS is not empty, build saver and attempt load
        brain_ckpt_prefix_name = BRAIN_PARAMS.get('save_ckpt_path', 'brain_model.ckpt')
        if type(brain_net_ckpt_to_load) == int:
            # If specifying an integer checkpoint, try to load it from the output directory
            brain_net_ckpt_basename = '{0:s}-{1:d}'.format(brain_ckpt_prefix_name, brain_net_ckpt_to_load)
            brain_net_ckpt_to_load = os.path.join(output_directory, brain_net_ckpt_basename)
        brain_var_scope = 'brain_network'
        brain_globals = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=brain_var_scope)
        brain_locals = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope=brain_var_scope)
        brain_variables =  brain_globals + brain_locals
        saver_brain_net, out_ckpt_loc_brain_net, brain_net_ckpt = fga.build_saver(
            sess,
            brain_variables,
            output_directory,
            restore_model_path=brain_net_ckpt_to_load,
            ckpt_prefix_name=brain_ckpt_prefix_name)
    
    ### PREPARE OUTPUT DICTIONARY ###
    output_dict = {}
    output_dict['tfrecords_regex'] = tfrecords_regex
    if BRAIN_PARAMS:
        output_dict['out_ckpt_loc_brain_net'] = out_ckpt_loc_brain_net
        output_dict['brain_net_ckpt'] = int(brain_net_ckpt)
    # Decide which of the tensors in all_evaluation_measures_dict should be included in the output file
    maindata_keys = []
    for key in all_evaluation_measures_dict.keys():
        if any([keypart in key for keypart in maindata_keyparts]):
            maindata_keys.append(key)
    # Initialize a list for each main and meta data key (also add metadata tensors to all_evaluation_measures_dict)
    for key in maindata_keys + metadata_keys:
        if key in all_evaluation_measures_dict.keys():
            output_dict[key] = []
        elif key in input_tensor_dict.keys():
            output_dict[key] = []
            all_evaluation_measures_dict[key] = tf.concat(input_tensor_dict[key], 0)
        else:
            print('Requested key `{}` not found in `input_tensor_dict` or `all_evaluation_measures_dict`'.format(key))
    # Down-cast tensors in all_evaluation_measures_dict to reasonable datatypes for JSON
    for key in all_evaluation_measures_dict.keys():
        if all_evaluation_measures_dict[key].dtype in [tf.int64, tf.bool]:
            all_evaluation_measures_dict[key] = tf.cast(all_evaluation_measures_dict[key], tf.int32)
        elif all_evaluation_measures_dict[key].dtype in [tf.float64]:
            all_evaluation_measures_dict[key] = tf.cast(all_evaluation_measures_dict[key], tf.float32)
    print('>>> `output_dict` structure <<<')
    for key in output_dict.keys(): print('-->', key, output_dict[key])
   
    evaluation_feed_dict = {
        controls_dict['batchnorm_flag_placeholder']:controls_dict['batchnorm_flag'],
        controls_dict['dropout_flag_placeholder']:controls_dict['dropout_flag']
    } 

    ### MAIN EVALUATION LOOP ###
    print('\n\n\n >>> begin evaluation routine <<< \n\n\n')
    step, example_num = 0, 0
    try:
        while True:
            # Evaluate all tensors in all_evaluation_measures_dict
            evaluated_batch = sess.run(all_evaluation_measures_dict, feed_dict=evaluation_feed_dict)
            # Store values in output_dict
            for key in set(output_dict.keys()).intersection(evaluated_batch.keys()):
                key_val = np.array(evaluated_batch[key]).tolist()
                if not isinstance(key_val, list): key_val = [key_val] # Handles special case of non-array values
                output_dict[key].extend(key_val)
            
            # Display progress
            step += 1
            if step % display_step == 0:
                print('### batch {:08}'.format(step))
                for batch_labels_key in batch_labels_dict.keys():
                    for name in list(all_evaluation_measures_dict.keys()):
                        if (batch_labels_key in name) and ('_update_op' not in name) and (':probs_out' not in name) and ('_audio' not in name):
                            print('    %s: %s'%(name, evaluated_batch[name])) 

    except tf.errors.OutOfRangeError:
        print('End of evaluation dataset reached.')
    
    finally:
        print('>>> `output_dict` structure <<<')
        for key in output_dict.keys():
            if type(output_dict[key]) is list: print('-->', key, type(output_dict[key]), len(output_dict[key]))
            else: print('-->', key, type(output_dict[key]), output_dict[key])
   
    # Set the batch norm values back to things that are json serializable (instead of placeholders)
    if BRAIN_PARAMS:
        if tf.contrib.framework.is_tensor(BRAIN_PARAMS['batchnorm_flag']):
            BRAIN_PARAMS['batchnorm_flag']=bool(sess.run(BRAIN_PARAMS['batchnorm_flag'], feed_dict=evaluation_feed_dict))
        if tf.contrib.framework.is_tensor(BRAIN_PARAMS['dropout_flag']):
            BRAIN_PARAMS['dropout_flag']=bool(sess.run(BRAIN_PARAMS['dropout_flag'], feed_dict=evaluation_feed_dict))
 
    sess.close()
    tf.reset_default_graph()
    return output_dict
