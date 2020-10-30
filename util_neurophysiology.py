import os
import sys
import json
import h5py
import glob
import copy
import pdb
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import pitchnet_evaluate_best

sys.path.append('/om2/user/msaddler/pitchnet/assets_psychophysics')
import util_figures_psychophysics

sys.path.append('/om2/user/msaddler/python-packages/msutil')
import util_figures
import util_misc

sys.path.append('/om4/group/mcdermott/user/msaddler/pitchnet_dataset/pitchnetDataset/pitchnetDataset')
import dataset_util

sys.path.append('ibmHearingAid/multi_gpu')
import functions_graph_assembly as fga


def store_network_activations(output_directory,
                              tfrecords_regex,
                              fn_activations='ACTIVATIONS.hdf5',
                              fn_config='config.json',
                              fn_valid_metrics='validation_metrics.json',
                              metadata_keys=['f0', 'low_harm', 'phase_mode', 'snr'],
                              maindata_keyparts=['relu'],
                              batch_size=128,
                              display_step=50):
    '''
    Evaluate network and return dictionary of activations and stimulus metadata.
    '''
    tf.reset_default_graph()
    
    if fn_activations == os.path.basename(fn_activations):
        fn_activations = os.path.join(output_directory, fn_activations)
    if fn_config == os.path.basename(fn_config):
        fn_config = os.path.join(output_directory, fn_config)
    if fn_valid_metrics == os.path.basename(fn_valid_metrics):
        fn_valid_metrics = os.path.join(output_directory, fn_valid_metrics)    
    with open(fn_config) as f:
        CONFIG = json.load(f)
    
    # Compute total number of examples from the tfrecords filenames
    fn_last_tfrecords = sorted(glob.glob(tfrecords_regex))[-1]
    fn_last_tfrecords = os.path.basename(fn_last_tfrecords)
    N = int(fn_last_tfrecords[fn_last_tfrecords.rfind('-')+1:fn_last_tfrecords.rfind('.')])
    
    # Build input data pipeline
    ITERATOR_PARAMS = CONFIG['ITERATOR_PARAMS']
    bytesList_decoding_dict = {"nervegram_meanrates": {"dtype": "tf.float32", "shape": [100, 1000]}}
    feature_parsing_dict = pitchnet_evaluate_best.get_feature_parsing_dict_from_tfrecords(
        tfrecords_regex,
        bytesList_decoding_dict)
    ITERATOR_PARAMS['feature_parsing_dict'] = feature_parsing_dict
    iterator, dataset, _ = fga.build_tfrecords_iterator(tfrecords_regex,
                                                        num_epochs=1,
                                                        shuffle_flag=False,
                                                        batch_size=batch_size,
                                                        iterator_type='one-shot',
                                                        **ITERATOR_PARAMS)
    input_tensor_dict = iterator.get_next()
    
    # Build network graph
    BRAIN_PARAMS = CONFIG['BRAIN_PARAMS']
    for key in sorted(BRAIN_PARAMS.keys()):
        if ('path' in key) or ('config' in key):
            dirname = os.path.dirname(BRAIN_PARAMS[key])
            if not dirname == output_directory:
                BRAIN_PARAMS[key] = BRAIN_PARAMS[key].replace(dirname, output_directory)
    N_CLASSES_DICT = CONFIG['N_CLASSES_DICT']
    batch_subbands = input_tensor_dict[ITERATOR_PARAMS['feature_signal_path']]
    while len(batch_subbands.shape) < 4:
        batch_subbands = tf.expand_dims(batch_subbands, axis=-1)
    batch_out_dict, brain_container = fga.build_brain_graph(batch_subbands,
                                                            N_CLASSES_DICT,
                                                            **BRAIN_PARAMS)
    
    # Start session and initialize variable
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess = tf.Session()
    sess.run(init_op)
    
    # Build saver graph and load network checkpoint
    ckpt_num = pitchnet_evaluate_best.get_best_checkpoint_number(fn_valid_metrics,
                                                                 metric_key='f0_label:accuracy',
                                                                 maximize=True,
                                                                 checkpoint_number_key='step')
    brain_var_scope = 'brain_network'
    brain_ckpt_prefix_name = BRAIN_PARAMS.get('save_ckpt_path', 'brain_model.ckpt')
    restore_model_path = os.path.join(output_directory, brain_ckpt_prefix_name + '-{}'.format(ckpt_num))
    brain_globals = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=brain_var_scope)
    brain_locals = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope=brain_var_scope)
    brain_variables =  brain_globals + brain_locals
    saver_brain_net, out_ckpt_loc_brain_net, brain_net_ckpt = fga.build_saver(
        sess, brain_variables, output_directory,
        restore_model_path=restore_model_path,
        ckpt_prefix_name=brain_ckpt_prefix_name)
    
    # Set up dictionary of tensors to evaluate
    tensors_to_evaluate = {}
    for key in sorted(set(input_tensor_dict.keys()).intersection(metadata_keys)):
        tensors_to_evaluate[key] = input_tensor_dict[key]
    for key in sorted(brain_container.keys()):
        for keypart in maindata_keyparts:
            if keypart in key:
                if len(brain_container[key].shape) == 4:
                    # Average activations across the time-axis when present
                    tensors_to_evaluate[key] = tf.reduce_mean(brain_container[key], axis=(2,))
                else:
                    tensors_to_evaluate[key] = brain_container[key]
                break
    
    # Main evaluation routine
    batch_count = 0
    try:
        while True:
            evaluated_batch = sess.run(tensors_to_evaluate)
            
            if batch_count == 0:
                print('[INITIALIZING]: {}'.format(fn_activations))
                data_key_pair_list = [(k, k) for k in sorted(evaluated_batch.keys())]
                data_dict = {k: evaluated_batch[k][0] for k in sorted(evaluated_batch.keys())}
                dataset_util.initialize_hdf5_file(fn_activations,
                                                  N,
                                                  data_dict,
                                                  file_mode='w',
                                                  data_key_pair_list=data_key_pair_list,
                                                  config_key_pair_list=[],
                                                  fillvalue=-1)
                hdf5_f = h5py.File(fn_activations, 'r+')
                for k in sorted(evaluated_batch.keys()):
                    print('[___', hdf5_f[k])
            
            for k in sorted(evaluated_batch.keys()):
                idx_start = batch_count * batch_size
                idx_end = idx_start + evaluated_batch[k].shape[0]
                hdf5_f[k][idx_start:idx_end] = evaluated_batch[k]
            
            if batch_count % display_step == 0:
                print('Evaluation step: {} (indexes {}-{})'.format(
                    batch_count, idx_start, idx_end))
            batch_count += 1
    except tf.errors.OutOfRangeError:
        print('End of evaluation dataset reached')
    
    print('[END]: {}'.format(fn_activations))
    for k in sorted(evaluated_batch.keys()):
        print('[___', hdf5_f[k])
    
    hdf5_f.close()
    return


def store_network_tuning_results(fn_input,
                                 fn_output,
                                 key_dim0='low_harm',
                                 key_dim1='f0_label',
                                 key_acts='relu',
                                 kwargs_f0_bins={}):
    '''
    Functions takes input hdf5 file of network activations, re-organizes
    activations to quantify tuning of each unit to two stimulus dimensions,
    and stores these tuning results in output hdf5 file. If `key_dim1` is
    "key_dim1", then octave tuning (relative to best F0) is also computed
    and stored.
    '''
    # Add f0_label to input hdf5 file if needed
    f = h5py.File(fn_input, 'r+')
    input_dataset_key_list = util_misc.get_hdf5_dataset_key_list(f)
    if 'f0_label' in [key_dim0, key_dim1]:
        f0_bins = dataset_util.get_f0_bins(**kwargs_f0_bins)
        f0_label_list = dataset_util.f0_to_label(f['f0'][:], f0_bins)
        if 'f0_label' in input_dataset_key_list:
            f['f0_label'][:] = f0_label_list
        else:
            f.create_dataset('f0_label',
                             f0_label_list.shape,
                             dtype=f0_label_list.dtype,
                             data=f0_label_list)
    f.close()
    f = h5py.File(fn_input, 'r')        
    
    # Compute unique values of stimulus dimensions
    unique_dim0, tuning_index_dim0 = np.unique(f[key_dim0][:], return_inverse=True)
    unique_dim1, tuning_index_dim1 = np.unique(f[key_dim1][:], return_inverse=True)
    
    # Initialize output hdf5 file
    print('[INITIALIZING] {}'.format(fn_output))
    f_output = h5py.File(fn_output, 'w')
    f_output.create_dataset(key_dim0, unique_dim0.shape, dtype=unique_dim0.dtype, data=unique_dim0)
    f_output.create_dataset(key_dim1, unique_dim1.shape, dtype=unique_dim1.dtype, data=unique_dim1)
    print(key_dim0, f_output[key_dim0])
    print(key_dim1, f_output[key_dim1])
    if 'f0_label' in [key_dim0, key_dim1]:
        f_output.create_dataset('f0_bins', f0_bins.shape, dtype=f0_bins.dtype, data=f0_bins)
        print('f0_bins', f_output['f0_bins'])
    
    # Iterate over activation keys and compute tuning to stimulus dimensions
    if isinstance(key_acts, str):
        key_acts = [k for k in input_dataset_key_list if key_acts in k]
    for k in key_acts:
        activations = f[k][:].reshape([tuning_index_dim0.shape[0], -1])
        shape = [activations.shape[-1], unique_dim0.shape[0], unique_dim1.shape[0]]
        tuning_array = np.zeros(shape, dtype=activations.dtype)
        tuning_count = np.zeros(shape, dtype=activations.dtype)
        print('... processing {} : {} --> {} --> {}'.format(
            k, str(f[k].shape), str(activations.shape), str(tuning_array.shape)))
        for idx_stim, (idx0, idx1) in enumerate(zip(tuning_index_dim0, tuning_index_dim1)):
            tuning_array[:, idx0, idx1] += activations[idx_stim, :]
            tuning_count[:, idx0, idx1] += 1
        tuning_array = tuning_array / tuning_count
        f_output.create_dataset(k,
                                tuning_array.shape,
                                dtype=tuning_array.dtype,
                                data=tuning_array)
        
        population_tuning_array = np.mean(tuning_array, axis=0)
        f_output.create_dataset(k + '_population_mean',
                                population_tuning_array.shape,
                                dtype=population_tuning_array.dtype,
                                data=population_tuning_array)
        
        # If key_dim1 == "f0_label", compute and store octave tuning
        if key_dim1 == 'f0_label':
            f0_label_list = f_output['f0_label'][:]
            f0_bins_used = f_output['f0_bins'][f0_label_list.min() : f0_label_list.max()+1]
            octave_max = np.log2(f0_bins_used[-1] / f0_bins_used[0])
            octave_bins = np.linspace(-octave_max, octave_max, 2*f0_bins_used.shape[0]-1)
            shape = [tuning_array.shape[0], octave_bins.shape[0]]
            octave_tuning_array = np.zeros(shape, dtype=tuning_array.dtype)
            octave_tuning_count = np.zeros(shape, dtype=tuning_array.dtype)
            print('... computing octave tuning', octave_bins.shape, octave_tuning_array.shape)
            f0_tuning_array = np.mean(tuning_array, axis=1)
            best_f0_indexes = np.argmax(f0_tuning_array, axis=1)
            best_octave_index = f0_bins_used.shape[0] - 1
            for index_unit, best_f0_bin_index in enumerate(best_f0_indexes):
                idx_start = best_octave_index - best_f0_bin_index
                idx_end = idx_start + f0_bins_used.shape[0]
                octave_tuning_array[index_unit, idx_start:idx_end] = f0_tuning_array[index_unit]
                octave_tuning_count[index_unit, idx_start:idx_end] += 1
            
            if 'octave_bins' in f_output:
                assert np.all(f_output['octave_bins'][:] == octave_bins)
            else:
                f_output.create_dataset('octave_bins',
                                        octave_bins.shape,
                                        dtype=octave_bins.dtype,
                                        data=octave_bins)
            f_output.create_dataset(k + '_octave_tuning',
                                    octave_tuning_array.shape,
                                    dtype=octave_tuning_array.dtype,
                                    data=octave_tuning_array)
            f_output.create_dataset(k + '_octave_tuning_count',
                                    octave_tuning_count.shape,
                                    dtype=octave_tuning_count.dtype,
                                    data=octave_tuning_count)
    
    print('[END] {}'.format(fn_output))
    for k in util_misc.get_hdf5_dataset_key_list(f_output):
        print(k, f_output[k])
    f.close()
    f_output.close()
    return


def get_results_dict_bendor_and_wang_2005():
    '''
    Results scanned from Bendor and Wang (2005, Nature), Figure 4C.
        n=50 pitch-selective neurons in marmoset auditory cortex
        Error bars indicate SEM
    '''
    results_dict = {
        'is_neural_data': True,
        'low_harm': np.array([1, 2, 3, 4, 5, 6, 8, 10, 12]),
        'yval': np.array([0.633677, 0.49193, 0.565334, 0.478132, 0.508102, 0.540082, 0.377799, 0.275124, 0.263335]),
        'yerr_max': np.array([0.700344, 0.556566, 0.639072, 0.528637, 0.580819, 0.597668, 0.440426, 0.331678, 0.324962]),
        'yerr_min': np.array([0.570041, 0.424253, 0.494627, 0.426617, 0.436385, 0.485536, 0.315184, 0.218536, 0.20072]),
    }
    yerr_estimate_min = np.abs(results_dict['yval'] - results_dict['yerr_min'])
    yerr_estimate_max = np.abs(results_dict['yval'] - results_dict['yerr_max'])
    results_dict['yerr'] = (yerr_estimate_min + yerr_estimate_max) / 2
    return results_dict


def get_results_dict_norman_haignere_2013():
    '''
    Results scanned from Norman-Haignere et al. (2013, JNeurosci), Figure 4C.
        n=12 human fMRI participants
        Error bars indicate one within-subject SEM
    '''
    results_dict = {
        'is_neural_data': True,
        'low_harm': np.array([3, 4, 5, 6, 8, 10, 12, 15]),
        'yval': np.array([0.89688, 0.95208, 0.89896, 0.89862, 0.81493, 0.75347, 0.64202, 0.58056]),
        'yerr_max': np.array([0.93021, 0.98542, 0.94063, 0.91806, 0.8566, 0.78958, 0.68091, 0.61944]),
        'noise_yval': np.array([0.45771]),
        'noise_yerr': np.array([0.49938-0.45771]),
    }
    results_dict['yerr'] = np.abs(results_dict['yval'] - results_dict['yerr_max'])
    return results_dict


def make_1d_tuning_plot(ax,
                        results_dict_input,
                        key_dim0='low_harm',
                        key_resp_list='relu_4_low_harm',
                        color_list=None,
                        include_yerr=True,
                        kwargs_plot_update={},
                        kwargs_legend_update={},
                        kwargs_bootstrap={'bootstrap_repeats': 1000, 'metric_function': 'mean'},
                        **kwargs_format_axes):
    '''
    '''
    if isinstance(results_dict_input, dict) and results_dict_input.get('is_neural_data', False):
        is_neural_data = True
        xval = results_dict_input[key_dim0]
        yval = results_dict_input['yval']
        yerr = results_dict_input['yerr']
    else:
        is_neural_data = False
    
    DATA = {}
    if is_neural_data:
        kwargs_plot = {
            'label': results_dict_input.get('label', None),
            'color': 'k',
            'ls': '-',
            'marker': '',
            'lw': 1,
        }
        kwargs_plot.update(kwargs_plot_update)
        if include_yerr:
            errorbar_kwargs = {
                'yerr': yerr,
                'fmt': 'none',
                'ecolor': 'k',
                'elinewidth': kwargs_plot.get('lw', 1),
                'capsize': 1.5 * kwargs_plot.get('lw', 1),
            }
            ax.errorbar(xval, yval, **errorbar_kwargs)
        ax.plot(xval, yval, **kwargs_plot)
        
#         noise_xval = results_dict_input.get('noise_xval', np.array([0, 31]))
#         noise_yval = results_dict_input.get('noise_yval', None)
#         noise_yerr = results_dict_input.get('noise_yerr', None)
#         if noise_yval is not None:
#             if (noise_yerr is not None) and include_yerr:
#                 ax.fill_between(noise_xval,
#                                 noise_yval-1*noise_yerr,
#                                 noise_yval+1*noise_yerr,
#                                 alpha=0.15,
#                                 facecolor=kwargs_plot.get('color', 'k'))
#             kwargs_plot['ls'] = '--'
#             kwargs_plot['dashes'] = (2,2)
#             kwargs_plot['marker'] = ''
#             kwargs_plot['label'] = results_dict_input.get('noise_label', 'Response to noise')
#             ax.plot(noise_xval * np.ones_like(noise_xval),
#                     noise_yval * np.ones_like(noise_xval),
#                     **kwargs_plot)

    else:
        if not isinstance(results_dict_input, list):
            results_dict_input = [results_dict_input]
        if not isinstance(key_resp_list, list):
            key_resp_list = [key_resp_list]
        if color_list is None:
            color_list = util_figures.get_color_list(len(key_resp_list), 'copper')
        if not isinstance(color_list, list):
            color_list = [color_list]
        for cidx, key_resp in enumerate(key_resp_list):
            yval_list = []
            for results_dict in results_dict_input:
                xval = np.array(results_dict[key_dim0])
                yval_tmp = np.array(results_dict[key_resp])
                assert np.all(yval_tmp.shape == xval.shape)
                yval_list.append(yval_tmp)
            yval_list = np.stack(yval_list, axis=0)
            DATA[key_dim0] = xval
            DATA[key_resp] = yval_list
            yval, yerr = util_figures_psychophysics.combine_subjects(
                yval_list, kwargs_bootstrap=kwargs_bootstrap)
            kwargs_plot = {
                'label': key_resp,
                'color': color_list[cidx],
                'ls': '-',
                'lw': 1,
                'marker': '',
            }
            kwargs_plot.update(kwargs_plot_update)
            if include_yerr:
                ax.fill_between(xval,
                                yval-2*yerr,
                                yval+2*yerr,
                                alpha=0.15,
                                facecolor=kwargs_plot.get('color', 'k'))
            ax.plot(xval, yval, **kwargs_plot)
    
    kwargs_legend = {
        'loc': 'upper right',
        'ncol': 1,
        'frameon': False,
        'fontsize': 12,
        'handlelength': 0.5,
        'borderpad': 0.5,
        'borderaxespad': 0.1,
    }
    kwargs_legend.update(kwargs_legend_update)
    leg = ax.legend(**kwargs_legend)
    for legobj in leg.legendHandles:
        legobj.set_linewidth(6.0)
    
    ax = util_figures.format_axes(ax, **kwargs_format_axes)
    return ax, DATA


def make_2d_tuning_plot(ax,
                        results_dict,
                        key_act='relu_4',
                        key_dim0='low_harm',
                        key_dim1='f0_label',
                        key_dim0_label=None,
                        key_dim1_label='f0_bins',
                        unit_idx=None,
                        num_ticks_dim0=5,
                        num_ticks_dim1=5,
                        kwargs_plot_update={},
                        kwargs_legend_update={},
                        **kwargs_format_axes):
    '''
    '''
    if isinstance(results_dict, list):
        print('Expected non-list input (using only first entry)')
        results_dict = results_dict[0]
    
    dim0_vals = results_dict[key_dim0][:]
    dim1_vals = results_dict[key_dim1][:]
    if key_dim0_label is not None:
        dim0_labels = results_dict[key_dim0_label][:]
        dim0_labels = dim0_labels[dim0_vals.astype(int)]
    else:
        dim0_labels = dim0_vals
    if key_dim1_label is not None:
        dim1_labels = results_dict[key_dim1_label][:]
        dim1_labels = dim1_labels[dim1_vals.astype(int)]
    else:
        dim1_labels = dim1_vals
    
    tuning_array = results_dict[key_act]
    if unit_idx is None:
        unit_idx = np.random.randint(0, tuning_array.shape[0], dtype=int)
        print('Randomly selected unit_idx={}'.format(unit_idx))
    
    # Plot 2d tuning array for a single unit
    im_data = tuning_array[unit_idx].T
    if im_data.max() > 0:
        im_data = im_data / im_data.max()
    IMG = ax.imshow(im_data,
                    origin=(0,0),
                    aspect='auto',
                    extent=[0, im_data.shape[1], 0, im_data.shape[0]],
                    cmap=plt.cm.gray)
    # Format axes
    dim0_ticks = np.linspace(0, dim0_vals.shape[0]-1, num=num_ticks_dim0, dtype=int)
    dim0_ticklabels = ['{:.0f}'.format(dim0_labels[tick]) for tick in dim0_ticks]
    dim1_ticks = np.linspace(0, dim1_vals.shape[0]-1, num=num_ticks_dim1, dtype=int)
    dim1_ticklabels = ['{:.0f}'.format(dim1_labels[tick]) for tick in dim1_ticks]
    kwargs = {
        'xticks': dim0_ticks,
        'yticks': dim1_ticks,
        'xticklabels': dim0_ticklabels,
        'yticklabels': dim1_ticklabels,

    }
    kwargs.update(kwargs_format_axes)
    if 'str_title' not in kwargs.keys():
        kwargs['str_title'] = 'unit {:04d}'.format(unit_idx)
    ax = util_figures.format_axes(ax, **kwargs)
    return ax, IMG


def make_low_harm_tuning_plot(ax, results_dict_input, key_resp_list=['relu_4'], **kwargs):
    '''
    '''
    if not isinstance(key_resp_list, list):
        key_resp_list = [key_resp_list]
    low_harm_key_resp_list = []
    for key in key_resp_list:
        if '_low_harm' in key:
            low_harm_key_resp_list.append(key)
        else:
            low_harm_key_resp_list.append(key + '_low_harm')
    
    kwargs_make_1d_tuning_plot = {
        'key_dim0': 'low_harm',
        'key_resp_list': low_harm_key_resp_list,
        'str_xlabel': 'Lowest harmonic number',
        'str_ylabel': 'Mean activation\n(normalized)',
        'xlimits': [0, 31],
        'xticks': np.arange(0, 31, 5),
        'xticks_minor': np.arange(0, 31, 1),
        'ylimits': [0, 1],
        'yticks': np.arange(0, 1.1, 0.2),
    }
    kwargs_make_1d_tuning_plot.update(kwargs)
    ax, DATA = make_1d_tuning_plot(ax, results_dict_input, **kwargs_make_1d_tuning_plot)
    return ax, DATA


def make_f0_tuning_plot(ax, results_dict_input, key_resp_list=['relu_4'], **kwargs):
    '''
    '''
    if not isinstance(key_resp_list, list):
        key_resp_list = [key_resp_list]
    f0_key_resp_list = []
    for key in key_resp_list:
        if '_f0_label' in key:
            f0_key_resp_list.append(key)
        else:
            f0_key_resp_list.append(key + '_f0_label')
    
    rd0 = results_dict_input
    if isinstance(rd0, list):
        rd0 = rd0[0]
    xval = rd0['f0_label']
    xval_labels = rd0['f0_bins']
    xtick_indexes = np.linspace(xval[0], xval[-1], 7, dtype=int)
    xticks = [xval[xti] for xti in xtick_indexes]
    xticklabels = ['{:.0f}'.format(xval_labels[xti]) for xti in xtick_indexes]
    kwargs_make_1d_tuning_plot = {
        'key_dim0': 'f0_label',
        'key_resp_list': f0_key_resp_list,
        'str_xlabel': 'F0 (Hz)',
        'str_ylabel': 'Mean activation\n(normalized)',
        'xlimits': [xval[0], xval[-1]],
        'xticks': xticks,
        'xticklabels': xticklabels,
    }
    kwargs_make_1d_tuning_plot.update(kwargs)
    ax, DATA = make_1d_tuning_plot(ax, results_dict_input, **kwargs_make_1d_tuning_plot)
    return ax, DATA


def make_octave_tuning_plot(ax, results_dict_input, key_resp_list=['relu_4'], **kwargs):
    '''
    '''
    if not isinstance(key_resp_list, list):
        key_resp_list = [key_resp_list]
    octave_tuning_key_resp_list = []
    for key in key_resp_list:
        if '_octave_tuning' in key:
            octave_tuning_key_resp_list.append(key)
        else:
            octave_tuning_key_resp_list.append(key + '_octave_tuning')
    kwargs_make_1d_tuning_plot = {
        'key_dim0': 'octave_bins',
        'key_resp_list': octave_tuning_key_resp_list,
        'str_xlabel': 'Octaves above best F0',
        'str_ylabel': 'Mean activation\n(normalized)',
    }
    kwargs_make_1d_tuning_plot.update(kwargs)
    ax, DATA = make_1d_tuning_plot(ax, results_dict_input, **kwargs_make_1d_tuning_plot)
    return ax, DATA


if __name__ == "__main__":
    ''' TEMPORARY COMMAND-LINE USAGE '''
    assert len(sys.argv) == 2, "scipt usage: python <script_name> <output_directory_regex>"
    output_directory_regex = str(sys.argv[1])
    
    tfrecords_regex = '/om/user/msaddler/data_pitchnet/bernox2005/neurophysiology_SlidingFixedFilter_lharm01to30_phase0_f0min080_f0max320/sr20000_cf100_species002_spont070_BW10eN1_IHC3000Hz_IHC7order/*.tfrecords'
    output_directory_list = sorted(glob.glob(output_directory_regex))
    
    print('output_directory_list:')
    for output_directory in output_directory_list:
        print('<> {}'.format(output_directory))
    
    for output_directory in output_directory_list:
        print('\n\n\nSTART: {}'.format(output_directory))
        
        fn_activations='NEUROPHYSIOLOGY_v01_bernox2005_activations.hdf5'
        fn_activations = os.path.join(output_directory, fn_activations)
        fn_tuning_results = fn_activations.replace('.hdf5', '_tuning_low_harm_f0.hdf5')

        if not os.path.exists(fn_activations):
            store_network_activations(output_directory,
                                      tfrecords_regex,
                                      fn_activations=fn_activations)
        store_network_tuning_results(fn_activations, fn_tuning_results)
        