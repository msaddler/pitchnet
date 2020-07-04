import os
import sys
import json
import glob
import copy
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


def get_network_activations(output_directory,
                            tfrecords_regex,
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
    
    if fn_config == os.path.basename(fn_config):
        fn_config = os.path.join(output_directory, fn_config)
    if fn_valid_metrics == os.path.basename(fn_valid_metrics):
        fn_valid_metrics = os.path.join(output_directory, fn_valid_metrics)    
    with open(fn_config) as f:
        CONFIG = json.load(f)
    
    # Build input data pipeline
    ITERATOR_PARAMS = CONFIG['ITERATOR_PARAMS']
    bytesList_decoding_dict = {"meanrates": {"dtype": "tf.float32", "shape": [100, 1000]}}
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
                    tensors_to_evaluate[key] = tf.reduce_mean(brain_container[key], axis=(1,2))
                else:
                    tensors_to_evaluate[key] = brain_container[key]
                break
    output_dict = {}
    for key in sorted(tensors_to_evaluate.keys()):
        print('[START] output_dict[`{}`]'.format(key))
        output_dict[key] = []
    
    # Main evaluation routine
    batch_count = 0
    try:
        while True:
            evaluated_batch = sess.run(tensors_to_evaluate)
            for key in set(output_dict.keys()).intersection(evaluated_batch.keys()):
                key_val = np.array(evaluated_batch[key]).tolist()
                if not isinstance(key_val, list): key_val = [key_val]
                output_dict[key].extend(key_val)
            batch_count += 1
            if batch_count % display_step == 0:
                print('Evaluation step: {}'.format(batch_count))
    except tf.errors.OutOfRangeError:
        print('End of evaluation dataset reached')
    
    for key in sorted(output_dict.keys()):
        output_dict[key] = np.array(output_dict[key])
        print('[END] output_dict[`{}`]'.format(key), output_dict[key].shape)
    return output_dict


def compute_1d_tuning(output_dict,
                      key_act='relu_0',
                      key_dim0='low_harm',
                      normalize_unit_activity=True,
                      tuning_dict={}):
    '''
    Network neurophysiology function for computing tuning of individual units
    along a single stimulus dimension as specified by key_dim0.
    
    Args
    ----
    output_dict (dict): dictionary of network activations and stimulus metadata
    key_act (str): output_dict key for activations of interest (specifies layer)
    key_dim0 (str): output_dict key for stimulus dimension of interest
    normalize_unit_activity (bool): if true, re-scale activations to fall between
        0 and 1 across all stimulus bins (each unit is scaled separately)
    tuning_dict (dict): dictionary that tuning results will be added to
    
    Returns
    -------
    tuning_dict (dict): dictionary of tuning results (contains stimulus dimension
        bins and mean / standard deviation / sample size of activations)
    '''
    dim0_values = output_dict[key_dim0]
    dim0_bins = np.unique(dim0_values)
    activations = output_dict[key_act]
    dim0_tuning_mean = np.zeros([dim0_bins.shape[0], activations.shape[1]])
    dim0_tuning_std = np.zeros([dim0_bins.shape[0], activations.shape[1]])
    dim0_tuning_n = np.zeros([dim0_bins.shape[0], activations.shape[1]])
    for dim0_index, dim0_bin_value in enumerate(dim0_bins):
        bin_indexes = dim0_values == dim0_bin_value
        bin_activations = activations[bin_indexes]
        dim0_tuning_mean[dim0_index, :] = np.mean(bin_activations, axis=0)
        dim0_tuning_std[dim0_index, :] = np.std(bin_activations, axis=0)
        dim0_tuning_n[dim0_index, :] = bin_activations.shape[0]
    if normalize_unit_activity:
        dim0_tuning_mean -= np.amin(dim0_tuning_mean, axis=0, keepdims=True)
        dim0_tuning_mean_max = np.amax(dim0_tuning_mean, axis=0, keepdims=True)
        dead_unit_indexes = dim0_tuning_mean_max == 0
        dim0_tuning_mean_max[dead_unit_indexes] = 1
        dim0_tuning_mean /= dim0_tuning_mean_max
        dim0_tuning_std /= dim0_tuning_mean_max
    tuning_dict['{}_bins'.format(key_dim0)] = dim0_bins
    tuning_dict['{}_tuning_mean'.format(key_dim0)] = dim0_tuning_mean
    tuning_dict['{}_tuning_std'.format(key_dim0)] = dim0_tuning_std
    tuning_dict['{}_tuning_n'.format(key_dim0)] = dim0_tuning_n
    return tuning_dict


def compute_2d_tuning(output_dict,
                      key_act='relu_0',
                      key_dim0='low_harm',
                      key_dim1='f0_label',
                      normalize_unit_activity=True,
                      tuning_dict={}):
    '''
    Network neurophysiology function for computing tuning of individual units
    along two simulus dimensions as specified by key_dim0 and key_dim1.
    
    Args
    ----
    output_dict (dict): dictionary of network activations and stimulus metadata
    key_act (str): output_dict key for activations of interest (specifies layer)
    key_dim0 (str): output_dict key for first stimulus dimension of interest
    key_dim0 (str): output_dict key for second stimulus dimension of interest
    normalize_unit_activity (bool): if true, re-scale activations to fall between
        0 and 1 across all stimulus bins (each unit is scaled separately)
    tuning_dict (dict): dictionary that tuning results will be added to
    
    Returns
    -------
    tuning_dict (dict): dictionary of tuning results (contains stimulus dimension
        bins and mean / standard deviation / sample size of activations)
    '''
    dim0_values = output_dict[key_dim0]
    dim0_bins = np.unique(dim0_values)
    dim1_values = output_dict[key_dim1]
    dim1_bins = np.unique(dim1_values)
    activations = output_dict[key_act]
    dim01_tuning_mean = np.zeros([dim0_bins.shape[0], dim1_bins.shape[0], activations.shape[1]])
    dim01_tuning_std = np.zeros([dim0_bins.shape[0], dim1_bins.shape[0], activations.shape[1]])
    dim01_tuning_n = np.zeros([dim0_bins.shape[0], dim1_bins.shape[0], activations.shape[1]])
    for dim0_index, dim0_bin_value in enumerate(dim0_bins):
        for dim1_index, dim1_bin_value in enumerate(dim1_bins):
            bin_indexes = np.logical_and(dim0_values == dim0_bin_value,
                                         dim1_values == dim1_bin_value)
            bin_activations = activations[bin_indexes]
            dim01_tuning_mean[dim0_index, dim1_index, :] = np.mean(bin_activations, axis=0)
            dim01_tuning_std[dim0_index, dim1_index, :] = np.std(bin_activations, axis=0)
            dim01_tuning_n[dim0_index, dim1_index, :] = bin_activations.shape[0]
    if normalize_unit_activity:
        dim01_tuning_mean -= np.amin(dim01_tuning_mean, axis=(0,1), keepdims=True)
        dim01_tuning_mean_max = np.amax(dim01_tuning_mean, axis=(0,1), keepdims=True)
        dead_unit_indexes = dim01_tuning_mean_max == 0
        dim01_tuning_mean_max[dead_unit_indexes] = 1
        dim01_tuning_mean /= dim01_tuning_mean_max
        dim01_tuning_std /= dim01_tuning_mean_max
    tuning_dict['{}_bins'.format(key_dim0)] = dim0_bins
    tuning_dict['{}_bins'.format(key_dim1)] = dim1_bins
    tuning_dict['{}_{}_tuning_mean'.format(key_dim0, key_dim1)] = dim01_tuning_mean
    tuning_dict['{}_{}_tuning_std'.format(key_dim0, key_dim1)] = dim01_tuning_std
    tuning_dict['{}_{}_tuning_n'.format(key_dim0, key_dim1)] = dim01_tuning_n
    return tuning_dict


def compute_f0_tuning_re_best(output_dict,
                              key_act='relu_0',
                              key_f0='f0',
                              key_f0_label='f0_label',
                              kwargs_f0_bins={},
                              normalize_unit_activity=True,
                              tuning_dict={}):
    '''
    Network neurophysiology function for computing tuning of individual units
    to f0. The resulting single-unit f0 tuning curves are also aligned according
    to their best f0s (tuning curves as a function of octaves-above-best-f0).
    
    Args
    ----
    output_dict (dict): dictionary of network activations and stimulus metadata
    key_act (str): output_dict key for activations of interest (specifies layer)
    key_f0 (str): output_dict key for stimulus f0 values
    key_f0_label (str): output_dict key to store binned f0 values
    kwargs_f0_bins (dict): keyword arguments for binning f0 values
    normalize_unit_activity (bool): if true, re-scale activations to fall between
        0 and 1 across all stimulus bins (each unit is scaled separately)
    tuning_dict (dict): dictionary that tuning results will be added to
    
    Returns
    -------
    tuning_dict (dict): dictionary of f0 tuning results
    '''
    f0_bins = dataset_util.get_f0_bins(**kwargs_f0_bins)
    output_dict[key_f0_label] = dataset_util.f0_to_label(output_dict[key_f0], f0_bins)
    tuning_dict = compute_1d_tuning(output_dict,
                                    key_act=key_act,
                                    key_dim0=key_f0_label,
                                    normalize_unit_activity=normalize_unit_activity,
                                    tuning_dict=tuning_dict)
    f0_label_bins = tuning_dict[key_f0_label + '_bins']
    assert_msg = "stimuli must tile contiguous f0 label bins (wider f0 bins may help)"
    assert np.max(np.diff(f0_label_bins)) == 1, assert_msg
    f0_tuning_mean = tuning_dict[key_f0_label + '_tuning_mean']
    f0_tuning_std = tuning_dict[key_f0_label + '_tuning_std']
    f0_tuning_n = tuning_dict[key_f0_label + '_tuning_n']
    f0_bins = f0_bins[np.min(f0_label_bins) : np.max(f0_label_bins)+1]
    octave_max = np.log2(f0_bins[-1] / f0_bins[0])
    octave_bins = np.linspace(-octave_max, octave_max, 2*f0_bins.shape[0] - 1)
    octave_tuning_mean = np.zeros([octave_bins.shape[0], f0_tuning_mean.shape[1]])
    octave_tuning_std = np.zeros([octave_bins.shape[0], f0_tuning_mean.shape[1]])
    octave_tuning_n = np.zeros([octave_bins.shape[0], f0_tuning_mean.shape[1]])
    best_octave_index = f0_bins.shape[0] - 1
    best_f0_indexes = np.argmax(f0_tuning_mean, axis=0)
    for unit_index, best_f0_bin_index in enumerate(best_f0_indexes):
        idxS = best_octave_index - best_f0_bin_index
        idxE = idxS + f0_bins.shape[0]
        octave_tuning_mean[idxS:idxE, unit_index] = f0_tuning_mean[:, unit_index]
        octave_tuning_std[idxS:idxE, unit_index] = f0_tuning_std[:, unit_index]
        octave_tuning_n[idxS:idxE, unit_index] = f0_tuning_n[:, unit_index]
    tuning_dict['f0_bins'] = f0_bins
    tuning_dict['octave_bins'] = octave_bins
    tuning_dict['octave_tuning_mean'] = octave_tuning_mean
    tuning_dict['octave_tuning_std'] = octave_tuning_std
    tuning_dict['octave_tuning_n'] = octave_tuning_n
    return tuning_dict


def make_1d_tuning_plot(ax,
                        results_dict_input,
                        key_dim0='low_harm',
                        restrict_conditions=None,
                        include_yerr=True,
                        n_subsample=None,
                        random_seed=32,
                        kwargs_plot_update={},
                        kwargs_legend_update={},
                        kwargs_bootstrap={'bootstrap_repeats': 1000, 'metric_function': 'mean'},
                        **kwargs_format_axes):
    '''
    '''
    if not isinstance(results_dict_input, list):
        results_dict_input = [results_dict_input]
    
    if restrict_conditions is None:
        restrict_conditions = sorted(results_dict_input[0].keys())
    
    color_list = util_figures.get_color_list(len(restrict_conditions), 'copper')
    DATA = {}
    for cidx, key_condition in enumerate(restrict_conditions):
        yval_list = []
        for results_dict in results_dict_input:
            tuning_dict = results_dict[key_condition]
            xval = np.array(tuning_dict['{}_bins'.format(key_dim0)])
            yval_tmp = np.array(tuning_dict['{}_tuning_mean'.format(key_dim0)])
            if n_subsample is not None:
                assert n_subsample <= yval_tmp.shape[1], "n_subsample exceeds number of units"
                IDX = np.arange(0, yval_tmp.shape[1], 1, dtype=int)
                np.random.seed(random_seed)
                np.random.shuffle(IDX)
                yval_tmp = yval_tmp[:, IDX[:n_subsample]]
            yval_list.append(np.mean(yval_tmp, axis=1))
        yval_list = np.stack(yval_list, axis=0)
        DATA[key_condition] = {
            '{}_bins'.format(key_dim0): xval,
            '{}_tuning_mean'.format(key_dim0): yval_list,
        }
        yval, yerr = util_figures_psychophysics.combine_subjects(yval_list,
                                                                 kwargs_bootstrap=kwargs_bootstrap)
        kwargs_plot = {
            'label': key_condition,
            'color': color_list[cidx],
            'ls': '-',
            'lw': 2,
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
        'loc': 'upper center',
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
        legobj.set_linewidth(8.0)
    
    ax = util_figures.format_axes(ax, **kwargs_format_axes)
    return ax, DATA


def make_2d_tuning_plot(ax,
                        results_dict,
                        key_act='relu_4',
                        key_dim0='low_harm',
                        key_dim1='f0_label',
                        key_dim0_label='low_harm',
                        key_dim1_label='f0',
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
    dim0_bins = np.array(results_dict[key_act]['{}_bins'.format(key_dim0)])
    dim1_bins = np.array(results_dict[key_act]['{}_bins'.format(key_dim1)])
    if key_dim0_label is not None:
        dim0_labels = np.array(results_dict[key_act]['{}_bins'.format(key_dim0_label)])
    else:
        dim0_labels = dim0_bins
    if key_dim1_label is not None:
        dim1_labels = np.array(results_dict[key_act]['{}_bins'.format(key_dim1_label)])
    else:
        dim1_labels = dim1_bins
    tuning_mean = np.array(results_dict[key_act]['{}_{}_tuning_mean'.format(key_dim0, key_dim1)])
    if unit_idx is None:
        unit_idx = np.random.randint(0, tuning_mean.shape[-1], dtype=int)
        print('Randomly selected unit_idx={}'.format(unit_idx))
    # Plot 2d tuning array for a single unit
    im_data = tuning_mean[:, :, unit_idx].T
    IMG = ax.imshow(im_data,
                    origin=(0,0),
                    aspect='auto',
                    extent=[0, im_data.shape[1], 0, im_data.shape[0]],
                    cmap=plt.cm.gray)
    # Format axes
    dim0_ticks = np.linspace(0, dim0_bins.shape[0]-1, num=num_ticks_dim0, dtype=int)
    dim0_ticklabels = ['{:.0f}'.format(dim0_labels[tick]) for tick in dim0_ticks]
    dim1_ticks = np.linspace(0, dim1_bins.shape[0]-1, num=num_ticks_dim1, dtype=int)
    dim1_ticklabels = ['{:.0f}'.format(dim1_labels[tick]) for tick in dim1_ticks]
    kwargs = {
        'xticks': dim0_ticks,
        'yticks': dim1_ticks,
        'xticklabels': dim0_ticklabels,
        'yticklabels': dim1_ticklabels,

    }
    kwargs.update(kwargs_format_axes)
    ax = util_figures.format_axes(ax, **kwargs)
    return ax, IMG


def make_low_harm_tuning_plot(ax, results_dict_input, **kwargs):
    '''
    '''
    kwargs_make_1d_tuning_plot = {
        'key_dim0': 'low_harm',
        'str_xlabel': 'Lowest harmonic number',
        'str_ylabel': 'Mean activation\n(normalized)',
        'xlimits': [0, 31],
        'ylimits': [0, 1],
        'xticks': np.arange(0, 31, 5),
        'xticks_minor': np.arange(0, 31, 1),
    }
    kwargs_make_1d_tuning_plot.update(kwargs)
    ax, DATA = make_1d_tuning_plot(ax, results_dict_input, **kwargs_make_1d_tuning_plot)
    return ax, DATA


def make_f0_tuning_plot(ax, results_dict_input, **kwargs):
    '''
    '''
    if isinstance(results_dict_input, list):
        rd0 = results_dict_input[0]
    else:
        rd0 = results_dict_input
    td0 = rd0[sorted(rd0.keys())[0]]
    xval = np.array(td0['f0_label_bins'])
    xval_labels = np.array(td0['f0_bins'])
    xtick_indexes = np.linspace(xval[0], xval[-1], 7, dtype=int)
    xticks = [xval[xti] for xti in xtick_indexes]
    xticklabels = ['{:.0f}'.format(xval_labels[xti]) for xti in xtick_indexes]
    kwargs_make_1d_tuning_plot = {
        'key_dim0': 'f0_label',
        'str_xlabel': 'F0 (Hz)',
        'str_ylabel': 'Mean activation\n(normalized)',
        'xlimits': [xval[0], xval[-1]],
        'ylimits': [0, 1],
        'xticks': xticks,
        'xticklabels': xticklabels,
    }
    kwargs_make_1d_tuning_plot.update(kwargs)
    ax, DATA = make_1d_tuning_plot(ax, results_dict_input, **kwargs_make_1d_tuning_plot)
    return ax, DATA


def make_octave_tuning_plot(ax, results_dict_input, **kwargs):
    '''
    '''
    kwargs_make_1d_tuning_plot = {
        'key_dim0': 'octave',
        'n_subsample': 32,
        'str_xlabel': 'Octaves above best F0',
        'str_ylabel': 'Mean activation\n(normalized)',
        'xlimits': [-2, 2],
        'ylimits': [0, 1],
    }
    kwargs_make_1d_tuning_plot.update(kwargs)
    ax, DATA = make_1d_tuning_plot(ax, results_dict_input, **kwargs_make_1d_tuning_plot)
    return ax, DATA


if __name__ == "__main__":
    ''' TEMPORARY COMMAND-LINE USAGE '''
    
    def run_network_neurophysiology(output_dict):
        '''
        '''
        results_dict = {}
        for key in sorted(output_dict.keys()):
            if ('relu' in key):
                print('processing {}'.format(key))
                tuning_dict = {}
                tuning_dict = compute_f0_tuning_re_best(output_dict,
                                                        key_act=key,
                                                        tuning_dict=tuning_dict)
                if 'low_harm' in output_dict.keys():
                    tuning_dict = compute_1d_tuning(output_dict,
                                                    key_act=key,
                                                    key_dim0='low_harm',
                                                    tuning_dict=tuning_dict)
                    tuning_dict = compute_2d_tuning(output_dict,
                                                    key_act=key,
                                                    key_dim0='low_harm',
                                                    key_dim1='f0_label',
                                                    tuning_dict=tuning_dict)
                results_dict[key] = copy.deepcopy(tuning_dict)
        return results_dict
    
    
    assert len(sys.argv) == 2, "scipt usage: python <script_name> <output_directory_regex>"
    output_directory_regex = str(sys.argv[1])
    
    tfrecords_regex = '/om/user/msaddler/data_pitchnet/neurophysiology/bernox2005_SlidingFixedFilter_lharm01to30_phase0_f0min080_f0max320/sr20000_cf100_species002_spont070_BW10eN1_IHC3000Hz_IHC7order/*.tfrecords'
    output_directory_list = sorted(glob.glob(output_directory_regex))
    print('output_directory_list:')
    for output_directory in output_directory_list:
        print('<> {}'.format(output_directory))
    
    for output_directory in output_directory_list:
        print('\n\n\nSTART: {}'.format(output_directory))
        output_dict = get_network_activations(output_directory, tfrecords_regex)
        results_dict = run_network_neurophysiology(output_dict)
        fn_results_dict = os.path.join(output_directory, 'NEUROPHYSIOLOGY_bernox2005.json')
        with open(fn_results_dict, 'w') as f:
            json.dump(results_dict, f, cls=util_misc.NumpyEncoder, sort_keys=True)
        print('WROTE: {}\n\n\n'.format(fn_results_dict))
