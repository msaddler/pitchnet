import os
import sys
import json
import copy
import numpy as np
sys.path.append('/om4/group/mcdermott/user/msaddler/pitchnet_dataset/pitchnetDataset/pitchnetDataset')
import dataset_util


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
