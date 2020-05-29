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
    key_dim0 (str): output_dict key for stimulus metadata dimension of interest
    normalize_unit_activity (bool): if true, re-scale activations to fall between
        0 and 1 across all stimuli (each unit is scaled separately)
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
        dim0_tuning_mean -= np.min(dim0_tuning_mean, axis=0, keepdims=True)
        dim0_tuning_mean_max = np.max(dim0_tuning_mean, axis=0, keepdims=True)
        dead_unit_indexes = dim0_tuning_mean_max == 0
        dim0_tuning_mean_max[dead_unit_indexes] = 1
        dim0_tuning_mean /= dim0_tuning_mean_max
        dim0_tuning_std /= dim0_tuning_mean_max
    tuning_dict[key_dim0 + '_bins'] = dim0_bins
    tuning_dict[key_dim0 + '_tuning_mean'] = dim0_tuning_mean
    tuning_dict[key_dim0 + '_tuning_std'] = dim0_tuning_std
    tuning_dict[key_dim0 + '_tuning_n'] = dim0_tuning_n
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
        0 and 1 across all stimuli (each unit is scaled separately)
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


def compute_tuning_tensor(output_dict,
                          key_act='relu_0',
                          key_x='low_harm',
                          key_y='f0_label',
                          normalize_act=True):
    '''
    '''
    x_unique = np.unique(output_dict[key_x])
    y_unique = np.unique(output_dict[key_y])
    shape = [output_dict[key_act].shape[1], x_unique.shape[0], y_unique.shape[0]]
    tuning_tensor = np.zeros(shape, dtype=output_dict[key_act].dtype)
    tuning_tensor_counts = np.zeros(shape[1:], dtype=int)
    activations = copy.deepcopy(output_dict[key_act])
    if normalize_act:
        activations -= np.amin(activations, axis=0)
        max_activations = np.amax(activations, axis=0)
        for idx, max_activation in enumerate(max_activations):
            if max_activation > 0:
                activations[:, idx] /= max_activation
    x_value_indexes = np.digitize(output_dict[key_x], x_unique, right=True)
    y_value_indexes = np.digitize(output_dict[key_y], y_unique, right=True)
    for idx in range(output_dict[key_act].shape[0]):
        x_idx = x_value_indexes[idx]
        y_idx = y_value_indexes[idx]
        tuning_tensor[:, x_idx, y_idx] += activations[idx]
        tuning_tensor_counts[x_idx, y_idx] += 1
    valid_indexes = tuning_tensor_counts > 0
    for unit_idx in range(tuning_tensor.shape[0]):
        tuning_tensor_unit = tuning_tensor[unit_idx, :, :]
        tuning_tensor_unit[valid_indexes] /= tuning_tensor_counts[valid_indexes]
        tuning_tensor[unit_idx, :, :] = tuning_tensor_unit
    return tuning_tensor


def get_octave_bins(octave_min=-2.0,
                    octave_max=2.0,
                    num_bins=4*12*16+1):
    '''
    '''
    return np.linspace(octave_min, octave_max, num_bins)


def compute_octave_tuning_array(output_dict,
                                key_act='relu_0',
                                kwargs_f0_bins={},
                                kwargs_octave_bins={},
                                shuffle=False,
                                n_subsample=None):
    '''
    '''
    ### Compute generic tuning tensor (low_harm and f0 tuning)
    f0_bins = dataset_util.get_f0_bins(**kwargs_f0_bins)
    output_dict['f0_label'] = dataset_util.f0_to_label(output_dict['f0'], f0_bins)
    tuning_tensor = compute_tuning_tensor(output_dict,
                                          key_act=key_act,
                                          key_x='low_harm',
                                          key_y='f0_label',
                                          normalize_act=True)
    
    ### If specified, subsample the tuning tensor
    if n_subsample is not None:
        IDX = np.arange(0, tuning_tensor.shape[0], 1, dtype=int)
        np.random.shuffle(IDX)
        tuning_tensor = tuning_tensor[IDX[:n_subsample], :, :]
    
    ### Collapse tuning tensor along low_harm axis to get f0 tuning
    f0_tuning_array = np.mean(tuning_tensor, axis=1) # Units by F0-bins array
    f0_bin_values = np.array([f0_bins[idx] for idx in np.unique(output_dict['f0_label'])])
    
    ### If specified, shuffle the f0 axis to get null distribution
    if shuffle:
        indexes = np.arange(0, f0_tuning_array.shape[1])
        np.random.shuffle(indexes)
        f0_tuning_array = f0_tuning_array[:, indexes]
    
    ### Compute best f0s and setup octave tuning array
    best_f0s = f0_bin_values[np.argmax(f0_tuning_array, axis=1)]
    octave_bins = get_octave_bins(**kwargs_octave_bins)
    octave_tuning_array = np.empty([f0_tuning_array.shape[0], octave_bins.shape[0]]) # Units by octave bins array
    
    ### Populate octave tuning array
    for itr_unit in range(octave_tuning_array.shape[0]):
        best_f0 = best_f0s[itr_unit]
        f0_tuning = f0_tuning_array[itr_unit, :]
        octaves_re_best_f0 = np.log2(f0_bin_values / best_f0)
        octave_indexes = np.digitize(octaves_re_best_f0, octave_bins)
        values = np.zeros_like(octave_bins)
        counts = np.zeros_like(octave_bins)
        for itr_bin, octave_index in enumerate(octave_indexes):
            values[octave_index] += f0_tuning_array[itr_unit, itr_bin]
            counts[octave_index] += 1
        valid_indexes = counts > 0
        octave_tuning_array[itr_unit, valid_indexes] = values[valid_indexes] / counts[valid_indexes]
    
    return octave_bins, octave_tuning_array


def average_tuning_array(bins, tuning_array, normalize=True):
    '''
    '''
    assert bins.shape[0] == tuning_array.shape[1]
    if normalize:
        for itr_unit in range(tuning_array.shape[0]):
            valid_indexes = ~np.isnan(tuning_array[itr_unit, :])
            if valid_indexes.sum() > 0:
                tuning_array[itr_unit, valid_indexes] -= np.min(tuning_array[itr_unit, valid_indexes])
                tuning_array[itr_unit, valid_indexes] /= np.max(tuning_array[itr_unit, valid_indexes])
    
    tuning_array_mean = np.empty_like(bins)
    tuning_array_err = np.empty_like(bins)
    for itr_bin in range(bins.shape[0]):
        valid_indexes = ~np.isnan(tuning_array[:, itr_bin])
        if valid_indexes.sum() > 0:
            tuning_array_mean[itr_bin] = np.mean(tuning_array[valid_indexes, itr_bin])
            tuning_array_err[itr_bin] = np.std(tuning_array[valid_indexes, itr_bin])
    
    valid_indexes = ~np.isnan(tuning_array_mean)
    tuning_array_mean = tuning_array_mean[valid_indexes]
    tuning_array_err = tuning_array_err[valid_indexes]
    tuning_array_mean_bins = bins[valid_indexes]
    return tuning_array_mean_bins, tuning_array_mean, tuning_array_err
