import os
import sys
import json
import numpy as np


def compute_tuning_tensor(output_dict,
                          key_act='relu_0',
                          key_x='low_harm',
                          key_y='f0_label_coarse',
                          normalize_act=True):
    '''
    '''
    x_unique = np.unique(output_dict[key_x])
    y_unique = np.unique(output_dict[key_y])
    shape = [x_unique.shape[0], y_unique.shape[0], output_dict[key_act].shape[1]]
    tuning_tensor = np.zeros(shape, dtype=output_dict[key_act].dtype)
    tuning_tensor_counts = np.zeros(shape[:-1] + [1], dtype=int)
    activations = output_dict[key_act]
    if normalize_act:
        activations -= np.amin(activations, axis=0)
        activations /= np.amax(activations, axis=0)
    x_value_indexes = np.digitize(output_dict[key_x], x_unique, right=True)
    y_value_indexes = np.digitize(output_dict[key_y], y_unique, right=True)
    for idx in range(output_dict[key_act].shape[0]):
        x_idx = x_value_indexes[idx]
        y_idx = y_value_indexes[idx]
        tuning_tensor[x_idx, y_idx, :] += activations[idx]
        tuning_tensor_counts[x_idx, y_idx] += 1
    tuning_tensor = tuning_tensor / tuning_tensor_counts
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
    coarse_f0_bins = dataset_util.get_f0_bins(**kwargs_f0_bins)
    output_dict['f0_label_coarse'] = dataset_util.f0_to_label(output_dict['f0'],
                                                              coarse_f0_bins)
    tuning_tensor = compute_tuning_tensor(output_dict,
                                          key_act=key_act,
                                          key_x='low_harm',
                                          key_y='f0_label_coarse')
    
    ### If specified, subsample the tuning tensor
    if n_subsample is not None:
        IDX = np.arange(0, tuning_tensor.shape[-1], 1, dtype=int)
        np.random.shuffle(IDX)
        tuning_tensor = tuning_tensor[:, :, IDX[:n_subsample]]
    
    ### Collapse tuning tensor along low_harm axis to get f0 tuning
    f0_tuning_array = np.mean(tuning_tensor, axis=0)
    print(tuning_tensor.shape, f0_tuning_array.shape)
    f0_bin_values = np.array([coarse_f0_bins[idx]
                              for idx in np.unique(output_dict['f0_label_coarse'])])
    
    ### If specified, shuffle the f0 axis to get null distribution
    if shuffle:
        indexes = np.arange(0, f0_tuning_array.shape[0])
        np.random.shuffle(indexes)
        f0_tuning_array = f0_tuning_array[indexes]
    
    ### Compute best f0s and setup octave tuning array
    best_f0s = f0_bin_values[np.argmax(f0_tuning_array, axis=0)]
    octave_bins = get_octave_bins(**kwargs_octave_bins)
    octave_tuning_array = -1 * np.ones([octave_bins.shape[0], f0_tuning_array.shape[1]])
    
    ### Populate octave tuning array
    for itr1 in range(f0_tuning_array.shape[1]):
        best_f0 = best_f0s[itr1]
        f0_tuning = f0_tuning_array[:, itr1]
        octaves_re_best_f0 = np.log2(f0_bin_values / best_f0)
        octave_indexes = np.digitize(octaves_re_best_f0, octave_bins)
        values = np.zeros_like(octave_bins)
        counts = np.zeros_like(octave_bins)
        for itr0, oct_idx in enumerate(octave_indexes):
            values[oct_idx] += f0_tuning_array[itr0, itr1]
            counts[oct_idx] += 1
        valid_indexes = counts > 0
        octave_tuning_array[valid_indexes, itr1] = values[valid_indexes] / counts[valid_indexes]

    return octave_bins, octave_tuning_array


def average_tuning_array(bins, tuning_array, normalize=True):
    '''
    '''
    assert bins.shape[0] == tuning_array.shape[0]
    if normalize:
        for itr1 in range(tuning_array.shape[1]):
            valid_indexes = tuning_array[:, itr1] >= 0
            tuning_array[valid_indexes, itr1] -= np.min(tuning_array[valid_indexes, itr1])
            tuning_array[valid_indexes, itr1] /= np.max(tuning_array[valid_indexes, itr1])
    
    tuning_array_mean = -1 * np.ones_like(bins)
    tuning_array_err = -1 * np.ones_like(bins)
    for itr0 in range(bins.shape[0]):
        valid_indexes = tuning_array[itr0, :] >= 0
        if any(valid_indexes):
            tuning_array_mean[itr0] = np.mean(tuning_array[itr0, valid_indexes])
            tuning_array_err[itr0] = np.std(tuning_array[itr0, valid_indexes])# / np.sqrt(np.sum(valid_indexes))
    
    valid_indexes = tuning_array_mean >= 0
    tuning_array_mean = tuning_array_mean[valid_indexes]
    tuning_array_err = tuning_array_err[valid_indexes]
    tuning_array_mean_bins = bins[valid_indexes]
    return tuning_array_mean_bins, tuning_array_mean, tuning_array_err
