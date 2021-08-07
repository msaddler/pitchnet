import os
import sys
import glob
import h5py
import warnings
import numpy as np


def get_dataset_paths_from_hdf5(f):
    '''
    Helper function to get list of paths to all h5py.Dataset objects in an open h5py.File object.
    '''
    hdf5_dataset_key_list = []
    def get_dataset_paths(name, node):
        if isinstance(node, h5py.Dataset):
            hdf5_dataset_key_list.append(name)
    f.visititems(get_dataset_paths)
    return hdf5_dataset_key_list


def write_example_to_hdf5(hdf5_f, data_dict, idx, data_key_pair_list=[]):
    '''
    Write individual example to open hdf5 file.
    
    Args
    ----
    hdf5_f (h5py.File object): open and writeable hdf5 file object
    data_dict (dict): dict containing data to write to hdf5 file
    idx (int): specifies row of hdf5 file to write to
    data_key_pair_list (list): list of tuples (hdf5_key, data_key)
    '''
    for (hdf5_key, data_key) in data_key_pair_list:
        try:
            hdf5_f[hdf5_key][idx] = np.squeeze(np.array(data_dict[data_key]))
        except:
            msg = "failed to write key `{}` for hdf5 file index `{}`"
            warnings.warn(msg.format(hdf5_key, idx))


def initialize_hdf5_file(hdf5_filename,
                         N,
                         data_dict,
                         data_key_pair_list=[],
                         config_key_pair_list=[],
                         file_mode='w',
                         vlen_data=False,
                         fillvalue=-1):
    '''
    Create a new hdf5 file and populate config parameters.
    
    Args
    ----
    hdf5_filename (str): filename for new hdf5 file
    N (int): number of examples that will be written to file (number of rows for datasets)
    data_dict (dict): dict containing auditory nerve model output and all metadata
    data_key_pair_list (list): list of tuples (hdf5_key, data_key) for datasets with N rows
    config_key_pair_list (list): list of tuples (hdf5_key, config_key) for config datasets (1 row)
    file_mode (str): 'w' = Create file, truncate if exists; 'w-' = Create file, fail if exists
    vlen_data (bool): if True, N-row datasets with more than 1 column will be variable length
    fillvalue (float or int): used to specify default value of fixed-length datasets with N rows 
    '''
    # Initialize hdf5 file
    f = h5py.File(hdf5_filename, file_mode)
    # Create the main output datasets
    for (hdf5_key, data_key) in data_key_pair_list:
        data_key_value = np.squeeze(np.array(data_dict[data_key]))
        is_str = False
        if not (np.issubdtype(data_key_value.dtype, np.number)):
            data_key_value = data_dict[data_key]
            is_str = True
        if vlen_data:
            # If variable-length data is allowed, 1-column non-NaN values will still be treated as fixed
            if is_str:
                data_key_dtype = h5py.special_dtype(vlen=type(data_key_value))
                f.create_dataset(hdf5_key, [N], dtype=data_key_dtype)
            elif (not is_str) and (len(data_key_value.shape) == 0) and (not np.isnan(data_key_value)):
                data_key_dtype = data_key_value.dtype
                f.create_dataset(hdf5_key, [N], dtype=data_key_dtype, fillvalue=fillvalue)
            else:
                data_key_dtype = h5py.special_dtype(vlen=data_key_value.dtype)
                f.create_dataset(hdf5_key, [N], dtype=data_key_dtype)
        else:
            # If variable-length data is not allowed, all datasets have fixed shape
            data_key_shape = [N] + list(data_key_value.shape)
            data_key_dtype = data_key_value.dtype
            f.create_dataset(hdf5_key, data_key_shape, dtype=data_key_dtype, fillvalue=fillvalue)
    # Create and populate the config datasets
    for (hdf5_key, config_key) in config_key_pair_list:
        config_key_value = data_dict[config_key]
        if isinstance(config_key_value, str):
            config_key_shape = [1]
            config_key_dtype = h5py.special_dtype(vlen=str)
        else:
            config_key_value = np.squeeze(np.array(config_key_value))
            config_key_shape = [1] + list(config_key_value.shape)
            config_key_dtype = config_key_value.dtype
        f.create_dataset(hdf5_key, config_key_shape, dtype=config_key_dtype, data=config_key_value)
    # Close the initialized hdf5 file
    f.close()


def check_hdf5_continuation(hdf5_filename,
                            check_key='parent_index',
                            check_key_fill_value=-1,
                            repeat_buffer=1):
    '''
    This function checks if the output dataset already exists and should be continued
    from the last populated row rather than restarted.
    
    Args
    ----
    hdf5_filename (str): filename for hdf5 dataset to check
    check_key (str): key in hdf5 file used to check for continuation (should be 1-dimensional dataset)
    check_key_fill_value (int or float): function will check for rows where check_key is equal to this value
    repeat_buffer (int): if continuing existing file, number of rows to be re-processed
    
    Returns
    -------
    continuation_flag (bool): True if hdf5 file exists and can be continued
    start_idx (int or None): row of hdf5 dataset at which to begin continuation
    '''
    continuation_flag = False
    start_idx = 0
    if os.path.isfile(hdf5_filename):
        f = h5py.File(hdf5_filename, 'r')
        if check_key in f:
            candidate_idxs = np.reshape(np.argwhere(f[check_key][:] == check_key_fill_value), [-1])
            continuation_flag = True
            if len(candidate_idxs > 0): start_idx = np.max([0, np.min(candidate_idxs)-repeat_buffer])
            else: start_idx = None
        else:
            warnings.warn('<<< check_key not found in hdf5 file; hdf5 dataset will be restarted >>>')
        f.close()
    return continuation_flag, start_idx


def get_f0_bins(f0_min=80., f0_max=1e3, binwidth_in_octaves=1/192):
    '''
    Get f0 bins for digitizing f0 values to log-spaced bins.
    
    Args
    ----
    f0_min (float): minimum f0 value, sets reference value for bins
    f0_max (float): maximum f0 value, determines number of bins
    binwidth_in_octaves (float): octaves above f0_min (1/192 = 1/16 semitone bins)
    
    Returns
    -------
    bins (np array): lower bounds of f0 bins
    '''
    max_octave = np.log2(f0_max / f0_min)
    bins = np.arange(0, max_octave, binwidth_in_octaves)
    bins = f0_min * (np.power(2, bins))
    return bins


def f0_to_label(f0_values, bins, right=False):
    '''
    Helper function to compute f0_labels from f0_values and bins.
    '''
    f0_labels = np.digitize(f0_values, bins, right=right) - 1
    return f0_labels


def label_to_f0(f0_labels, bins):
    '''
    Helper function to compute f0_value estimates from f0_labels (estimate is bin lower bound).
    '''
    assert np.all(f0_labels >= 0), "f0_labels must be positive"
    f0_values = bins[np.array(f0_labels).astype(int)]
    return f0_values


def f0_to_octave(f0_values, f0_ref=1.0):
    '''
    Helper function to convert f0_values to octaves above f0_ref.
    '''
    assert np.all(f0_values > 0), "f0_values must be greater than 0"
    return np.log2(f0_values / f0_ref)


def octave_to_f0(f0_octave, f0_ref=1.0):
    '''
    Helper function to convert octaves above f0_ref to f0_values.
    '''
    return f0_ref * (np.power(2, f0_octave))


def f0_to_normalized(f0_values, f0_min=80.0, f0_max=1001.3714, log_scale=True):
    '''
    Helper function to convert f0_values to [0, 1] range for regression.
    '''
    assert np.all(f0_values >= f0_min), "f0_values must be greater than or equal to f0_min"
    assert np.all(f0_values <= f0_max), "f0_values must be less than or equal to f0_max"
    if log_scale:
        f0_normalized = np.log(f0_values/f0_min) / np.log(f0_max/f0_min)
    else:
        f0_normalized = (f0_values - f0_min) / (f0_max - f0_min)
    return f0_normalized


def normalized_to_f0(f0_normalized, f0_min=80.0, f0_max=1001.3714, log_scale=True):
    '''
    Helper function to convert normalized f0_values back to Hz.
    '''
    if log_scale: f0_values = f0_min * np.exp(f0_normalized * np.log(f0_max/f0_min))
    else: f0_values = f0_normalized * (f0_max - f0_min) + f0_min
    return f0_values
