import sys
import os
import h5py
import glob
import numpy as np


def get_f0_bins(f0_min=80, f0_max=1e3, binwidth_in_octaves=1/192):
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


def f0_to_label(f0_values, bins):
    '''
    Helper function to compute f0_labels from f0_values and bins.
    '''
    f0_labels = np.digitize(f0_values, bins) - 1
    assert np.all(f0_labels >= 0), "f0_values below lowest bin"
    return f0_labels


def label_to_f0(f0_labels, bins, strict_bin_minimum=True):
    '''
    Helper function to compute f0_value estimates from f0_labels (estimate is bin lower bound).
    '''
    assert np.all(f0_labels >= 0), "f0_labels must be positive"
    f0_values = bins[np.array(f0_labels).astype(int)]
    return f0_values


def normalize_f0(f0_values, f0_min=80., f0_max=1e3, log_scale=True):
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


def add_f0_label_to_hdf5(hdf5_filename, f0_key='f0', f0_label_key='f0_label', f0_normal_key='f0_lognormal',
                         f0_label_dtype=np.int64, f0_bin_kwargs={}, f0_normalization_kwargs={}):
    '''
    Function adds or recomputes f0 labels dataset within specified hdf5 file.
    
    Args
    ----
    hdf5_filename (str)
    f0_key (str): hdf5 path to f0 values (hdf5 dataset should have shape (N,))
    f0_label_key (str): hdf5 path to f0 labels (this dataset will be added or overwritten)
    f0_normal_key (str): hdf5 path to normalized f0 values (this dataset will be added or overwritten)
    f0_label_dtype (np.dtype): datatype for f0 label dataset
    f0_bin_kwargs (dict): kwargs for `get_f0_bins()` (parameters for computing f0 label bins)
    f0_normalization_kwargs (dict): kwargs for `f0_normalization_kwargs()` (parameters for normalizing f0 values)
    '''
    bins = get_f0_bins(**f0_bin_kwargs)
    print('[ADDING F0 LABELS]: {}'.format(hdf5_filename))
    print('[ADDING F0 LABELS]: f0_key={}, f0_label_key={}'.format(f0_key, f0_label_key))
    hdf5_f = h5py.File(hdf5_filename, 'r+')
    f0_values = hdf5_f[f0_key][:]
    f0_labels = f0_to_label(f0_values, bins)
    f0_normal = normalize_f0(f0_values, **f0_normalization_kwargs)
    # Write the f0 bin labels to hdf5 file
    if f0_label_key in hdf5_f:
        print('[OVERWRITING DATASET]: {}'.format(f0_label_key))
        hdf5_f[f0_label_key][:] = f0_labels
    else:
        print('[INITIALIZING DATASET]: {}'.format(f0_label_key))
        hdf5_f.create_dataset(f0_label_key, f0_labels.shape, dtype=f0_label_dtype, data=f0_labels)
    # Write the f0 normalized values to hdf5 file
    if f0_normal_key in hdf5_f:
        print('[OVERWRITING DATASET]: {}'.format(f0_normal_key))
        hdf5_f[f0_normal_key][:] = f0_normal
    else:
        print('[INITIALIZING DATASET]: {}'.format(f0_normal_key))
        hdf5_f.create_dataset(f0_normal_key, f0_normal.shape, dtype=f0_values.dtype, data=f0_normal)
    hdf5_f.close()
    print('[END]: {}'.format(hdf5_filename))


if __name__ == "__main__":
    ''' TEMPORARY COMMAND-LINE USAGE '''
    assert len(sys.argv) == 2, "scipt usage: python <script_name> <hdf5_filename>"
    hdf5_fn_regex = str(sys.argv[1])
    hdf5_fn_list = sorted(glob.glob(hdf5_fn_regex))
    for hdf5_filename in hdf5_fn_list:
        print('|===| {} |===|'.format(hdf5_filename))
        add_f0_label_to_hdf5(hdf5_filename, f0_key='f0', f0_label_key='f0_label', f0_normal_key='f0_lognormal',
                             f0_label_dtype=np.int64, f0_bin_kwargs={}, f0_normalization_kwargs={})
