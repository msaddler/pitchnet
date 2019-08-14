import sys
import os
import h5py
import glob
import numpy as np
import argparse
sys.path.append('/om4/group/mcdermott/user/msaddler/pitchnet_dataset/pitchnetDataset/pitchnetDataset')
from dataset_util import get_f0_bins, f0_to_label, f0_to_octave, f0_to_normalized
from dataset_util import label_to_f0, octave_to_f0, normalized_to_f0


def add_f0_label_to_hdf5(hdf5_filename, source_f0_key,
                         f0_key='f0', f0_label_key='f0_label', f0_octave_key='f0_log2', f0_normal_key='f0_lognormal',
                         f0_label_dtype=np.int64, f0_bin_kwargs={}, f0_octave_kwargs={}, f0_normalization_kwargs={}):
    '''
    Function adds or recomputes f0 labels and normalized values in specified hdf5 file.
    
    Args
    ----
    hdf5_filename (str): source filename (this hdf5 file will be modified)
    source_f0_key (str): source path to f0 values in the hdf5 dataset
    f0_key (str): hdf5 output path for f0 values (dataset will be added or overwritten)
    f0_label_key (str): output path path for f0 labels (dataset will be added or overwritten)
    f0_octave_key (str): output path for f0 octave values (dataset will be added or overwritten)
    f0_normal_key (str): output path for normalized f0 values (dataset will be added or overwritten)
    f0_label_dtype (np.dtype): datatype for f0 label dataset
    f0_bin_kwargs (dict): kwargs for `get_f0_bins()` (parameters for computing f0 label bins)
    f0_octave_kwargs (dict): kwargs for `f0_to_octave()` (f0_ref for Hz to octave conversion, default is f0_ref=1.0)
    f0_normalization_kwargs (dict): kwargs for `f0_to_normalized()` (parameters for normalizing f0 values)
    '''
    print('[ADDING F0 LABELS]: {}'.format(hdf5_filename))
    print('source_f0_key=`{}`'.format(source_f0_key))
    print('f0_key=`{}`, f0_label_key=`{}`, f0_normal_key=`{}`'.format(f0_key, f0_label_key, f0_normal_key))
    hdf5_f = h5py.File(hdf5_filename, 'r+')
    
    f0_bins = get_f0_bins(**f0_bin_kwargs)
    output_dict = {
        f0_key: hdf5_f[source_f0_key][:],
        f0_label_key: f0_to_label(hdf5_f[source_f0_key][:], f0_bins),
        f0_octave_key: f0_to_octave(hdf5_f[source_f0_key][:], **f0_octave_kwargs),
        f0_normal_key: f0_to_normalized(hdf5_f[source_f0_key][:], **f0_normalization_kwargs),
    }
    
    for key in output_dict.keys():
        if (key in hdf5_f) and (not key == source_f0_key):
            print('overwriting dataset: {}'.format(key))
            hdf5_f[key][:] = output_dict[key]
        elif (key in hdf5_f) and (key == source_f0_key):
            print('source_f0_key and f0_key are equal: {}'.format(key))
        else:
            print('initializing dataset: {}'.format(key))
            if key == f0_label_key: dtype = f0_label_dtype
            else: dtype = output_dict[key].dtype
            hdf5_f.create_dataset(key, output_dict[key].shape, dtype=dtype, data=output_dict[key])
        print('... key=`{}`, min_value={}, max_value={}'.format(key, np.min(output_dict[key]), np.max(output_dict[key])))
    hdf5_f.close()
    print('[END]: {}'.format(hdf5_filename))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add f0 labels to dataset")
    parser.add_argument('-r', '--hdf5_fn_regex', type=str,
                        help='regexp that globs all hdf5 files to process')
    parser.add_argument('-skf', '--source_f0_key', type=str, 
                        help='source path for f0 values')
    parser.add_argument('-kf', '--f0_key', type=str, default='f0',
                        help='destination path for f0 values (if different from source_f0_key)')
    parser.add_argument('-kfl', '--f0_label_key', type=str, default='f0_label',
                        help='destination path for f0 label values')
    parser.add_argument('-kfo', '--f0_octave_key', type=str, default='f0_log2',
                        help='destination path for f0 octave values')
    parser.add_argument('-kfn', '--f0_normal_key', type=str, default='f0_lognormal',
                        help='destination path for f0 normalized values')
    
    args = parser.parse_args()
    assert args.hdf5_fn_regex is not None, "-r (--hdf5_fn_regex) is a required argument"
    assert args.source_f0_key is not None, "-skf (--source_f0_key) is a required argument"
    
    hdf5_fn_list = sorted(glob.glob(args.hdf5_fn_regex))
    for hdf5_filename in hdf5_fn_list:
        print('=== {} ==='.format(hdf5_filename))
        add_f0_label_to_hdf5(hdf5_filename, args.source_f0_key,
                             f0_key=args.f0_key,
                             f0_label_key=args.f0_label_key,
                             f0_octave_key=args.f0_octave_key,
                             f0_normal_key=args.f0_normal_key,
                             f0_label_dtype=np.int64,
                             f0_bin_kwargs={},
                             f0_normalization_kwargs={})
