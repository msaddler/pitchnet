import sys
import os
import h5py
import tensorflow as tf
import numpy as np
import glob


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))    


def create_tfrecords(output_fn, source_file, feature_paths={}, idx_start=0, idx_end=None, disp_step=500):
    """
    This function writes data from an open hdf5 file to a new tfrecords file.
    The tfrecords file will have the same keys as the hdf5 file.
    
    Args
    ----
    output_fn (string): filename for output tfrecords file
    source_file (h5py File object): open and readable hdf5 file
    feature_paths (dict): keys are ['bytes_list', 'int_list', 'float_list'],
        fields are lists of key paths that point to fields in `source_file`.
    idx_start (int): determines which row in hdf5 to start reading from
    idx_end (int or None): determines which row in hdf5 to stop reading from (None --> last)
    disp_step (int): every disp_step, progress is displayed
    """
    options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    writer = tf.python_io.TFRecordWriter(output_fn, options=options)
    if idx_end == None:
        idx_end = source_file[feature_paths['bytes_list'][0]].shape[0]
        print('Setting `idx_end={}`'.format(idx_end))
    for idx in np.arange(idx_start, idx_end):
        # Initialize feature dict for each example
        feature = {}
        # Populate the feature dict using bytes, int, and float features
        for key_path in feature_paths.get('bytes_list', []):
            if key_path in source_file:
                feature_data = source_file[key_path][idx]
                if feature_data.dtype == np.float64: # Down-cast float64 to float32 for tensorflow
                    feature_data = feature_data.astype(np.float32)
                feature[key_path] = _bytes_feature(tf.compat.as_bytes(feature_data.tostring()))
            elif idx == idx_start: print('IGNORING `{}` (not found in source_file)'.format(key_path))
        for key_path in feature_paths.get('int_list', []):
            if key_path in source_file:
                feature[key_path] = _int64_feature(source_file[key_path][idx])
            elif idx == idx_start: print('IGNORING `{}` (not found in source_file)'.format(key_path))
        for key_path in feature_paths.get('float_list', []):
            if key_path in source_file:
                feature[key_path] = _float_feature(source_file[key_path][idx])
            elif idx == idx_start: print('IGNORING `{}` (not found in source_file)'.format(key_path))
        if idx % disp_step == 0:
            print('idx_start: {:09} | idx_end: {:09} | idx: {:09}'.format(idx_start, idx_end, idx))
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        # Serialize to string and write on the file
        writer.write(example.SerializeToString())
    writer.close()
    sys.stdout.flush()


def get_feature_paths_from_source_file(source_file, groups_to_search=['/']):
    """
    Function searches for suitable keys in the provided hdf5 file object to generate a
    feature_paths dictionary for tfrecord writing. Suitable keys are those corresponding
    to datasets with first dimension greater than 1.
    
    Args
    ----
    source_file (h5py File object): open and readable hdf5 file
    groups_to_search (list): list of groups in `source_file` to search for features
    
    Returns
    -------
    feature_paths (dict): keys are ['bytes_list', 'int_list', 'float_list'],
        fields are lists of key paths that point to fields in `source_file`.
    """
    feature_paths = {
        'bytes_list': [],
        'int_list': [],
        'float_list': [],
    }
    key_candidate_list = []
    for group in groups_to_search:
        if group in source_file:
            key_candidate_list.extend(list(source_file[group].keys()))
    for key in key_candidate_list:
        if isinstance(source_file[key], h5py.Dataset):
            if source_file[key].shape[0] > 1:
                if len(source_file[key].shape) > 1:
                    feature_paths['bytes_list'].append(key)
                else:
                    if source_file[key].dtype == np.integer:
                        feature_paths['int_list'].append(key)
                    else:
                        feature_paths['float_list'].append(key)
    return feature_paths


def parallel_run_tfrecords(source_regex, job_idx=0, jobs_per_source_file=1, groups_to_search=['/']):
    """
    Wrapper function to easily parallelize `create_tfrecords()`.
    
    Args
    ----
    source_regex (str): regular expression that globs all source hdf5 filenames
    job_idx (int): index of current job
    jobs_per_source_file (int): number of jobs each source file is split into
    groups_to_search (list): list of groups in `source_file` to search for features
        (argument for `get_feature_paths_from_source_file()`)
    """
    # Determine the source_hdf5_filename using source_regex, job_idx, and jobs_per_source_file
    source_fn_list = sorted(glob.glob(source_regex))
    assert len(source_fn_list) > 0, "source_regex did not match any files"
    source_file_idx = job_idx // jobs_per_source_file
    assert source_file_idx < len(source_fn_list), "source_file_idx out of range"
    source_hdf5_filename = source_fn_list[source_file_idx]
    # Open the source hdf5 file and determine the feature paths from hdf5 paths
    source_hdf5_f = h5py.File(source_hdf5_filename, 'r')
    feature_paths = get_feature_paths_from_source_file(source_hdf5_f, groups_to_search=groups_to_search)
    print('>>> [PARALLEL_RUN] feature_dict:')
    for key in feature_paths.keys():
        print('--| {}:'.format(key))
        for feat_path in feature_paths[key]:
            print('-----|', feat_path, source_hdf5_f[feat_path].shape)
    # Compute idx_start and idx_end within source_hdf5_filename for the given job_idx
    N = source_hdf5_f[feature_paths['bytes_list'][0]].shape[0]
    idx_splits = np.linspace(0, N, jobs_per_source_file + 1, dtype=int)
    idx_start = idx_splits[job_idx % jobs_per_source_file]
    idx_end = idx_splits[(job_idx % jobs_per_source_file) + 1]
    # Design output filename for the tfrecords
    sidx = source_hdf5_filename.rfind('.')
    if jobs_per_source_file > 1:
        output_tfrecords_fn = source_hdf5_filename[:sidx] + '_{:06d}-{:06d}.tfrecords'.format(idx_start, idx_end)
    else:
        output_tfrecords_fn = source_hdf5_filename[:sidx] + '.tfrecords'
    # Call `create_tfrecords()` to generate the tfrecords file
    print('>>> [PARALLEL_RUN] job_idx: {}, source_file_idx: {} of {}, jobs_per_source_file: {}'.format(
        job_idx, source_file_idx, len(source_fn_list), jobs_per_source_file))
    print('>>> [PARALLEL_RUN] source_hdf5_filename: {}'.format(source_hdf5_filename))
    print('>>> [PARALLEL_RUN] output_tfrecords_fn: {}'.format(output_tfrecords_fn))
    print('>>> [PARALLEL_RUN] feature_paths:', feature_paths)
    print('>>> [PARALLEL_RUN] idx_start: {}, idx_end: {}'.format(idx_start, idx_end))
    create_tfrecords(output_tfrecords_fn, source_hdf5_f,
                     feature_paths=feature_paths,
                     idx_start=idx_start, idx_end=idx_end)
    # Close the source hdf5 file
    source_hdf5_f.close()
    print('>>> [END] {}'.format(output_tfrecords_fn))


if __name__ == "__main__":
    ''' TEMPORARY COMMAND LINE USAGE '''
    assert len(sys.argv) == 4, "scipt usage: python <script_name> <source_regex> <int> <int>"
    source_regex = str(sys.argv[1])
    job_idx = int(sys.argv[2])
    jobs_per_source_file = int(sys.argv[3])
    
    parallel_run_tfrecords(source_regex,
                           job_idx=job_idx,
                           jobs_per_source_file=jobs_per_source_file,
                           groups_to_search=['/'])
