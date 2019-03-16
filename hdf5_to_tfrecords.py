import h5py
import tensorflow as tf
import numpy as np
import sys
import glob


# Tensorflow datatype conversion functions
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def create_record(output_fn, source_file, feature_paths):
    options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    writer = tf.python_io.TFRecordWriter(output_fn, options=options)

    N = source_file[feature_paths['bytes_list'][0][0]].shape[0]
    for idx in range(N):
        # Initialize feature dict for each example
        feature = {}

        # Populate the feature dict using bytes, int, and float features
        for (hdf5_path, tfr_path) in feature_paths['bytes_list']:
            feature[tfr_path] = _bytes_feature(tf.compat.as_bytes(source_file[hdf5_path][idx].tostring()))
        for (hdf5_path, tfr_path) in feature_paths['int_list']:
            feature[tfr_path] = _int64_feature(source_file[hdf5_path][idx])
        for (hdf5_path, tfr_path) in feature_paths['float_list']:
            feature[tfr_path] = _float_feature(source_file[hdf5_path][idx])

        if idx % 100 == 0:
            print('writing example {:06} of {:06}'.format(idx, N))

        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Serialize to string and write on the file
        writer.write(example.SerializeToString())

    writer.close()
    sys.stdout.flush()


if __name__ == "__main__":
    # SET UP SCRIPT TO RUN FROM COMMAND LINE

    assert len(sys.argv) == 2, 'COMMAND LINE USAGE: run <script_name.py> <dset_id>'
    # Get dset_id from command line
    dset_id = int(sys.argv[1])

    # Filename handling: provide source filename regex and specify output filename
    source_regex = '/om/scratch/Tue/msaddler/ANmodel_data_BEZ_10kHz/pitchdatasetsSoundDetection/bernox2005_puretone_detection_threshold_expt_f0s0200-6000Hz-cutoff-300Hz-rolloff-4_CF50-SR70-0fGn_sp2_filt00_*.hdf5'
    #source_regex = '/om/scratch/Tue/msaddler/JSIN_all__run_*.h5'
    source_fn_list =  sorted(glob.glob(source_regex))
    source_fn = source_fn_list[dset_id]
    output_fn = source_fn.replace('.hdf5', '.tfrecords') # <--- Set how .tfrecords output_fn is set from source_fn

    # Specify which features to store in the .tfrecords file (supports bytes, int, and float features)
    # - Comment out unneeded features to speed up (e.g. noise-less-signal and noise-only-signal)
    # - Tuples contain pairs of strings: (hdf5_path, tfrecords_path)
    feature_paths = {} # The following lists are (hdf5_path, tfrecords_path) pairs for each feature
    feature_paths['bytes_list'] = [
        ('/meanrates', '/meanrates'),
    ]
    feature_paths['int_list'] = [
        ('/sound_detection/present_flag', '/sound_detection/present_flag')
    ]
    feature_paths['float_list'] = [
        ('/f0', '/f0'),
        ('/augmentation/snr', '/augmentation/snr'),
        ('/augmentation/pin_dBSPL', '/augmentation/pin_dBSPL'),
        ('/diagnostic/signal_dBSPL', '/diagnostic/signal_dBSPL'),
    ]

    # Open the source hdf5 file
    source_f = h5py.File(source_fn, 'r')

    # Call tfrecords writer function
    print('[START]', output_fn)
    create_record(output_fn, source_f, feature_paths)
    print('[END]', output_fn)

    # Close the source hdf5 file
    source_f.close()
