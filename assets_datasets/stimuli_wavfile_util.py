import sys
import os
import glob
import numpy as np
import h5py
import librosa
import re

sys.path.append('/om2/user/msaddler/python-packages/bez2018model')
from bez2018model_run_hdf5_dataset import initialize_hdf5_file, write_example_to_hdf5


def wav_files_to_hdf5(wav_regex, hdf5_filename, fs=32e3, disp_step=1):
    '''
    Function copies data from equally sized wav files to a single hdf5 file.
    NOTE: function is currently specific to wav files named `Recording##_Segment##.wav`
    
    Args
    ----
    wav_regex (str): regular expression that globs all source wav files
    hdf5_filename (str): filename for output hdf5 file
    fs (int): sampling rate; note that audio files will be resampled by librosa (Hz)
    disp_step (int): progress is displayed every disp_step
    '''
    # Collect the filenames of the source wav files
    wav_fn_list = sorted(glob.glob(wav_regex))
    N = len(wav_fn_list)
    fs = int(fs)
    # Prepare data_dict and config_key_pair_list for hdf5 filewriting
    data_dict = {'signal_fs': fs}
    config_key_pair_list = [(k, k) for k in data_dict.keys()]
    data_key_pair_list = []
    # Main loop for loading each wav file and writing it to hdf5 file
    for itrN in range(0, N):
        signal, _ = librosa.load(wav_fn_list[itrN], sr=fs)
        wav_fn_basename = os.path.basename(wav_fn_list[itrN])
        wav_fn_numbers = [int(n) for n in re.findall(r"(\d+)", wav_fn_basename)]
        data_dict['signal'] = signal.astype(np.float32)
        data_dict['recording'] = wav_fn_numbers[0]
        data_dict['segment'] = wav_fn_numbers[1]
        # Initialize the hdf5 file on the first iteration
        if itrN == 0:
            print('[INITIALIZING]: {}'.format(hdf5_filename))
            for k in data_dict.keys():
                if not (k, k) in config_key_pair_list:
                    data_key_pair_list.append((k, k))
            initialize_hdf5_file(hdf5_filename, N, data_dict, file_mode='w',
                                 data_key_pair_list=data_key_pair_list,
                                 config_key_pair_list=config_key_pair_list,
                                 dtype=np.float32, cast_data=False, cast_config=False)
            hdf5_f = h5py.File(hdf5_filename, 'r+')
        # Write each data_dict to hdf5 file
        write_example_to_hdf5(hdf5_f, data_dict, itrN, data_key_pair_list=data_key_pair_list)
        if itrN % disp_step == 0:
            print('... signal {} of {} : {}'.format(itrN, N, wav_fn_basename))
    # Close hdf5 file
    hdf5_f.close()
    print('[END]: {}'.format(hdf5_filename))


if __name__ == "__main__":
    ''' TEMPORARY COMMAND-LINE USAGE '''
    assert len(sys.argv) == 3, "scipt usage: python <script_name> <wav_regex> <hdf5_filename>"
    wav_regex = str(sys.argv[1])
    hdf5_filename = str(sys.argv[2])
    wav_files_to_hdf5(wav_regex, hdf5_filename, fs=32e3, disp_step=1)