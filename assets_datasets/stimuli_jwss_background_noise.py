import sys
import os
import glob
import numpy as np
import h5py
import librosa
import re

from stimuli_util import combine_signal_and_noise

sys.path.append('/om2/user/msaddler/python-packages/bez2018model')
from bez2018model_run_hdf5_dataset import initialize_hdf5_file, write_example_to_hdf5


def wav_files_to_hdf5(wav_regex, hdf5_filename, fs=32e3, disp_step=1):
    '''
    Function copies data from equally sized wav files to a single hdf5 file.
    NOTE:
        function is currently specific to wav files named `Recording##_Segment##.wav`
        function designed for storing Jarrod and Wiktor's background sounds as hdf5
    
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


def add_noise_to_tone_file(tone_hdf5_filename, noise_hdf5_filename, random_seed=None,
                           tone_signal_key='tone', tone_fs_key='config_tone/fs',
                           noise_signal_key='signal', noise_fs_key='signal_fs',
                           output_combined_key='tone_in_noise', output_noise_key='noise',
                           output_snr_key='snr', output_config_prefix='config_noise/',
                           buffer_dur=2., snr_range=[-10., 3.], dBSPL_range=[30., 90.],
                           dtype=np.float32, disp_step=100):
    '''
    Function adds noise from a source hdf5 file to a clean tone hdf5 dataset at a
    range of uniformly sampled SNRs and SPLs.
    
    Args
    ----
    tone_hdf5_filename (str):
    noise_hdf5_filename (str):
    random_seed (None or int):
    tone_signal_key (str):
    tone_fs_key (str):
    noise_signal_key (str):
    noise_fs_key (str):
    output_combined_key (str):
    output_noise_key (str):
    output_snr_key (str):
    output_config_prefix (str):
    buffer_dur (float):
    snr_range (list):
    dBSPL_range (list):
    dtype (np dtype):
    disp_step (int):
    '''
    if random_seed is not None: np.random.seed(random_seed)
    # Open the tone and noise hdf5 files
    print('[TONE_FILE]: {}'.format(tone_hdf5_filename))
    print('[NOISE_FILE]: {}'.format(noise_hdf5_filename))
    tone_f = h5py.File(tone_hdf5_filename, 'r+')
    noise_f = h5py.File(noise_hdf5_filename, 'r')
    # Check sampling rates are compatible
    assert_msg = "tone and noise hdf5 files must have the same sampling rate"
    assert tone_f[tone_fs_key][0] == noise_f[noise_fs_key][0], assert_msg
    # Check signal lengths are compatible
    (N, tone_length) = tone_f[tone_signal_key].shape
    (noise_N, noise_length) = noise_f[noise_signal_key].shape
    buffer_length = int(buffer_dur * noise_f[noise_fs_key][0])
    assert_msg = "noise_length - 2 * buffer_length must be greater than tone_length"
    assert noise_length - 2*buffer_length > tone_length, assert_msg
    clip_noise_range = [buffer_length, noise_length-buffer_length-tone_length]
    # Main loop to iterate over all signals in the tone hdf5 file
    for itrN in range(0, N):
        # Randomly sample a noise snoise_idx and clip_start_idx
        noise_idx = np.random.randint(0, noise_N)
        clip_start_idx = np.random.randint(clip_noise_range[0], clip_noise_range[1])
        clip_end_idx = clip_start_idx + tone_length
        noise = noise_f[noise_signal_key][noise_idx, clip_start_idx:clip_end_idx]
        # Randomly sample SNR and SPL
        snr = np.random.uniform(low=snr_range[0], high=snr_range[1])
        dBSPL = np.random.uniform(low=dBSPL_range[0], high=dBSPL_range[1])
        rms_out = 20e-6 * np.power(10, dBSPL/20)
        # Combine tone and noise with specified SNR and SPL
        tone = tone_f[tone_signal_key][itrN]
        tone_in_noise = combine_signal_and_noise(tone, noise, snr, rms_out=rms_out)
        data_dict = {
            output_combined_key: tone_in_noise,
            output_noise_key: noise,
            output_snr_key: snr,
            output_combined_key + '_rms': rms_out,
            output_combined_key + '_dBSPL': dBSPL,
            output_config_prefix + 'noise_idx': noise_idx,
            output_config_prefix + 'clip_start_idx': clip_start_idx,
        }
        # Initialize new datasets on the first iteration
        if itrN == 0:
            data_key_pair_list = []
            for key in data_dict.keys():
                data_key_pair_list.append((key, key))
                if not key in tone_f:
                    data_key_value = np.squeeze(np.array(data_dict[key]))
                    if not data_key_value.dtype == np.integer:
                        data_key_value = data_key_value.astype(dtype)
                    data_key_shape = [N] + list(data_key_value.shape)
                    tone_f.create_dataset(key, data_key_shape, dtype=data_key_value.dtype)
                    print('[INITIALIZED DATASET]: {}'.format(key), data_key_shape, data_key_value.dtype)
        # Write each data_dict to hdf5 file
        write_example_to_hdf5(tone_f, data_dict, itrN, data_key_pair_list=data_key_pair_list)
        if itrN % disp_step == 0:
            print('... signal {} of {}'.format(itrN, N))
    # Close the hdf5 files
    tone_f.close()
    noise_f.close()
    print('[END]: {}'.format(tone_hdf5_filename))


if __name__ == "__main__":
    ''' TEMPORARY COMMAND-LINE USAGE '''
    assert len(sys.argv) == 3, "scipt usage: python <script_name> <tone_hdf5_filename> <noise_hdf5_filename>"
    tone_hdf5_filename = str(sys.argv[1])
    noise_hdf5_filename = str(sys.argv[2])
    add_noise_to_tone_file(tone_hdf5_filename, noise_hdf5_filename)