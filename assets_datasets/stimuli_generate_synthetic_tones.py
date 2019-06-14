import sys
import os
import numpy as np
import h5py
import scipy.signal

from stimuli_util import complex_tone

sys.path.append('/om2/user/msaddler/python-packages/bez2018model')
from bez2018model_run_hdf5_dataset import initialize_hdf5_file, write_example_to_hdf5


def lowpass_complex_tone(f0, fs, dur, attenuation_start=1000., attenuation_slope=0.,
                         phase_mode='sine', amp_noise=0.):
    '''
    Function generates a lowpass complex tone with harmonics attenuated
    linearly on a log scale above specified cutoff.
    
    Args
    ----
    f0 (float): fundamental frequency (Hz)
    fs (int): sampling rate (Hz)
    dur (float): duration of tone (s)
    attenuation_start (float): cutoff frequency for start of attenuation (Hz)
    attenuation_slope (float): slope in units of dB/octave above attenuation_start
    phase_mode (str): specifies relative phase between harmonics
    amp_noise (float): if > 0, scales noise multiplied with harmonic amplitudes
    
    Returns
    -------
    '''
    # Compute harmonic numbers and relative amplitude of each harmonic
    harmonics = np.arange(f0, fs/2, f0)
    harmonic_numbers = harmonics / f0
    dB_attenuation = -attenuation_slope * np.log2(harmonics / attenuation_start)
    dB_attenuation[dB_attenuation > 0] = 0
    amplitudes = np.power(10, (dB_attenuation/20))
    # Jitter the harmonic amplitudes with multiplicative noise
    if amp_noise > 0:
        amplitude_noise = np.random.uniform(1-amp_noise, 1+amp_noise, amplitudes.shape)
        amplitudes = amplitude_noise * amplitudes
    # Build and return the complex tone
    signal = complex_tone(f0, fs, dur, harmonic_numbers=harmonic_numbers,
                          amplitudes=amplitudes,
                          phase_mode=phase_mode,
                          offset_start=True, strict_nyquist=True)
    return signal


def generate_lowpass_complex_tone_dataset(hdf5_filename, N, fs=32e3, dur=0.150,
                                          disp_step=100, f0_min=80., f0_max=1e3,
                                          amp_noise=0., phase_mode_list=['sine'],
                                          atten_start_min=10., atten_start_max=4e3,
                                          atten_slope_min=10., atten_slope_max=60.):
    '''
    Main routine for generating a dataset of randomly-sampled lowpass complex tones.
    
    Args
    ----
    hdf5_filename (str): 
    N (int):
    fs (int): sampling rate (Hz)
    dur (float): duration of tones (s)
    disp_step (int): every disp_step, progress is displayed
    f0_min (float):
    f0_max (float):
    amp_noise (float):
    phase_mode_list (list):
    atten_start_min (float):
    atten_start_max (float):
    atten_slope_min (float):
    atten_slope_max (float):
    '''
    # Randomly sample f0s, phase_modes, and attenuation parameters
    f0_list = np.exp(np.random.uniform(np.log(f0_min), np.log(f0_max), size=[N]))
    phase_mode_list = np.random.choice(phase_mode_list, size=[N])
    attenuation_start_list = np.exp(np.random.uniform(np.log(atten_start_min), np.log(atten_start_max), size=[N]))
    attenuation_slope_list = np.random.uniform(atten_slope_min, atten_slope_max, size=[N])
    # Define encoding / decoding dictionaries for phase_mode
    phase_mode_encoding = {'sine':0, 'rand':1, 'sch':2, 'cos':3, 'alt':4}
    phase_mode_decoding = {0:'sine', 1:'rand', 2:'sch', 3:'cos', 4:'alt'}
    # Prepare data_dict and config_key_pair_list for hdf5 filewriting
    data_dict = {
        'config_tone/fs': fs,
        'config_tone/dur': dur,
        'config_tone/f0_min': f0_min,
        'config_tone/f0_max': f0_max,
        'config_tone/atten_start_min': atten_start_min,
        'config_tone/atten_start_max': atten_start_max,
        'config_tone/atten_slope_min': atten_slope_min,
        'config_tone/atten_slope_max': atten_slope_max,
        'config_tone/amp_noise': amp_noise,
    }
    config_key_pair_list = [(k, k) for k in data_dict.keys()]
    data_key_pair_list = [] # Will be populated right before initializing hdf5 file
    # Main loop to generate lowpass complex tones
    for itrN in range(0, N):
        signal = lowpass_complex_tone(f0_list[itrN], fs, dur,
                                      attenuation_start=attenuation_start_list[itrN],
                                      attenuation_slope=attenuation_slope_list[itrN],
                                      phase_mode=phase_mode_list[itrN],
                                      amp_noise=amp_noise)
        data_dict['tone'] = signal
        data_dict['f0'] = f0_list[itrN]
        data_dict['attenuation_start'] = attenuation_start_list[itrN]
        data_dict['attenuation_slope'] = attenuation_slope_list[itrN]
        data_dict['phase_mode'] = phase_mode_encoding[phase_mode_list[itrN]]
        # Initialize the hdf5 file on the first iteration
        if itrN == 0:
            print('[INITIALIZING]: {}'.format(hdf5_filename))
            for k in data_dict.keys():
                if not (k, k) in config_key_pair_list:
                    data_key_pair_list.append((k, k))
            initialize_hdf5_file(hdf5_filename, N, data_dict, file_mode='w',
                                 data_key_pair_list=data_key_pair_list,
                                 config_key_pair_list=config_key_pair_list,
                                 dtype=np.float32, cast_data=True, cast_config=False)
            hdf5_f = h5py.File(hdf5_filename, 'r+')
        # Write each data_dict to hdf5 file
        write_example_to_hdf5(hdf5_f, data_dict, itrN, data_key_pair_list=data_key_pair_list)
        if itrN % disp_step == 0:
            print('... signal {} of {}'.format(itrN, N))
    # Close hdf5 file
    hdf5_f.close()
    print('[END]: {}'.format(hdf5_filename))


if __name__ == "__main__":
    ''' TEMPORARY COMMAND-LINE USAGE '''
    assert len(sys.argv) == 3, "scipt usage: python <script_name> <hdf5_filename> <N>"
    hdf5_filename = str(sys.argv[1])
    N = int(sys.argv[2])
    generate_lowpass_complex_tone_dataset(hdf5_filename, N)