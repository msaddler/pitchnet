import sys
import os
import numpy as np
import h5py
import scipy.signal
import itertools
import pdb

import stimuli_util

sys.path.append('/om4/group/mcdermott/user/msaddler/pitchnet_dataset/pitchnetDataset/pitchnetDataset')
from dataset_util import initialize_hdf5_file, write_example_to_hdf5


def get_bandpass_filter_frequency_response(fl, fh, fs=32e3, order=4):
    '''
    Returns a function that computes the frequency response in dB of a bandpass
    filter with -3dB cutoff frequencies [fl, fh].
    
    Args
    ----
    fl (float): -3dB cutoff frequency of the highpass filter (Hz) before re-scaling
    fh (float): -3dB cutoff frequency of the lowpass filter (Hz) before re-scaling
    fs (int): sampling rate (Hz); NOTE: returned function is only valid w.r.t. fs
    order (int): order of the butterworth filters
    
    Returns
    -------
    frequency_response_in_dB (function): frequency response function (dB)
    '''
    assert fl <= fh, "The highpass cutoff must be less than the lowpass cutoff"
    # Design a lowpass and a highpass butterworth filter
    [Bl, Al] = scipy.signal.butter(order, fh / (fs/2), btype='low')
    [Bh, Ah] = scipy.signal.butter(order, fl / (fs/2), btype='high')
    # Create the frequency-amplitude response function and return
    def frequency_response_in_dB(freqs):
        ''' Amplitude response of the double filtering operation in dB '''
        [_, Hh] = scipy.signal.freqz(Bh, a=Ah, worN=2*np.pi*freqs/fs, whole=False)
        [_, Hl] = scipy.signal.freqz(Bl, a=Al, worN=2*np.pi*freqs/fs, whole=False)
        return 20.* np.log10(np.abs(Hl * Hh))
    return frequency_response_in_dB


def generate_AltPhase_dataset(hdf5_filename, fs=32000, dur=0.150, phase_modes=['alt', 'sine'],
                              f0_min=80.0, f0_max=320.0, step_size_in_octaves=1/(192*16),
                              include_2xF0=False, filter_order=8, passband_component_dBSPL=50.0,
                              noise_dBHzSPL=15.0, noise_attenuation_start=600.0,
                              noise_attenuation_slope=2, disp_step=100):
    '''
    Main routine for generating Shackleton & Carlyon (1994, JASA) ALT phase dataset.
    
    Args
    ----
    '''
    # Define encoding / decoding dictionaries for phase_mode
    phase_mode_encoding = {'sine':0, 'rand':1, 'sch':2, 'cos':3, 'alt':4}
    phase_mode_decoding = {0:'sine', 1:'rand', 2:'sch', 3:'cos', 4:'alt'}
    
    # Define stimulus-specific parameters
    list_f0 = f0_min * (np.power(2, np.arange(0, np.log2(f0_max / f0_min), step_size_in_octaves)))
    list_filter_params = [
        {'fl': 125.0, 'fh': 625.0, 'fs': fs, 'order':filter_order},
        {'fl': 1375.0, 'fh': 1875.0, 'fs': fs, 'order':filter_order},
        {'fl': 3900.0, 'fh': 5400.0, 'fs': fs, 'order':filter_order},
    ]
    list_phase = np.array([phase_mode_encoding[p] for p in phase_modes])
    N = len(list_f0) * len(list_filter_params) * len(list_phase)
    if (include_2xF0) and (0 in list_phase):
        N = N + len(list_f0) * len(list_filter_params)
    
    # Build frequency response functions for the different filter conditions
    list_filters = []
    for filter_params in list_filter_params:
        frequency_response_in_dB = get_bandpass_filter_frequency_response(**filter_params)
        list_filters.append({'freq_response': frequency_response_in_dB, **filter_params})
    
    # Prepare config_dict with config values
    config_dict = {
        'sr': fs,
        'config_tone/fs': fs,
        'config_tone/dur': dur,
        'config_tone/filter_order': filter_order,
        'config_tone/passband_component_dBSPL': passband_component_dBSPL,
        'config_tone/noise_dBHzSPL': noise_dBHzSPL,
        'config_tone/noise_attenuation_start': noise_attenuation_start,
        'config_tone/noise_attenuation_slope': noise_attenuation_slope,
    }
    config_key_pair_list = [(k, k) for k in config_dict.keys()]
    
    # Iterate over all combinations of stimulus-specific parameters
    itrN = 0
    for phase in list_phase:
        for filter_dict in list_filters:
            # Get the frequency response function for the current filter condition
            frequency_response_in_dB = filter_dict['freq_response']
            for f0 in list_f0:
                # Build signal with specified phase, filter, and f0
                harmonic_freqs = np.arange(f0, fs/2, f0)
                harmonic_numbers = harmonic_freqs / f0
                harmonic_dBSPL = passband_component_dBSPL + frequency_response_in_dB(harmonic_freqs)
                amplitudes = 20e-6 * np.power(10, (harmonic_dBSPL/20))
                signal = stimuli_util.complex_tone(f0, fs, dur, harmonic_numbers=harmonic_numbers,
                                                   amplitudes=amplitudes,
                                                   phase_mode=phase_mode_decoding[phase])
                # Construct modified uniform masking noise
                noise = stimuli_util.modified_uniform_masking_noise(
                    fs, dur, dBHzSPL=noise_dBHzSPL,
                    attenuation_start=noise_attenuation_start,
                    attenuation_slope=noise_attenuation_slope)
                # Add signal + noise and metadata to data_dict for hdf5 filewriting
                tone_in_noise = signal + noise
                data_dict = {
                    'tone_in_noise': tone_in_noise.astype(np.float32),
                    'f0': np.float32(f0),
                    'phase_mode': int(phase),
                    'filter_fl': np.float32(filter_dict['fl']),
                    'filter_fh': np.float32(filter_dict['fh']),
                }
                # Initialize output hdf5 dataset on first iteration
                if itrN == 0:
                    print('[INITIALIZING]: {}'.format(hdf5_filename))
                    data_dict.update(config_dict)
                    data_key_pair_list = [(k, k) for k in set(data_dict.keys()).difference(config_dict.keys())]
                    initialize_hdf5_file(hdf5_filename, N, data_dict, file_mode='w',
                                         data_key_pair_list=data_key_pair_list,
                                         config_key_pair_list=config_key_pair_list)
                    hdf5_f = h5py.File(hdf5_filename, 'r+')
                # Write each data_dict to hdf5 file
                write_example_to_hdf5(hdf5_f, data_dict, itrN, data_key_pair_list=data_key_pair_list)
                if itrN % disp_step == 0: print('... signal {} of {}, (f0={})'.format(itrN, N, f0))
                itrN += 1
    
    # If `include_2xF0` is specified, generate sine phase stimuli with 2x F0
    if include_2xF0:
        phase = 0
        list_f0 = 2 * list_f0
        for filter_dict in list_filters:
            # Get the frequency response function for the current filter condition
            frequency_response_in_dB = filter_dict['freq_response']
            for f0 in list_f0:
                # Build signal with specified phase, filter, and f0
                harmonic_freqs = np.arange(f0, fs/2, f0)
                harmonic_numbers = harmonic_freqs / f0
                harmonic_dBSPL = passband_component_dBSPL + frequency_response_in_dB(harmonic_freqs)
                amplitudes = 20e-6 * np.power(10, (harmonic_dBSPL/20))
                signal = stimuli_util.complex_tone(f0, fs, dur, harmonic_numbers=harmonic_numbers,
                                                   amplitudes=amplitudes,
                                                   phase_mode=phase_mode_decoding[phase])
                # Construct modified uniform masking noise
                noise = stimuli_util.modified_uniform_masking_noise(
                    fs, dur, dBHzSPL=noise_dBHzSPL,
                    attenuation_start=noise_attenuation_start,
                    attenuation_slope=noise_attenuation_slope)
                # Add signal + noise and metadata to data_dict for hdf5 filewriting
                tone_in_noise = signal + noise
                data_dict = {
                    'tone_in_noise': tone_in_noise.astype(np.float32),
                    'f0': np.float32(f0),
                    'phase_mode': int(phase),
                    'filter_fl': np.float32(filter_dict['fl']),
                    'filter_fh': np.float32(filter_dict['fh']),
                }
                # Write each data_dict to hdf5 file
                write_example_to_hdf5(hdf5_f, data_dict, itrN, data_key_pair_list=data_key_pair_list)
                if itrN % disp_step == 0: print('... signal {} of {}, (f0={})'.format(itrN, N, f0))
                itrN += 1
    
    # Close hdf5 file
    hdf5_f.close()
    print('[END]: {}'.format(hdf5_filename))


if __name__ == "__main__":
    ''' TEMPORARY COMMAND-LINE USAGE '''
    assert len(sys.argv) == 2, "scipt usage: python <script_name> <hdf5_filename>"
    hdf5_filename = str(sys.argv[1])
    
    generate_AltPhase_dataset(hdf5_filename)