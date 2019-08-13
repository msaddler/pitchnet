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
    signal (np array): sound waveform
    '''
    # Compute harmonic numbers and relative amplitude of each harmonic
    harmonics = np.arange(f0, fs/2, f0)
    harmonic_numbers = harmonics / f0
    dB_attenuation = -attenuation_slope * np.log2(harmonics / attenuation_start)
    dB_attenuation[dB_attenuation > 0] = 0
    amplitudes = 20e-6 * np.power(10, (dB_attenuation/20))
    # Jitter the harmonic amplitudes with multiplicative noise
    if amp_noise > 0:
        amplitude_noise = np.random.uniform(1-amp_noise, 1+amp_noise, amplitudes.shape)
        amplitudes = amplitude_noise * amplitudes
    # Build and return the complex tone
    signal = stimuli_util.complex_tone(f0, fs, dur, harmonic_numbers=harmonic_numbers,
                                       amplitudes=amplitudes, phase_mode=phase_mode,
                                       offset_start=True, strict_nyquist=True)
    return signal


def generate_lowpass_complex_tone_dataset(hdf5_filename, N, fs=32e3, dur=0.150,
                                          disp_step=100, f0_min=80., f0_max=1e3,
                                          amp_noise=0., phase_modes=['sine'],
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
    phase_modes (list):
    atten_start_min (float):
    atten_start_max (float):
    atten_slope_min (float):
    atten_slope_max (float):
    '''
    # Randomly sample f0s, phase_modes, and attenuation parameters
    f0_list = np.exp(np.random.uniform(np.log(f0_min), np.log(f0_max), size=[N]))
    phase_mode_list = np.random.choice(phase_modes, size=[N])
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
        data_dict['tone'] = signal.astype(np.float32)
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
                                 config_key_pair_list=config_key_pair_list)
            hdf5_f = h5py.File(hdf5_filename, 'r+')
        # Write each data_dict to hdf5 file
        write_example_to_hdf5(hdf5_f, data_dict, itrN, data_key_pair_list=data_key_pair_list)
        if itrN % disp_step == 0:
            print('... signal {} of {}'.format(itrN, N))
    # Close hdf5 file
    hdf5_f.close()
    print('[END]: {}'.format(hdf5_filename))


def get_bandpass_filter_frequency_response(fl, fh, fs=32e3, order=4):
    '''
    Returns a function that computes the frequency response in dB of a bandpass
    filter with cutoff frequencies [fl, fh]. This function is based on 
    "filterbw.m" from Joshua Bernstein and Andrew Oxenham (2005).
    
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
    [w, Hl] = scipy.signal.freqz(Bl, a=Al, worN=int(fs), whole=False, plot=None)
    [w, Hh] = scipy.signal.freqz(Bh, a=Ah, worN=int(fs), whole=False, plot=None)
    # Compute frequency response of the double filtering operation
    H = Hl * Hh
    # Re-scale one of the filter's weights so double filtering operation gives 0-dB max response
    Bl = Bl / np.max(np.abs(H))
    # Create the frequency-amplitude response function and return
    def frequency_response_in_dB(freqs):
        ''' Amplitude response of the double filtering operation in dB '''
        [_, Hh] = scipy.signal.freqz(Bh, a=Ah, worN=2*np.pi*freqs/fs, whole=False)
        [_, Hl] = scipy.signal.freqz(Bl, a=Al, worN=2*np.pi*freqs/fs, whole=False)
        return 20.* np.log10(np.abs(Hl * Hh))
    return frequency_response_in_dB


def bernox2005_bandpass_complex_tone(f0, fs, dur, frequency_response_in_dB=None,
                                     threshold_dBSPL=33.3, component_dBSL=15.,
                                     **kwargs_complex_tone):
    '''
    Generates a bandpass filtered complex tone with component levels as determined by
    Berntein and Oxenhamm (2005, JASA).
    
    Args
    ----
    f0 (float): fundamental frequency (Hz)
    fs (int): sampling rate (Hz)
    dur (float): duration of tone (s)
    frequency_response_in_dB (function): see `get_bandpass_filter_frequency_response()`
    threshold_dBSPL (float): audible threshold in units of dB re 20e-6 Pa
    component_dBSL (float): "sensation level" in units of dB above audible threshold
    **kwargs_complex_tone (kwargs): passed directly to `complex_tone()`
    
    Returns
    -------
    signal (np array): sound waveform (Pa)
    audible_harmonic_numbers (np array): harmonic numbers presented above threshold_dBSPL
    '''
    harmonics = np.arange(f0, fs/2, f0)
    harmonic_numbers = harmonics / f0
    harmonic_dBSPL = threshold_dBSPL + component_dBSL + frequency_response_in_dB(harmonics)
    amplitudes = 20e-6 * np.power(10, (harmonic_dBSPL/20))
    signal = stimuli_util.complex_tone(f0, fs, dur, harmonic_numbers=harmonic_numbers,
                                       amplitudes=amplitudes, **kwargs_complex_tone)
    audible_harmonic_numbers = harmonic_numbers[harmonic_dBSPL >= threshold_dBSPL]
    return signal, audible_harmonic_numbers


def shift_bandpass_filter_frequency_response(desired_fl, desired_fl_gain_in_dB,
                                             fs=32000, unshifted_passband=None,
                                             frequency_response_in_dB=None):
    '''
    '''
    assert frequency_response_in_dB is not None, "`frequency_response_in_dB` must be specified"
    # Compute the `desired_fl_gain_in_dB` passband of the unshifted filter
    if unshifted_passband is None:
        freq_vector = np.arange(0, fs/2)
        gain_vector = frequency_response_in_dB(freq_vector)
        passed_freqs = freq_vector[gain_vector > desired_fl_gain_in_dB]
        unshifted_passband = [np.min(passed_freqs), np.max(passed_freqs)]
    # Compute offset between unshifted and desired filter
    fl_offset = desired_fl - unshifted_passband[0]
    # Return frequency response function of shifted filter
    def shifted_frequency_response_in_dB(freqs):
        shifted_freqs = freqs - fl_offset
        shifted_freqs[shifted_freqs  < 0.] = 0.
        shifted_freqs[shifted_freqs  > fs/2] = fs/2
        return frequency_response_in_dB(shifted_freqs)
    return shifted_frequency_response_in_dB


def bernoxMovingFilter_bandpass_complex_tone(f0, fs, dur, low_harm, frequency_response_in_dB=None,
                                             threshold_dBSPL=33.3, component_dBSL=15.,
                                             **kwargs_complex_tone):
    '''
    '''
    # Define the harmonic stack
    harmonics = np.arange(f0, fs/2, f0)
    harmonic_numbers = np.rint(harmonics / f0).astype(int)
    # Shift the baseline frequency response function to the desired low_harm number and gain
    desired_fl = f0 * low_harm
    desired_fl_gain_in_dB = -1 * component_dBSL
    shifted_frequency_response_in_dB = shift_bandpass_filter_frequency_response(
        desired_fl, desired_fl_gain_in_dB, fs=fs, unshifted_passband=None,
        frequency_response_in_dB=frequency_response_in_dB)
    # Generate harmonic complex tone with the shifted frequency response function
    harmonic_dBSPL = threshold_dBSPL + component_dBSL + shifted_frequency_response_in_dB(harmonics)
    amplitudes = 20e-6 * np.power(10, (harmonic_dBSPL/20))
    signal = stimuli_util.complex_tone(f0, fs, dur, harmonic_numbers=harmonic_numbers,
                                       amplitudes=amplitudes, **kwargs_complex_tone)
    audible_harmonic_numbers = harmonic_numbers[harmonic_dBSPL >= threshold_dBSPL]
    # Assert that the lowest audible harmonic number is indeed the specified low_harm number
    assert int(np.min(audible_harmonic_numbers)) == low_harm
    return signal, audible_harmonic_numbers


def generate_bernoxMovingFilter_dataset(hdf5_filename, fs=32e3, dur=0.150, f0_min=100., f0_max=300.,
                                        step_size_in_octaves=4/768, phase_modes=['sine', 'rand'],
                                        low_harm_min=1, low_harm_max=30,
                                        highpass_filter_cutoff=2.5e3, lowpass_filter_cutoff=3.5e3,
                                        filter_order=4, threshold_dBSPL=33.3, component_dBSL=15.,
                                        noise_dBHzSPL=15., noise_attenuation_start=600.,
                                        noise_attenuation_slope=2, disp_step=100):
    '''
    Main routine for generating modified form of Bernstein and Oxenham (2005, JASA)
    diagnostic dataset. The bandpass filter from Bernstein and Oxenham (2005, JASA)
    is translated linearly along the frequency axis to manipulate the lowest audible
    harmonic number in the resulting stimuli.
    
    Args
    ----
    hdf5_filename (str):
    fs (int): sampling rate (Hz)
    dur (float): duration of tone (s)
    f0_min (float):
    f0_max (float):
    step_size_in_octaves (float):
    phase_modes (list):
    low_harm_min (int):
    low_harm_max (int):
    highpass_filter_cutoff (float):
    lowpass_filter_cutoff (float):
    filter_order (int):
    threshold_dBSPL (float):
    component_dBSL (float):
    noise_dBHzSPL (float):
    noise_attenuation_start (float):
    noise_attenuation_slope (float):
    disp_step (int):
    '''
    # Define encoding / decoding dictionaries for phase_mode
    phase_mode_encoding = {'sine':0, 'rand':1, 'sch':2, 'cos':3, 'alt':4}
    phase_mode_decoding = {0:'sine', 1:'rand', 2:'sch', 3:'cos', 4:'alt'}
    # Determine lists of unique low harm numbers, f0s, and phase values
    unique_low_harm_numbers = np.arange(low_harm_min, low_harm_max + 1)
    unique_f0s = np.arange(0, np.log2(f0_max / f0_min), step_size_in_octaves)
    unique_f0s = f0_min * (np.power(2, unique_f0s))
    unique_phase_modes = np.array([phase_mode_encoding[p] for p in phase_modes])
    # Generate parameter lists to iterate over (all combinations of low_harm, f0, and phase_mode)
    (list_lh, list_f0, list_ph) = ([], [], [])
    for (lh, f0, ph) in itertools.product(unique_low_harm_numbers, unique_f0s, unique_phase_modes):
        list_lh.append(lh)
        list_f0.append(f0)
        list_ph.append(ph)
    N = len(list_lh)
    # Calculate the "unshifted" bandpass filter frequency response function to use as a baseline
    frequency_response_in_dB = get_bandpass_filter_frequency_response(highpass_filter_cutoff,
                                                                      lowpass_filter_cutoff,
                                                                      fs=fs, order=filter_order)
    # Prepare data_dict and config_key_pair_list for hdf5 filewriting
    data_dict = {
        'config_tone/fs': fs,
        'config_tone/dur': dur,
        'config_tone/f0_min': f0_min,
        'config_tone/f0_max': f0_max,
        'config_tone/step_size_in_octaves': step_size_in_octaves,
        'config_tone/low_harm_min': low_harm_min,
        'config_tone/low_harm_max': low_harm_max,
        'config_tone/unshifted_highpass_filter_cutoff': highpass_filter_cutoff,
        'config_tone/unshifted_lowpass_filter_cutoff': lowpass_filter_cutoff,
        'config_tone/unshifted_filter_order': filter_order,
        'config_tone/threshold_dBSPL': threshold_dBSPL,
        'config_tone/component_dBSL': component_dBSL,
        'config_tone/noise_dBHzSPL': noise_dBHzSPL,
        'config_tone/noise_attenuation_start': noise_attenuation_start,
        'config_tone/noise_attenuation_slope': noise_attenuation_slope,
    }
    config_key_pair_list = [(k, k) for k in data_dict.keys()]
    data_key_pair_list = [] # Will be populated right before initializing hdf5 file
    # Main loop to generate the bandpass filtered tones
    for itrN in range(0, N):
        
        signal, audible_harmonic_numbers = bernoxMovingFilter_bandpass_complex_tone(
            list_f0[itrN], fs, dur, list_lh[itrN],
            frequency_response_in_dB=frequency_response_in_dB,
            threshold_dBSPL=threshold_dBSPL,
            component_dBSL=component_dBSL,
            phase_mode=phase_mode_decoding[list_ph[itrN]])
        
        noise = stimuli_util.modified_uniform_masking_noise(fs, dur, dBHzSPL=noise_dBHzSPL,
                                                            attenuation_start=noise_attenuation_start,
                                                            attenuation_slope=noise_attenuation_slope)
        
        tone_in_noise = signal + noise
        data_dict['tone_in_noise'] = tone_in_noise.astype(np.float32)
        data_dict['f0'] = list_f0[itrN]
        data_dict['phase_mode'] = int(list_ph[itrN])
        data_dict['low_harm'] = int(np.min(audible_harmonic_numbers))
        data_dict['upp_harm'] = int(np.max(audible_harmonic_numbers))
        
        # Initialize the hdf5 file on the first iteration
        if itrN == 0:
            print('[INITIALIZING]: {}'.format(hdf5_filename))
            for k in data_dict.keys():
                if not (k, k) in config_key_pair_list:
                    data_key_pair_list.append((k, k))
            initialize_hdf5_file(hdf5_filename, N, data_dict, file_mode='w',
                                 data_key_pair_list=data_key_pair_list,
                                 config_key_pair_list=config_key_pair_list)
            hdf5_f = h5py.File(hdf5_filename, 'r+')
        # Write each data_dict to hdf5 file
        write_example_to_hdf5(hdf5_f, data_dict, itrN, data_key_pair_list=data_key_pair_list)
        if itrN % disp_step == 0:
            print('... signal {} of {}'.format(itrN, N))
    # Close hdf5 file
    hdf5_f.close()
    print('[END]: {}'.format(hdf5_filename))


def generate_bernox2005_dataset(hdf5_filename, fs=32e3, dur=0.150, f0_min=80., f0_max=1e3,
                                step_size_in_octaves=1/1536, phase_modes=['sine', 'rand'],
                                highpass_filter_cutoff=2.5e3, lowpass_filter_cutoff=3.5e3,
                                filter_order=4, threshold_dBSPL=33.3, component_dBSL=15.,
                                noise_dBHzSPL=15., noise_attenuation_start=600.,
                                noise_attenuation_slope=2, disp_step=100):
    '''
    Main routine for generating Bernstein and Oxenham (2005, JASA) diagnostic dataset.
    
    Args
    ----
    hdf5_filename (str):
    fs (int): sampling rate (Hz)
    dur (float): duration of tone (s)
    f0_min (float):
    f0_max (float):
    step_size_in_octaves (float):
    phase_modes (list):
    highpass_filter_cutoff (float):
    lowpass_filter_cutoff (float):
    filter_order (int):
    threshold_dBSPL (float):
    component_dBSL (float):
    noise_dBHzSPL (float):
    noise_attenuation_start (float):
    noise_attenuation_slope (float):
    disp_step (int):
    '''
    # Define encoding / decoding dictionaries for phase_mode
    phase_mode_encoding = {'sine':0, 'rand':1, 'sch':2, 'cos':3, 'alt':4}
    phase_mode_decoding = {0:'sine', 1:'rand', 2:'sch', 3:'cos', 4:'alt'}
    # Determine list of f0 and phase values
    f0s = np.arange(0, np.log2(f0_max / f0_min), step_size_in_octaves)
    f0s = f0_min * (np.power(2, f0s))
    f0_list = np.tile(f0s, len(phase_modes))
    phase_list = []
    for phase in phase_modes:
        phase_list.append(phase_mode_encoding[phase] * np.ones_like(f0s))
    phase_list = np.concatenate(phase_list, axis=0)
    N = f0_list.shape[0]
    # Calculate the fixed bandpass filter frequency response function
    frequency_response_in_dB = get_bandpass_filter_frequency_response(highpass_filter_cutoff,
                                                                      lowpass_filter_cutoff,
                                                                      fs=fs, order=filter_order)
    # Prepare data_dict and config_key_pair_list for hdf5 filewriting
    data_dict = {
        'config_tone/fs': fs,
        'config_tone/dur': dur,
        'config_tone/f0_min': f0_min,
        'config_tone/f0_max': f0_max,
        'config_tone/step_size_in_octaves': step_size_in_octaves,
        'config_tone/highpass_filter_cutoff': highpass_filter_cutoff,
        'config_tone/lowpass_filter_cutoff': lowpass_filter_cutoff,
        'config_tone/filter_order': filter_order,
        'config_tone/threshold_dBSPL': threshold_dBSPL,
        'config_tone/component_dBSL': component_dBSL,
        'config_tone/noise_dBHzSPL': noise_dBHzSPL,
        'config_tone/noise_attenuation_start': noise_attenuation_start,
        'config_tone/noise_attenuation_slope': noise_attenuation_slope,
    }
    config_key_pair_list = [(k, k) for k in data_dict.keys()]
    data_key_pair_list = [] # Will be populated right before initializing hdf5 file
    # Main loop to generate the bandpass filtered tones
    for itrN in range(0, N):
        signal, audible_harmonic_numbers = bernox2005_bandpass_complex_tone(
            f0_list[itrN], fs, dur,
            frequency_response_in_dB=frequency_response_in_dB,
            threshold_dBSPL=threshold_dBSPL,
            component_dBSL=component_dBSL,
            phase_mode=phase_mode_decoding[phase_list[itrN]])
        noise = stimuli_util.modified_uniform_masking_noise(fs, dur, dBHzSPL=noise_dBHzSPL,
                                                            attenuation_start=noise_attenuation_start,
                                                            attenuation_slope=noise_attenuation_slope)
        tone_in_noise = signal + noise
        data_dict['tone_in_noise'] = tone_in_noise.astype(np.float32)
        data_dict['f0'] = f0_list[itrN]
        data_dict['phase_mode'] = int(phase_list[itrN])
        data_dict['low_harm'] = int(np.min(audible_harmonic_numbers))
        data_dict['upp_harm'] = int(np.max(audible_harmonic_numbers))
        
        # Initialize the hdf5 file on the first iteration
        if itrN == 0:
            print('[INITIALIZING]: {}'.format(hdf5_filename))
            for k in data_dict.keys():
                if not (k, k) in config_key_pair_list:
                    data_key_pair_list.append((k, k))
            initialize_hdf5_file(hdf5_filename, N, data_dict, file_mode='w',
                                 data_key_pair_list=data_key_pair_list,
                                 config_key_pair_list=config_key_pair_list)
            hdf5_f = h5py.File(hdf5_filename, 'r+')
        # Write each data_dict to hdf5 file
        write_example_to_hdf5(hdf5_f, data_dict, itrN, data_key_pair_list=data_key_pair_list)
        if itrN % disp_step == 0:
            print('... signal {} of {}'.format(itrN, N))
    # Close hdf5 file
    hdf5_f.close()
    print('[END]: {}'.format(hdf5_filename))


def generate_bandpass_complex_tone_dataset(hdf5_filename, N, fs=32e3, dur=0.150, f0_min=80., f0_max=1e3,
                                           phase_modes=['sine', 'rand'], rms_out=0.02,
                                           highpass_filter_cutoff_range=[7e1, 7e3],
                                           bandwidth_range=[7e1, 7e3],
                                           filter_order_min=1, filter_order_max=12, disp_step=100):
    '''
    '''
    # Randomly sample f0s, phase_modes, and attenuation parameters
    f0_list = np.exp(np.random.uniform(np.log(f0_min), np.log(f0_max), size=[N]))
    phase_mode_list = np.random.choice(phase_modes, size=[N])
    highpass_filter_cutoff_list = np.random.uniform(highpass_filter_cutoff_range[0],
                                                    highpass_filter_cutoff_range[1], size=[N])
    bandwidth_list = np.random.uniform(bandwidth_range[0], bandwidth_range[1], size=[N])
    lowpass_filter_cutoff_list = highpass_filter_cutoff_list + bandwidth_list
    filter_order_list = np.random.randint(filter_order_min, high=filter_order_max+1, size=[N])
    # Define encoding / decoding dictionaries for phase_mode
    phase_mode_encoding = {'sine':0, 'rand':1, 'sch':2, 'cos':3, 'alt':4}
    phase_mode_decoding = {0:'sine', 1:'rand', 2:'sch', 3:'cos', 4:'alt'}
    # Prepare data_dict and config_key_pair_list for hdf5 filewriting
    data_dict = {
        'config_tone/fs': fs,
        'config_tone/dur': dur,
        'config_tone/f0_min': f0_min,
        'config_tone/f0_max': f0_max,
        'config_tone/highpass_filter_cutoff_min': highpass_filter_cutoff_range[0],
        'config_tone/highpass_filter_cutoff_max': highpass_filter_cutoff_range[1],
        'config_tone/bandwidth_min': bandwidth_range[0],
        'config_tone/bandwidth_max': bandwidth_range[0],
        'config_tone/filter_order_min': filter_order_min,
        'config_tone/filter_order_max': filter_order_max,
    }
    config_key_pair_list = [(k, k) for k in data_dict.keys()]
    data_key_pair_list = [] # Will be populated right before initializing hdf5 file
    # Main loop to generate bandpass complex tones
    np.seterr(all='raise')
    for itrN in range(0, N):
        signal = np.nan
        while np.any(np.isnan(signal)):
            try:
                # Calculate bandpass filter frequency response function
                freq_resp_in_dB = get_bandpass_filter_frequency_response(highpass_filter_cutoff_list[itrN],
                                                                         lowpass_filter_cutoff_list[itrN],
                                                                         fs=fs, order=filter_order_list[itrN])
                # Create the bandpass filtered complex tone
                signal, audible_harmonic_numbers = bernox2005_bandpass_complex_tone(f0_list[itrN], fs, dur,
                                                                                    frequency_response_in_dB=freq_resp_in_dB,
                                                                                    phase_mode=phase_mode_list[itrN])
                signal = (signal / stimuli_util.rms(signal)) * rms_out
            except Exception as e:
                print(e, '------> RESAMPLING FILTER')
                highpass_filter_cutoff_list[itrN] = np.random.uniform(highpass_filter_cutoff_range[0],
                                                                      highpass_filter_cutoff_range[1])
                bandwidth_list[itrN] = np.random.uniform(bandwidth_range[0], bandwidth_range[1])
                lowpass_filter_cutoff_list[itrN] = highpass_filter_cutoff_list[itrN] + bandwidth_list[itrN]
                filter_order_list[itrN] = np.random.randint(filter_order_min, high=filter_order_max+1)

        data_dict['tone'] = signal.astype(np.float32)
        data_dict['f0'] = f0_list[itrN]
        data_dict['phase_mode'] = phase_mode_encoding[phase_mode_list[itrN]]
        data_dict['filter_highpass_cutoff'] = highpass_filter_cutoff_list[itrN]
        data_dict['filter_lowpass_cutoff'] = lowpass_filter_cutoff_list[itrN]
        data_dict['filter_bandwidth'] = bandwidth_list[itrN]
        data_dict['filter_order'] = filter_order_list[itrN]
        # Initialize the hdf5 file on the first iteration
        if itrN == 0:
            print('[INITIALIZING]: {}'.format(hdf5_filename))
            for k in data_dict.keys():
                if not (k, k) in config_key_pair_list:
                    data_key_pair_list.append((k, k))
            initialize_hdf5_file(hdf5_filename, N, data_dict, file_mode='w',
                                 data_key_pair_list=data_key_pair_list,
                                 config_key_pair_list=config_key_pair_list)
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
#     generate_bandpass_complex_tone_dataset(hdf5_filename, N)
#     generate_lowpass_complex_tone_dataset(hdf5_filename, N)
    generate_bernox2005_dataset(hdf5_filename, fs=32e3, dur=0.150, f0_min=80., f0_max=1e3,
                                step_size_in_octaves=1/(12*16*64), phase_modes=['sine', 'rand'],
                                highpass_filter_cutoff=5e3, lowpass_filter_cutoff=7e3,
                                filter_order=4, threshold_dBSPL=33.3, component_dBSL=15.,
                                noise_dBHzSPL=15., noise_attenuation_start=600.,
                                noise_attenuation_slope=2, disp_step=100)

#     generate_bernoxMovingFilter_dataset(hdf5_filename, fs=32e3, dur=0.150, f0_min=80., f0_max=640.,
#                                         step_size_in_octaves=1/768, phase_modes=['sine', 'rand'],
#                                         low_harm_min=1, low_harm_max=15,
#                                         highpass_filter_cutoff=2.5e3, lowpass_filter_cutoff=3.5e3,
#                                         filter_order=4, threshold_dBSPL=33.3, component_dBSL=15.,
#                                         noise_dBHzSPL=15., noise_attenuation_start=600.,
#                                         noise_attenuation_slope=2, disp_step=100)
