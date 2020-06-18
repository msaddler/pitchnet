import sys
import os
import numpy as np
import h5py
import scipy.signal
import itertools
import pdb

sys.path.append('/om2/user/msaddler/python-packages/msutil')
import util_stimuli

sys.path.append('/om4/group/mcdermott/user/msaddler/pitchnet_dataset/pitchnetDataset/pitchnetDataset')
from dataset_util import initialize_hdf5_file, write_example_to_hdf5


def get_bandpass_filter_frequency_response(fl, fh, fs=20e3, order=4):
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


def shift_bandpass_filter_frequency_response(desired_fl,
                                             desired_fl_gain_in_dB,
                                             fs=20e3,
                                             unshifted_passband=None,
                                             frequency_response_in_dB=None):
    '''
    Accepts a frequency response function and translates it along the frequency axis
    to a specified low frequency cutoff.
    
    Args
    ----
    desired_fl (float): low frequency cutoff of the shifted filter (Hz)
    desired_fl_gain_in_dB (float): desired gain of filter at desired_fl (dB) 
    fs (int): sampling rate (Hz)
    unshifted_passband (list or None): if passband (low, high) is not specified, it will be computed
    frequency_response_in_dB (function): frequency response function to be shifted along frequency axis
    
    Returns
    -------
    shifted_frequency_response_in_dB (function): frequency response function (dB)
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


def harmonic_or_inharmonic_complex_tone(f0,
                                        fs=20e3,
                                        dur=2e0,
                                        inharmonic_jitter_pattern=None,
                                        frequency_response_in_dB=None,
                                        threshold_dBSPL=33.3,
                                        component_dBSL=15.0,
                                        **kwargs_complex_tone):
    '''
    Generates a complex tone (either harmonic or inharmonic) with component
    levels determined by frequency_response_in_dB, threshold_dBSPL, and
    component_dBSL (adapted from function designed to generate bandpass-
    filtered harmonic tones used by Bernstein and Oxenham, 2005 JASA).
    
    Args
    ----
    f0 (float): fundamental frequency (Hz)
    fs (int): sampling rate (Hz)
    dur (float): duration of tone (s)
    inharmonic_jitter_pattern (np.ndarray): vector of jitter values to apply to 
        harmonic frequencies to make an inharmonic tone (if None, tone is harmonic)
    frequency_response_in_dB (function): see `get_bandpass_filter_frequency_response()`
    threshold_dBSPL (float): audible threshold in units of dB re 20e-6 Pa
    component_dBSL (float): "sensation level" in units of dB above audible threshold
    **kwargs_complex_tone (kwargs): passed directly to `complex_tone()`
    
    Returns
    -------
    signal (np.ndarray): sound waveform (Pa)
    signal_metadata (dict): signal generation parameters
    '''
    # Define component frequencies
    harmonic_freqs = np.arange(f0, fs/2, f0)
    nominal_harmonic_numbers = harmonic_freqs / f0
    if inharmonic_jitter_pattern is None:
        # Component frequencies are harmonic frequencies
        component_freqs = harmonic_freqs
    else:
        # Component frequencies are harmonic frequencies * inharmonic_jitter_pattern
        N_component_freqs = nominal_harmonic_numbers.shape[0]
        msg = "inharmonic_jitter_pattern must be at least as long as nominal_harmonic_numbers"
        assert len(inharmonic_jitter_pattern.shape) == 1, msg
        assert inharmonic_jitter_pattern.shape[0] >= N_component_freqs, msg
        component_freqs = harmonic_freqs * inharmonic_jitter_pattern[:N_component_freqs]
        component_freqs[component_freqs > fs/2] = fs/2
    # Define component levels
    if frequency_response_in_dB is None:
        # Return equal-amplitude harmonics if frequency_response_in_dB is not specified
        frequency_response_in_dB = lambda f: np.zeros_like(f)
    component_dBSPL = threshold_dBSPL + component_dBSL + frequency_response_in_dB(component_freqs)
    component_amplitudes = 20e-6 * np.power(10, (component_dBSPL/20))
    # Return signal waveform and dictionary of metadata
    signal = util_stimuli.complex_tone(f0,
                                       fs,
                                       dur,
                                       harmonic_numbers=None,
                                       frequencies=component_freqs,
                                       amplitudes=component_amplitudes,
                                       **kwargs_complex_tone)
    signal_metadata = {
        'f0': f0,
        'fs': fs,
        'dur': dur,
        'nominal_harmonic_numbers': nominal_harmonic_numbers,
        'jitter_pattern': component_freqs / harmonic_freqs,
        'component_freqs': component_freqs,
        'component_dBSPL': component_dBSPL,
        'component_dBSL': component_dBSL,
        'threshold_dBSPL': threshold_dBSPL,
    }
    signal_metadata.update(kwargs_complex_tone)
    return signal, signal_metadata


def generate_BernsteinOxenhamFixedFilter_dataset(hdf5_filename,
                                                 fs=32e3,
                                                 dur=0.150,
                                                 phase_modes=['sine', 'rand'],
                                                 low_harm_min=1,
                                                 low_harm_max=30,
                                                 base_f0_min=100.0,
                                                 base_f0_max=300.0,
                                                 base_f0_n=10,
                                                 delta_f0_min=0.94,
                                                 delta_f0_max=1.06,
                                                 delta_f0_n=121,
                                                 highpass_filter_cutoff=2.5e3,
                                                 lowpass_filter_cutoff=3.5e3,
                                                 filter_order=4,
                                                 threshold_dBSPL=33.3,
                                                 component_dBSL=15.0,
                                                 noise_dBHzSPL=15.0,
                                                 noise_attenuation_start=600.0,
                                                 noise_attenuation_slope=2,
                                                 disp_step=100):
    '''
    '''
    # Define encoding / decoding dictionaries for phase_mode
    phase_mode_encoding = {'sine':0, 'rand':1, 'sch':2, 'cos':3, 'alt':4}
    phase_mode_decoding = {0:'sine', 1:'rand', 2:'sch', 3:'cos', 4:'alt'}
    # Define lists of unique phase modes and low harm numbers
    unique_ph_list = np.array([phase_mode_encoding[p] for p in phase_modes])
    unique_lh_list = np.arange(low_harm_min, low_harm_max + 1)
    # Define list of "base f0" values (Hz), which are used to set the filters
    base_f0_list = np.power(2, np.linspace(np.log2(base_f0_min), np.log2(base_f0_max), base_f0_n))
    # Define list of "delta f0" values (fraction of f0)
    delta_f0_list = np.linspace(delta_f0_min, delta_f0_max, delta_f0_n)
    # Compute number of stimuli (all combinations of low_harm, base_f0, delta_f0, and phase_mode)
    N = len(unique_ph_list) * len(unique_lh_list) * len(base_f0_list) * len(delta_f0_list)
    # Calculate the "unshifted" bandpass filter frequency response function to use as a baseline
    baseline_freq_response = get_bandpass_filter_frequency_response(highpass_filter_cutoff,
                                                                    lowpass_filter_cutoff,
                                                                    fs=fs, order=filter_order)
    # Prepare data_dict and config_key_pair_list for hdf5 filewriting
    data_dict = {
        'config_tone/fs': fs,
        'config_tone/dur': dur,
        'config_tone/low_harm_min': low_harm_min,
        'config_tone/low_harm_max': low_harm_max,
        'config_tone/base_f0_min': base_f0_min,
        'config_tone/base_f0_max': base_f0_max,
        'config_tone/base_f0_n': base_f0_n,
        'config_tone/delta_f0_min': delta_f0_min,
        'config_tone/delta_f0_max': delta_f0_max,
        'config_tone/delta_f0_n': delta_f0_n,
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
    itrN = 0
    for lh in unique_lh_list:
        for base_f0 in base_f0_list:
            # Compute fixed filter's frequency response using low harm number and base_f0
            desired_fl = base_f0 * lh
            desired_fl_gain_in_dB = -1 * component_dBSL
            fixed_freq_response = shift_bandpass_filter_frequency_response(
                desired_fl, desired_fl_gain_in_dB, fs=fs, unshifted_passband=None,
                frequency_response_in_dB=baseline_freq_response)
            for delta_f0 in delta_f0_list:
                for ph in unique_ph_list:
                    # Construct signal with specified f0 and phase mode
                    f0 = base_f0 * delta_f0
                    signal, audible_harmonic_numbers = bernox2005_bandpass_complex_tone(
                        f0, fs, dur, frequency_response_in_dB=fixed_freq_response,
                        threshold_dBSPL=threshold_dBSPL, component_dBSL=component_dBSL,
                        phase_mode=phase_mode_decoding[ph])
                    # Construct modified uniform masking noise
                    noise = util_stimuli.modified_uniform_masking_noise(
                        fs, dur, dBHzSPL=noise_dBHzSPL,
                        attenuation_start=noise_attenuation_start,
                        attenuation_slope=noise_attenuation_slope)
                    # Add signal + noise and metadata to data_dict for hdf5 filewriting
                    tone_in_noise = signal + noise
                    data_dict['tone_in_noise'] = tone_in_noise.astype(np.float32)
                    data_dict['f0'] = f0
                    data_dict['base_f0'] = base_f0
                    data_dict['delta_f0'] = delta_f0
                    data_dict['phase_mode'] = int(ph)
                    data_dict['low_harm'] = int(lh)
                    data_dict['min_audible_harm'] = int(np.min(audible_harmonic_numbers))
                    data_dict['max_audible_harm'] = int(np.max(audible_harmonic_numbers))
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
                    write_example_to_hdf5(hdf5_f, data_dict, itrN,
                                          data_key_pair_list=data_key_pair_list)
                    if itrN % disp_step == 0:
                        print('... signal {} of {}'.format(itrN, N))
                    itrN += 1
    # Close hdf5 file
    hdf5_f.close()
    print('[END]: {}'.format(hdf5_filename))


if __name__ == "__main__":
    ''' TEMPORARY COMMAND-LINE USAGE '''
    assert len(sys.argv) == 2, "scipt usage: python <script_name> <hdf5_filename>"
    hdf5_filename = str(sys.argv[1])
    
#     generate_BernsteinOxenhamFixedFilter_dataset(hdf5_filename,
#                                                  fs=32e3,
#                                                  dur=0.150,
#                                                  phase_modes=['sine', 'rand'],
#                                                  low_harm_min=1,
#                                                  low_harm_max=30,
#                                                  base_f0_min=100.0,
#                                                  base_f0_max=300.0,
#                                                  base_f0_n=10,
#                                                  delta_f0_min=0.94,
#                                                  delta_f0_max=1.06,
#                                                  delta_f0_n=121,
#                                                  highpass_filter_cutoff=2.5e3,
#                                                  lowpass_filter_cutoff=3.5e3,
#                                                  filter_order=4,
#                                                  threshold_dBSPL=33.3,
#                                                  component_dBSL=15.0,
#                                                  noise_dBHzSPL=15.0,
#                                                  noise_attenuation_start=600.0,
#                                                  noise_attenuation_slope=2,
#                                                  disp_step=100)
    generate_BernsteinOxenhamFixedFilter_dataset(hdf5_filename,
                                                 fs=32e3,
                                                 dur=0.150,
                                                 phase_modes=['sine'],
                                                 low_harm_min=1,
                                                 low_harm_max=30,
                                                 base_f0_min=80.0,
                                                 base_f0_max=320.0,
                                                 base_f0_n=192*2*4,
                                                 delta_f0_min=1,
                                                 delta_f0_max=1,
                                                 delta_f0_n=1,
                                                 highpass_filter_cutoff=2.5e3,
                                                 lowpass_filter_cutoff=3.5e3,
                                                 filter_order=4,
                                                 threshold_dBSPL=33.3,
                                                 component_dBSL=15.0,
                                                 noise_dBHzSPL=15.0,
                                                 noise_attenuation_start=600.0,
                                                 noise_attenuation_slope=2,
                                                 disp_step=100)
