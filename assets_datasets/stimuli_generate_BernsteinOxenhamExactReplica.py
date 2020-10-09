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


def bernox2005_bandpass_complex_tone(f0,
                                     fs,
                                     dur,
                                     frequency_response_in_dB=None,
                                     threshold_dBSPL=33.3,
                                     component_dBSL=15.,
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
    harmonic_freqs = np.arange(f0, fs/2, f0)
    harmonic_numbers = harmonic_freqs / f0
    harmonic_dBSPL = threshold_dBSPL + component_dBSL + frequency_response_in_dB(harmonic_freqs)
    amplitudes = 20e-6 * np.power(10, (harmonic_dBSPL/20))
    signal = util_stimuli.complex_tone(f0, fs, dur, harmonic_numbers=harmonic_numbers,
                                       amplitudes=amplitudes, **kwargs_complex_tone)
    audible_harmonic_numbers = harmonic_numbers[harmonic_dBSPL >= threshold_dBSPL]
    return signal, audible_harmonic_numbers


def generate_BernsteinOxenhamExactReplica_dataset(hdf5_filename,
                                                  fs=32e3,
                                                  dur=0.150,
                                                  phase_modes=['sine', 'rand'],
                                                  unique_fl_list=[1250, 2500, 3750, 5000],
                                                  unique_fh_list=[1750, 3500, 5250, 7000],
                                                  f0_min=080.0,
                                                  f0_max=640.0,
                                                  step_size_in_octaves=1/(12*16*16),
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
    # Define list of unique f0 values
    unique_f0_list = np.arange(0, np.log2(f0_max / f0_min), step_size_in_octaves)
    unique_f0_list = f0_min * (np.power(2, unique_f0_list))
    # Compute number of stimuli (all combinations of phase_mode, f0, and filter position)
    N = len(unique_ph_list) * len(unique_f0_list) * len(unique_fl_list)
    # Prepare data_dict and config_key_pair_list for hdf5 filewriting
    data_dict = {
        'sr': fs,
        'config_tone/fs': fs,
        'config_tone/dur': dur,
        'config_tone/unique_ph_list': unique_ph_list,
        'config_tone/f0_min': f0_min,
        'config_tone/f0_max': f0_max,
        'config_tone/step_size_in_octaves': step_size_in_octaves,
        'config_tone/filter_order': filter_order,
        'config_tone/unique_fl_list': unique_fl_list,
        'config_tone/unique_fh_list': unique_fh_list,
        'config_tone/threshold_dBSPL': threshold_dBSPL,
        'config_tone/component_dBSL': component_dBSL,
        'config_tone/noise_dBHzSPL': noise_dBHzSPL,
        'config_tone/noise_attenuation_start': noise_attenuation_start,
        'config_tone/noise_attenuation_slope': noise_attenuation_slope,
    }
    config_key_pair_list = [(k, k) for k in data_dict.keys()]
    data_key_pair_list = [] # Will be populated right before initializing hdf5 file
    # Construct modified uniform masking noise buffer to sample from
    if np.isinf(noise_dBHzSPL):
        print('------> USING ZERO NOISE <------')
        noise_buffer = np.zeros(100*dur*fs)
    noise_buffer = util_stimuli.modified_uniform_masking_noise(
        fs,
        100*dur,
        dBHzSPL=noise_dBHzSPL,
        attenuation_start=noise_attenuation_start,
        attenuation_slope=noise_attenuation_slope)
    # Main loop to generate the bandpass filtered tones
    itrN = 0
    for (fl, fh) in zip(unique_fl_list, unique_fh_list):
        frequency_response_in_dB = get_bandpass_filter_frequency_response(fl,
                                                                          fh,
                                                                          fs=fs,
                                                                          order=filter_order)
        for f0 in unique_f0_list:
            for ph in unique_ph_list:
                signal, audible_harmonic_numbers = bernox2005_bandpass_complex_tone(
                    f0,
                    fs,
                    dur,
                    frequency_response_in_dB=frequency_response_in_dB,
                    threshold_dBSPL=threshold_dBSPL,
                    component_dBSL=component_dBSL,
                    phase_mode=phase_mode_decoding[ph])
                
                noise_idx_start = np.random.randint(len(signal),
                                                    len(noise_buffer)-2*len(signal))
                noise = noise_buffer[noise_idx_start : noise_idx_start+len(signal)]
                # Add signal + noise and metadata to data_dict for hdf5 filewriting
                tone_in_noise = signal + noise
                data_dict['tone_in_noise'] = tone_in_noise.astype(np.float32)
                data_dict['f0'] = f0
                data_dict['fl'] = fl
                data_dict['fh'] = fh
                data_dict['phase_mode'] = int(ph)
                data_dict['low_harm'] = int(np.min(audible_harmonic_numbers))
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
                    print('... signal {} of {}, f0={}, filter=[{},{}], audible_harmonics=[{},{}]'.format(
                        itrN, N, f0, fl, fh, audible_harmonic_numbers[0], audible_harmonic_numbers[-1]))
                itrN += 1
    
    # Close hdf5 file
    hdf5_f.close()
    print('[END]: {}'.format(hdf5_filename))
    

if __name__ == "__main__":
    ''' TEMPORARY COMMAND-LINE USAGE '''
    assert len(sys.argv) == 2, "scipt usage: python <script_name> <hdf5_filename>"
    hdf5_filename = str(sys.argv[1])
    
    generate_BernsteinOxenhamExactReplica_dataset(hdf5_filename)
