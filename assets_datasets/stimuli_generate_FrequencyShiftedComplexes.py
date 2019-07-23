import os
import sys
import h5py
import numpy as np
import pdb
import stimuli_util

# sys.path.append('/om2/user/msaddler/python-packages/bez2018model')
# from bez2018model_run_hdf5_dataset import initialize_hdf5_file, write_example_to_hdf5
sys.path.append('/om4/group/mcdermott/user/msaddler/pitchnet_dataset/pitchnetDataset/pitchnetDataset')
from dataset_util import initialize_hdf5_file, write_example_to_hdf5


def get_MooreMoore2003_spectral_envelope(f0, f_center, f_bandwidth):
    '''
    Returns a function to compute spectral envelope of Moore & Moore (2003)
    frequency-shifted complexes (see left column of page 979).
    
    Args
    ----
    f0 (float): fundamental frequency used to set sloping regions (Hz)
    f_center (float): center frequency of the bandpass filter (Hz)
    f_bandwidth (float): bandwidth of the flat region (Hz)
    
    Returns
    -------
    spectral_envelope (function): returns amplitude envelope as a function of frequency (in Hz)
    '''
    
    f_high_edge = f_center + f_bandwidth/2
    f_low_edge = f_center - f_bandwidth/2
    
    def sloping_region(f, f_edge, f0):
        x = 1. - np.abs((f - f_edge) / (1.5 * f0))
        x[x < 0] = 0
        return (np.power(10, x) - 1) / 9
    
    def spectral_envelope(f):
        envelope = np.ones_like(f).astype(np.float32)
        envelope[f <= f_low_edge] = sloping_region(f[f <= f_low_edge], f_low_edge, f0)
        envelope[f >= f_high_edge] = sloping_region(f[f >= f_high_edge], f_high_edge, f0)
        envelope[envelope < 0] = 0
        return envelope
    
    return spectral_envelope


def get_MooreMoore2003_complex_tone(f0, f0_shift=0.0, spectral_envelope=None,
                                    fs=32000, dur=0.150, dBSPL=70.0, phase_mode='cos'):
    '''
    Returns a frequency-shifted complex with the specified spectral envelope.
    
    Args
    ----
    f0 (float): fundamental frequency of complex tone before frequency-shifting (Hz)
    f0_shift (float): frequency shift applied to all components (fraction of f0)
    spectral_envelope (function): returns amplitude envelope as a function of frequency (in Hz)
    fs (int): sampling rate (Hz)
    dur (float): duration (s)
    dBSPL (float): overall sound presentation level in dB re 20e-6 Pa
    phase_mode (str): specify starting phase of each component
    
    Returns
    -------
    y (np array): frequency-shifted complex tone (units Pa)
    '''
    assert spectral_envelope is not None, "`spectral_envelope` is a required argument"
    frequencies = np.arange(f0, fs/2, f0, dtype=np.float32)
    frequencies = frequencies + f0*f0_shift
    amplitudes = spectral_envelope(frequencies)
    y = stimuli_util.complex_tone(f0, fs, dur, harmonic_numbers=None,
                                  frequencies=frequencies, amplitudes=amplitudes,
                                  phase_mode=phase_mode, offset_start=True, strict_nyquist=False)
    y = stimuli_util.set_dBSPL(y, dBSPL)
    return y


def generate_MooreMoore2003_dataset(hdf5_filename, fs=32000, dur=0.150, dBSPL=70.0, disp_step=10):
    '''
    Main routine for generating Moore & Moore (2003, JASA) frequency-shifted complex tone dataset.
    
    Args
    ----
    '''
    # Define encoding / decoding dictionaries for phase_mode
    phase_mode_encoding = {'sine':0, 'rand':1, 'sch':2, 'cos':3, 'alt':4}
    phase_mode_decoding = {0:'sine', 1:'rand', 2:'sch', 3:'cos', 4:'alt'}
    phase_mode = 'cos'
    
    # Define stimulus-specific parameters
    list_f0_shift = np.arange(0, 0.25, 0.1, dtype=np.float32)
    f0_min=80.
    f0_max=500
    step_size_in_octaves=1/(12*16*2)
    f0s = np.arange(0, np.log2(f0_max / f0_min), step_size_in_octaves)
    f0s = f0_min * (np.power(2, f0s))
    list_f0 = list(f0s)
    list_spectral_envelope_params = [
        {'spectral_envelope_centered_harmonic': 5, 'spectral_envelope_bandwidth_in_harmonics': 3}, # "RES"
        {'spectral_envelope_centered_harmonic': 11, 'spectral_envelope_bandwidth_in_harmonics': 5}, # "INT"
        {'spectral_envelope_centered_harmonic': 16, 'spectral_envelope_bandwidth_in_harmonics': 5}, # "UNRES"
    ]
    
    # Prepare config_dict with config values
    config_dict = {
        'sr': fs,
        'config_tone/fs': fs,
        'config_tone/dur': dur,
        'config_tone/dBSPL': dBSPL,
        'config_tone/phase_mode': phase_mode_encoding[phase_mode],
    }
    config_key_pair_list = [(k, k) for k in config_dict.keys()]
    
    # Iterate over all combinations of stimulus-specific parameters
    N = len(list_f0_shift) * len(list_f0) * len(list_spectral_envelope_params)
    itrN = 0
    for spectral_envelope_params in list_spectral_envelope_params:
        for f0 in list_f0:
            # Spectral envelope depends only on f0 and the spectral envelope condition
            harmonic_centered = spectral_envelope_params['spectral_envelope_centered_harmonic']
            harmonic_bandwidth = spectral_envelope_params['spectral_envelope_bandwidth_in_harmonics']
            f_center = f0 * harmonic_centered
            f_bandwidth = f0 * harmonic_bandwidth
            spectral_envelope = get_MooreMoore2003_spectral_envelope(f0, f_center, f_bandwidth)
            for f0_shift in list_f0_shift:
                # Generate stimulus and store alongside metadata in data_dict
                y = get_MooreMoore2003_complex_tone(f0, f0_shift=f0_shift, spectral_envelope=spectral_envelope,
                                                    fs=fs, dur=dur, dBSPL=dBSPL, phase_mode=phase_mode)
                y = stimuli_util.set_dBSPL(y, dBSPL)
                data_dict = {
                    'stimuli/signal': y.astype(np.float32),
                    'f0': np.float32(f0),
                    'f0_shift': np.float32(f0_shift),
                    'spectral_envelope_f_center': np.float32(f_center),
                    'spectral_envelope_f_bandwidth': np.float32(f_bandwidth),
                    'spectral_envelope_centered_harmonic': int(harmonic_centered),
                    'spectral_envelope_bandwidth_in_harmonics': int(harmonic_bandwidth),
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
                if itrN % disp_step == 0: print('... signal {} of {}'.format(itrN, N))
                itrN += 1
    
    # Close hdf5 file
    hdf5_f.close()
    print('[END]: {}'.format(hdf5_filename))
