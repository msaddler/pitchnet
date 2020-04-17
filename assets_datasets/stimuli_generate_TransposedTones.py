import os
import sys
import h5py
import numpy as np
import pdb
import scipy.signal

sys.path.append('/om2/user/msaddler/python-packages/msutil')
import util_stimuli

sys.path.append('/om4/group/mcdermott/user/msaddler/pitchnet_dataset/pitchnetDataset/pitchnetDataset')
from dataset_util import initialize_hdf5_file, write_example_to_hdf5


def get_Oxenham2004_transposed_tone(f_carrier, f_envelope, fs=32000, dur=0.150, buffer_dur=1.0,
                                    dBSPL=70.0, offset_start=True, lowpass_filter_envelope=True):
    '''
    Returns a transposed tone with specified carrier and envelope frequencies
    as described by Oxenham et al. (2004, PNAS). If `f_carrier` is set to 0,
    this function will return a pure tone with frequency `f_envelope`.
    
    Args
    ----
    f_carrier (float): carrier frequency of transposed tone (Hz)
    f_envelope (float): envelope frequency of transposed tone (Hz)
    spectral_envelope (function): returns amplitude envelope as a function of frequency (in Hz)
    fs (int): sampling rate (Hz)
    dur (float): duration of output signal (s)
    buffer_dur (float): duration of buffer added to signal to avoid filter edge effects (s)
    dBSPL (float): overall sound presentation level in dB re 20e-6 Pa
    offset_start (bool): if True, starting phase is offset by np.random.rand() * dur
    lowpass_filter_envelope (bool): if True, lowpass filter is applied to envelope after HWR
    
    Returns
    -------
    signal (np array): transposed tone (units Pa)
    '''
    t = np.arange(0, dur + buffer_dur, 1 / fs)
    if offset_start: t = t + np.random.rand() * dur
    envelope = np.sin(2 * np.pi * f_envelope * t)
    
    if f_carrier > 0:
        envelope[envelope<0] = 0 # Half-wave rectify the envelope
        if lowpass_filter_envelope:
            N = 4 / 2 # Apply a 4th order lowpass Butterworth filter with `filtfilt`
            Wn = 0.2 * f_carrier / (fs / 2) # Cutoff frequency is 0.2 * f_carrier
            b, a = scipy.signal.butter(N, Wn, btype='low')
            envelope = scipy.signal.filtfilt(b, a, envelope)
        carrier = np.sin(2 * np.pi * f_carrier * t)
        signal = carrier * envelope
    else:
        # If f_carrier = 0, return a pure tone at envelope frequency
        print('f_carrier = 0 --> generating pure tone at f_envelope')
        signal = envelope
    start_index = int(buffer_dur/2 * fs)
    end_index = start_index + int(dur * fs)
    signal = signal[start_index:end_index]
    signal = util_stimuli.set_dBSPL(signal, dBSPL)
    return signal


def generate_Oxenham2004_dataset(hdf5_filename, fs=32000, dur=0.150, buffer_dur=1.0,
                                 dBSPL=70.0, offset_start=True, lowpass_filter_envelope=True,
                                 list_f_carrier = [0.0, 4000.0, 6350.0, 10080.0],
                                 f0_min=80.0, f0_max=320., step_size_in_octaves=1/(12*16*16),
                                 disp_step=100):
    '''
    Main routine for generating Oxenham et al. (2004, PNAS) transposed tone dataset.
    
    Args
    ----
    '''
    # Define stimulus-specific parameters
    f0s = np.arange(0, np.log2(f0_max / f0_min), step_size_in_octaves)
    f0s = f0_min * (np.power(2, f0s))
    list_f_envelope = list(f0s)
    list_f_carrier = list(list_f_carrier)
    
    # Prepare config_dict with config values
    config_dict = {
        'sr': fs,
        'config_tone/fs': fs,
        'config_tone/dur': dur,
        'config_tone/dBSPL': dBSPL,
    }
    config_key_pair_list = [(k, k) for k in config_dict.keys()]
    
    # Iterate over all combinations of stimulus-specific parameters
    N = len(list_f_carrier) * len(list_f_envelope)
    itrN = 0
    for f_carrier in list_f_carrier:
        for f_envelope in list_f_envelope:
            # Generate stimulus and store alongside metadata in data_dict
            y = get_Oxenham2004_transposed_tone(f_carrier, f_envelope, fs=fs, dur=dur, buffer_dur=buffer_dur,
                                                dBSPL=dBSPL, offset_start=offset_start,
                                                lowpass_filter_envelope=lowpass_filter_envelope)
            data_dict = {
                'stimuli/signal': y.astype(np.float32),
                'f0': np.float32(f_envelope),
                'f_envelope': np.float32(f_envelope),
                'f_carrier': np.float32(f_carrier),
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
