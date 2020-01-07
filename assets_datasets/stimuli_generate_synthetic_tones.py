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
from augment_dataset import sample_and_apply_random_filter


def random_filtered_complex_tone(f0, fs, dur,
                                 phase_mode='sine',
                                 amplitude_jitter=0.5,
                                 augmentation_filter_params={}):
    '''
    '''
    frequencies = np.arange(f0, fs/2, f0)
    amplitudes = np.random.uniform(low=1-amplitude_jitter,
                                   high=1+amplitude_jitter,
                                   size=frequencies.shape)
    signal = stimuli_util.complex_tone(f0, fs, dur,
                                       harmonic_numbers=None,
                                       frequencies=frequencies,
                                       amplitudes=amplitudes,
                                       phase_mode=phase_mode,
                                       offset_start=True,
                                       strict_nyquist=True)
    signal, signal_filter_params = sample_and_apply_random_filter(signal, fs,
                                                                  **augmentation_filter_params)
    return signal, signal_filter_params


def random_filtered_complex_tone_dataset(hdf5_filename, N,
                                         fs=32e3,
                                         dur=0.150,
                                         amplitude_jitter=0.5,
                                         phase_modes=['sine'],
                                         augmentation_filter_params={},
                                         kwargs_modified_uniform_masking_noise={},
                                         range_f0=[80.0, 1000.0],
                                         range_snr=[-10., 10.],
                                         range_dbspl=[30., 90.],
                                         out_combined_key='stimuli/signal_in_noise',
                                         out_signal_key='stimuli/signal',
                                         out_noise_key='stimuli/noise',
                                         out_snr_key='snr',
                                         out_augmentation_prefix='augmentation/',
                                         random_seed=858,
                                         disp_step=1000):
    '''
    '''
    # Set random seed
    np.random.seed(random_seed)
    # Randomly sample f0s and phase modes
    list_f0 = np.exp(np.random.uniform(low=np.log(range_f0[0]),
                                       high=np.log(range_f0[1]),
                                       size=[N]))
    list_phase_mode = np.random.choice(phase_modes, size=[N])
    list_snr = np.random.uniform(low=range_snr[0],
                                 high=range_snr[1],
                                 size=[N])
    list_dbspl = np.random.uniform(low=range_dbspl[0],
                                   high=range_dbspl[1],
                                   size=[N])
    # Define encoding / decoding dictionaries for phase_mode
    phase_mode_encoding = {'sine':0, 'rand':1, 'sch':2, 'cos':3, 'alt':4}
    phase_mode_decoding = {0:'sine', 1:'rand', 2:'sch', 3:'cos', 4:'alt'}
    # Prepare data_dict and config_key_pair_list for hdf5 filewriting
    data_dict = {
        'sr': fs,
        'config_tone/fs': fs,
        'config_tone/dur': dur,
        'config_tone/f0_min': range_f0[0],
        'config_tone/f0_max': range_f0[1],
        'config_tone/amplitude_jitter': amplitude_jitter,
    }
    for key in kwargs_modified_uniform_masking_noise.keys():
        noise_augmentation_key = out_augmentation_prefix + 'modified_uniform_masking_noise_' + key
        data_dict[noise_augmentation_key] = kwargs_modified_uniform_masking_noise[key]
    config_key_pair_list = [(k, k) for k in data_dict.keys()]
    data_key_pair_list = [] # Will be populated right before initializing hdf5 file
    # Main loop to generate dataset
    for itrN in range(0, N):
        # Generate signal and noise with desired properties
        f0 = list_f0[itrN]
        phase_mode = list_phase_mode[itrN]
        snr = list_snr[itrN]
        dbspl = list_dbspl[itrN]
        signal, signal_filter_params = random_filtered_complex_tone(f0,
                                                                    fs,
                                                                    dur,
                                                                    phase_mode=phase_mode,
                                                                    amplitude_jitter=amplitude_jitter,
                                                                    augmentation_filter_params=augmentation_filter_params)
        noise = stimuli_util.modified_uniform_masking_noise(fs, dur, **kwargs_modified_uniform_masking_noise)
        signal_and_noise = stimuli_util.combine_signal_and_noise(signal=signal,
                                                                 noise=noise,
                                                                 snr=snr,
                                                                 rms_out=20e-6*np.power(10, dbspl/20))
        # Prepare data_dict for hdf5 filewriting
        data_dict['f0'] = f0
        data_dict['phase_mode'] = int(phase_mode_encoding[phase_mode])
        data_dict[out_combined_key + '_dBSPL'] = dbspl
        data_dict[out_snr_key] = snr
        data_dict[out_combined_key] = signal_and_noise.astype(np.float32)
        if out_signal_key is not None: data_dict[out_signal_key] = signal.astype(np.float32)
        if out_noise_key is not None: data_dict[out_noise_key] = noise.astype(np.float32)
        for key in signal_filter_params.keys():
            data_dict[out_augmentation_prefix + 'signal_' + key] = signal_filter_params[key]
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
            print('... signal {} of {} (f0={:.2f}, dbspl={:.2f}, snr={:.2f})'.format(itrN, N, f0, dbspl, snr),
                  signal_filter_params)
    # Close hdf5 file
    hdf5_f.close()
    print('[END]: {}'.format(hdf5_filename))



if __name__ == "__main__":
    ''' TEMPORARY COMMAND-LINE USAGE '''
    assert len(sys.argv) == 3, "scipt usage: python <script_name> <hdf5_filename> <N>"
    hdf5_filename = str(sys.argv[1])
    N = int(sys.argv[2])
    
#     augmentation_filter_params = {
#        'filter_signal': True, # filter_signalBPv00
#         'filter_noise': False,
#         'btype': 'bandpass',
#         'sampling_kwargs': {
#             'filter_fraction': 1.0,
#             'N_range': [1, 5],
#             'fc_range': [1e2, 5e3],
#             'bw_range': [2e3, 1e4],
#             'fc_log_scale': True,
#             'bw_log_scale': False
#         },
#     }
    augmentation_filter_params = {
        'filter_signal': True, # filter_signalHPv00
        'filter_noise': False,
        'btype': 'highpass',
        'sampling_kwargs': {
           'filter_fraction': 1.0,
           'N_range': [1, 5],
           'fc_range': [1e3, 1e4],
           'fc_log_scale': True,
        },
    }
#     augmentation_filter_params = {
#         'filter_signal': True, # filter_signalLPv00
#         'filter_noise': False,
#         'btype': 'lowpass',
#         'sampling_kwargs': {
#            'filter_fraction': 1.0,
#            'N_range': [1, 5],
#            'fc_range': [1e3, 1e4],
#            'fc_log_scale': True,
#         },
#     }
    
    kwargs_modified_uniform_masking_noise = {
        'dBHzSPL': 15.0,
        'attenuation_start': 600.0,
        'attenuation_slope': 2.0,
    }
    
    random_filtered_complex_tone_dataset(hdf5_filename, N,
                                         fs=32e3,
                                         dur=0.150,
                                         amplitude_jitter=0.5,
                                         phase_modes=['sine'],
                                         augmentation_filter_params=augmentation_filter_params,
                                         kwargs_modified_uniform_masking_noise=kwargs_modified_uniform_masking_noise,
                                         range_f0=[80.0, 1000.0],
                                         range_snr=[-10., 10.],
                                         range_dbspl=[30., 90.],
                                         out_combined_key='stimuli/signal_in_noise',
                                         out_signal_key=None,
                                         out_noise_key=None,
                                         out_snr_key='snr',
                                         out_augmentation_prefix='augmentation/',
                                         random_seed=858,
                                         disp_step=1000)
