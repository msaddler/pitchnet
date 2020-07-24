import sys
import os
import h5py
import json
import numpy as np
import scipy.signal
import librosa
import itertools
import argparse
import pdb

sys.path.append('/om2/user/msaddler/python-packages/msutil')
import util_stimuli

sys.path.append('/om4/group/mcdermott/user/msaddler/pitchnet_dataset/pitchnetDataset/pitchnetDataset')
from dataset_util import initialize_hdf5_file, write_example_to_hdf5
from augment_dataset import sample_and_apply_random_filter


def sample_and_apply_stable_filter(b_lp, a_lp_mean, a_lp_cov, x, invert_filter=False):
    '''
    Sample filter transfer function coefficients from multivariate Gaussian
    (filter stability ensured via rejection sampling) and apply filter to
    given signal.
    
    Args
    ----
    b_lp (array_like): numerator coefficients of filter to apply
    a_lp_mean (array_like): denominator coef mean
    a_lp_cov (array_like): denominator coef covariance matrix
    x (array_like): input signal to be filterd
    invert_filter (bool): if True, filter is inverted by switching
        numerator and denominator coefficients before applying filter
    
    Returns
    -------
    scipy.signal.lfilter(b, a, x): filtered signal
    b (array_like): numerator coefficients of applied filter
    a (array_like): denominator coefficients of applied filter
    '''
    filter_is_unstable = True
    while filter_is_unstable:
        a_lp = np.random.multivariate_normal(a_lp_mean, a_lp_cov)
        z, p, k = scipy.signal.tf2zpk(b_lp, a_lp)
        filter_is_unstable = np.any(np.abs(p) >= 1)
    if invert_filter:
        b = a_lp
        a = b_lp
    else:
        b = b_lp
        a = a_lp
    return scipy.signal.lfilter(b, a, x), b, a


def spectrally_shaped_synthetic_dataset(hdf5_filename,
                                        N,
                                        spectral_statistics_filename,
                                        fs=32e3,
                                        dur=0.150,
                                        phase_modes=['sine'],
                                        range_f0=[80.0, 1001.3713909809752],
                                        range_snr=[-10., 10.],
                                        range_dbspl=[30., 90.],
                                        n_mfcc=12,
                                        invert_signal_filter=False,
                                        invert_noise_filter=False,
                                        out_combined_key='stimuli/signal_in_noise',
                                        out_signal_key='stimuli/signal',
                                        out_noise_key='stimuli/noise',
                                        out_snr_key='snr',
                                        out_augmentation_prefix='augmentation/',
                                        random_seed=858,
                                        disp_step=1000):
    '''
    '''
    # Gather mean and covariance of signal and noise MFCCs
    with open(spectral_statistics_filename, 'r') as f:
        spectral_statistics_dict = json.load(f)
    signal_mfcc_mean = np.array(spectral_statistics_dict[out_signal_key]['mfcc_mean'])
    signal_mfcc_cov = np.array(spectral_statistics_dict[out_signal_key]['mfcc_cov'])
    noise_mfcc_mean = np.array(spectral_statistics_dict[out_noise_key]['mfcc_mean'])
    noise_mfcc_cov = np.array(spectral_statistics_dict[out_noise_key]['mfcc_cov'])
    assert fs == spectral_statistics_dict[out_signal_key]['sr']
    assert fs == spectral_statistics_dict[out_noise_key]['sr']
    assert np.all(noise_mfcc_mean.shape == signal_mfcc_mean.shape)
    print('Loaded spectral_statistics_dict from: {}'.format(spectral_statistics_filename))
    
    # Multiply mean MFCCs by -1 to invert spectral envelopes
    if invert_signal_filter:
        print('<><><> INVERTING `signal_mfcc_mean` <><><>')
        signal_mfcc_mean = -1 * signal_mfcc_mean
    if invert_noise_filter:
        print('<><><> INVERTING `noise_mfcc_mean` <><><>')
        noise_mfcc_mean = -1 * noise_mfcc_mean
    
    # Define inverse Mel-filterbank
    n_fft = int(fs * dur)
    n_mels = len(signal_mfcc_mean)
    M = librosa.filters.mel(fs, n_fft, n_mels=n_mels)
    Minv = np.linalg.pinv(M)
    n_fft_freqs = np.fft.rfftfreq(n_fft, d=1/fs)
    
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
        out_augmentation_prefix + 'signal_mfcc_mean': signal_mfcc_mean,
        out_augmentation_prefix + 'signal_mfcc_cov': signal_mfcc_cov,
        out_augmentation_prefix + 'noise_mfcc_mean': noise_mfcc_mean,
        out_augmentation_prefix + 'noise_mfcc_cov': noise_mfcc_cov,
        out_augmentation_prefix + 'n_fft': n_fft,
        out_augmentation_prefix + 'n_mels': n_mels,
        out_augmentation_prefix + 'n_mfcc': n_mfcc,
    }
    config_key_pair_list = [(k, k) for k in data_dict.keys()]
    data_key_pair_list = [] # Will be populated right before initializing hdf5 file
    print('|======================== config ========================|')
    for k in sorted(data_dict.keys()):
        print(k, data_dict[k])
    print('|======================== config ========================|')
    
    # Main loop to generate dataset
    for itrN in range(0, N):
        # Generate signal and noise with desired properties
        f0 = list_f0[itrN]
        phase_mode = list_phase_mode[itrN]
        snr = list_snr[itrN]
        dbspl = list_dbspl[itrN]
        
        # Generate harmonic signal with sampled spectral envelope (MFCCs drawn from multivariate normal)
        signal_mfcc = np.random.multivariate_normal(signal_mfcc_mean, signal_mfcc_cov)
        signal_mfcc[n_mfcc:] = 0
        signal_power_spectrum = util_stimuli.get_power_spectrum_from_mfcc(signal_mfcc, Minv) 
        frequencies = np.arange(f0, fs/2, f0)
        amplitudes = np.interp(frequencies,
                               n_fft_freqs, 
                               np.sqrt(signal_power_spectrum))
        signal = util_stimuli.complex_tone(f0,
                                           fs,
                                           dur,
                                           harmonic_numbers=None,
                                           frequencies=frequencies,
                                           amplitudes=amplitudes,
                                           phase_mode=phase_mode,
                                           offset_start=True,
                                           strict_nyquist=True)
        signal = util_stimuli.set_dBSPL(signal, 60.0)
        
        # Generate white noise and impose sampled spectral envelope (MFCCs drawn from multivariate normal)
        noise = np.random.randn(n_fft)
        noise_mfcc = np.random.multivariate_normal(noise_mfcc_mean, noise_mfcc_cov)
        noise_mfcc[n_mfcc:] = 0
        noise_power_spectrum = util_stimuli.get_power_spectrum_from_mfcc(noise_mfcc, Minv) 
        noise = util_stimuli.impose_power_spectrum(noise, noise_power_spectrum)
        noise = util_stimuli.set_dBSPL(noise, 60.0)
        
        # Combine signal and noise at desired SNR and dB SPL
        signal_and_noise = util_stimuli.combine_signal_and_noise(signal, noise, snr)
        signal_and_noise = util_stimuli.set_dBSPL(signal_and_noise, dbspl)
        
        # Prepare data_dict for hdf5 filewriting
        data_dict['f0'] = f0
        data_dict['phase_mode'] = int(phase_mode_encoding[phase_mode])
        data_dict[out_combined_key + '_dBSPL'] = dbspl
        data_dict[out_snr_key] = snr
        data_dict[out_combined_key] = signal_and_noise.astype(np.float32)
        data_dict[out_signal_key] = signal.astype(np.float32)
        data_dict[out_noise_key] = noise.astype(np.float32)
        data_dict[out_augmentation_prefix + 'signal_mfcc'] = signal_mfcc
        data_dict[out_augmentation_prefix + 'noise_mfcc'] = noise_mfcc
        
        # Initialize the hdf5 file on the first iteration
        if itrN == 0:
            print('[INITIALIZING]: {}'.format(hdf5_filename))
            for k in data_dict.keys():
                if not (k, k) in config_key_pair_list:
                    data_key_pair_list.append((k, k))
            initialize_hdf5_file(hdf5_filename,
                                 N,
                                 data_dict,
                                 file_mode='w',
                                 data_key_pair_list=data_key_pair_list,
                                 config_key_pair_list=config_key_pair_list)
            hdf5_f = h5py.File(hdf5_filename, 'r+')
        
        # Write each data_dict to hdf5 file
        write_example_to_hdf5(hdf5_f, data_dict, itrN, data_key_pair_list=data_key_pair_list)
        if itrN % disp_step == 0:
            print('... signal {} of {} (f0={:.2f}, dbspl={:.2f}, snr={:.2f})'.format(itrN, N, f0, dbspl, snr))
    
    # Close hdf5 file
    hdf5_f.close()
    print('[END]: {}'.format(hdf5_filename))
    return


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
    signal = util_stimuli.complex_tone(f0, fs, dur,
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
                                         range_f0=[80.0, 1001.3713909809752],
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
        noise = util_stimuli.modified_uniform_masking_noise(fs, dur, **kwargs_modified_uniform_masking_noise)
        signal_and_noise = util_stimuli.combine_signal_and_noise(signal, noise, snr)
        signal_and_noise = util_stimuli.set_dBSPL(signal_and_noise, dbspl)
        
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
    return



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel add background noise to dataset")
    parser.add_argument('-d', '--dest_filename', type=str, help='destination dataset filename')
    parser.add_argument('-j', '--job_idx', type=int, default=0, help='job index')
    parser.add_argument('-npj', '--num_parallel_jobs', type=int, default=1, help='number of parallel jobs')
    parser.add_argument('-nts', '--num_total_stimuli', type=int, default=2100000, help='total number of stimuli')
    parser.add_argument('-isf', '--invert_signal_filter', type=int, default=0, help='invert signal spectral envelopes')
    args = parser.parse_args()
    assert args.dest_filename is not None, "-d (--dest_filename) is a required argument"
    
    # Modify dest_filename to reflect parallelization
    partitions = np.linspace(0, args.num_total_stimuli, args.num_parallel_jobs+1, dtype=int)
    idx_start = partitions[args.job_idx]
    idx_end = partitions[args.job_idx + 1]
    N = idx_end - idx_start
    dest_filename = args.dest_filename
    sidx = dest_filename.rfind('.')
    dest_filename = dest_filename[:sidx] + '_{:07d}-{:07d}' + dest_filename[sidx:]
    dest_filename = dest_filename.format(idx_start, idx_end)
    print('[START] {}'.format(dest_filename))
    print('job_idx={}, N={}'.format(args.job_idx, N))
    
    spectral_statistics_filename = '/om/scratch/Mon/msaddler/data_pitchnet/PND_v08/noise_TLAS_snr_neg10pos10/SPECTRAL_STATISTICS_v00/results_dict.json'
    spectrally_shaped_synthetic_dataset(dest_filename,
                                        N,
                                        spectral_statistics_filename,
                                        fs=32e3,
                                        dur=0.150,
                                        phase_modes=['cos'],
                                        range_f0=[80.0, 1001.3713909809752],
                                        range_snr=[-10., 10.],
                                        range_dbspl=[30., 90.],
                                        n_mfcc=12,
                                        invert_signal_filter=bool(args.invert_signal_filter),
                                        invert_noise_filter=False,
                                        out_combined_key='stimuli/signal_in_noise',
                                        out_signal_key='stimuli/signal',
                                        out_noise_key='stimuli/noise',
                                        out_snr_key='snr',
                                        out_augmentation_prefix='augmentation/',
                                        random_seed=args.job_idx,
                                        disp_step=50)
    
#     augmentation_filter_params = {
#         'filter_signal': True, # filter_signalBPv00
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
#     augmentation_filter_params = {
#         'filter_signal': True, # filter_signalHPv00
#         'filter_noise': False,
#         'btype': 'highpass',
#         'sampling_kwargs': {
#             'filter_fraction': 1.0,
#             'N_range': [1, 5],
#             'fc_range': [1e3, 1e4],
#             'fc_log_scale': True,
#         },
#     }
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
    
#     kwargs_modified_uniform_masking_noise = {
#         'dBHzSPL': 15.0,
#         'attenuation_start': 600.0,
#         'attenuation_slope': 2.0,
#     }
    
#     for key in sorted(augmentation_filter_params.keys()):
#         print('augmentation_filter_params/', key, augmentation_filter_params[key])
#     for key in sorted(kwargs_modified_uniform_masking_noise.keys()):
#         print('kwargs_modified_uniform_masking_noise/', key, kwargs_modified_uniform_masking_noise[key])
#     random_filtered_complex_tone_dataset(dest_filename, N,
#                                          fs=32e3,
#                                          dur=0.150,
#                                          amplitude_jitter=0.5,
#                                          phase_modes=['sine', 'rand'],
#                                          augmentation_filter_params=augmentation_filter_params,
#                                          kwargs_modified_uniform_masking_noise=kwargs_modified_uniform_masking_noise,
#                                          range_f0=[80.0, 1001.3713909809752],
#                                          range_snr=[-10., 10.],
#                                          range_dbspl=[30., 90.],
#                                          out_combined_key='stimuli/signal_in_noise',
#                                          out_signal_key='stimuli/signal',
#                                          out_noise_key='stimuli/noise',
#                                          out_snr_key='snr',
#                                          out_augmentation_prefix='augmentation/',
#                                          random_seed=args.job_idx,
#                                          disp_step=1000)
