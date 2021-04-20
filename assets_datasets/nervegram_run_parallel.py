import os
import sys
import numpy as np
import glob
import argparse

sys.path.append('/om2/user/msaddler/python-packages/bez2018model')
from bez2018model_run_hdf5_dataset import parallel_run_dataset_generation


def get_source_keys_to_copy():
    '''
    Helper function to return exhaustive list of possible stimulus metadata
    keys to be copied from source dataset to destination dataset.
    '''
    source_keys_to_copy = [
        'f0',
        'f0_label',
        'f0_log2',
        'f0_lognormal',
        'snr',
        'phase_mode',
        'low_harm',
        'upp_harm',
        'max_audible_harm',
        'min_audible_harm',
        'base_f0',
        'delta_f0',
        'nopad_f0_mean',
        'nopad_f0_median',
        'nopad_f0_stddev',
        'f0_shift',
        'spectral_envelope_bandwidth_in_harmonics',
        'spectral_envelope_centered_harmonic',
        'spectral_envelope_f_bandwidth',
        'spectral_envelope_f_center',
        'f_carrier',
        'f_envelope',
        'filter_fl',
        'filter_fh',
        'mistuned_harm',
        'mistuned_pct',
        'snr_per_component',
        'dbspl',
        'fl',
        'fh',
        'bandpass_fl',
        'bandpass_fh',
        'jitter_mode',
        'jitter_pattern',
    ]
    return source_keys_to_copy


if __name__ == "__main__":
    # Parse commandline arguments
    parser = argparse.ArgumentParser(description="run bez2018model in parallel")
    parser.add_argument('-s', '--source_regex', type=str, default=None,
                        help='regex for source dataset (audio)')
    parser.add_argument('-d', '--dest_filename', type=str, default=None,
                        help='destination filename for output dataset (nervegram)')
    parser.add_argument('-j', '--job_idx', type=int, default=None,
                        help='index of current job')
    parser.add_argument('-jps', '--jobs_per_source_file', type=int, default=None, 
                        help='number of jobs per source dataset file')
    parser.add_argument('-sks', '--source_key_signal', type=str, default='auto',
                        help='key for signal in source dataset')
    parser.add_argument('-sksr', '--source_key_sr', type=str, default='sr',
                        help='key for signal sampling rate in source dataset')
    parser.add_argument('-mrsr', '--meanrates_sr', type=float, default=10e3,
                        help='sampling rate for auditory nerve firing rates (Hz)')
    parser.add_argument('-bwsf', '--bandwidth_scale_factor', type=float, default=1.0,
                        help='scales cochlear filter bandwidths in the Carney model')
    parser.add_argument('-lpf', '--lowpass_filter_cutoff', type=float, default=3000.0,
                        help='IHC lowpass filter cutoff frequency')
    parser.add_argument('-lpfo', '--lowpass_filter_order', type=int, default=7,
                        help='IHC lowpass filter order')
    parser.add_argument('-spont', '--spont_rate', type=float, default=70.0,
                        help='ANF spontaneous rate: options are 70.0, 4.0, or 0.1 Hz')
    parser.add_argument('-ncf', '--num_cf', type=int, default=100,
                        help='number of auditory nerve center frequencies')
    parser.add_argument('-nst', '--num_spike_trains', type=int, default=1,
                        help='number of auditory nerve fiber spike trains to sample')
    args = parser.parse_args()
    # Check commandline arguments
    assert args.source_regex is not None
    assert args.dest_filename is not None
    assert args.job_idx is not None
    assert args.jobs_per_source_file is not None
    # Set bez2018model nervegram parameters
    kwargs_nervegram = {
        'nervegram_dur': 0.050,
        'nervegram_fs': args.meanrates_sr,
        'buffer_start_dur': 0.070,
        'buffer_end_dur': 0.010,
        'pin_fs': 100e3,
        'pin_dBSPL_flag': 0,
        'pin_dBSPL': None,
        'species': 2,
        'bandwidth_scale_factor': args.bandwidth_scale_factor,
        'cf_list': None,
        'num_cf': args.num_cf,
        'min_cf': 125.0,
        'max_cf': 14e3,
        'max_spikes_per_train': 500,
        'num_spike_trains': args.num_spike_trains,
        'cohc': 0.0,
        'cihc': 1.0,
        'IhcLowPass_cutoff': args.lowpass_filter_cutoff,
        'IhcLowPass_order': args.lowpass_filter_order,
        'spont': args.spont_rate,
        'noiseType': 0,
        'implnt': 0,
        'tabs': 6e-4,
        'trel': 6e-4,
        'random_seed': None,
        'return_vihcs': False,
        'return_meanrates': True,
        'return_spike_times': False,
        'return_spike_tensor_sparse': False,
        'return_spike_tensor_dense': False,
        'nervegram_spike_tensor_fs': 100e3,
    }
    print(args.dest_filename)
    print(args.source_regex)
    print("### bez2018model nervegram parameters ###")
    for key in sorted(kwargs_nervegram.keys()):
        if isinstance(kwargs_nervegram[key], dict):
            for sub_key in sorted(kwargs_nervegram[key].keys()):
                print('#', key, sub_key, kwargs_nervegram[key][sub_key])
        else:
            print('#', key, kwargs_nervegram[key])
    print("### bez2018model nervegram parameters ###")
    
    # Quick check to ensure nervegram parameters match those advertised in dest_filename
    if kwargs_nervegram['spont'] > 1:
        spont_str = '{:03d}'.format(int(kwargs_nervegram['spont']))
    else:
        spont_str = '{:d}eN1'.format(int(kwargs_nervegram['spont'] * 10))
    fn_check = 'sr{:d}_cf{:03d}_species{:03d}_spont{}_BW{:02d}eN1_IHC{:04d}Hz_IHC{:d}order'.format(
        int(kwargs_nervegram['nervegram_fs']),
        int(kwargs_nervegram['num_cf']),
        int(kwargs_nervegram['species']),
        spont_str,
        int(kwargs_nervegram['bandwidth_scale_factor'] * 10),
        int(kwargs_nervegram['IhcLowPass_cutoff']),
        int(kwargs_nervegram['IhcLowPass_order']),
    )
    print(fn_check)
    assert fn_check in args.dest_filename, "FAILED DEST FILENAME CHECK"
    
    # Automating selection of source keys
    source_fn_list = sorted(glob.glob(args.source_regex))
    source_fn = source_fn_list[args.job_idx // args.jobs_per_source_file]
    source_key_signal = args.source_key_signal
    source_key_signal_fs = args.source_key_sr
    if source_key_signal.lower() == 'auto':
        if 'bernox2005' in source_fn:
            source_key_signal = 'tone_in_noise'
            source_key_signal_fs = 'config_tone/fs'
        elif 'moore1985' in source_fn:
            source_key_signal = 'stimuli/signal'
            source_key_signal_fs = 'config_tone/fs'
        elif 'mooremoore2003' in source_fn:
            source_key_signal = 'stimuli/signal'
            source_key_signal_fs = 'config_tone/fs'
        elif 'oxenham2004' in source_fn:
            source_key_signal = 'stimuli/signal_in_noise'
            source_key_signal_fs = 'config_tone/fs'
        elif 'shackcarl1994' in source_fn:
            source_key_signal = 'tone_in_noise'
            source_key_signal_fs = 'config_tone/fs'
        elif 'mcpherson2020' in source_fn:
            source_key_signal = 'tone_in_noise'
            source_key_signal_fs = 'config_tone/fs'
        else:
            source_key_signal = 'stimuli/signal_in_noise'
            source_key_signal_fs = 'sr'
        print('AUTO-SELECTED source keys for source_fn={}'.format(source_fn))
        print('\t source_key_signal={}'.format(source_key_signal))
        print('\t source_key_signal_fs={}'.format(source_key_signal_fs))
    
    # Automating generation of destination filename
    dest_filename = args.dest_filename
    if not os.path.dirname(dest_filename):
        dest_filename = os.path.join(os.path.dirname(source_fn), dest_filename, 'bez2018meanrates.hdf5')
        print('AUTO-GENERATED dest_filename={}'.format(dest_filename))
    
    # Run bez2018 model
    np.random.seed(args.job_idx)
    parallel_run_dataset_generation(args.source_regex,
                                    dest_filename,
                                    job_idx=args.job_idx,
                                    jobs_per_source_file=args.jobs_per_source_file,
                                    source_key_signal=source_key_signal,
                                    source_key_signal_fs=source_key_signal_fs,
                                    source_keys_to_copy=get_source_keys_to_copy(),
                                    kwargs_nervegram=kwargs_nervegram,
                                    range_dbspl=None)
