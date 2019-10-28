import os
import sys
import numpy as np
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
    parser.add_argument('-sks', '--source_key_signal', type=str, default='stimuli/signal_in_noise',
                        help='regex for source dataset (audio)')
    parser.add_argument('-sksr', '--source_key_sr', type=str, default='sr',
                        help='regex for source dataset (audio)')
    parser.add_argument('-lpf', '--lowpass_filter_cutoff', type=float, default=3000.0,
                        help='cutoff frequency of lowpass filter applied to nervegrams')
    parser.add_argument('-lpfo', '--lowpass_filter_order', type=int, default=7,
                    help='order of lowpass filter applied to nervegrams')
    args = parser.parse_args()
    # Check commandline arguments
    assert args.source_regex is not None
    assert args.dest_filename is not None
    assert args.job_idx is not None
    assert args.jobs_per_source_file is not None
    # Set bez2018model nervegram parameters
    kwargs_nervegram_meanrates = {
        'meanrates_params': {
            'dur': 0.050,
            'buffer_start_dur': 0.070,
            'buffer_end_dur': 0.010
        },
        'ANmodel_params': {
            'num_cfs': 100,
            'min_cf':125,
            'max_cf':14e3,
            'IhcLowPass_cutoff':args.lowpass_filter_cutoff,
            'IhcLowPass_order': args.lowpass_filter_order
        },
    }
    print("### bez2018model nervegram parameters ###")
    for key in kwargs_nervegram_meanrates.keys():
        if instance(kwargs_nervegram_meanrates[key], dict):
            for sub_key in kwargs_nervegram_meanrates[key].keys():
                print('#', key, sub_key, kwargs_nervegram_meanrates[key][sub_key])
        else:
            print('#', key, kwargs_nervegram_meanrates[key])
    print("### bez2018model nervegram parameters ###")
    # Run bez2018 model
    parallel_run_dataset_generation(args.source_regex,
                                    args.dest_filename,
                                    job_idx=args.job_idx,
                                    jobs_per_source_file=args.jobs_per_source_file,
                                    source_key_signal=args.source_key_signal,
                                    source_key_signal_fs=args.source_key_sr,
                                    source_keys_to_copy=get_source_keys_to_copy(),
                                    kwargs_nervegram_meanrates=kwargs_nervegram_meanrates)
    