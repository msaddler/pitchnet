import os
import sys
import numpy as np

sys.path.append('/om2/user/msaddler/python-packages/bez2018model')
from bez2018model_run_hdf5_dataset import parallel_run_dataset_generation


if __name__ == "__main__":
    '''
    TEMPORARY COMMAND LINE USAGE
    TODO: argparser and config files for nervegram parameters
    '''
    assert len(sys.argv) == 5, "scipt usage: python <script_name> <source_regex> <dest_filename> <int> <int>"
    source_regex = str(sys.argv[1])
    dest_filename = str(sys.argv[2])
    job_idx = int(sys.argv[3])
    jobs_per_source_file = int(sys.argv[4])
    
#     source_key_signal = 'stimuli/signal'
#     source_key_signal = 'stimuli/signal_in_noise'
#     source_key_signal_fs = 'sr'
    source_key_signal = 'tone_in_noise'
    source_key_signal_fs = 'config_tone/fs'
    source_keys_to_copy = [
        'f0',
        'f0_label',
        'f0_log2',
        'f0_lognormal',
        'phase_mode',
        'low_harm',
        'upp_harm',
        'max_audible_harm',
        'min_audible_harm',
        'snr',
        'nopad_f0_mean', 'nopad_f0_median', 'nopad_f0_stddev',
        'f0_shift', 'spectral_envelope_bandwidth_in_harmonics', 'spectral_envelope_centered_harmonic',
        'spectral_envelope_f_bandwidth', 'spectral_envelope_f_center',
        'f_carrier', 'f_envelope',
    ]
    kwargs_nervegram_meanrates = {
        'meanrates_params': {'dur': 0.050, 'buffer_start_dur': 0.070, 'buffer_end_dur': 0.010},
        'ANmodel_params': {'num_cfs': 100, 'min_cf':125, 'max_cf':14e3},
    }
    
    parallel_run_dataset_generation(source_regex, dest_filename,
                                    job_idx=job_idx, jobs_per_source_file=jobs_per_source_file,
                                    source_key_signal=source_key_signal,
                                    source_key_signal_fs=source_key_signal_fs,
                                    source_keys_to_copy=source_keys_to_copy,
                                    kwargs_nervegram_meanrates=kwargs_nervegram_meanrates)
    