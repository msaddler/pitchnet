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
    
    source_key_signal = 'tone_in_noise'
    source_key_signal_fs = 'config_tone/fs'
    source_keys_to_copy = [
        'f0',
        'f0_label',
        'phase_mode',
        'low_harm',
        'upp_harm',
        'snr',
    ]
    kwargs_nervegram_meanrates = {
        'meanrates_params': {'dur': 0.05}
    }
    
    parallel_run_dataset_generation(source_regex, dest_filename,
                                    job_idx=job_idx, jobs_per_source_file=jobs_per_source_file,
                                    source_key_signal=source_key_signal,
                                    source_key_signal_fs=source_key_signal_fs,
                                    source_keys_to_copy=source_keys_to_copy,
                                    kwargs_nervegram_meanrates=kwargs_nervegram_meanrates)