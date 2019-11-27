import sys
import os
import glob
import json
import numpy as np
import pdb
import argparse

import arch_generate_output_directory

sys.path.append('/code_location/multi_gpu/')
import run_train_or_eval


def train_random_arch(output_dir_pattern, source_config_fn, job_idx,
                      tfrecords_regex_train=None, tfrecords_regex_eval=None):
    '''
    '''
    output_dir = output_dir_pattern.format(job_idx)
    if os.path.exists(output_dir):
        assert os.path.exists(os.path.join(output_dir, 'config.json'))
        config_filename = None
    else:
        assert source_config_fn is not None
        config_filename = arch_generate_output_directory.generate_output_directory(
            output_dir, source_config_fn=source_config_fn)
    
    assert tfrecords_regex_train is not None
    assert tfrecords_regex_eval is not None
    print('OUTPUT_DIR:', output_dir)
    print('DATA_TRAIN:', tfrecords_regex_train)
    print('DATA_EVAL:', tfrecords_regex_eval)
    print('CONFIG:', config_filename)
    run_train_or_eval.run_train_or_eval(output_dir,
                                        tfrecords_regex_train,
                                        tfrecords_regex_eval,
                                        config_filename=config_filename,
                                        eval_only_mode=False, 
                                        force_overwrite=False)


def main():
    '''
    '''
    parser = argparse.ArgumentParser(description="launch random architecture search training routine")
    parser.add_argument('-o', '--output_dir_pattern', type=str, default=None,
                        help='formattable string for naming new output directory')
    parser.add_argument('-c', '--source_config_fn', type=str, default=None,
                        help='filename for base config file migrated to each new output directory')
    parser.add_argument('-j', '--job_idx', type=int, default=None,
                        help='job index used to name current output directory')
    parser.add_argument('-dt', '--tfrecords_regex_train', type=str, default=None, 
                        help='regex that globs tfrecords for model training data')
    parser.add_argument('-de', '--tfrecords_regex_eval', type=str, default=None,
                        help='regex that globs tfrecords for model evaluation data')
    train_random_arch(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
