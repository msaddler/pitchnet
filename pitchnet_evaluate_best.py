import sys
import os
import glob
import json
import numpy as np
import argparse

sys.path.append('/code_location/multi_gpu')
from run_train_or_eval import run_train_or_eval


def get_best_checkpoint_number(validation_metrics_fn, metric_key='f0_label:accuracy', maximize=True,
                               checkpoint_number_key='step'):
    '''
    '''
    with open(validation_metrics_fn) as f: validation_metrics_dict = json.load(f)
    checkpoint_numbers = validation_metrics_dict[checkpoint_number_key]
    metric_values = validation_metrics_dict[metric_key]
    if maximize: bci = np.argmax(metric_values)
    else: bci = np.argmin(metric_values)
    print('Selecting checkpoint {} ({}={})'.format(checkpoint_numbers[bci], metric_key, metric_values[bci]))
    return checkpoint_numbers[bci]


def get_feature_parsing_dict_from_tfrecords(eval_regex):
    '''
    TODO: fix / automate this.
    '''
    if ('bernox2005' in eval_regex) and ('cf100' in eval_regex):
        feature_parsing_dict = {
            "f0": { "dtype": "tf.float32", "shape": [] },
            "f0_label": { "dtype": "tf.int64", "shape": [] },
            "f0_lognormal": { "dtype": "tf.float32", "shape": [] },
            "meanrates": { "dtype": "tf.float32", "shape": [100, 500] },
            "pin_dBSPL": { "dtype": "tf.float32", "shape": [] },
            "low_harm": { "dtype": "tf.int64", "shape": [] },
            "upp_harm": { "dtype": "tf.int64", "shape": [] },
            "phase_mode": { "dtype": "tf.int64", "shape": [] }
        }
    else:
        raise ValueError("Unrecognized `eval_regex`: {}".format(eval_regex))
    return feature_parsing_dict


def create_temporary_config(output_directory, eval_regex,
                            config_filename='config.json', 
                            temporary_config_filename='EVAL_config.json'):
    '''
    '''
    with open(os.path.join(output_directory, config_filename)) as f: CONFIG = json.load(f)
    feature_parsing_dict = get_feature_parsing_dict_from_tfrecords(eval_regex)
    CONFIG["ITERATOR_PARAMS"]["feature_parsing_dict"] = feature_parsing_dict
    out_filename = os.path.join(output_directory, temporary_config_filename)
    print('Writing temporary eval config file: {}'.format(out_filename))
    with open(out_filename, 'w') as f: json.dump(CONFIG, f, sort_keys=True, indent=4)
    return out_filename


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run evaluation using best checkpoint in output directory")
    parser.add_argument('-o', '--outputdir', type=str, default=None,
                        help='output directory for model')
    parser.add_argument('-c', '--configfile', type=str, default=None,
                        help='location of the config file to use for the model (default is to look in output directory)')
    parser.add_argument('-de', '--tfrecordsregexeval', type=str, default=None,
                        help='regex that globs tfrecords for model evaluation data')
    parser.add_argument('-ebc', '--eval_brain_ckpt', type=int, default=None, 
                        help='if specified, load the specified brain net ckpt number instead of the best one')
    parser.add_argument('-efn', '--eval_output_fn', type=str, default='EVAL.json', 
                        help='JSON filename to store evaluation outputs (eval_only_mode), do not include path')
    parser.add_argument('-vfn', '--validation_metrics_fn', type=str, default='validation_metrics.json', 
                        help='JSON filename where validation metrics are stored, do not include path')
    parser.add_argument('-vk', '--validation_metrics_key', type=str, default='f0_label:accuracy', 
                        help='key in validation_metrics_fn to use when selecting best checkpoint')
    args = parser.parse_args()
    
    assert args.outputdir is not None
    assert args.tfrecordsregexeval is not None
    output_directory = args.outputdir
    eval_only_mode = True
    train_regex = None
    eval_regex = args.tfrecordsregexeval
    
    validation_metrics_fn = os.path.join(output_directory, args.validation_metrics_fn)
    validation_metrics_key = args.validation_metrics_key
    if 'loss' in validation_metrics_key: maximize = False
    else: maximize = True
    
    eval_brain_ckpt = args.eval_brain_ckpt
    if eval_brain_ckpt is None:
        eval_brain_ckpt = get_best_checkpoint_number(validation_metrics_fn, metric_key=validation_metrics_key,
                                                     maximize=maximize, checkpoint_number_key='step')
    eval_output_fn = args.eval_output_fn
    
    config_filename = args.configfile
    if config_filename is None:
        config_filename = create_temporary_config(output_directory, eval_regex,
                                                  config_filename='config.json', 
                                                  temporary_config_filename='EVAL_config.json')
    
    run_train_or_eval(output_directory, train_regex, eval_regex,
                      config_filename=config_filename,
                      eval_only_mode=True,
                      force_overwrite=False, 
                      eval_brain_ckpt=eval_brain_ckpt,
                      eval_wavenet_ckpt=None,
                      eval_output_fn=eval_output_fn,
                      write_audio_files=False)
    