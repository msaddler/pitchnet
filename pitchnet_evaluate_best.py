import sys
import os
import glob
import json
import copy
import numpy as np
import pdb
import argparse

import tensorflow as tf
from google.protobuf.json_format import MessageToJson

sys.path.append('/code_location/multi_gpu')
from run_train_or_eval import run_train_or_eval


def get_best_checkpoint_number(validation_metrics_fn,
                               metric_key='f0_label:accuracy',
                               maximize=True,
                               checkpoint_number_key='step'):
    '''
    '''
    with open(validation_metrics_fn) as f:
        validation_metrics_dict = json.load(f)
    valid_step = validation_metrics_dict[checkpoint_number_key]
    if metric_key not in validation_metrics_dict.keys():
        # If metric_key does not exist in validation_metrics_dict, look for a similarly named key
        for available_key in validation_metrics_dict.keys():
            if all([mkp in available_key for mkp in metric_key.split(':')]):
                print('Did not find key `{}`; using `{}` instead'.format(metric_key, available_key))
                metric_key = available_key
                break
    metric_values = validation_metrics_dict[metric_key]
    ### START: WORK AROUND FOR BUG CAUSED BY PREEMPTING AND RESTARTING TRAINING (valid step is reset)
    checkpoint_numbers = [valid_step[0]]
    for idx, diff in enumerate(np.diff(valid_step)):
        if diff > 0: checkpoint_numbers.append(checkpoint_numbers[-1] + diff)
        else: checkpoint_numbers.append(checkpoint_numbers[-1] + valid_step[idx+1])
    assert len(checkpoint_numbers) == len(valid_step)
    assert len(checkpoint_numbers) == len(metric_values)
    ### END: WORK AROUND FOR BUG CAUSED BY PREEMPTING AND RESTARTING TRAINING (valid step is reset)
    if maximize: bci = np.argmax(metric_values)
    else: bci = np.argmin(metric_values)
    print('Selecting checkpoint {} ({}={})'.format(checkpoint_numbers[bci], metric_key, metric_values[bci]))
    return int(checkpoint_numbers[bci])


def get_feature_parsing_dict_from_tfrecords(eval_regex, bytesList_decoding_dict={}):
    '''
    TODO: fix / automate bytesList_decoding_dict.
    '''
    tfrecords_fn_list = sorted(glob.glob(eval_regex))
    options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    for example in tf.python_io.tf_record_iterator(tfrecords_fn_list[0], options=options):
        jsonMessage = MessageToJson(tf.train.Example.FromString(example))
        jsonDict = json.loads(jsonMessage)
        break
    raw_feature_parsing_dict = jsonDict['features']['feature']
    feature_parsing_dict = {}
    for key in raw_feature_parsing_dict.keys():
        print(key, raw_feature_parsing_dict[key].keys())
        dtype_key = list(raw_feature_parsing_dict[key].keys())[0]
        if dtype_key == 'floatList':
            feature_parsing_dict[key] = { "dtype": "tf.float32", "shape": [] }
        elif dtype_key == 'int64List':
            feature_parsing_dict[key] = { "dtype": "tf.int64", "shape": [] }
        elif dtype_key == 'bytesList':
            if key in bytesList_decoding_dict.keys():
                feature_parsing_dict[key] = bytesList_decoding_dict[key]
            else:
                print("Ignoring tfrecords_key `{}` (not found in bytesList_decoding_dict)".format(key))
    return feature_parsing_dict


def create_temporary_config(output_directory,
                            eval_regex,
                            config_filename='config.json', 
                            temporary_config_filename='EVAL_config.json',
                            bytesList_keys=['signal', 'meanrates', 'nervegram_meanrates', 'nervegram_spike_times']):
    '''
    '''
    with open(os.path.join(output_directory, config_filename)) as f: CONFIG = json.load(f)
    original_feature_parsing_dict = CONFIG["ITERATOR_PARAMS"]["feature_parsing_dict"]
    bytesList_decoding_dict = {}
    for key in set(bytesList_keys).intersection(set(original_feature_parsing_dict.keys())):
        bytesList_decoding_dict[key] = copy.deepcopy(original_feature_parsing_dict[key])
    feature_parsing_dict = get_feature_parsing_dict_from_tfrecords(eval_regex, bytesList_decoding_dict)
    CONFIG["ITERATOR_PARAMS"]["feature_parsing_dict"] = feature_parsing_dict
    assert CONFIG["ITERATOR_PARAMS"]["feature_signal_path"] in feature_parsing_dict.keys()
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
    parser.add_argument('-wpo', '--write_probs_out', type=int, default=0,
                        help='if specified, final activations will be included in output file (eval_only_mode)')
    parser.add_argument('-f', '--force_overwrite', action='store_true', default=False,
                        help='if specified, force overwrite eval_output_fn')
    args = parser.parse_args()
    
    assert args.outputdir is not None
    assert args.tfrecordsregexeval is not None
    output_directory = args.outputdir
    eval_only_mode = True
    train_regex = None
    eval_regex = args.tfrecordsregexeval
    
    eval_output_fn = args.eval_output_fn
    tmp_eval_output_fn = eval_output_fn
    if os.path.basename(tmp_eval_output_fn) == tmp_eval_output_fn:
        tmp_eval_output_fn = os.path.join(output_directory, eval_output_fn)
    if os.path.exists(tmp_eval_output_fn):
        if not args.force_overwrite:
            raise SystemExit("eval_output_fn={} already exists".format(tmp_eval_output_fn))
    
    validation_metrics_fn = os.path.join(output_directory, args.validation_metrics_fn)
    validation_metrics_key = args.validation_metrics_key
    if 'loss' in validation_metrics_key: maximize = False
    else: maximize = True
    
    eval_brain_ckpt = args.eval_brain_ckpt
    if eval_brain_ckpt is None:
        eval_brain_ckpt = get_best_checkpoint_number(validation_metrics_fn,
                                                     metric_key=validation_metrics_key,
                                                     maximize=maximize,
                                                     checkpoint_number_key='step')
    
    config_filename = args.configfile
    if config_filename is None:
        config_filename = create_temporary_config(output_directory,
                                                  eval_regex,
                                                  config_filename='config.json', 
                                                  temporary_config_filename='EVAL_config.json')
    
    run_train_or_eval(output_directory,
                      train_regex,
                      eval_regex,
                      config_filename=config_filename,
                      eval_only_mode=True,
                      force_overwrite=False, 
                      eval_brain_ckpt=eval_brain_ckpt,
                      eval_output_fn=eval_output_fn,
                      write_probs_out=bool(args.write_probs_out))
    