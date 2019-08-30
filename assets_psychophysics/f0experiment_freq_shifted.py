import sys
import os
import json
import numpy as np
import glob
import argparse
import pdb
import f0dl_bernox


def compute_f0_shift_curve(expt_dict, filter_key, filter_value, f0_min=80.0, f0_max=1e3):
    '''
    '''
    # Identify trials where filter_key = filter_value and stimulus is in f0 range 
    indexes = expt_dict[filter_key] == filter_value
    indexes = np.logical_and(indexes, np.logical_and(expt_dict['f0'] >= f0_min, expt_dict['f0'] <= f0_max))
    # Compute f0 shifts
    f0_shift = expt_dict['f0_shift'][indexes]
    f0_pred_shift = (expt_dict['f0_pred'][indexes] - expt_dict['f0'][indexes]) / expt_dict['f0'][indexes]
    # For each unique f0 shift, compute the mean, median, stddev predicted f0 shift 
    f0_shift_unique = np.unique(f0_shift)
    f0_pred_shift_mean = np.zeros_like(f0_shift_unique)
    f0_pred_shift_median = np.zeros_like(f0_shift_unique)
    f0_pred_shift_stddev = np.zeros_like(f0_shift_unique)
    for idx, f0_shift_value in enumerate(f0_shift_unique):
        current_value_indexes = f0_shift == f0_shift_value
        f0_pred_shift_mean[idx] = np.mean(f0_pred_shift[current_value_indexes])
        f0_pred_shift_median[idx] = np.median(f0_pred_shift[current_value_indexes])
        f0_pred_shift_stddev[idx] = np.std(f0_pred_shift[current_value_indexes])
    # Return results in dictionary (units converted to percent)
    sub_results_dict = {
        'f0_shift': 100.0 * f0_shift_unique,
        'f0_pred_shift_mean': 100.0 * f0_pred_shift_mean,
        'f0_pred_shift_median': 100.0 * f0_pred_shift_median,
        'f0_pred_shift_stddev': 100.0 * f0_pred_shift_stddev,
    }
    return sub_results_dict


def run_f0experiment_freq_shifted(json_fn, filter_key='spectral_envelope_centered_harmonic',
                                  f0_label_pred_key='f0_label:labels_pred',
                                  f0_label_true_key='f0_label:labels_true',
                                  f0_min=None, f0_max=None):
    '''
    '''
    # Load JSON file of model predictions into `expt_dict`
    metadata_key_list = [
        'f0',
        'f0_shift',
        'spectral_envelope_centered_harmonic',
        'spectral_envelope_bandwidth_in_harmonics',
    ]
    expt_dict = f0dl_bernox.load_f0_expt_dict_from_json(json_fn,
                                                        f0_label_true_key=f0_label_true_key,
                                                        f0_label_pred_key=f0_label_pred_key,
                                                        metadata_key_list=metadata_key_list)
    expt_dict = f0dl_bernox.add_f0_estimates_to_expt_dict(expt_dict,
                                                          f0_label_true_key=f0_label_true_key,
                                                          f0_label_pred_key=f0_label_pred_key)
    # Initialize dictionary to hold psychophysical results
    if f0_min is None: f0_min = np.min(expt_dict['f0'])
    if f0_max is None: f0_max = np.max(expt_dict['f0'])
    results_dict = {filter_key: {}, 'f0_min':f0_min, 'f0_max':f0_max}
    for filter_value in np.unique(expt_dict[filter_key]):
        results_dict[filter_key][int(filter_value)] = compute_f0_shift_curve(expt_dict,
                                                                             filter_key,
                                                                             filter_value,
                                                                             f0_min=f0_min,
                                                                             f0_max=f0_max)
    # Return dictionary of psychophysical experiment results
    return results_dict


def main(json_eval_fn, json_results_dict_fn=None, save_results_to_file=False,
         filter_key='spectral_envelope_centered_harmonic',
         f0_label_pred_key='f0_label:labels_pred',
         f0_label_true_key='f0_label:labels_true',
         f0_min=None, f0_max=None):
    '''
    '''
    # Run the Moore and Moore (2003) freq-shifted complexes experiment; results stored in results_dict
    results_dict = run_f0experiment_freq_shifted(json_eval_fn,
                                                 filter_key=filter_key,
                                                 f0_label_pred_key=f0_label_pred_key,
                                                 f0_label_true_key=f0_label_true_key,
                                                 f0_min=f0_min, f0_max=f0_max)
    results_dict['json_eval_fn'] = json_eval_fn
    # If specified, save results_dict to file
    if save_results_to_file:
        # Check filename for results_dict
        if json_results_dict_fn is None:
            json_results_dict_fn = json_eval_fn.replace('.json', '_results_dict.json')
        assert not json_results_dict_fn == json_eval_fn, "json_results_dict_fn must not overwrite json_eval_fn"
        # Define helper class to JSON serialize the results_dict
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray): return obj.tolist()
                if isinstance(obj, np.int64): return int(obj)  
                return json.JSONEncoder.default(self, obj)
        # Write results_dict to json_results_dict_fn
        with open(json_results_dict_fn, 'w') as f: json.dump(results_dict, f, cls=NumpyEncoder)
        print('[END] wrote results_dict to {}'.format(json_results_dict_fn))
    return results_dict


if __name__ == "__main__":
    '''
    '''
    parser = argparse.ArgumentParser(description="run Moore and Moore (2003) freq-shifted complexes experiment")
    parser.add_argument('-r', '--regex_json_eval_fn', type=str, default=None,
                        help='regex that globs list of json_eval_fn to process')
    parser.add_argument('-j', '--job_idx', type=int, default=None,
                        help='job index used to select json_eval_fn from list')
    parsed_args_dict = vars(parser.parse_args())
    assert parsed_args_dict['regex_json_eval_fn'] is not None, "regex_json_eval_fn is a required argument"
    assert parsed_args_dict['job_idx'] is not None, "job_idx is a required argument"
    list_json_eval_fn = sorted(glob.glob(parsed_args_dict['regex_json_eval_fn']))
    json_eval_fn = list_json_eval_fn[parsed_args_dict['job_idx']]
    print('Processing file {} of {}'.format(parsed_args_dict['job_idx'], len(list_json_eval_fn)))
    print('Processing file: {}'.format(json_eval_fn))
    main(json_eval_fn, save_results_to_file=True)
