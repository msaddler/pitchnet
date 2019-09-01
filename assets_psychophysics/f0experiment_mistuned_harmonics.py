import sys
import os
import json
import numpy as np
import glob
import argparse
import pdb
import f0dl_bernox


def compute_f0_pred_percent_shift(expt_dict):
    f0_true = expt_dict['f0']
    f0_pred = expt_dict['f0_pred']
    expt_dict['f0_pred_pct'] = 100.0 * (f0_pred - f0_true) / f0_true
    return expt_dict


def compute_mistuning_shifts(expt_dict, key_mistuned_harm='mistuned_harm', key_mistuned_pct='mistuned_pct',
                             f0_min=-np.inf, f0_max=np.inf):
    expt_dict = compute_f0_pred_percent_shift(expt_dict)
    unique_harm = np.unique(expt_dict[key_mistuned_harm])
    unique_pct = np.unique(expt_dict[key_mistuned_pct])
    results_dict = {}
    for harm in unique_harm:
        results_dict[harm] = {
            'f0_pred_pct_median': [],
            'f0_pred_pct_mean': [],
            'f0_pred_pct_stddev': [],
            'mistuned_pct': [],
            'mistuned_harm': harm
        }
        for pct in unique_pct:
            filter_dict = {
                key_mistuned_harm: harm,
                key_mistuned_pct: pct,
                'f0': [f0_min, f0_max],
            }
            sub_expt_dict = f0dl_bernox.filter_expt_dict(expt_dict, filter_dict=filter_dict)
            results_dict[harm]['f0_pred_pct_median'].append(np.median(sub_expt_dict['f0_pred_pct']))
            results_dict[harm]['f0_pred_pct_mean'].append(np.mean(sub_expt_dict['f0_pred_pct']))
            results_dict[harm]['f0_pred_pct_stddev'].append(np.std(sub_expt_dict['f0_pred_pct']))
            results_dict[harm]['mistuned_pct'].append(pct)
    return results_dict


def run_f0experiment_mistuned_harmonics(json_fn, filter_key='spectral_envelope_centered_harmonic',
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
    parser = argparse.ArgumentParser(description="run Moore et al. (1985) harmonic mistuning experiment")
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
