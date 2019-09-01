import sys
import os
import json
import numpy as np
import glob
import argparse
import pdb
import f0dl_bernox


def compute_mistuning_shifts(expt_dict, key_mistuned_harm='mistuned_harm', key_mistuned_pct='mistuned_pct',
                             f0_ref=100.0, f0_ref_width=0.04, use_relative_shift=True):
    '''
    '''
    # Filter expt_dict to f0 range specified by f0_ref and f0_ref_width
    f0_min = f0_ref * (1.0 - f0_ref_width)
    f0_max = f0_ref * (1.0 + f0_ref_width)
    sub_expt_dict = f0dl_bernox.filter_expt_dict(expt_dict, filter_dict={'f0': [f0_min, f0_max]})
    # Compute f0_pred_pct (shift in f0_pred relative to f0_true)
    f0_true = sub_expt_dict['f0']
    f0_pred = sub_expt_dict['f0_pred']
    if use_relative_shift:
        # If use_relative_shift, adjust all f0_pred using the mean offset from stimuli with 0% mistuning
        harmonic_idx = sub_expt_dict[key_mistuned_pct] == 0.0
        harmonic_f0_shift = f0_pred[harmonic_idx] - f0_true[harmonic_idx]
        f0_pred = f0_pred - np.mean(harmonic_f0_shift)
    sub_expt_dict['f0_pred_pct'] = 100.0 * (f0_pred - f0_true) / f0_true
    # For each individual harmonic and each percent mistuning, compute mean, median, stddev f0_pred_pct
    unique_harm = np.unique(sub_expt_dict[key_mistuned_harm])
    unique_pct = np.unique(sub_expt_dict[key_mistuned_pct])
    sub_results_dict = {key_mistuned_harm:{}, 'f0_min':f0_min, 'f0_max':f0_max}
    for harm in unique_harm:
        sub_results_dict[key_mistuned_harm][int(harm)] = {
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
            harm_pct_dict = f0dl_bernox.filter_expt_dict(sub_expt_dict, filter_dict=filter_dict)
            sub_results_dict[key_mistuned_harm][harm]['f0_pred_pct_median'].append(np.median(harm_pct_dict['f0_pred_pct']))
            sub_results_dict[key_mistuned_harm][harm]['f0_pred_pct_mean'].append(np.mean(harm_pct_dict['f0_pred_pct']))
            sub_results_dict[key_mistuned_harm][harm]['f0_pred_pct_stddev'].append(np.std(harm_pct_dict['f0_pred_pct']))
            sub_results_dict[key_mistuned_harm][harm]['mistuned_pct'].append(pct)
    return sub_results_dict


def run_f0experiment_mistuned_harmonics(json_fn, use_relative_shift=True,
                                        f0_label_pred_key='f0_label:labels_pred',
                                        f0_label_true_key='f0_label:labels_true',
                                        f0_ref_list=[100.0, 200.0, 400.0], f0_ref_width=0.04):
    '''
    '''
    # Load JSON file of model predictions into `expt_dict`
    metadata_key_list = ['f0', 'mistuned_harm', 'mistuned_pct']
    expt_dict = f0dl_bernox.load_f0_expt_dict_from_json(json_fn,
                                                        f0_label_true_key=f0_label_true_key,
                                                        f0_label_pred_key=f0_label_pred_key,
                                                        metadata_key_list=metadata_key_list)
    expt_dict = f0dl_bernox.add_f0_estimates_to_expt_dict(expt_dict,
                                                          f0_label_true_key=f0_label_true_key,
                                                          f0_label_pred_key=f0_label_pred_key)
    # Initialize dictionary to hold psychophysical results
    results_dict = {'f0_ref':{}, 'f0_ref_list':f0_ref_list, 'f0_ref_width':f0_ref_width}
    for f0_ref in f0_ref_list:
        results_dict['f0_ref'][float(f0_ref)] = compute_mistuning_shifts(expt_dict,
                                                                         key_mistuned_harm='mistuned_harm',
                                                                         key_mistuned_pct='mistuned_pct',
                                                                         f0_ref=f0_ref,
                                                                         f0_ref_width=f0_ref_width,
                                                                         use_relative_shift=use_relative_shift)
    # Return dictionary of psychophysical experiment results
    return results_dict


def main(json_eval_fn, json_results_dict_fn=None, save_results_to_file=False,
         use_relative_shift=True,
         f0_label_pred_key='f0_label:labels_pred',
         f0_label_true_key='f0_label:labels_true',
         f0_ref_list=[100.0, 200.0, 400.0], f0_ref_width=0.04):
    '''
    '''
    # Run the Moore et al. (1985) harmonic mistuning experiment; results stored in results_dict
    results_dict = run_f0experiment_mistuned_harmonics(json_eval_fn,
                                                       use_relative_shift=use_relative_shift,
                                                       f0_label_pred_key=f0_label_pred_key,
                                                       f0_label_true_key=f0_label_true_key,
                                                       f0_ref_list=f0_ref_list,
                                                       f0_ref_width=f0_ref_width)
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
