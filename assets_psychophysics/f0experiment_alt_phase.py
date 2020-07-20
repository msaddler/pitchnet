import sys
import os
import json
import numpy as np
import glob
import argparse
import pdb
import f0dl_bernox


def compute_f0_pred_ratio(expt_dict_list, phase_mode=4,
                          f0_bin_centers=[80, 125, 250],
                          f0_bin_width=0.04):
    '''
    Helper function parses evaluation dictionaries from the Shackleton
    and Carlyon (1994) alt-phase stimuli and computes histograms of
    f0_pred / f0_true ratios for different f0 and filter conditions.
    '''
    if not isinstance(expt_dict_list, list):
        expt_dict_list = [expt_dict_list]
    filter_conditions = np.unique(expt_dict_list[0]['filter_fl'])
    f0_pred_ratio_list = []
    f0_condition_list = []
    filter_condition_list = []
    for filt_cond in filter_conditions:
        for f0_center in f0_bin_centers:
            f0_range = [f0_center*(1-f0_bin_width), f0_center*(1+f0_bin_width)]
            f0_pred_ratio_sublist = []
            for expt_dict in expt_dict_list:
                filter_dict={
                    'filter_fl': filt_cond,
                    'f0': f0_range, 
                    'phase_mode': phase_mode
                }
                sub_expt_dict = f0dl_bernox.filter_expt_dict(expt_dict,
                                                             filter_dict=filter_dict)
                sub_expt_dict['f0_pred_ratio'] = sub_expt_dict['f0_pred'] / sub_expt_dict['f0']
                f0_pred_ratio_sublist.extend(sub_expt_dict['f0_pred_ratio'].tolist())
            f0_pred_ratio_list.append(f0_pred_ratio_sublist)
            filter_condition_list.append(filt_cond)
            f0_condition_list.append(f0_center)
    return filter_condition_list, f0_condition_list, f0_pred_ratio_list


def compute_f0_2x_pref(expt_dict, use_log_scale=True):
    '''
    '''
    f0_true = expt_dict['f0']
    f0_true_2x = 2.0 * expt_dict['f0']
    f0_pred = expt_dict['f0_pred']
    if use_log_scale:
        f0_mid_point = np.exp((np.log(f0_true) + np.log(f0_true_2x)) / 2)
    else:
        f0_mid_point = (f0_true + f0_true_2x) / 2
    expt_dict['f0_2x_pref'] = np.zeros_like(expt_dict['f0_pred'])
    expt_dict['f0_2x_pref'][f0_pred > f0_mid_point] = 1
    expt_dict['f0_2x_pref'][f0_pred <= f0_mid_point] = -1
    return expt_dict


def run_f0experiment_alt_phase(json_fn, phase_mode=4, f0_min=None, f0_max=None, f0_nbins=12,
                               f0_label_pred_key='f0_label_coarse:labels_pred',
                               f0_label_true_key='f0_label_coarse:labels_true',
                               f0_label_prob_key='f0_label_coarse:probs_out',
                               kwargs_f0_pred_ratio={
                                   'f0_bin_centers': [80, 125, 250],
                                   'f0_bin_width': 0.10,
                               },
                               kwargs_f0_prior={},
                               use_log_scale=True):
    '''
    '''
    # Load JSON file of model predictions into `expt_dict`
    metadata_key_list = ['f0', 'phase_mode', 'filter_fl', 'filter_fh']
    expt_dict = f0dl_bernox.load_f0_expt_dict_from_json(json_fn,
                                                        f0_label_true_key=f0_label_true_key,
                                                        f0_label_pred_key=f0_label_pred_key,
                                                        f0_label_prob_key=f0_label_prob_key,
                                                        metadata_key_list=metadata_key_list)
    expt_dict = f0dl_bernox.add_f0_estimates_to_expt_dict(expt_dict,
                                                          f0_label_true_key=f0_label_true_key,
                                                          f0_label_pred_key=f0_label_pred_key,
                                                          kwargs_f0_prior=kwargs_f0_prior)
    expt_dict = compute_f0_2x_pref(expt_dict, use_log_scale=use_log_scale)
    # Design f0 bins for digitizing x-axis and pooling f0s
    if f0_min is None: f0_min = np.min(expt_dict['f0'])
    if f0_max is None: f0_max = np.max(expt_dict['f0'])
    bins = np.exp(np.linspace(np.log(f0_min), np.log(f0_max), f0_nbins + 1))
    bin_centers = (bins[:-1] + bins[1:]) / 2
    # Initialize dictionary to hold psychophysical results
    results_dict = {'f0_bin_centers': bin_centers, 'filter_fl_bin_means':{}}
    # Iterate over experimental conditions and compute alt phase f0 preferences
    for filter_fl in np.unique(expt_dict['filter_fl']):
        filter_dict={'filter_fl': filter_fl, 'f0': [f0_min, f0_max], 'phase_mode': phase_mode}
        sub_expt_dict = f0dl_bernox.filter_expt_dict(expt_dict, filter_dict=filter_dict)
        bin_idx = np.digitize(sub_expt_dict['f0'], bins, right=False) - 1
        bin_idx[bin_idx == len(bin_centers)] = len(bin_centers) - 1
        assert np.all(bin_idx < len(bin_centers))
        bin_judgments = np.zeros_like(bin_centers)
        bin_counts = np.zeros_like(bin_centers)
        for itr0, bi in enumerate(bin_idx):
            bin_judgments[bi] += sub_expt_dict['f0_2x_pref'][itr0]
            bin_counts[bi] += 1
        assert np.all(bin_counts > 0)
        bin_means = bin_judgments / bin_counts
        results_dict['filter_fl_bin_means'][filter_fl] = bin_means
    # If kwargs_f0_pred_ratio is specified, include f0 pred ratios for histograms
    if kwargs_f0_pred_ratio:
        filter_condition_list, f0_condition_list, f0_pred_ratio_list = compute_f0_pred_ratio(
            expt_dict, phase_mode=phase_mode, **kwargs_f0_pred_ratio)
        results_dict['f0_pred_ratio_results'] = {
            'filter_condition_list': filter_condition_list,
            'f0_condition_list': f0_condition_list,
            'f0_pred_ratio_list': f0_pred_ratio_list,
            'kwargs_f0_pred_ratio': kwargs_f0_pred_ratio,
        }
    # Return dictionary of psychophysical experiment results
    return results_dict


def main(json_eval_fn, json_results_dict_fn=None, save_results_to_file=False,
         phase_mode=4, f0_min=None, f0_max=None, f0_nbins=12,
         f0_label_pred_key='f0_label_coarse:labels_pred',
         f0_label_true_key='f0_label_coarse:labels_true',
         f0_label_prob_key='f0_label_coarse:probs_out',
         kwargs_f0_prior={},
         use_log_scale=True):
    '''
    '''
    # Run the Oxenham et al. (2004) transposed tones F0DL experiment; results stored in results_dict
    results_dict = run_f0experiment_alt_phase(json_eval_fn, phase_mode=phase_mode,
                                              f0_min=f0_min, f0_max=f0_max, f0_nbins=f0_nbins,
                                              f0_label_pred_key=f0_label_pred_key,
                                              f0_label_true_key=f0_label_true_key,
                                              f0_label_prob_key=f0_label_prob_key,
                                              kwargs_f0_prior=kwargs_f0_prior,
                                              use_log_scale=use_log_scale)
    results_dict['json_eval_fn'] = json_eval_fn
    results_dict['kwargs_f0_prior'] = kwargs_f0_prior
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
    parser = argparse.ArgumentParser(description="run Shackleton and Carlyon (1994) alt phase experiment")
    parser.add_argument('-r', '--regex_json_eval_fn', type=str, default=None,
                        help='regex that globs list of json_eval_fn to process')
    parser.add_argument('-j', '--job_idx', type=int, default=None,
                        help='job index used to select json_eval_fn from list')
    parser.add_argument('-p', '--prior_range_in_octaves', type=float, default=0,
                        help='sets octave_range in `kwargs_f0_prior`: [#, #]')
    parsed_args_dict = vars(parser.parse_args())
    assert parsed_args_dict['regex_json_eval_fn'] is not None, "regex_json_eval_fn is a required argument"
    assert parsed_args_dict['job_idx'] is not None, "job_idx is a required argument"
    list_json_eval_fn = sorted(glob.glob(parsed_args_dict['regex_json_eval_fn']))
    json_eval_fn = list_json_eval_fn[parsed_args_dict['job_idx']]
    print('Processing file {} of {}'.format(parsed_args_dict['job_idx'], len(list_json_eval_fn)))
    print('Processing file: {}'.format(json_eval_fn))
    
    if parsed_args_dict['prior_range_in_octaves'] > 0:
        kwargs_f0_prior = {
            'f0_label_prob_key': 'f0_label_coarse:probs_out',
            'f0_prior_ref_key': 'f0',
            'octave_range': [
                -parsed_args_dict['prior_range_in_octaves'] - 1,
                parsed_args_dict['prior_range_in_octaves'] + 1
            ],
        }
    else:
        kwargs_f0_prior = {}
    
    main(json_eval_fn, save_results_to_file=True, kwargs_f0_prior=kwargs_f0_prior)
