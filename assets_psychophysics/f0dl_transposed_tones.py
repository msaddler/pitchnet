import sys
import os
import json
import numpy as np
import glob
import argparse
import pdb
import scipy.optimize
import scipy.stats
import f0dl_bernox


def run_f0dl_experiment(json_fn,
                        max_pct_diff=6.0,
                        noise_stdev=1e-12,
                        bin_width=5e-2,
                        mu=0.0,
                        threshold_value=0.707,
                        use_empirical_f0dl_if_possible=False,
                        f0_label_true_key='f0_label:labels_true',
                        f0_label_pred_key='f0_label:labels_pred',
                        f0_label_prob_key='f0_label:probs_out',
                        kwargs_f0_bins={},
                        kwargs_f0_octave={},
                        kwargs_f0_normalization={},
                        kwargs_f0_prior={},
                        f0_ref_min=80.0,
                        f0_ref_max=320.0,
                        f0_ref_n_step=5,
                        metadata_key_list=['f_carrier', 'f_envelope', 'f0']):
    '''
    '''
    # Load JSON file of model predictions into `expt_dict`
    expt_dict = f0dl_bernox.load_f0_expt_dict_from_json(json_fn,
                                                        f0_label_true_key=f0_label_true_key,
                                                        f0_label_pred_key=f0_label_pred_key,
                                                        f0_label_prob_key=f0_label_prob_key,
                                                        metadata_key_list=metadata_key_list)
    # Define list of reference F0s at which to measure discrimination thresholds
    f0_ref_list = np.power(2, np.linspace(np.log2(f0_ref_min), np.log2(f0_ref_max), f0_ref_n_step))
    unique_f_carrier_list = np.unique(expt_dict['f_carrier'])
    N = len(unique_f_carrier_list) * len(f0_ref_list)
    # Add list of nearest f0_ref values for centering prior (defined as the nearest reference F0)
    nearest_f0_ref_bins = [-np.inf]
    for itr0 in range(1, f0_ref_list.shape[0]):
        f0_low = f0_ref_list[itr0 - 1]
        f0_high = f0_ref_list[itr0]
        nearest_f0_ref_bins.append(np.exp(np.mean(np.log([f0_low, f0_high]))))
    nearest_f0_ref_bins.append(np.inf)
    nearest_f0_ref_bins = np.array(nearest_f0_ref_bins)
    f0_ref_indexes = np.digitize(expt_dict['f0'], nearest_f0_ref_bins) - 1
    expt_dict['nearest_f0_ref'] = f0_ref_list[f0_ref_indexes]
    # Add f0 estimates to expt_dict (possibly using prior)
    expt_dict = f0dl_bernox.add_f0_estimates_to_expt_dict(expt_dict,
                                                          f0_label_true_key=f0_label_true_key,
                                                          f0_label_pred_key=f0_label_pred_key,
                                                          kwargs_f0_bins=kwargs_f0_bins,
                                                          kwargs_f0_octave=kwargs_f0_octave,
                                                          kwargs_f0_normalization=kwargs_f0_normalization,
                                                          kwargs_f0_prior=kwargs_f0_prior)
    # Initialize dictionary to hold psychophysical results
    results_dict = {
        'f_carrier': [None]*N,
        'f0_ref': [None]*N,
        'f0dl': [None]*N,
        'psychometric_function': [None]*N,
    }
    itr0 = 0
    for f_carrier in unique_f_carrier_list:
        for f0_ref in f0_ref_list:
            # Simulate f0 discrimination experiment for limited f0 range
            f0_range = [f0_ref * (1.0-max_pct_diff/100.0), f0_ref * (1.0+max_pct_diff/100.0)]
            sub_expt_dict = f0dl_bernox.filter_expt_dict(expt_dict, filter_dict={'f_carrier': f_carrier, 'f0': f0_range})
            sub_expt_dict = f0dl_bernox.add_f0_judgments_to_expt_dict(sub_expt_dict, f0_true_key='f0', f0_pred_key='f0_pred',
                                                                      max_pct_diff=max_pct_diff, noise_stdev=noise_stdev)
            pct_diffs = sub_expt_dict['pairwise_pct_diffs'].reshape([-1])
            pct_diffs = pct_diffs[~np.isnan(pct_diffs)]
            judgments = sub_expt_dict['pairwise_judgments'].reshape([-1])
            judgments = judgments[~np.isnan(judgments)]
            # Fit the empirical psychometric function and compute a threshold
            bins, bin_means = f0dl_bernox.get_empirical_psychometric_function(pct_diffs, judgments, bin_width=bin_width)
            sigma_opt, sigma_opt_cov = f0dl_bernox.fit_normcdf(bins, bin_means, mu=mu)
            f0dl = scipy.stats.norm(mu, sigma_opt).ppf(threshold_value)
            # Replace fit-computed f0dl with the empirical threshold if empirical psychometric function passes threshold
            if use_empirical_f0dl_if_possible:
                above_threshold_bin_indexes = np.logical_and(bins >= 0, bin_means > threshold_value)
                if np.sum(above_threshold_bin_indexes) > 0:
                    f0dl = bins[above_threshold_bin_indexes][0]
            # Organize psychophysical results to return
            psychometric_function_dict = {
                'bins': bins.tolist(),
                'bin_means': bin_means.tolist(),
                'sigma': sigma_opt,
                'sigma_cov': sigma_opt_cov,
                'mu': mu,
                'threshold_value': threshold_value,
            }
            results_dict['f_carrier'][itr0] = f_carrier
            results_dict['f0_ref'][itr0] = f0_ref
            results_dict['f0dl'][itr0] = f0dl
            results_dict['psychometric_function'][itr0] = psychometric_function_dict
            itr0 = itr0 + 1
    # Return dictionary of psychophysical experiment results
    return results_dict


def main(json_eval_fn,
         json_results_dict_fn=None,
         save_results_to_file=False,
         max_pct_diff=6.0,
         noise_stdev=1e-12,
         bin_width=5e-2,
         mu=0.0,
         threshold_value=0.707,
         use_empirical_f0dl_if_possible=False,
         f0_label_true_key='f0_label:labels_true',
         f0_label_pred_key='f0_label:labels_pred',
         f0_label_prob_key='f0_label:probs_out',
         kwargs_f0_bins={},
         kwargs_f0_octave={},
         kwargs_f0_normalization={},
         kwargs_f0_prior={},
         f0_ref_min=80.0,
         f0_ref_max=320.0,
         f0_ref_n_step=5,
         metadata_key_list=['f_carrier', 'f_envelope', 'f0']):
    '''
    '''
    # Run the Oxenham et al. (2004) transposed tones F0DL experiment; results stored in results_dict
    results_dict = run_f0dl_experiment(json_eval_fn,
                                       max_pct_diff=max_pct_diff,
                                       noise_stdev=noise_stdev,
                                       bin_width=bin_width,
                                       mu=mu, threshold_value=threshold_value,
                                       use_empirical_f0dl_if_possible=use_empirical_f0dl_if_possible,
                                       f0_label_true_key=f0_label_true_key,
                                       f0_label_pred_key=f0_label_pred_key,
                                       f0_label_prob_key=f0_label_prob_key,
                                       kwargs_f0_bins=kwargs_f0_bins,
                                       kwargs_f0_octave=kwargs_f0_octave,
                                       kwargs_f0_normalization=kwargs_f0_normalization,
                                       kwargs_f0_prior=kwargs_f0_prior,
                                       f0_ref_min=f0_ref_min,
                                       f0_ref_max=f0_ref_max,
                                       f0_ref_n_step=f0_ref_n_step,
                                       metadata_key_list=metadata_key_list)
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
    parser = argparse.ArgumentParser(description="run Oxenham et al. (2004) transposed tones F0DL experiment")
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
            'f0_label_prob_key': 'f0_label:probs_out',
            'f0_prior_ref_key': 'nearest_f0_ref',
            'octave_range': [
                -parsed_args_dict['prior_range_in_octaves'],
                parsed_args_dict['prior_range_in_octaves']
            ],
        }
    else:
        kwargs_f0_prior = {}
    
    main(json_eval_fn, save_results_to_file=True, kwargs_f0_prior=kwargs_f0_prior)
