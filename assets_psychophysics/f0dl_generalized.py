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

sys.path.append('/om2/user/msaddler/python-packages/msutil')
import util_misc


def run_f0dl_experiment(json_fn,
                        key_condition,
                        max_pct_diff=100/6,
                        noise_stdev=1e-12,
                        bin_width=5e-2,
                        mu=0.0,
                        threshold_value=0.707,
                        use_empirical_f0dl_if_possible=False,
                        f0_label_true_key='f0_label_coarse:labels_true',
                        f0_label_pred_key='f0_label_coarse:labels_pred',
                        f0_label_prob_key='f0_label_coarse:probs_out',
                        f0_true_key='f0',
                        f0_pred_key='f0_pred',
                        kwargs_f0_bins={'f0_min':80., 'f0_max':1e3, 'binwidth_in_octaves':1/48},
                        kwargs_f0_octave={},
                        kwargs_f0_normalization={},
                        kwargs_f0_prior={},
                        verbose=True):
    '''
    '''
    # Load JSON file of model predictions into `expt_dict`
    metadata_key_list = [key_condition, f0_true_key]
    expt_dict = f0dl_bernox.load_f0_expt_dict_from_json(json_fn,
                                                        f0_label_true_key=f0_label_true_key,
                                                        f0_label_pred_key=f0_label_pred_key,
                                                        f0_label_prob_key=f0_label_prob_key,
                                                        metadata_key_list=metadata_key_list)
    expt_dict = f0dl_bernox.add_f0_estimates_to_expt_dict(expt_dict,
                                                          f0_label_true_key=f0_label_true_key,
                                                          f0_label_pred_key=f0_label_pred_key,
                                                          kwargs_f0_bins=kwargs_f0_bins,
                                                          kwargs_f0_octave=kwargs_f0_octave,
                                                          kwargs_f0_normalization=kwargs_f0_normalization,
                                                          kwargs_f0_prior=kwargs_f0_prior)
    condition_list = np.unique(expt_dict[key_condition])
    N = len(condition_list)
    if verbose:
        print('key_condition: {} ({} conditions)'.format(key_condition,  N))
    # Initialize dictionary to hold psychophysical results
    results_dict = {
        key_condition: [None]*N,
        'f0dl': [None]*N,
        'psychometric_function': [None]*N,
    }
    for itr0, condition in enumerate(condition_list):
        if verbose:
            print('... {}={} (condition {} of {})'.format(key_condition,  condition, itr0+1, N))
        # Simulate f0 discrimination experiment for each condition
        sub_expt_dict = f0dl_bernox.filter_expt_dict(expt_dict, filter_dict={key_condition: condition})
        sub_expt_dict = f0dl_bernox.add_f0_judgments_to_expt_dict(sub_expt_dict,
                                                                  f0_true_key=f0_true_key,
                                                                  f0_pred_key=f0_pred_key,
                                                                  max_pct_diff=max_pct_diff,
                                                                  noise_stdev=noise_stdev)
        pct_diffs = sub_expt_dict['pairwise_pct_diffs'].reshape([-1])
        pct_diffs = pct_diffs[~np.isnan(pct_diffs)]
        judgments = sub_expt_dict['pairwise_judgments'].reshape([-1])
        judgments = judgments[~np.isnan(judgments)]
        # Fit the empirical psychometric function and compute a threshold
        bins, bin_means = f0dl_bernox.get_empirical_psychometric_function(pct_diffs,
                                                                          judgments,
                                                                          bin_width=bin_width)
        sigma_opt, sigma_opt_cov = f0dl_bernox.fit_normcdf(bins, bin_means, mu=mu)
        f0dl = scipy.stats.norm(mu, sigma_opt).ppf(threshold_value)
        # Replace computed f0dl with empirical f0dl if empirical psychometric function passes threshold
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
        results_dict[key_condition][itr0] = condition
        results_dict['f0dl'][itr0] = f0dl
        results_dict['psychometric_function'][itr0] = psychometric_function_dict
    
    # Return dictionary of psychophysical experiment results
    return results_dict


def main(json_eval_fn,
         json_results_dict_fn=None,
         save_results_to_file=False,
         key_condition=None,
         kwargs_f0_prior={},
         **kwargs_run_f0dl_experiment):
    '''
    '''
    # Run F0 discrimination threshold experiment; results stored in results_dict
    results_dict = run_f0dl_experiment(json_eval_fn,
                                       key_condition,
                                       kwargs_f0_prior=kwargs_f0_prior,
                                       **kwargs_run_f0dl_experiment)
    results_dict['json_eval_fn'] = json_eval_fn
    results_dict['kwargs_f0_prior'] = kwargs_f0_prior
    # If specified, save results_dict to file
    if save_results_to_file:
        # Check filename for results_dict
        if json_results_dict_fn is None:
            json_results_dict_fn = json_eval_fn.replace('.json', '_results_dict.json')
        assert not json_results_dict_fn == json_eval_fn, "json_results_dict_fn must not overwrite json_eval_fn"
        # Write results_dict to json_results_dict_fn
        with open(json_results_dict_fn, 'w') as f:
            json.dump(results_dict, f, cls=util_misc.NumpyEncoder)
        print('[END] wrote results_dict to {}'.format(json_results_dict_fn))
    return results_dict


if __name__ == "__main__":
    '''
    '''
    parser = argparse.ArgumentParser(description="run generalized F0DL experiment")
    parser.add_argument('-r', '--regex_json_eval_fn', type=str, default=None,
                        help='regex that globs list of json_eval_fn to process')
    parser.add_argument('-j', '--job_idx', type=int, default=None,
                        help='job index used to select json_eval_fn from list')
    parser.add_argument('-k', '--key_condition', type=str, default=None,
                        help='key in json_eval_fn that specifies experiment condition')
    parser.add_argument('-p', '--prior_range_in_octaves', type=float, default=0,
                        help='sets octave_range in `kwargs_f0_prior`: [#, #]')
    parsed_args_dict = vars(parser.parse_args())
    assert parsed_args_dict['regex_json_eval_fn'] is not None, "regex_json_eval_fn is a required argument"
    assert parsed_args_dict['job_idx'] is not None, "job_idx is a required argument"
    assert parsed_args_dict['key_condition'] is not None, "key_condition is a required argument"
    list_json_eval_fn = sorted(glob.glob(parsed_args_dict['regex_json_eval_fn']))
    json_eval_fn = list_json_eval_fn[parsed_args_dict['job_idx']]
    key_condition = parsed_args_dict['key_condition']
    print('Processing file {} of {}'.format(parsed_args_dict['job_idx'], len(list_json_eval_fn)))
    print('Processing file: {}'.format(json_eval_fn))
    
    if parsed_args_dict['prior_range_in_octaves'] > 0:
        kwargs_f0_prior = {
            'f0_label_prob_key': 'f0_label_coarse:probs_out',
            'f0_prior_ref_key': 'f0', # Note: using true F0 may slightly bias up/down judgments
            'octave_range': [
                -parsed_args_dict['prior_range_in_octaves'],
                parsed_args_dict['prior_range_in_octaves']
            ],
        }
    else:
        kwargs_f0_prior = {}
    
    main(json_eval_fn,
         json_results_dict_fn=None,
         save_results_to_file=True,
         key_condition=key_condition,
         kwargs_f0_prior=kwargs_f0_prior)
