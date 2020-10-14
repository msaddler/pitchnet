import sys
import os
import json
import numpy as np
import glob
import argparse
import pdb
import scipy.optimize
import scipy.stats

import itertools
import functools
import multiprocessing

import f0dl_bernox

sys.path.append('/om2/user/msaddler/pitchnet/assets_datasets/')
import stimuli_f0_labels


def add_f0_judgments_to_expt_dict(expt_dict,
                                  f0_true_key='f0',
                                  f0_pred_key='f0_pred',
                                  max_pct_diff=100/6,
                                  noise_stdev=1e-12):
    '''
    Function simulates f0 discrimination experiment given a list of true and predicted f0s.
    
    Args
    ----
    expt_dict (dict): f0 experiment data dict (`pairwise_pct_diffs` and `pairwise_judgments` will be added)
    f0_true_key (str): key for f0_true values in expt_dict
    f0_pred_key (str): key for f0_pred values in expt_dict
    max_pct_diff (float): pairs of f0 predictions are only compared if their pct_diff <= max_pct_diff
    noise_stdev (float): standard deviation of decision-stage noise
    
    Returns
    -------
    expt_dict (dict): includes large `pairwise_pct_diffs` and `pairwise_judgments` matrices
    '''
    sort_idx = np.argsort(expt_dict[f0_true_key])
    f0_true = expt_dict[f0_true_key][sort_idx]
    f0_pred = expt_dict[f0_pred_key][sort_idx]
    # Initialize f0 percent difference and psychophysical judgment arrays (fill with NaN)
    pairwise_pct_diffs = np.full([f0_true.shape[0], f0_true.shape[0]], np.nan, dtype=np.float32)
    pairwise_judgments = np.full([f0_true.shape[0], f0_true.shape[0]], np.nan, dtype=np.float32)
    # Iterate over all true f0 values
    for idx_ref in range(f0_true.shape[0]):
        f0_ref = f0_true[idx_ref] # Each f0_true will be used as the reference once
        f0_pred_ref = f0_pred[idx_ref] # Predicted f0 for the current f0_true
        # Compute vector of pct_diffs (compare f0_ref against all of f0_true)
        pct_diffs = 100. * (f0_true - f0_ref) / f0_ref
        # Find pct_diffs within the range specified by max_pct_diff
        comparable_idxs = np.logical_and(pct_diffs >= -max_pct_diff, pct_diffs <= max_pct_diff)
        pairwise_pct_diffs[idx_ref, comparable_idxs] = pct_diffs[comparable_idxs]
        # Compute the percent differences between the predictions
        pred_pct_diffs = 100. * (f0_pred[comparable_idxs] - f0_pred_ref) / f0_pred_ref
        pred_decision_noise = noise_stdev * np.random.randn(pred_pct_diffs.shape[0])
        # Judgment is 1. if model predicts f0_pred_ref > f0_pred
        # Judgment is 0. if model predicts f0_pred_ref < f0_pred
        # Decision stage Gaussian noise is used to break ties
        tmp_judgments = np.array(pred_pct_diffs > pred_decision_noise, dtype=np.float32)
        pairwise_judgments[idx_ref, comparable_idxs] = tmp_judgments
    # Store the (largely NaN) pairwise_pct_diffs and pairwise_judgments matrices in expt_dict
    expt_dict['pairwise_pct_diffs'] = pairwise_pct_diffs
    expt_dict['pairwise_judgments'] = pairwise_judgments
    return expt_dict


def parallel_run_f0dl_experiment(par_idx,
                                 expt_dict,
                                 unique_phase_mode_list,
                                 unique_low_harm_list,
                                 max_pct_diff=100/6,
                                 noise_stdev=1e-12,
                                 bin_width=5e-2,
                                 mu=0.0,
                                 threshold_value=0.707,
                                 use_empirical_f0dl_if_possible=False):
    '''
    This function runs the f0 discrimination threshold experiment using a subset of the trials in
    `expt_dict` and is designed to be parallelized over phase and lowest harmonic number conditions.
    
    Args
    ----
    par_idx (int): process index
    expt_dict (dict): dictionary with true / predicted f0 values and metadata stored as np arrays
    unique_phase_mode_list (np array): list of unique phase modes (i.e. [0, 1] for ['sine', 'rand'])
    unique_low_harm_list (np array): list of unique lowest harmonic numbers
    max_pct_diff (float): pairs of f0 predictions are only compared if their pct_diff <= max_pct_diff
    noise_stdev (float): standard deviation of decision-stage noise
    bin_width (float): width of bin used to digitize pct_diffs
    mu (float): fixed mean of fitted normcdf
    threshold_value (float): value of the fitted normcdf used to compute f0 difference limen
    use_empirical_f0dl_if_possible (bool): if True, empirical f0dl will attempt to overwrite one computed from fit
    
    Returns
    -------
    par_idx (int): process index
    sub_results_dict (dict): contains psychophysical results for a single phase and lowest harmonic number
    '''
    # Generate master list of experimental conditions and select one using `par_idx`
    (ph, lh) = list(itertools.product(unique_phase_mode_list, unique_low_harm_list))[par_idx]
    # Measure f0 discrimination psychometric function for single condition
    sub_expt_dict = f0dl_bernox.filter_expt_dict(expt_dict, filter_dict={'phase_mode':ph, 'low_harm':lh})
    sub_expt_dict = add_f0_judgments_to_expt_dict(sub_expt_dict, f0_true_key='f0', f0_pred_key='f0_pred',
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
    sub_results_dict = {
        'phase_mode': ph,
        'low_harm': lh,
        'f0dl': f0dl,
        'psychometric_function': psychometric_function_dict
    }
    return par_idx, sub_results_dict


def run_f0dl_experiment(json_fn,
                        max_pct_diff=100/6,
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
                        f0_min=-np.inf,
                        f0_max=np.inf,
                        metadata_key_list=['low_harm', 'phase_mode', 'f0'],
                        max_processes=60):
    '''
    Main routine for simulating f0 discrimination experiment from Bernstein & Oxenham (2005, JASA).
    Function computes f0 discrimination thresholds as a function of lowest harmonic number and
    phase mode.
    
    Args
    ----
    json_fn (str): json filename to load
    max_pct_diff (float): pairs of f0 predictions are only compared if their pct_diff <= max_pct_diff
    noise_stdev (float): standard deviation of decision-stage noise
    bin_width (float): width of bin used to digitize pct_diffs
    mu (float): fixed mean of fitted normcdf
    threshold_value (float): value of the fitted normcdf used to compute f0 difference limen
    use_empirical_f0dl_if_possible (bool): if True, empirical f0dl will attempt to overwrite one computed from fit
    f0_label_true_key (str): key for f0_label_true in the json file
    f0_label_pred_key (str): key for f0_label_pred in the json file
    f0_label_prob_key (str): key for f0_label_pred probabilities in the json file
    kwargs_f0_bins (dict): kwargs for computing f0 bins (lower bound used as estimate)
    kwargs_f0_octave (dict): kwargs for converting f0s from Hz to octaves
    kwargs_f0_normalization (dict): kwargs for normalizing f0s
    kwargs_f0_prior (dict): kwargs for using a prior to compute f0_pred
    f0_min (float): use this argument to limit the f0 range used to compute thresholds (Hz)
    f0_max (float): use this argument to limit the f0 range used to compute thresholds (Hz)
    metadata_key_list (list): metadata keys in json file to use for experiment (see `load_f0_expt_dict_from_json()`)
    max_processes (int): use this argument to cap the number of parallel processes
    
    Returns
    -------
    results_dict (dict): contains lists of thresholds and psychometric functions for all conditions
    '''
    # Load JSON file of model predictions into `expt_dict`
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
    if 'base_f0' in expt_dict.keys():
        expt_dict = f0dl_bernox.filter_expt_dict(expt_dict, filter_dict={'base_f0':[f0_min, f0_max]})
    else:
        expt_dict = f0dl_bernox.filter_expt_dict(expt_dict, filter_dict={'f0':[f0_min, f0_max]})
    unique_phase_mode_list = np.unique(expt_dict['phase_mode'])
    unique_low_harm_list = np.unique(expt_dict['low_harm'])
    N = len(unique_phase_mode_list) * len(unique_low_harm_list)
    # Initialize dictionary to hold psychophysical results
    results_dict = {
        'phase_mode': [None]*N,
        'low_harm': [None]*N,
        'f0dl': [None]*N,
        'psychometric_function': [None]*N,
    }
    # Define a pickle-able wrapper for `parallel_run_f0dl_experiment` using functools
    parallel_run_wrapper = functools.partial(parallel_run_f0dl_experiment,
                                             expt_dict=expt_dict,
                                             unique_phase_mode_list=unique_phase_mode_list,
                                             unique_low_harm_list=unique_low_harm_list,
                                             max_pct_diff=max_pct_diff,
                                             noise_stdev=noise_stdev,
                                             bin_width=bin_width,
                                             mu=mu,
                                             threshold_value=threshold_value,
                                             use_empirical_f0dl_if_possible=use_empirical_f0dl_if_possible)
    # Call the wrapper in parallel processes using multiprocessing.Pool
    with multiprocessing.Pool(processes=np.min([N, max_processes])) as pool:    
        parallel_results = pool.map(parallel_run_wrapper, range(0, N))
        for (par_idx, sub_results_dict) in parallel_results:
            for key in sub_results_dict.keys():
                results_dict[key][par_idx] = sub_results_dict[key]
    # Return dictionary of psychophysical experiment results
    return results_dict


def main(json_eval_fn,
         json_results_dict_fn=None,
         save_results_to_file=False,
         max_pct_diff=100/6,
         bin_width=5e-2,
         use_empirical_f0dl_if_possible=False,
         f0_label_true_key='f0_label:labels_true',
         f0_label_pred_key='f0_label:labels_pred',
         f0_label_prob_key='f0_label:probs_out',
         kwargs_f0_bins={},
         kwargs_f0_octave={},
         kwargs_f0_normalization={},
         kwargs_f0_prior={},
         f0_min=-np.inf,
         f0_max=np.inf,
         max_processes=60):
    '''
    '''
    # Run the Bernstein and Oxenham (2005) F0DL experiment; results stored in results_dict
    metadata_key_list=['low_harm', 'phase_mode', 'f0', 'base_f0']
    results_dict = run_f0dl_experiment(json_eval_fn,
                                       max_pct_diff=max_pct_diff,
                                       bin_width=bin_width,
                                       use_empirical_f0dl_if_possible=use_empirical_f0dl_if_possible,
                                       metadata_key_list=metadata_key_list,
                                       f0_label_pred_key=f0_label_pred_key,
                                       f0_label_true_key=f0_label_true_key,
                                       f0_label_prob_key=f0_label_prob_key,
                                       kwargs_f0_bins=kwargs_f0_bins,
                                       kwargs_f0_octave=kwargs_f0_octave,
                                       kwargs_f0_normalization=kwargs_f0_normalization,
                                       kwargs_f0_prior=kwargs_f0_prior,
                                       f0_min=f0_min,
                                       f0_max=f0_max,
                                       max_processes=max_processes)
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
        print('[END] wrote results_dict to {}'.format(json_results_dict_fn), flush=True)
    return results_dict


if __name__ == "__main__":
    '''
    '''
    parser = argparse.ArgumentParser(description="run Bernstein and Oxenham (2005) F0DL experiment")
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
            'f0_prior_ref_key': 'f0'
            'octave_range': [
                -parsed_args_dict['prior_range_in_octaves'],
                parsed_args_dict['prior_range_in_octaves']
            ],
        }
    else:
        kwargs_f0_prior = {}
    
    main(json_eval_fn, save_results_to_file=True, kwargs_f0_prior=kwargs_f0_prior)
