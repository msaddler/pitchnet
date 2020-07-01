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

sys.path.append('/om2/user/msaddler/pitchnet/assets_datasets/')
import stimuli_f0_labels


def load_f0_expt_dict_from_json(json_fn,
                                f0_label_true_key='f0_label:labels_true',
                                f0_label_pred_key='f0_label:labels_pred',
                                f0_label_prob_key='f0_label:probs_out',
                                metadata_key_list=['low_harm', 'phase_mode', 'f0']):
    '''
    Function loads a json file and returns a dictionary with np array keys.
    
    Args
    ----
    json_fn (str): json filename to load
    f0_label_true_key (str): key for f0_label_true in the json file
    f0_label_pred_key (str): key for f0_label_pred in the json file
    f0_label_prob_key (str): key for f0_label_pred probabilities in the json file
    metadata_key_list (list): keys in the json file to copy to returned dictionary
    
    Returns
    -------
    expt_dict (dict): dictionary with true/pred labels and metadata stored as np arrays
    '''
    # Load the entire json file as a dictionary
    with open(json_fn, 'r') as f: json_dict = json.load(f)
    # Return dictionary (expt_dict) with only specified fields
    expt_dict = {}
    assert f0_label_true_key in json_dict.keys(), "f0_label_true_key not found in json file"
    assert f0_label_pred_key in json_dict.keys(), "f0_label_pred_key not found in json file"
    sort_idx = np.argsort(json_dict['f0'])
    expt_dict = {
        f0_label_true_key: np.array(json_dict[f0_label_true_key])[sort_idx],
        f0_label_pred_key: np.array(json_dict[f0_label_pred_key])[sort_idx],
    }
    # Include predicted label probabilities if f0_label_prob_key is in json_dict
    if f0_label_prob_key in json_dict.keys():
        f0_label_prob_value = json_dict[f0_label_prob_key]
        if isinstance(f0_label_prob_value, str):
            f0_label_prob_key_fn = os.path.join(os.path.dirname(json_fn), f0_label_prob_value)
            print('Loading f0_label_prob from {}'.format(f0_label_prob_key_fn), flush=True)
            expt_dict[f0_label_prob_key] = np.load(f0_label_prob_key_fn)
            print('Loaded f0_label_prob from {}'.format(f0_label_prob_key_fn), flush=True)
        expt_dict[f0_label_prob_key] = np.array(expt_dict[f0_label_prob_key])[sort_idx]
    # Populate expt_dict with metadata required for specific experiment
    for key in metadata_key_list:
        if key in json_dict.keys():
            if isinstance(json_dict[key], list):
                expt_dict[key] = np.array(json_dict[key])[sort_idx]
            else:
                expt_dict[key] = json_dict[key]
        else:
            print("metadata key `{}` not found in json file".format(key))
    return expt_dict


def compute_f0_pred_with_prior(expt_dict,
                               f0_bins,
                               f0_label_prob_key='f0_label:probs_out',
                               f0_prior_ref_key='base_f0',
                               octave_range=[-1, 1],
                               use_octave_folding_prior=False):
    '''
    Computes predicted f0 values from a probability distribution
    over f0 classes and a specified prior.
    
    Args
    ----
    expt_dict (dict): f0 experiment data dict
    f0_bins (np array): f0 bins in Hz
    f0_label_prob_key (str): key for f0_label_pred probabilities in expt_dict
    f0_prior_ref_key (str): key for f0_prior_ref (reference f0 for the prior)
    octave_range (list): limits for the uniform prior (in octaves re: f0_prior_ref)
    use_octave_folding_prior (bool) if True, f0 predictions are multiplied by powers
        of two to return a prediction in the same octave_range as f0_prior_ref_key
    
    Returns
    -------
    f0_pred (np array): predicted f0 values in Hz
    '''
    assert f0_label_prob_key in expt_dict.keys(), "f0_label_prob_key not found in expt_dict"
    assert f0_prior_ref_key in expt_dict.keys(), "f0_prior_ref_key not found in expt_dict"
    f0_pred_prob = expt_dict[f0_label_prob_key]
    msg = "f0_pred_prob ({}) and f0_bins ({}) must match in shape"
    assert f0_pred_prob.shape[1] == f0_bins.shape[0], msg.format(f0_pred_prob.shape, f0_bins.shape)
    f0_prior_ref = expt_dict[f0_prior_ref_key]
    f0_pred = np.zeros_like(f0_prior_ref)
    print('Computing f0_pred using uniform prior: {} octaves'.format(octave_range), flush=True)
    counter = 0
    for stimulus_idx, f0 in enumerate(f0_prior_ref):
        f0_prior_range = f0 * np.power(2, np.array(octave_range, dtype=float))
        bin_mask = np.logical_and(f0_bins >= f0_prior_range[0],
                                  f0_bins <= f0_prior_range[1]).astype(float)
        probs_masked = bin_mask * f0_pred_prob[stimulus_idx]
        f0_pred[stimulus_idx] = f0_bins[np.argmax(probs_masked)]
        if not np.argmax(probs_masked) == np.argmax(f0_pred_prob[stimulus_idx]):
            counter += 1
    print('Computed f0_pred using uniform prior: {} octaves'.format(octave_range), flush=True)
    print('Prior adjusted {} f0 predictions'.format(counter), flush=True)
    
    if use_octave_folding_prior:
        f0_pred_tmp = f0_pred.reshape([-1, 1])
        f0_prior_ref_tmp = f0_prior_ref.reshape([-1, 1])
        pows2 = np.power(2.0, np.arange(octave_range[0], octave_range[1]+0.1, 1)).reshape([1, -1])
        f0_pred_oct = f0_pred_tmp * pows2
        f0_err_vals = np.abs(f0_pred_oct - f0_prior_ref_tmp) / f0_prior_ref_tmp
        octave_fold_indexes = np.argmin(f0_err_vals, axis=1)
        for itr0 in range(f0_pred.shape[0]):
            f0_pred[itr0] = f0_pred_oct[itr0, octave_fold_indexes[itr0]]
        print('Applied octave folding prior', flush=True)
        print(f0_pred_tmp.shape, pows2.shape, f0_pred_oct.shape, f0_err_vals.shape, f0_pred.shape)
    return f0_pred


def add_f0_estimates_to_expt_dict(expt_dict,
                                  f0_label_true_key='f0_label:labels_true',
                                  f0_label_pred_key='f0_label:labels_pred',
                                  kwargs_f0_bins={},
                                  kwargs_f0_octave={},
                                  kwargs_f0_normalization={},
                                  kwargs_f0_prior={}):
    '''
    Function computes f0 estimates corresponding to f0 labels in expt_dict.
    
    Args
    ----
    expt_dict (dict): f0 experiment data dict (f0 and f0_pred keys will be added)
    f0_label_true_key (str): key for f0_label_true in expt_dict
    f0_label_pred_key (str): key for f0_label_pred in expt_dict
    kwargs_f0_bins (dict): kwargs for computing f0 bins (lower bound used as estimate)
    kwargs_f0_octave (dict): kwargs for converting f0s from Hz to octaves
    kwargs_f0_normalization (dict): kwargs for normalizing f0s
    kwargs_f0_prior (dict): kwargs for using a prior to compute f0_pred
    
    Returns
    -------
    expt_dict (dict): F0 experiment data dict (includes f0 and f0_pred keys)
    '''
    if 'log2' in f0_label_pred_key:
        if 'f0' not in expt_dict.keys():
            expt_dict['f0'] = stimuli_f0_labels.octave_to_f0(expt_dict[f0_label_true_key],
                                                             **kwargs_f0_octave)
        if not 'f0_pred' in expt_dict.keys():
            expt_dict['f0_pred'] = stimuli_f0_labels.octave_to_f0(expt_dict[f0_label_pred_key],
                                                                  **kwargs_f0_octave)
    elif 'normal' in f0_label_pred_key:
        if 'f0' not in expt_dict.keys():
            expt_dict['f0'] = stimuli_f0_labels.normalized_to_f0(expt_dict[f0_label_true_key],
                                                                 **kwargs_f0_normalization)
        if not 'f0_pred' in expt_dict.keys():
            expt_dict['f0_pred'] = stimuli_f0_labels.normalized_to_f0(expt_dict[f0_label_pred_key],
                                                                      **kwargs_f0_normalization)
    else:
        f0_bins = stimuli_f0_labels.get_f0_bins(**kwargs_f0_bins)
        if not 'f0' in expt_dict.keys():
            expt_dict['f0'] = stimuli_f0_labels.label_to_f0(expt_dict[f0_label_true_key], f0_bins)
        if kwargs_f0_prior:
            expt_dict['f0_pred'] = compute_f0_pred_with_prior(expt_dict, f0_bins, **kwargs_f0_prior)
        else:
            if not 'f0_pred' in expt_dict.keys():
                expt_dict['f0_pred'] = stimuli_f0_labels.label_to_f0(
                    expt_dict[f0_label_pred_key], f0_bins)
    return expt_dict


def filter_expt_dict(expt_dict, filter_dict={'phase_mode': 0}):
    '''
    Helper function for filtering expt dict to rows that match specified values
    or fall between specified [min, max] ranges.
    
    Args
    ----
    expt_dict (dict): f0 experiment data dict (will not be modified)
    filter_dict (dict): contains key-value pairs used to filter `expt_dict`
    
    Returns
    -------
    filtered_expt_dict (dict): f0 experiment dict filtered to only include matches to `filter_dict`
    '''
    keep_idx = np.ones_like(expt_dict[sorted(filter_dict.keys())[0]])
    for key in sorted(filter_dict.keys()):
        fvalue = filter_dict[key]
        if hasattr(fvalue, '__len__'):
            assert len(fvalue) == 2, "filter_dict values must be single value or [min, max] range"
            keep_idx = np.logical_and(keep_idx, np.logical_and(expt_dict[key] >= fvalue[0], expt_dict[key] <= fvalue[1]))
        else:
            keep_idx = np.logical_and(keep_idx, expt_dict[key] == fvalue)
    filtered_expt_dict = expt_dict.copy()
    for key in expt_dict.keys():
        if isinstance(expt_dict[key], np.ndarray):
            if np.all(expt_dict[key].shape == keep_idx.shape):
                filtered_expt_dict[key] = expt_dict[key][keep_idx]
    return filtered_expt_dict


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
    if 'base_f0' in expt_dict.keys():
        base_f0 = expt_dict['base_f0'][sort_idx]
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
        if 'base_f0' in expt_dict.keys():
            same_filter_idxs = base_f0 == base_f0[idx_ref]
            comparable_idxs = np.logical_and(comparable_idxs, same_filter_idxs)
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


def get_empirical_psychometric_function(pct_diffs, judgments, bin_width=5e-2):
    '''
    Function estimates the empirical psychometric function by binning the independent
    variable (pct_diffs) and computing the mean of the judgments.
    
    Args
    ----
    pct_diffs (np array): independent variable (1-dimensional vector)
    judgments (np array): response variable (1-dimensional vector)
    bin_width (float): width of bin used to digitize pct_diffs
    
    Returns
    -------
    bins (np array): vector of pct_diffs bins
    bin_means (np array): vector of mean judgments corresponding to each bin
    '''
    assert np.all(pct_diffs.shape == judgments.shape), "pct_diffs and judgments must have same shape"
    bins = np.arange(np.min(pct_diffs), np.max(pct_diffs), bin_width)
    bin_idx = np.digitize(pct_diffs, bins)
    bin_judgments = np.zeros(bins.shape)
    bin_counts = np.zeros(bins.shape)
    for itr0, bi in enumerate(bin_idx):
        bin_judgments[bi-1] += judgments[itr0]
        bin_counts[bi-1] += 1
    keep_idx = bin_counts > 0
    bin_judgments = bin_judgments[keep_idx]
    bin_counts = bin_counts[keep_idx]
    bins = bins[keep_idx]
    bin_means = bin_judgments / bin_counts
    return bins, bin_means


def fit_normcdf(xvals, yvals, mu=0.):
    '''
    Helper function for fitting normcdf (with fixed mean) to data.
    
    Args
    ----
    xvals (np array): independent variable
    yvals (np array): dependent variable
    mu (float): fixed mean of fitted normcdf
    
    Returns
    -------
    sigma_opt (float): value of sigma that minimizes normcdf sum squared residuals
    sigma_opt_cov (float): estimated covariance of popt
    '''
    normcdf = lambda x, sigma: scipy.stats.norm(mu, sigma).cdf(x)
    sigma_opt, sigma_opt_cov = scipy.optimize.curve_fit(normcdf, xvals, yvals)
    return np.squeeze(sigma_opt), np.squeeze(sigma_opt_cov)


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
    sub_expt_dict = filter_expt_dict(expt_dict, filter_dict={'phase_mode':ph, 'low_harm':lh})
    sub_expt_dict = add_f0_judgments_to_expt_dict(sub_expt_dict, f0_true_key='f0', f0_pred_key='f0_pred',
                                                  max_pct_diff=max_pct_diff, noise_stdev=noise_stdev)
    pct_diffs = sub_expt_dict['pairwise_pct_diffs'].reshape([-1])
    pct_diffs = pct_diffs[~np.isnan(pct_diffs)]
    judgments = sub_expt_dict['pairwise_judgments'].reshape([-1])
    judgments = judgments[~np.isnan(judgments)]
    # Fit the empirical psychometric function and compute a threshold
    bins, bin_means = get_empirical_psychometric_function(pct_diffs, judgments, bin_width=bin_width)
    sigma_opt, sigma_opt_cov = fit_normcdf(bins, bin_means, mu=mu)
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
    expt_dict = load_f0_expt_dict_from_json(json_fn,
                                            f0_label_true_key=f0_label_true_key,
                                            f0_label_pred_key=f0_label_pred_key,
                                            f0_label_prob_key=f0_label_prob_key,
                                            metadata_key_list=metadata_key_list)
    expt_dict = add_f0_estimates_to_expt_dict(expt_dict,
                                              f0_label_true_key=f0_label_true_key,
                                              f0_label_pred_key=f0_label_pred_key,
                                              kwargs_f0_bins=kwargs_f0_bins,
                                              kwargs_f0_octave=kwargs_f0_octave,
                                              kwargs_f0_normalization=kwargs_f0_normalization,
                                              kwargs_f0_prior=kwargs_f0_prior)
    if 'base_f0' in expt_dict.keys():
        expt_dict = filter_expt_dict(expt_dict, filter_dict={'base_f0':[f0_min, f0_max]})
    else:
        expt_dict = filter_expt_dict(expt_dict, filter_dict={'f0':[f0_min, f0_max]})
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


def parallel_compute_confusion_matrices(par_idx, expt_dict, unique_phase_mode_list, unique_low_harm_list):
    '''
    '''
    # Generate master list of experimental conditions and select one using `par_idx`
    (ph, lh) = list(itertools.product(unique_phase_mode_list, unique_low_harm_list))[par_idx]
    # Filter stimuli for single condition
    sub_expt_dict = filter_expt_dict(expt_dict, filter_dict={'phase_mode':ph, 'low_harm':lh})
    sub_results_dict = {
        'phase_mode': ph,
        'low_harm': lh,
        'f0_true': sub_expt_dict['f0'],
        'f0_pred': sub_expt_dict['f0_pred'],
    }
    return par_idx, sub_results_dict


def compute_confusion_matrices(json_fn, f0_label_true_key='f0_label:labels_true',
                               f0_label_pred_key='f0_label:labels_pred',
                               kwargs_f0_bins={}, kwargs_f0_octave={}, kwargs_f0_normalization={},
                               f0_min=-np.inf, f0_max=np.inf, max_processes=60):
    '''
    '''
    # Load JSON file of model predictions into `expt_dict`
    expt_dict = load_f0_expt_dict_from_json(json_fn,
                                            f0_label_true_key=f0_label_true_key,
                                            f0_label_pred_key=f0_label_pred_key,
                                            metadata_key_list=['low_harm', 'phase_mode', 'f0'])
    expt_dict = add_f0_estimates_to_expt_dict(expt_dict,
                                              f0_label_true_key=f0_label_true_key,
                                              f0_label_pred_key=f0_label_pred_key,
                                              kwargs_f0_bins=kwargs_f0_bins,
                                              kwargs_f0_octave=kwargs_f0_octave,
                                              kwargs_f0_normalization=kwargs_f0_normalization)
    expt_dict = filter_expt_dict(expt_dict, filter_dict={'f0':[f0_min, f0_max]})
    unique_phase_mode_list = np.unique(expt_dict['phase_mode'])
    unique_low_harm_list = np.unique(expt_dict['low_harm'])
    N = len(unique_phase_mode_list) * len(unique_low_harm_list)
    # Initialize dictionary to hold confusion matrices
    results_dict = {
        'phase_mode': [None]*N,
        'low_harm': [None]*N,
        'f0_true': [None]*N,
        'f0_pred': [None]*N,
        'f0_min': np.min([np.min(expt_dict['f0']), np.min(expt_dict['f0_pred'])]),
        'f0_max': np.max([np.max(expt_dict['f0']), np.max(expt_dict['f0_pred'])]),
    }
    # Define a pickle-able wrapper for `parallel_compute_confusion_matrices` using functools
    parallel_run_wrapper = functools.partial(parallel_compute_confusion_matrices,
                                             expt_dict=expt_dict,
                                             unique_phase_mode_list=unique_phase_mode_list,
                                             unique_low_harm_list=unique_low_harm_list)
    # Call the wrapper in parallel processes using multiprocessing.Pool
    with multiprocessing.Pool(processes=np.min([N, max_processes])) as pool:    
        parallel_results = pool.map(parallel_run_wrapper, range(0, N))
        for (par_idx, sub_results_dict) in parallel_results:
            for key in sub_results_dict.keys():
                results_dict[key][par_idx] = sub_results_dict[key]
    # Return dictionary of confusion matrices
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
            'f0_prior_ref_key': 'base_f0', # Use base_f0, so prior does not bias up/down judgments
            'octave_range': [
                -parsed_args_dict['prior_range_in_octaves'],
                parsed_args_dict['prior_range_in_octaves']
            ],
        }
    else:
        kwargs_f0_prior = {}
    
    main(json_eval_fn, save_results_to_file=True, kwargs_f0_prior=kwargs_f0_prior)
