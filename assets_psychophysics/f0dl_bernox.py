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

sys.path.append('../assets_datasets/')
import stimuli_f0_labels


def bernox2005_human_results_dict():
    '''
    Returns psychophysical results dictionary of Bernstein & Oxenham (2005, JASA) human data
    '''
    ### LOW SPECTRUM CONDITION ###
    bernox_sine_f0dl_LowSpec = [6.011, 4.6697, 4.519, 1.4067, 0.66728, 0.40106]
    bernox_rand_f0dl_LowSpec = [12.77, 10.0439, 8.8367, 1.6546, 0.68939, 0.515]
    bernox_sine_f0dl_stddev_LowSpec = [2.2342, 0.90965, 1.5508, 0.68285, 0.247, 0.13572]
    bernox_rand_f0dl_stddev_LowSpec = [4.7473, 3.2346, 2.477, 0.85577, 0.2126, 0.19908]
    unique_low_harm_list_LowSpec = 1560 / np.array([50, 75, 100, 150, 200, 300])
    unique_phase_mode_list = [0, 1]
    results_dict_LowSpec = {
        'phase_mode': [],
        'low_harm': [],
        'f0dl': [],
        'f0dl_stddev': [],
    }
    for phase_mode in unique_phase_mode_list:
        for lhidx, low_harm in enumerate(unique_low_harm_list_LowSpec):
            results_dict_LowSpec['phase_mode'].append(phase_mode)
            results_dict_LowSpec['low_harm'].append(low_harm)
            if phase_mode == 0:
                results_dict_LowSpec['f0dl'].append(bernox_sine_f0dl_LowSpec[lhidx])
                results_dict_LowSpec['f0dl_stddev'].append(bernox_sine_f0dl_stddev_LowSpec[lhidx])
            elif phase_mode == 1:
                results_dict_LowSpec['f0dl'].append(bernox_rand_f0dl_LowSpec[lhidx])
                results_dict_LowSpec['f0dl_stddev'].append(bernox_rand_f0dl_stddev_LowSpec[lhidx])
            else:
                raise ValueError("ERROR OCCURRED IN `bernox2005_human_results_dict`")
    
    ### HIGH SPECTRUM CONDITION ###
    bernox_sine_f0dl_HighSpec = [5.5257, 5.7834, 4.0372, 1.7769, 0.88999, 0.585]
    bernox_rand_f0dl_HighSpec = [13.4933, 12.0717, 11.5717, 6.1242, 0.94167, 0.53161]
    bernox_sine_f0dl_stddev_HighSpec = [2.0004, 1.4445, 1.1155, 1.0503, 0.26636, 0.16206]
    bernox_rand_f0dl_stddev_HighSpec = [3.4807, 2.3967, 2.3512, 3.2997, 0.37501, 0.24618]
    unique_low_harm_list_HighSpec = 3280 / np.array([100, 150, 200, 300, 400, 600])
    unique_phase_mode_list = [0, 1]
    results_dict_HighSpec = {
        'phase_mode': [],
        'low_harm': [],
        'f0dl': [],
        'f0dl_stddev': [],
    }
    for phase_mode in unique_phase_mode_list:
        for lhidx, low_harm in enumerate(unique_low_harm_list_HighSpec):
            results_dict_HighSpec['phase_mode'].append(phase_mode)
            results_dict_HighSpec['low_harm'].append(low_harm)
            if phase_mode == 0:
                results_dict_HighSpec['f0dl'].append(bernox_sine_f0dl_HighSpec[lhidx])
                results_dict_HighSpec['f0dl_stddev'].append(bernox_sine_f0dl_stddev_HighSpec[lhidx])
            elif phase_mode == 1:
                results_dict_HighSpec['f0dl'].append(bernox_rand_f0dl_HighSpec[lhidx])
                results_dict_HighSpec['f0dl_stddev'].append(bernox_rand_f0dl_stddev_HighSpec[lhidx])
            else:
                raise ValueError("ERROR OCCURRED IN `bernox2005_human_results_dict`")
    
    results_dict = {}
    for key in set(results_dict_LowSpec.keys()).intersection(results_dict_HighSpec.keys()):
        results_dict[key] = results_dict_LowSpec[key] + results_dict_HighSpec[key]
    sort_idx = np.argsort(results_dict['low_harm'])
    for key in results_dict.keys():
        results_dict[key] = np.array(results_dict[key])[sort_idx].tolist()
    return results_dict


def load_f0_expt_dict_from_json(json_fn,
                                f0_label_true_key='f0_label:labels_true',
                                f0_label_pred_key='f0_label:labels_pred',
                                metadata_key_list=['low_harm', 'phase_mode', 'f0']):
    '''
    Function loads a json file and returns a dictionary with np array keys.
    
    Args
    ----
    json_fn (str): json filename to load
    f0_label_true_key (str): key for f0_label_true in the json file
    f0_label_pred_key (str): key for f0_label_pred in the json file
    metadata_key_list (list): keys in the json file to copy to returned dictionary
    
    Returns
    -------
    expt_dict (dict): dictionary with true/pred labels and metadata stored as np arrays
    '''
    # Load the entire json file as a dictionary
    with open(json_fn, 'r') as f: json_dict = json.load(f)
    # Return dict with only specified fields
    expt_dict = {}
    assert f0_label_true_key in json_dict.keys(), "f0_label_true_key not found in json file"
    assert f0_label_pred_key in json_dict.keys(), "f0_label_pred_key not found in json file"
    sort_idx = np.argsort(json_dict['f0'])
    expt_dict = {
        f0_label_true_key: np.array(json_dict[f0_label_true_key])[sort_idx],
        f0_label_pred_key: np.array(json_dict[f0_label_pred_key])[sort_idx],
    }
    for key in metadata_key_list:
        assert key in json_dict.keys(), "metadata key `{}` not found in json file".format(key)
        if isinstance(json_dict[key], list): expt_dict[key] = np.array(json_dict[key])[sort_idx]
        else: expt_dict[key] = json_dict[key]
    return expt_dict


def add_f0_estimates_to_expt_dict(expt_dict,
                                  f0_label_true_key='f0_label:labels_true',
                                  f0_label_pred_key='f0_label:labels_pred',
                                  kwargs_f0_bins={}, kwargs_f0_octave={}, kwargs_f0_normalization={}):
    '''
    Function computes f0 estimates corresponding to f0 labels in expt_dict.
    
    Args
    ----
    expt_dict (dict): f0 experiment data dict (f0 and f0_pred keys will be added)
    kwargs_f0_bins (dict): kwargs for computing f0 bins (lower bound used as estimate)
    kwargs_f0_octave (dict): kwargs for converting f0s from Hz to octaves
    kwargs_f0_normalization (dict): kwargs for normalizing f0s
    
    Returns
    -------
    expt_dict (dict): F0 experiment data dict (includes f0 and f0_pred keys)
    '''
    if 'log2' in f0_label_pred_key:
        if not 'f0_pred' in expt_dict.keys():
            expt_dict['f0_pred'] = stimuli_f0_labels.octave_to_f0(expt_dict[f0_label_pred_key],
                                                                  **kwargs_f0_octave)
        if 'f0' not in expt_dict.keys():
            expt_dict['f0'] = stimuli_f0_labels.octave_to_f0(expt_dict[f0_label_true_key],
                                                             **kwargs_f0_octave)
    elif 'normal' in f0_label_pred_key:
        if not 'f0_pred' in expt_dict.keys():
            expt_dict['f0_pred'] = stimuli_f0_labels.normalized_to_f0(expt_dict[f0_label_pred_key],
                                                                      **kwargs_f0_normalization)
        if 'f0' not in expt_dict.keys():
            expt_dict['f0'] = stimuli_f0_labels.normalized_to_f0(expt_dict[f0_label_true_key],
                                                                 **kwargs_f0_normalization)
    else:
        bins = stimuli_f0_labels.get_f0_bins(**kwargs_f0_bins)
        if not 'f0_pred' in expt_dict.keys():
            expt_dict['f0_pred'] = stimuli_f0_labels.label_to_f0(expt_dict[f0_label_pred_key], bins)
        if not 'f0' in expt_dict.keys():
            expt_dict['f0'] = stimuli_f0_labels.label_to_f0(expt_dict[f0_label_true_key], bins)
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


def add_f0_judgments_to_expt_dict(expt_dict, f0_true_key='f0', f0_pred_key='f0_pred',
                                  max_pct_diff=6., noise_stdev=1e-12):
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


def get_empirical_psychometric_function(pct_diffs, judgments, bin_width=1e-2):
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


def parallel_run_f0dl_experiment(par_idx, expt_dict, unique_phase_mode_list, unique_low_harm_list,
                                 max_pct_diff=6., noise_stdev=1e-12, bin_width=1e-2,
                                 mu=0.0, threshold_value=0.707, use_empirical_f0dl_if_possible=False):
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


def run_f0dl_experiment(json_fn, max_pct_diff=np.inf, noise_stdev=1e-12, bin_width=5e-2, mu=0.0,
                        threshold_value=0.707, use_empirical_f0dl_if_possible=False,
                        f0_label_true_key='f0_label:labels_true', f0_label_pred_key='f0_label:labels_pred',
                        kwargs_f0_bins={}, kwargs_f0_octave={}, kwargs_f0_normalization={},
                        f0_min=-np.inf, f0_max=np.inf, metadata_key_list=['low_harm', 'phase_mode', 'f0'],
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
    kwargs_f0_bins (dict): kwargs for computing f0 bins (lower bound used as estimate)
    kwargs_f0_octave (dict): kwargs for converting f0s from Hz to octaves
    kwargs_f0_normalization (dict): kwargs for normalizing f0s
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
                                            metadata_key_list=metadata_key_list)
    expt_dict = add_f0_estimates_to_expt_dict(expt_dict,
                                              f0_label_true_key=f0_label_true_key,
                                              f0_label_pred_key=f0_label_pred_key,
                                              kwargs_f0_bins=kwargs_f0_bins,
                                              kwargs_f0_octave=kwargs_f0_octave,
                                              kwargs_f0_normalization=kwargs_f0_normalization)
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


def main(json_eval_fn, json_results_dict_fn=None, save_results_to_file=False,
         f0_label_pred_key='f0_label:labels_pred', f0_label_true_key='f0_label:labels_true',
         max_pct_diff=3, bin_width=5e-2, use_empirical_f0dl_if_possible=False,
         f0_min=-np.inf, f0_max=np.inf, max_processes=60):
    '''
    '''
    # Run the Bernstein and Oxenham (2005) F0DL experiment; results stored in results_dict
    metadata_key_list=['low_harm', 'phase_mode', 'f0']
    if 'FixedFilter' in json_eval_fn: metadata_key_list = metadata_key_list + ['base_f0']
    results_dict = run_f0dl_experiment(json_eval_fn,
                                       max_pct_diff=max_pct_diff, bin_width=bin_width,
                                       f0_label_pred_key=f0_label_pred_key,
                                       f0_label_true_key=f0_label_true_key,
                                       use_empirical_f0dl_if_possible=use_empirical_f0dl_if_possible,
                                       metadata_key_list=metadata_key_list,
                                       f0_min=f0_min, f0_max=f0_max, max_processes=max_processes)
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
    parser = argparse.ArgumentParser(description="run Bernstein and Oxenham (2005) F0DL experiment")
    parser.add_argument('-r', '--regex_json_eval_fn', type=str, default=None,
                        help='regex that globs')
    parser.add_argument('-j', '--job_idx', type=int, default=None,
                        help='job index used to name current output directory')
    parsed_args_dict = vars(parser.parse_args())
    assert parsed_args_dict['regex_json_eval_fn'] is not None, "regex_json_eval_fn is a required argument"
    assert parsed_args_dict['job_idx'] is not None, "job_idx is a required argument"
    list_json_eval_fn = sorted(glob.glob(parsed_args_dict['regex_json_eval_fn']))
    json_eval_fn = list_json_eval_fn[parsed_args_dict['job_idx']]
    print('Processing file {} of {}'.format(parsed_args_dict['job_idx'], len(list_json_eval_fn)))
    print('Processing file: {}'.format(json_eval_fn))
    main(json_eval_fn, save_results_to_file=True)
    