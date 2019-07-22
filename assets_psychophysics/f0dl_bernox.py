import sys
import os
import json
import numpy as np
import pdb
import scipy.optimize
import scipy.stats

sys.path.append('../assets_datasets/')
import stimuli_f0_labels


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
                                  kwargs_f0_bins={}, kwargs_f0_normalization={}):
    '''
    Function computes f0 estimates corresponding to f0 labels in expt_dict.
    
    Args
    ----
    expt_dict (dict): f0 experiment data dict (f0 and f0_pred keys will be added)
    kwargs_f0_bins (dict): kwargs for computing f0 bins (lower bound used as estimate)
    kwargs_f0_normalization (dict): kwargs for normalizing f0s
    
    Returns
    -------
    expt_dict (dict): f0 experiment data dict (includes f0 and f0_pred keys)
    '''
    
    if 'normal' in f0_label_pred_key:
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


def get_sub_expt_dict_by_key_value(expt_dict, split_key='phase_mode', split_value=0):
    '''
    Helper function for splitting experiment data dict according to key-value pair.
    
    Args
    ----
    expt_dict (dict): f0 experiment data dict (will not be modified)
    split_key (str): points to array in expt_dict that will be used to split expt_dict
    split_value (float or int): arrays in returned dict will only contain rows where
        the `split_key` array is equal to `split_value`
    
    Returns
    -------
    sub_expt_dict (dict): f0 experiment data dict filtered by split_key and split_value
    '''
    sub_expt_dict = expt_dict.copy()
    keep_idx = np.argwhere(sub_expt_dict[split_key] == split_value).reshape([-1])
    for key in expt_dict.keys():
        if isinstance(expt_dict[key], np.ndarray):
            if np.all(expt_dict[key].shape == expt_dict[split_key].shape):
                sub_expt_dict[key] = sub_expt_dict[key][keep_idx]
    return sub_expt_dict


def add_f0_judgments_to_expt_dict(expt_dict, f0_true_key='f0', f0_pred_key='f0_pred',
                                  max_pct_diff=6., noise_stdev=1e-12):
    '''
    Function simulates F0 discrimination experiment given a list of true and predicted F0s.
    
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
        within_range_idxs = np.logical_and(pct_diffs >= -max_pct_diff, pct_diffs <= max_pct_diff)
        pairwise_pct_diffs[idx_ref, within_range_idxs] = pct_diffs[within_range_idxs]
        # Compute the percent differences between the predictions
        pred_pct_diffs = 100. * (f0_pred[within_range_idxs] - f0_pred_ref) / f0_pred_ref
        pred_decision_noise = noise_stdev * np.random.randn(pred_pct_diffs.shape[0])
        # Judgment is 1. if model predicts f0_pred_ref > f0_pred
        # Judgment is 0. if model predicts f0_pred_ref < f0_pred
        # Decision stage Gaussian noise is used to break ties
        tmp_judgments = np.array(pred_pct_diffs > pred_decision_noise, dtype=np.float32)
        pairwise_judgments[idx_ref, within_range_idxs] = tmp_judgments
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


def compute_f0_thresholds_for_each_low_harm(expt_dict, threshold_value=0.707, mu=0.,
                                            bin_width=1e-2, return_psychometric_functions=True):
    '''
    Function computes F0 difference limens (in percent F0) as a function of the lowest harmonic
    number by fitting normcdf to estimated psychometric functions.
    
    Args
    ----
    expt_dict (dict): f0 experiment data dict (must contain `pairwise_pct_diffs`, `pairwise_judgments`,
                                               and `low_harm`)
    threshold_value (float): value of the fitted normcdf used to compute F0 difference limen
    mu (float): fixed mean of normcdf used to fit psychometric functions
    bin_width (float): bin width in percent F0 used to digitize pairwise_pct_diffs for psychometric function
    return_psychometric_functions (bool): if True, estimated psychometric functions will be returned
    
    Returns
    -------
    results_dict (dict): contains `low_harm`, `f0dl`, and `psychometric_functions`
    '''
    pairwise_pct_diffs = expt_dict['pairwise_pct_diffs']
    pairwise_judgments = expt_dict['pairwise_judgments']
    assert_msg = "pairwise_pct_diffs and pairwise_judgments must have same shape"
    assert np.all(pairwise_pct_diffs.shape == pairwise_judgments.shape), assert_msg
    # Initialize dictionary used to store normcdf fit information
    psychometric_functions = {
        'sigma': [],
        'sigma_cov': [],
        'bins': [],
        'bin_means': [],
        'threshold_value': threshold_value,
        'mu': mu,
        'bin_width': bin_width,
    }
    # F0 difference limens will be computed for each unique lowest harmonic number
    unique_low_harm_numbers = np.unique(expt_dict['low_harm'])
    low_harm_thresholds = np.zeros_like(unique_low_harm_numbers, dtype=np.float64)
    for idx_low_harm, low_harm in enumerate(unique_low_harm_numbers):
        # Rows of pairwise_pct_diffs corresponding to same low_harm will be combined
        idx_f0_list = np.argwhere(expt_dict['low_harm'] == low_harm).reshape([-1])
        pct_diffs_list = []
        judgments_list = []
        for idx_f0 in idx_f0_list:
            within_range_idxs = np.logical_not(np.isnan(pairwise_pct_diffs[idx_f0]))
            pct_diffs_list.append(pairwise_pct_diffs[idx_f0, within_range_idxs])
            judgments_list.append(pairwise_judgments[idx_f0, within_range_idxs])
        # Estimate the psychometric function from the combined judgments
        bins, bin_means = get_empirical_psychometric_function(np.concatenate(pct_diffs_list),
                                                              np.concatenate(judgments_list),
                                                              bin_width=bin_width)
        # Fit normcdf to the estimated psychometric function and compute threshold
        sigma_opt, sigma_opt_cov = fit_normcdf(bins, bin_means, mu=mu)
        low_harm_thresholds[idx_low_harm] = scipy.stats.norm(mu, sigma_opt).ppf(threshold_value)
        psychometric_functions['sigma'].append(sigma_opt)
        psychometric_functions['sigma_cov'].append(sigma_opt_cov)
#         # HACK BELOW
#         above_threshold_bin_indexes = bin_means > threshold_value
#         if np.sum(above_threshold_bin_indexes) > 0:
#             print('USING EMPIRICAL PSYCHOMETRIC FUNCTION RATHER THAN FIT')
#             print(bins[above_threshold_bin_indexes][0], 'inplace of', low_harm_thresholds[idx_low_harm])
#             low_harm_thresholds[idx_low_harm] = bins[above_threshold_bin_indexes][0]
#         else: low_harm_thresholds[idx_low_harm] = 100
        # Save the empirical psychometric functions if specified
        if return_psychometric_functions:
            psychometric_functions['bins'].append(bins.tolist())
            psychometric_functions['bin_means'].append(bin_means.tolist())
    # Return dict containing low harm numbers, thresholds, and psychometric function information
    results_dict = {
        'low_harm': unique_low_harm_numbers,
        'f0dl': low_harm_thresholds,
        'psychometric_functions': psychometric_functions,
    }
    return results_dict
