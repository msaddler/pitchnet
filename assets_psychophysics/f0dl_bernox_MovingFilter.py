import sys
import os
import json
import numpy as np
import pdb
import scipy.optimize
import scipy.stats

import itertools
import functools
import multiprocessing

sys.path.append('../assets_datasets/')
import stimuli_f0_labels
import f0dl_bernox


def bernox2005_human_results_dict():
    '''
    Creates psychophysical results dictionary from Bernstein & Oxenham (2005 JASA) human data
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


def parallel_run_f0dl_experiment(par_idx, expt_dict, unique_phase_mode_list, unique_low_harm_list,
                                 max_pct_diff=6., noise_stdev=1e-12, bin_width=1e-2,
                                 mu=0.0, threshold_value=0.707):
    '''
    '''
    (ph, lh) = list(itertools.product(unique_phase_mode_list, unique_low_harm_list))[par_idx]
    
    sub_expt_dict = f0dl_bernox.filter_expt_dict(expt_dict, filter_dict={'phase_mode':ph, 'low_harm':lh})
    sub_expt_dict = f0dl_bernox.add_f0_judgments_to_expt_dict(sub_expt_dict, f0_true_key='f0', f0_pred_key='f0_pred',
                                                              max_pct_diff=max_pct_diff, noise_stdev=noise_stdev)
    pct_diffs = sub_expt_dict['pairwise_pct_diffs'].reshape([-1])
    pct_diffs = pct_diffs[~np.isnan(pct_diffs)]
    judgments = sub_expt_dict['pairwise_judgments'].reshape([-1])
    judgments = judgments[~np.isnan(judgments)]
    
    bins, bin_means = f0dl_bernox.get_empirical_psychometric_function(pct_diffs, judgments, bin_width=bin_width)
    sigma_opt, sigma_opt_cov = f0dl_bernox.fit_normcdf(bins, bin_means, mu=mu)
    f0dl = scipy.stats.norm(mu, sigma_opt).ppf(threshold_value)
            
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


def main(json_fn, max_pct_diff=6., noise_stdev=1e-12, bin_width=1e-2, mu=0.0, threshold_value=0.707,
         f0_label_true_key='f0_label:labels_true', f0_label_pred_key='f0_label:labels_pred',
         f0_min=-np.inf, f0_max=np.inf, max_processes=60):
    '''
    '''
    expt_dict = f0dl_bernox.load_f0_expt_dict_from_json(json_fn,
                                                        f0_label_true_key=f0_label_true_key,
                                                        f0_label_pred_key=f0_label_pred_key,
                                                        metadata_key_list=['low_harm', 'phase_mode', 'f0'])
    expt_dict = f0dl_bernox.add_f0_estimates_to_expt_dict(expt_dict,
                                                          f0_label_true_key=f0_label_true_key,
                                                          f0_label_pred_key=f0_label_pred_key,
                                                          kwargs_f0_bins={}, kwargs_f0_normalization={})
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
                                             threshold_value=threshold_value)
    # Call the wrapper in parallel processes using multiprocessing.Pool
    with multiprocessing.Pool(processes=np.min([N, max_processes])) as pool:    
        parallel_results = pool.map(parallel_run_wrapper, range(0, N))
        for (par_idx, sub_results_dict) in parallel_results:
            for key in results_dict.keys():
                results_dict[key][par_idx] = sub_results_dict[key]
    # Return dictionary of psychophysical experiment results
    return results_dict
