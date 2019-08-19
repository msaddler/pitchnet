import sys
import os
import json
import numpy as np
import pdb
import scipy.optimize
import scipy.stats

sys.path.append('../assets_datasets/')
import stimuli_f0_labels
import f0dl_bernox


def run_f0dl_experiment(json_fn, max_pct_diff=6, noise_stdev=1e-12, bin_width=5e-2, mu=0.0,
                        threshold_value=0.707, use_empirical_f0dl_if_possible=False,
                        f0_label_true_key='f0_label:labels_true', f0_label_pred_key='f0_label:labels_pred',
                        kwargs_f0_bins={}, kwargs_f0_octave={}, kwargs_f0_normalization={},
                        f0_ref_min=80.0, f0_ref_max=320.0, f0_ref_n_step=5,
                        metadata_key_list=['f_carrier', 'f_envelope', 'f0']):
    '''
    '''
    # Load JSON file of model predictions into `expt_dict`
    expt_dict = f0dl_bernox.load_f0_expt_dict_from_json(json_fn,
                                                        f0_label_true_key=f0_label_true_key,
                                                        f0_label_pred_key=f0_label_pred_key,
                                                        metadata_key_list=metadata_key_list)
    expt_dict = f0dl_bernox.add_f0_estimates_to_expt_dict(expt_dict,
                                                          f0_label_true_key=f0_label_true_key,
                                                          f0_label_pred_key=f0_label_pred_key,
                                                          kwargs_f0_bins=kwargs_f0_bins,
                                                          kwargs_f0_octave=kwargs_f0_octave,
                                                          kwargs_f0_normalization=kwargs_f0_normalization)
    unique_f_carrier_list = np.unique(expt_dict['f_carrier'])
    f0_ref_list = np.power(2, np.linspace(np.log2(f0_ref_min), np.log2(f0_ref_max), f0_ref_n_step))
    N = len(unique_f_carrier_list) * len(f0_ref_list)
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
