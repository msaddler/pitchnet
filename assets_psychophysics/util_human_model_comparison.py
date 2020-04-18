import sys
import os
import json
import numpy as np
import glob
import pdb
import scipy.interpolate
import scipy.stats
import copy


def get_human_results_dict_bernox2005(average_conditions=True):
    '''
    Returns psychophysical results dictionary of Bernstein and Oxenham (2005, JASA)
    human data (Fig 1)
    '''
    # "Low spectrum" condition
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
                raise ValueError("phase_mode={} is not supported".format(phase_mode))
    
    # "High spectrum" condition
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
                raise ValueError("phase_mode={} is not supported".format(phase_mode))
    
    # Combine results of both conditions in a single results_dict 
    results_dict = {}
    if average_conditions:
        # Average the harmonic numbers and f0 discrimination thresholds between conditions
        for key in set(results_dict_LowSpec.keys()).intersection(results_dict_HighSpec.keys()):
            results_dict[key] = (np.array(results_dict_LowSpec[key]) + np.array(results_dict_HighSpec[key])) / 2
            results_dict[key] = results_dict[key].tolist()
    else:
        # Include both conditions in the same set of curves
        for key in set(results_dict_LowSpec.keys()).intersection(results_dict_HighSpec.keys()):
            results_dict[key] = results_dict_LowSpec[key] + results_dict_HighSpec[key]
    sort_idx = np.argsort(results_dict['low_harm'])
    for key in results_dict.keys():
        results_dict[key] = np.array(results_dict[key])[sort_idx].tolist()
    return results_dict


def get_human_results_dict_transposedtones():
    '''
    Returns psychophysical results dictionary of Oxenham et al. (2004, PNAS)
    human data (Fig 2B)
    '''
    results_dict = {
        'f_carrier': [],
        'f0_ref': [],
        'f0dl': [],
    }
    f0_ref_list = [55.60, 80.62, 125.44, 200.77, 320.00] # x-axis values
    # Pure tone discrimination thresholds
    results_dict['f_carrier'].extend([0.0] * len(f0_ref_list))
    results_dict['f0_ref'].extend(f0_ref_list)
    results_dict['f0dl'].extend([6.00, 4.11, 2.36, 1.52, 1.37])
    # 4000Hz TT discrimination thresholds
    results_dict['f_carrier'].extend([4000.0] * len(f0_ref_list))
    results_dict['f0_ref'].extend(f0_ref_list)
    results_dict['f0dl'].extend([13.35, 11.53, 9.21, 5.32, 2.43])
    # 6350Hz TT discrimination thresholds
    results_dict['f_carrier'].extend([6350.0] * len(f0_ref_list))
    results_dict['f0_ref'].extend(f0_ref_list)
    results_dict['f0dl'].extend([18.60, 13.14, 13.74, 7.53, 4.92])
    # 10080Hz TT discrimination thresholds
    results_dict['f_carrier'].extend([10080.0] * len(f0_ref_list))
    results_dict['f0_ref'].extend(f0_ref_list)
    results_dict['f0dl'].extend([20.55, 18.58, 13.53, 10.73, 8.77])
    return results_dict


def get_human_results_dict_freqshiftedcomplexes(average_conditions=True):
    '''
    Returns psychophysical results dictionary of Moore and Moore (2003, JASA)
    human data (Fig 3)
    '''
    # F0=400 Hz condition
    results_dict_400Hz = {
        'f0_max': 400.0,
        'f0_min': 400.0,
        'spectral_envelope_centered_harmonic': {
            '5': {
                'f0_pred_shift_mean': [0.80290, 3.39820, 5.99310, 6.53730],
                'f0_pred_shift_median': [0.80290, 3.39820, 5.99310, 6.53730],
                'f0_pred_shift_stddev': [None, None, None, None],
                'f0_shift': [0.0, 8.0, 16.0, 24.0],
            },
            '11': {
                'f0_pred_shift_mean': [0.11960, 0.62110, 1.37910, 1.92310],
                'f0_pred_shift_median': [0.11960, 0.62110, 1.37910, 1.92310],
                'f0_pred_shift_stddev': [None, None, None, None],
                'f0_shift': [0.0, 8.0, 16.0, 24.0],
            },
            '16': {
                'f0_pred_shift_mean': [-0.73000, -1.17000, -0.24400, -0.00058],
                'f0_pred_shift_median': [-0.73000, -1.17000, -0.24400, -0.00058],
                'f0_pred_shift_stddev': [None, None, None, None],
                'f0_shift': [0.0, 8.0, 16.0, 24.0],
            },
        },
    }
    # F0=200 Hz condition
    results_dict_200Hz = {
        'f0_max': 200.0,
        'f0_min': 200.0,
        'spectral_envelope_centered_harmonic': {
            '5': {
                'f0_pred_shift_mean': [0.6951, 3.4079, 5.6098, 6.9581],
                'f0_pred_shift_median': [0.6951, 3.4079, 5.6098, 6.9581],
                'f0_pred_shift_stddev': [None, None, None, None],
                'f0_shift': [0.0, 8.0, 16.0, 24.0],
            },
            '11': {
                'f0_pred_shift_mean': [0.0566, 0.8515, 1.8173, 2.7402],
                'f0_pred_shift_median': [0.0566, 0.8515, 1.8173, 2.7402],
                'f0_pred_shift_stddev': [None, None, None, None],
                'f0_shift': [0.0, 8.0, 16.0, 24.0],
            },
            '16': {
                'f0_pred_shift_mean': [-0.029, 0.1271, 0.6242, 0.2688],
                'f0_pred_shift_median': [-0.029, 0.1271, 0.6242, 0.2688],
                'f0_pred_shift_stddev': [None, None, None, None],
                'f0_shift': [0.0, 8.0, 16.0, 24.0],
            },
        },
    }
    # F0=100 Hz condition
    results_dict_100Hz = {
        'f0_max': 100.0,
        'f0_min': 100.0,
        'spectral_envelope_centered_harmonic': {
            '5': {
                'f0_pred_shift_mean': [1.3547, 3.5363, 5.1610, 6.6987],
                'f0_pred_shift_median': [1.3547, 3.5363, 5.1610, 6.6987],
                'f0_pred_shift_stddev': [None, None, None, None],
                'f0_shift': [0.0, 8.0, 16.0, 24.0],
            },
            '11': {
                'f0_pred_shift_mean': [0.1967, 0.9196, 2.0293, 2.7524],
                'f0_pred_shift_median': [0.1967, 0.9196, 2.0293, 2.7524],
                'f0_pred_shift_stddev': [None, None, None, None],
                'f0_shift': [0.0, 8.0, 16.0, 24.0],
            },
            '16': {
                'f0_pred_shift_mean': [1.2260, 1.0914, 0.9140, 1.3366],
                'f0_pred_shift_median': [1.2260, 1.0914, 0.9140, 1.3366],
                'f0_pred_shift_stddev': [None, None, None, None],
                'f0_shift': [0.0, 8.0, 16.0, 24.0],
            },
        },
    }
    # Results averaged across F0=400,200,100Hz conditions
    results_dict = {
        'f0_max': 400.0,
        'f0_min': 100.0,
        'spectral_envelope_centered_harmonic': {
            '5': {
                'f0_pred_shift_mean': [0.950900, 3.447467, 5.587967, 6.731367],
                'f0_pred_shift_median': [0.950900, 3.447467, 5.587967, 6.731367],
                'f0_pred_shift_stddev': [None, None, None, None],
                'f0_shift': [0.0, 8.0, 16.0, 24.0],
            },
            '11': {
                'f0_pred_shift_mean': [0.124300, 0.797400, 1.741900, 2.471900],
                'f0_pred_shift_median': [0.124300, 0.797400, 1.741900, 2.471900],
                'f0_pred_shift_stddev': [None, None, None, None],
                'f0_shift': [0.0, 8.0, 16.0, 24.0],
            },
            '16': {
                'f0_pred_shift_mean': [0.155667, 0.016167, 0.431400, 0.534940],
                'f0_pred_shift_median': [0.155667, 0.016167, 0.431400, 0.534940],
                'f0_pred_shift_stddev': [None, None, None, None],
                'f0_shift': [0.0, 8.0, 16.0, 24.0],
            },
        },
    }
    if average_conditions:
        return results_dict
    else:
        return [results_dict_400Hz, results_dict_200Hz, results_dict_100Hz]


def get_human_results_dict_mistunedharmonics():
    '''
    Returns psychophysical results dictionary of Moore et al. (1985, JASA)
    human data, as extracted from figures 2, 3, and 4 (2020-04-17 MRS).
    '''
    with open('human_data_moore1985.json', 'r') as f:
        results_dict = json.load(f)
    return results_dict


def get_human_results_dict_altphasecomplexes(average_conditions=True):
    '''
    Returns psychophysical results dictionary of Shackleton and Carlyon (1994, JASA)
    human data (Fig 2 and 3)
    '''
    # Subject 1
    results_dict_subjectTS = {
        'f0_bin_centers': [62.5, 74.3, 88.4, 105.1, 125.0, 148.7, 176.8, 210.2, 250.0],
        'filter_fl_bin_means': {
            '125.0': [-69.81, -99.09, -99.07, -99.91, -97.31, -95.99, -98.99, -98.97, -90.32],
            '1375.0': [99.81, 98.53, 98.56, 17.97, -44.94, -89.31, -97.05, -99.61, -99.16],
            '3900.0': [99.45, 99.01, 97.28, 99.85, 98.98, 97.67, 96.80, 97.65, 96.35],
        },
    }
    # Subject 2
    results_dict_subjectJS = {
        'f0_bin_centers': [62.5, 74.3, 88.4, 105.1, 125.0, 148.7, 176.8, 210.2, 250.0],
        'filter_fl_bin_means': {
            '125.0': [-96.10, -99.96, -98.64, -99.91, -99.46, -93.41, -98.99, -98.97, -88.60],
            '1375.0': [95.93, 95.95, 96.40, 91.25, 59.38, 1.21, -81.53, -84.96, -98.73],
            '3900.0': [99.45, 99.44, 97.28, 99.85, 97.25, 97.67, 94.65, 97.22, 95.06],
        },
    }
    # Subject 3
    results_dict_subjectSD = {
        'f0_bin_centers': [62.5, 74.3, 88.4, 105.1, 125.0, 148.7, 176.8, 210.2, 250.0],
        'filter_fl_bin_means': {
            '125.0': [-87.04, -99.96, -99.94, -99.91, -100.00, -100.00, -99.85, -96.81, -97.65],
            '1375.0': [97.22, 94.22, 94.25, 71.42, 0.32, -93.62, -97.91, -97.89, -100.00],
            '3900.0': [96.87, 93.41, 97.28, 96.83, 99.41, 97.67, 95.94, 98.51, 96.06],
        },
    }
    # Subject 4
    results_dict_subjectRB = {
        'f0_bin_centers': [62.5, 74.3, 88.4, 105.1, 125.0, 148.7, 176.8, 210.2, 250.0],
        'filter_fl_bin_means': {
            '125.0': [-5.58, -37.46, -76.66, -96.03, -96.44, -97.71, -97.26, -94.22, -89.89],
            '1375.0': [100.00, 97.67, 98.99, 54.18, 49.89, -16.47, -54.81, -87.11, -97.00],
            '3900.0': [100.00, 100.00, 99.43, 96.40, 100.00, 98.97, 93.36, 89.47, 83.43],
        },
    }
    # Mean across subjects 1-4
    results_dict = {
        'f0_bin_centers': [62.5, 74.3, 88.4, 105.1, 125.0, 148.7, 176.8, 210.2, 250.0],
        'filter_fl_bin_means': {
            '125.0': [-64.63, -84.12, -93.58, -98.94, -98.30, -96.78, -98.77, -97.24, -91.62],
            '1375.0': [98.24, 96.59, 97.05, 58.71, 16.16, -49.55, -82.83, -92.39, -98.72],
            '3900.0': [98.94, 97.97, 97.82, 98.23, 98.91, 98.00, 95.19, 95.71, 92.73],
        },
    }
    # Convert values from percent to fraction
    for rd in [results_dict_subjectTS,
               results_dict_subjectJS,
               results_dict_subjectSD,
               results_dict_subjectRB,
               results_dict]:
        for key in rd['filter_fl_bin_means'].keys():
            values = rd['filter_fl_bin_means'][key]
            rd['filter_fl_bin_means'][key] = [v/100 for v in values]
    
    # Pitch-match histogram data from Figure 2 (extracted from alt_hist.ps)
    # Email communication from Bob Carlyon and Trevor Shackleton, 2020JAN10
    bins = [0.9]
    while bins[-1] < 2.3:
        bins.append(bins[-1] * (1.01))
    bins = np.array(bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_widths = bins[1:] - bins[:-1]
    filter_conditions = np.array([ 125., 125., 125., 1375., 1375., 1375., 3900., 3900., 3900.])
    f0_conditions = np.array([ 62.5, 125., 250., 62.5, 125., 250., 62.5, 125., 250.])
    bin_heights_array = np.array([[ 0. ,  0. ,  0.7,  0. ,  0. ,  0.7,  1.4,  1.4,  7.6,  9. , 13.2,
                                14.6, 11.8, 11.1, 10.4,  6.9,  4.2,  1.4,  0.7,  1.4,  1.4,  0. ,
                                 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0.7,  0. ,  0. ,  0. ,
                                 0.7,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0.7,
                                 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
                                 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
                                 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
                                 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
                                 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
                               [ 0. ,  0. ,  0. ,  0. ,  0. ,  0.7,  2.8,  8.3, 13.9, 20.8, 16. ,
                                16.7,  9. ,  6.9,  1.4,  0.7,  0.7,  0.7,  0. ,  0.7,  0. ,  0. ,
                                 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
                                 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
                                 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
                                 0. ,  0. ,  0. ,  0.7,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
                                 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
                                 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
                                 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
                               [ 0. ,  0. ,  0. ,  0.7,  0.7,  1.4,  2.8,  2.1, 22.2, 22.9, 25.7,
                                13.2,  6.3,  0.7,  0.7,  0. ,  0.7,  0. ,  0. ,  0. ,  0. ,  0. ,
                                 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
                                 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
                                 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
                                 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
                                 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
                                 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
                                 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
                               [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
                                 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
                                 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
                                 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0.7,  0. ,  0. ,
                                 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0.7,  0. ,  0. ,  0. ,  0. ,
                                 0.7,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0.7,  0. ,  0. ,  0. ,
                                 1.4,  0. ,  0.7,  1.4,  0.7,  0.7,  4.2,  3.5,  6.3,  2.1,  1.4,
                                 6.3, 12.5, 11.8,  7.6, 10.4,  8.3,  5.6,  3.5,  2.8,  1.4,  2.1,
                                 0.7,  0.7,  0. ,  0. ,  0.7,  0. ,  0.7],
                               [ 0. ,  0.7,  0. ,  0.7,  0. ,  0.7,  0. ,  0. ,  0. ,  0.7,  0.7,
                                 2.1,  2.1,  2.8,  1.4,  2.8,  3.5,  5.6,  2.8,  1.4,  3.5,  1.4,
                                 0. ,  0.7,  1.4,  0.7,  0.7,  2.8,  1.4,  0. ,  2.1,  0. ,  1.4,
                                 1.4,  0. ,  0. ,  0. ,  1.4,  1.4,  0.7,  0.7,  1.4,  2.1,  0. ,
                                 0. ,  0. ,  2.1,  0. ,  0. ,  0. ,  0. ,  0.7,  1.4,  0.7,  1.4,
                                 0. ,  2.1,  0. ,  2.1,  0.7,  0.7,  2.8,  3.5,  4.9,  2.1,  2.1,
                                 4.2,  0. ,  2.1,  2.8,  2.1,  0.7,  2.1,  0. ,  0.7,  0. ,  0. ,
                                 0.7,  1.4,  0.7,  0. ,  0. ,  0.7,  0. ,  0. ,  0.7,  0. ,  0. ,
                                 0. ,  0.7,  0.7,  0. ,  0. ,  0. ,  0. ],
                               [ 1.4,  0. ,  1.4,  0.7,  2.8,  0. ,  3.5, 12.5, 11.8, 17.4, 17.4,
                                13.2,  9. ,  2.8,  2.1,  1.4,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
                                 0. ,  0. ,  0.7,  0. ,  0. ,  1.4,  0. ,  0. ,  0.7,  0. ,  0. ,
                                 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
                                 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
                                 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
                                 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
                                 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
                                 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
                               [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
                                 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
                                 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
                                 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  1.9,  0. ,  0. ,  0. ,  0. ,
                                 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
                                 0. ,  0. ,  0. ,  1.9,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  1.9,
                                 3.7,  3.7,  3.7,  0. ,  0. ,  7.4,  1.9,  5.6,  9.3, 11.1,  9.3,
                                 9.3,  3.7, 16.7,  5.6,  3.7,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
                                 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
                               [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
                                 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
                                 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
                                 0. ,  0. ,  0. ,  0. ,  0.7,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
                                 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0.7,
                                 0. ,  0.7,  0.7,  0.7,  0. ,  0.7,  0.7,  0.7,  0. ,  0.7,  0.7,
                                 1.4,  0.7,  0.7,  1.4,  1.4,  2.8,  1.4,  6.3,  7.6,  8.3,  8.3,
                                 6.9,  5.6,  5.6,  8.3,  4.2,  3.5,  1.4,  1.4,  2.1,  4.2,  4.2,
                                 2.8,  0.7,  1.4,  0. ,  0. ,  0. ,  0.7],
                               [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
                                 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
                                 0. ,  1.4,  0. ,  0. ,  1.4,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
                                 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  2.8,  2.1,
                                 0. ,  0. ,  0. ,  0. ,  2.8,  1.4,  0. ,  3.5,  1.4,  0.7,  0.7,
                                 0. ,  0. ,  0. ,  0. ,  0.7,  2.1,  2.1,  2.8,  4.2,  2.8,  4.9,
                                 1.4,  5.6,  3.5,  6.3,  1.4,  3.5,  4.9,  3.5,  2.1,  0.7,  4.9,
                                 1.4,  2.1,  1.4,  2.1,  0.7,  2.8,  2.1,  1.4,  0. ,  1.4,  0.7,
                                 2.8,  1.4,  0. ,  1.4,  0.7,  0.7,  2.1]])
    # Add pitch-match histogram data to each results_dict
    for rd in [results_dict_subjectTS,
               results_dict_subjectJS,
               results_dict_subjectSD,
               results_dict_subjectRB,
               results_dict]:
        rd['filter_conditions'] = filter_conditions
        rd['f0_conditions'] = f0_conditions
        rd['bins'] = bins
        rd['bin_centers'] = bin_centers
        rd['bin_widths'] = bin_widths
        rd['bin_heights_array'] = bin_heights_array
    
    if average_conditions:
        return results_dict
    else:
        return [results_dict_subjectTS, results_dict_subjectJS, results_dict_subjectSD, results_dict_subjectRB]


def extract_data_from_alt_phase_histogram_ps_file(fn='alt_hist.ps'):
    '''
    Helper script to extract alt-phase pitch-match data from histograms
    (Figure 2 of Shackleton & Carlyon, 1994 JASA). Argument is filename of
    post-script file received by email from Carlyon & Shackleton (2020JAN10).
    '''
    # Load post-script file as text file
    with open(fn, 'r') as f:
        line_list = f.readlines()
    relevant_line_list = []
    # Hacked-together code to extract all relevant lines
    for tmp_line in line_list:
        x = tmp_line.strip()
        if len(x) > 4:
            if (x[0] == '(') and (x[3] == ')'):
                relevant_line_list.append(x[1])
            if (x[-2:] == ' h'):
                relevant_line_list.append(x.replace(' h', ''))
    # Sort relevant lines into labels (keys) and values (floats)
    results_dict = {}
    current_key = None
    for tmp_line in relevant_line_list:
        if '.' not in tmp_line:
            results_dict[tmp_line] = []
            current_key = tmp_line
        else:
            assert current_key is not None
            results_dict[current_key].append(float(tmp_line))
    for key in results_dict.keys():
        results_dict[key] = np.array(results_dict[key])
    # Organize output in human_hist_results_dict for re-plotting
    bin_heights_array = np.zeros([9, 95])
    for idx, key in enumerate(sorted(results_dict.keys())):
        bin_heights_array[idx, :] = results_dict[key]
    bins = [0.9]
    while bins[-1] < 2.3:
        bins.append(bins[-1] * (1.01))
    bins = np.array(bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_widths = bins[1:] - bins[:-1]
    filter_conditions = np.array([ 125., 125., 125., 1375., 1375., 1375., 3900., 3900., 3900.])
    f0_conditions = np.array([ 62.5, 125., 250., 62.5, 125., 250., 62.5, 125., 250.])
    human_hist_results_dict = {
        'filter_conditions': filter_conditions,
        'f0_conditions': f0_conditions,
        'bins': bins,
        'bin_centers': bin_centers,
        'bin_widths': bin_widths,
        'bin_heights_array': bin_heights_array,
    }
    return human_hist_results_dict


def get_mistuned_harmonics_bar_graph_results_dict(results_dict,
                                                  mistuned_pct=3.0,
                                                  pitch_shift_key='f0_pred_pct_median',
                                                  pitch_shift_err_key=None,
                                                  harmonic_list=None,
                                                  use_relative_shift=False):
    '''
    This helper function parses a results_dict from the Moore et al. (1985, JASA)
    mistuned harmonics experiment into a smaller bar_graph_results_dict, which
    allows for easier plotting of the Meddis and O'Mard (1997, JASA) summary bar
    graph (Fig 8B)
    '''
    if pitch_shift_err_key is None: pitch_shift_err_key = pitch_shift_key + '_err'
    f0_ref_list = results_dict['f0_ref_list']
    bar_graph_results_dict = {}
    if harmonic_list is None:
        harm_key_list = results_dict['f0_ref'][str(f0_ref_list[0])]['mistuned_harm'].keys()
        harmonic_list = sorted([int(harm_key) for harm_key in harm_key_list])
    for harm in harmonic_list:
        harm_key = str(harm)
        bar_graph_results_dict[harm_key] = {
            'f0_ref': [],
            pitch_shift_key: [],
            pitch_shift_err_key: [],
        }
        for f0_ref in f0_ref_list:
            f0_ref_key = str(f0_ref)
            sub_results_dict = results_dict['f0_ref'][f0_ref_key]['mistuned_harm'][harm_key]
            mp_idx = sub_results_dict['mistuned_pct'].index(mistuned_pct)
            pitch_shift = sub_results_dict[pitch_shift_key][mp_idx]
            if pitch_shift_err_key in sub_results_dict.keys():
                pitch_shift_err = sub_results_dict[pitch_shift_err_key][mp_idx]
            else:
                pitch_shift_err = np.zeros_like(pitch_shift).tolist()
            if use_relative_shift:
                if 0.0 in sub_results_dict['mistuned_pct']:
                    unshifted_idx = sub_results_dict['mistuned_pct'].index(0.0)
                    pitch_shift = pitch_shift - sub_results_dict[pitch_shift_key][unshifted_idx]
            bar_graph_results_dict[harm_key]['f0_ref'].append(f0_ref)
            bar_graph_results_dict[harm_key][pitch_shift_key].append(pitch_shift)
            bar_graph_results_dict[harm_key][pitch_shift_err_key].append(pitch_shift_err)
    return bar_graph_results_dict


def get_altphase_histogram_results_dict(results_dict,
                                        bin_step=0.0201,
                                        bin_limits=[0.9, 2.3]):
    '''
    This helper function parses a results_dict from the Shackleton and Carlyon (1994, JASA)
    alternating phase experiment into a smaller histogram_results_dict, which allows for
    easier plotting of the Shackleton & Carlyon (1994, JASA) pitch match histograms (Fig 2).
    '''
    # Create the histogram bins (shared for all conditions)
    bins = [bin_limits[0]]
    while bins[-1] < bin_limits[1]:
        bins.append(bins[-1] * (1.0+bin_step))
    bins = np.array(bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_widths = bins[1:] - bins[:-1]
    if 'bin_heights_array' not in results_dict.keys():
        # Compute histogram results if they do not already exist in results_dict
        filter_conditions = np.array(results_dict['f0_pred_ratio_results']['filter_condition_list'])
        f0_conditions = np.array(results_dict['f0_pred_ratio_results']['f0_condition_list'])
        f0_pred_ratio_list = results_dict['f0_pred_ratio_results']['f0_pred_ratio_list']
        assert len(f0_pred_ratio_list) == len(filter_conditions)
        assert len(f0_pred_ratio_list) == len(f0_conditions)
        bin_heights_array = np.zeros([len(f0_pred_ratio_list), len(bin_centers)])
        # Manually compute histogram and convert to percentage for each condition
        for idx in range(len(f0_pred_ratio_list)):
            bin_counts, bin_edges = np.histogram(f0_pred_ratio_list[idx], bins=bins)
            bin_percentages = 100.0 * bin_counts / np.sum(bin_counts)
            bin_heights_array[idx, :] = bin_percentages
        # Return outputs in an orderly dictionary, ready for plotting / averaging
        histogram_results_dict = {
            'filter_conditions': filter_conditions,
            'f0_conditions': f0_conditions,
            'bins': bins,
            'bin_centers': bin_centers,
            'bin_widths': bin_widths,
            'bin_heights_array': bin_heights_array,
        }
    else:
        # Do not compute histogram results if they already exist in results_dict
        histogram_results_dict = copy.deepcopy(results_dict)
        if not np.array_equal(bins, histogram_results_dict['bins']):
            original_bin_centers = histogram_results_dict['bin_centers']
            new_bin_center_indexes = np.digitize(original_bin_centers, bins, right=False) - 1
            new_bin_heights_array = np.zeros([histogram_results_dict['bin_heights_array'].shape[0],
                                              len(bin_centers)])
            for old_idx, new_idx in enumerate(new_bin_center_indexes):
                new_bin_heights_array[:, new_idx] += histogram_results_dict['bin_heights_array'][:, old_idx]
            histogram_results_dict['bins'] = bins
            histogram_results_dict['bin_centers'] = bin_centers
            histogram_results_dict['bin_widths'] = bin_widths
            histogram_results_dict['bin_heights_array'] = new_bin_heights_array
    return histogram_results_dict


def interpolate_data(xvals,
                     yvals,
                     interp_xvals,
                     kind='linear',
                     bounds_error=True,
                     fill_value=np.nan):
    '''
    '''
    interp_fcn = scipy.interpolate.interp1d(xvals, yvals,
                                            kind=kind,
                                            bounds_error=bounds_error,
                                            fill_value=fill_value)
    if bounds_error:
        interp_xvals = interp_xvals[interp_xvals >= np.min(xvals)]
        interp_xvals = interp_xvals[interp_xvals <= np.max(xvals)]
    return interp_xvals, interp_fcn(interp_xvals)


def compare_human_model_data(results_vector_human,
                             results_vector_model,
                             metric='spearmanr',
                             log_scale=False):
    '''
    '''
    metric_functions = {
        'pearsonr': scipy.stats.pearsonr,
        'spearmanr': scipy.stats.spearmanr,
    }
    assert results_vector_human.shape == results_vector_model.shape
    assert metric in metric_functions.keys(), "metric=`{}` is not supported".format(metric)
    if log_scale:
        results_vector_human = np.log(results_vector_human)
        results_vector_model = np.log(results_vector_model)
    return metric_functions[metric](results_vector_human, results_vector_model)


def compare_bernox2005(human_results_dict,
                       model_results_dict,
                       extrapolate_lowest_harm=True,
                       kwargs_interp={},
                       kwargs_compare={'log_scale':True}):
    '''
    '''
    human_phase_mode_list = np.array(human_results_dict['phase_mode'])
    model_phase_mode_list = np.array(model_results_dict['phase_mode'])
    assert np.array_equal(np.unique(human_phase_mode_list), np.unique(model_phase_mode_list))
    unique_phase_modes = np.unique(human_phase_mode_list)
    
    results_vector_human = []
    results_vector_model = []
    for phase_mode in unique_phase_modes:
        human_low_harm = np.array(human_results_dict['low_harm'])[human_phase_mode_list == phase_mode]
        human_f0dl = np.array(human_results_dict['f0dl'])[human_phase_mode_list == phase_mode]
        model_low_harm = np.array(model_results_dict['low_harm'])[model_phase_mode_list == phase_mode]
        model_f0dl = np.array(model_results_dict['f0dl'])[model_phase_mode_list == phase_mode]
        
        if extrapolate_lowest_harm:
            lowest_harm_index = np.argmin(human_low_harm)
            kwargs_interp.update({'bounds_error': False,
                                  'fill_value': (human_f0dl[lowest_harm_index], np.nan)})
        interp_human_low_harm, interp_human_f0dl = interpolate_data(human_low_harm,
                                                                    human_f0dl,
                                                                    model_low_harm,
                                                                    **kwargs_interp)
        interp_human_low_harm = interp_human_low_harm.tolist()
        interp_human_f0dl = interp_human_f0dl.tolist()
        model_low_harm = model_low_harm.tolist()
        model_f0dl = model_f0dl.tolist()
        
        for idx_human, low_harm in enumerate(interp_human_low_harm):
            idx_model = model_low_harm.index(low_harm)
            results_vector_human.append(interp_human_f0dl[idx_human])
            results_vector_model.append(model_f0dl[idx_model])
    
    results_vector_human = np.array(results_vector_human)
    results_vector_model = np.array(results_vector_model)
    return compare_human_model_data(results_vector_human, results_vector_model,
                                    **kwargs_compare)


def compare_transposedtones(human_results_dict,
                            model_results_dict,
                            kwargs_interp={},
                            kwargs_compare={'log_scale':True}):
    '''
    '''
    human_f_carrier_list = np.array(human_results_dict['f_carrier'])
    model_f_carrier_list = np.array(model_results_dict['f_carrier'])
    assert np.array_equal(np.unique(human_f_carrier_list), np.unique(model_f_carrier_list))
    unique_f_carriers = np.unique(human_f_carrier_list)
    
    results_vector_human = []
    results_vector_model = []
    for f_carrier in unique_f_carriers:
        human_f0_ref = np.array(human_results_dict['f0_ref'])[human_f_carrier_list == f_carrier]
        human_f0dl = np.array(human_results_dict['f0dl'])[human_f_carrier_list == f_carrier]
        model_f0_ref = np.array(model_results_dict['f0_ref'])[model_f_carrier_list == f_carrier]
        model_f0dl = np.array(model_results_dict['f0dl'])[model_f_carrier_list == f_carrier]
        
        interp_human_f0_ref, interp_human_f0dl = interpolate_data(human_f0_ref,
                                                                  human_f0dl,
                                                                  model_f0_ref,
                                                                  **kwargs_interp)
        interp_human_f0_ref = interp_human_f0_ref.tolist()
        interp_human_f0dl = interp_human_f0dl.tolist()
        model_f0_ref = model_f0_ref.tolist()
        model_f0dl = model_f0dl.tolist()
        
        for idx_human, f0_ref in enumerate(interp_human_f0_ref):
            idx_model = model_f0_ref.index(f0_ref)
            results_vector_human.append(interp_human_f0dl[idx_human])
            results_vector_model.append(model_f0dl[idx_model])
    
    results_vector_human = np.array(results_vector_human)
    results_vector_model = np.array(results_vector_model)
    return compare_human_model_data(results_vector_human, results_vector_model,
                                    **kwargs_compare)


def compare_freqshiftedcomplexes(human_results_dict,
                                 model_results_dict,
                                 pitch_shift_key='f0_pred_shift_median',
                                 restrict_conditions=['5', '11', '16'],
                                 kwargs_interp={},
                                 kwargs_compare={'log_scale':False}):
    '''
    '''
    human_conditions = human_results_dict['spectral_envelope_centered_harmonic'].keys()
    model_conditions = model_results_dict['spectral_envelope_centered_harmonic'].keys()
    assert np.array_equal(human_conditions, model_conditions)
    condition_list = human_conditions
    results_vector_human = []
    results_vector_model = []
    if restrict_conditions is not None:
        condition_list = restrict_conditions
    for condition_key in condition_list:
        human_xvals = np.array(human_results_dict['spectral_envelope_centered_harmonic'][condition_key]['f0_shift'])
        human_yvals = np.array(human_results_dict['spectral_envelope_centered_harmonic'][condition_key][pitch_shift_key])
        model_xvals = np.array(model_results_dict['spectral_envelope_centered_harmonic'][condition_key]['f0_shift'])
        model_yvals = np.array(model_results_dict['spectral_envelope_centered_harmonic'][condition_key][pitch_shift_key])
        
        interp_human_xvals, interp_human_yvals = interpolate_data(human_xvals,
                                                                  human_yvals,
                                                                  model_xvals,
                                                                  **kwargs_interp)
        interp_human_xvals = interp_human_xvals.tolist()
        interp_human_yvals = interp_human_yvals.tolist()
        model_xvals = model_xvals.tolist()
        model_yvals = model_yvals.tolist()
        
        for idx_human, xval in enumerate(interp_human_xvals):
            idx_model = model_xvals.index(xval)
            results_vector_human.append(interp_human_yvals[idx_human])
            results_vector_model.append(model_yvals[idx_model])
    
    results_vector_human = np.array(results_vector_human)
    results_vector_model = np.array(results_vector_model)
    return compare_human_model_data(results_vector_human, results_vector_model,
                                    **kwargs_compare)


def compare_mistunedharmonics(human_results_dict,
                              model_results_dict,
                              kwargs_bar_graph={},
                              kwargs_compare={'log_scale':False}):
    '''
    '''
    human_bar_graph_results_dict = get_mistuned_harmonics_bar_graph_results_dict(
        human_results_dict, **kwargs_bar_graph)
    model_bar_graph_results_dict = get_mistuned_harmonics_bar_graph_results_dict(
        model_results_dict, **kwargs_bar_graph)
    pitch_shift_key=kwargs_bar_graph.get('pitch_shift_key', 'f0_pred_pct_median')
    human_conditions = human_bar_graph_results_dict.keys()
    model_conditions = model_bar_graph_results_dict.keys()
    
    results_vector_human = []
    results_vector_model = []
    for condition_key in model_conditions:
        if condition_key in human_conditions:
            human_xvals = human_bar_graph_results_dict[condition_key]['f0_ref']
            human_yvals = human_bar_graph_results_dict[condition_key][pitch_shift_key]
            model_xvals = model_bar_graph_results_dict[condition_key]['f0_ref']
            model_yvals = model_bar_graph_results_dict[condition_key][pitch_shift_key]
            assert np.array_equal(human_xvals, model_xvals)
            results_vector_human.extend(list(human_yvals))
            results_vector_model.extend(list(model_yvals))
    
    results_vector_human = np.array(results_vector_human)
    results_vector_model = np.array(results_vector_model)
    return compare_human_model_data(results_vector_human, results_vector_model,
                                    **kwargs_compare)


def compare_altphasecomplexes_line(human_results_dict,
                                   model_results_dict,
                                   kwargs_interp={},
                                   kwargs_compare={'log_scale':False}):
    '''
    '''
    human_conditions = human_results_dict['filter_fl_bin_means'].keys()
    model_conditions = model_results_dict['filter_fl_bin_means'].keys()
    assert np.array_equal(human_conditions, model_conditions)
    
    results_vector_human = []
    results_vector_model = []
    for condition_key in human_conditions:
        human_xvals = np.array(human_results_dict['f0_bin_centers'])
        human_yvals = np.array(human_results_dict['filter_fl_bin_means'][condition_key])
        model_xvals = np.array(model_results_dict['f0_bin_centers'])
        model_yvals = np.array(model_results_dict['filter_fl_bin_means'][condition_key])
        
        interp_human_xvals, interp_human_yvals = interpolate_data(human_xvals,
                                                                  human_yvals,
                                                                  model_xvals,
                                                                  **kwargs_interp)
        interp_human_xvals = interp_human_xvals.tolist()
        interp_human_yvals = interp_human_yvals.tolist()
        model_xvals = model_xvals.tolist()
        model_yvals = model_yvals.tolist()
        
        for idx_human, xval in enumerate(interp_human_xvals):
            idx_model = model_xvals.index(xval)
            results_vector_human.append(interp_human_yvals[idx_human])
            results_vector_model.append(model_yvals[idx_model])
    
    results_vector_human = np.array(results_vector_human)
    results_vector_model = np.array(results_vector_model)
    return compare_human_model_data(results_vector_human, results_vector_model,
                                    **kwargs_compare)


def compare_altphasecomplexes_hist(human_results_dict,
                                   model_results_dict,
                                   restrict_conditions_filter=[125.0, 1375.0, 3900.0],
                                   restrict_conditions_f0=[125.0, 250.0],
                                   kwargs_histogram={},
                                   kwargs_compare={'log_scale':False}):
    '''
    '''
    human_hist_results_dict = get_altphase_histogram_results_dict(human_results_dict,
                                                                  **kwargs_histogram)
    model_hist_results_dict = get_altphase_histogram_results_dict(model_results_dict,
                                                                  **kwargs_histogram)
    
    human_filter_conditions = human_hist_results_dict['filter_conditions']
    human_f0_conditions = human_hist_results_dict['f0_conditions']
    human_bin_centers = human_hist_results_dict['bin_centers']
    human_bin_widths = human_hist_results_dict['bin_widths']
    human_bin_heights_array = human_hist_results_dict['bin_heights_array']
    
    model_filter_conditions = model_hist_results_dict['filter_conditions']
    model_f0_conditions = model_hist_results_dict['f0_conditions']
    model_bin_centers = model_hist_results_dict['bin_centers']
    model_bin_widths = model_hist_results_dict['bin_widths']
    model_bin_heights_array = model_hist_results_dict['bin_heights_array']
    
    distance_metrics = []
    results_vector_human = []
    results_vector_model = []
    for f0_val in restrict_conditions_f0:
        for filter_val in restrict_conditions_filter:
            idx_human = np.logical_and(human_f0_conditions==f0_val, human_filter_conditions==filter_val)
            if np.all(f0_val < model_f0_conditions):
                adjusted_f0_val = np.min(model_f0_conditions)
                idx_model = np.logical_and(model_f0_conditions==adjusted_f0_val, model_filter_conditions==filter_val)
            else:
                idx_model = np.logical_and(model_f0_conditions==f0_val, model_filter_conditions==filter_val)
            assert np.array_equal(idx_human, idx_model)
            assert np.sum(idx_human) == 1
            idx = list(idx_human).index(True)
            human_dist = human_bin_heights_array[idx] / np.sum(human_bin_heights_array[idx])
            model_dist = model_bin_heights_array[idx] / np.sum(model_bin_heights_array[idx])
            results_vector_human.extend(list(human_dist))
            results_vector_model.extend(list(model_dist))
#             distance_metrics.append(-np.log(np.sum(np.sqrt(human_dist * model_dist)))) # Bhattacharyya distance
#             distance_metrics.append(np.sum(np.sqrt(human_dist * model_dist))) # Bhattacharyya coefficient
    
    results_vector_human = np.array(results_vector_human)
    results_vector_model = np.array(results_vector_model)
#     distance_metrics = np.sum(np.array(distance_metrics))
#     return distance_metrics, None
    return compare_human_model_data(results_vector_human, results_vector_model,
                                    **kwargs_compare)
