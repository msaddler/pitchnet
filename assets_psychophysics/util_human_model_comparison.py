import sys
import os
import json
import numpy as np
import glob


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
                raise ValueError("ERROR OCCURRED IN `bernox2005_human_results_dict`")
    
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
                raise ValueError("ERROR OCCURRED IN `bernox2005_human_results_dict`")
    
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
            5: {
                'f0_pred_shift_mean': [0.80290, 3.39820, 5.99310, 6.53730],
                'f0_pred_shift_median': [0.80290, 3.39820, 5.99310, 6.53730],
                'f0_pred_shift_stddev': [None, None, None, None],
                'f0_shift': [0.0, 8.0, 16.0, 24.0],
            },
            11: {
                'f0_pred_shift_mean': [0.11960, 0.62110, 1.37910, 1.92310],
                'f0_pred_shift_median': [0.11960, 0.62110, 1.37910, 1.92310],
                'f0_pred_shift_stddev': [None, None, None, None],
                'f0_shift': [0.0, 8.0, 16.0, 24.0],
            },
            16: {
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
            5: {
                'f0_pred_shift_mean': [0.6951, 3.4079, 5.6098, 6.9581],
                'f0_pred_shift_median': [0.6951, 3.4079, 5.6098, 6.9581],
                'f0_pred_shift_stddev': [None, None, None, None],
                'f0_shift': [0.0, 8.0, 16.0, 24.0],
            },
            11: {
                'f0_pred_shift_mean': [0.0566, 0.8515, 1.8173, 2.7402],
                'f0_pred_shift_median': [0.0566, 0.8515, 1.8173, 2.7402],
                'f0_pred_shift_stddev': [None, None, None, None],
                'f0_shift': [0.0, 8.0, 16.0, 24.0],
            },
            16: {
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
            5: {
                'f0_pred_shift_mean': [1.3547, 3.5363, 5.1610, 6.6987],
                'f0_pred_shift_median': [1.3547, 3.5363, 5.1610, 6.6987],
                'f0_pred_shift_stddev': [None, None, None, None],
                'f0_shift': [0.0, 8.0, 16.0, 24.0],
            },
            11: {
                'f0_pred_shift_mean': [0.1967, 0.9196, 2.0293, 2.7524],
                'f0_pred_shift_median': [0.1967, 0.9196, 2.0293, 2.7524],
                'f0_pred_shift_stddev': [None, None, None, None],
                'f0_shift': [0.0, 8.0, 16.0, 24.0],
            },
            16: {
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
            5: {
                'f0_pred_shift_mean': [0.950900, 3.447467, 5.587967, 6.731367],
                'f0_pred_shift_median': [0.950900, 3.447467, 5.587967, 6.731367],
                'f0_pred_shift_stddev': [None, None, None, None],
                'f0_shift': [0.0, 8.0, 16.0, 24.0],
            },
            11: {
                'f0_pred_shift_mean': [0.124300, 0.797400, 1.741900, 2.471900],
                'f0_pred_shift_median': [0.124300, 0.797400, 1.741900, 2.471900],
                'f0_pred_shift_stddev': [None, None, None, None],
                'f0_shift': [0.0, 8.0, 16.0, 24.0],
            },
            16: {
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
    human data, as summarized by Meddis and O'Mard (1997, JASA) (Fig 8B)
    '''
    pass


def get_human_results_dict_altphasecomplexes():
    '''
    Returns psychophysical results dictionary of Shackleton and Carlyon (1994, JASA)
    human data (Fig 3)
    '''
    pass
