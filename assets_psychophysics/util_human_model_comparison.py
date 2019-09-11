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
    results_dict = {
        'f0_ref': {
            '100.0': {
                'f0_max': 100.0,
                'f0_min': 100.0,
                'mistuned_harm': {
                    '1': {
                        'mistuned_harm': 1,
                        'mistuned_pct': [3.0],
                        'f0_pred_pct_mean': [0.1505],
                        'f0_pred_pct_median': [0.1505],
                        'f0_pred_pct_stddev': [None],
                    },
                    '2': {
                        'mistuned_harm': 2,
                        'mistuned_pct': [3.0],
                        'f0_pred_pct_mean': [0.7229],
                        'f0_pred_pct_median': [0.7229],
                        'f0_pred_pct_stddev': [None],
                    },
                    '3': {
                        'mistuned_harm': 3,
                        'mistuned_pct': [3.0],
                        'f0_pred_pct_mean': [0.3954],
                        'f0_pred_pct_median': [0.3954],
                        'f0_pred_pct_stddev': [None],
                    },
                    '4': {
                        'mistuned_harm': 4,
                        'mistuned_pct': [3.0],
                        'f0_pred_pct_mean': [0.7252],
                        'f0_pred_pct_median': [0.7252],
                        'f0_pred_pct_stddev': [None],
                    },
                    '5': {
                        'mistuned_harm': 5,
                        'mistuned_pct': [3.0],
                        'f0_pred_pct_mean': [0.5120],
                        'f0_pred_pct_median': [0.5120],
                        'f0_pred_pct_stddev': [None],
                    },
                    '6': {
                        'mistuned_harm': 6,
                        'mistuned_pct': [3.0],
                        'f0_pred_pct_mean': [0.2713],
                        'f0_pred_pct_median': [0.2713],
                        'f0_pred_pct_stddev': [None],
                    },
                },
            },
            '200.0': {
                'f0_max': 200.0,
                'f0_min': 200.0,
                'mistuned_harm': {
                    '1': {
                        'mistuned_harm': 1,
                        'mistuned_pct': [3.0],
                        'f0_pred_pct_mean': [0.7842],
                        'f0_pred_pct_median': [0.7842],
                        'f0_pred_pct_stddev': [None],
                    },
                    '2': {
                        'mistuned_harm': 2,
                        'mistuned_pct': [3.0],
                        'f0_pred_pct_mean': [0.8426],
                        'f0_pred_pct_median': [0.8426],
                        'f0_pred_pct_stddev': [None],
                    },
                    '3': {
                        'mistuned_harm': 3,
                        'mistuned_pct': [3.0],
                        'f0_pred_pct_mean': [0.7530],
                        'f0_pred_pct_median': [0.7530],
                        'f0_pred_pct_stddev': [None],
                    },
                    '4': {
                        'mistuned_harm': 4,
                        'mistuned_pct': [3.0],
                        'f0_pred_pct_mean': [0.5125],
                        'f0_pred_pct_median': [0.5125],
                        'f0_pred_pct_stddev': [None],
                    },
                    '5': {
                        'mistuned_harm': 5,
                        'mistuned_pct': [3.0],
                        'f0_pred_pct_mean': [0.2735],
                        'f0_pred_pct_median': [0.2735],
                        'f0_pred_pct_stddev': [None],
                    },
                    '6': {
                        'mistuned_harm': 6,
                        'mistuned_pct': [3.0],
                        'f0_pred_pct_mean': [0.0907],
                        'f0_pred_pct_median': [0.0907],
                        'f0_pred_pct_stddev': [None],
                    },
                },
            },
            '400.0': {
                'f0_max': 400.0,
                'f0_min': 400.0,
                'mistuned_harm': {
                    '1': {
                        'mistuned_harm': 1,
                        'mistuned_pct': [3.0],
                        'f0_pred_pct_mean': [0.6343],
                        'f0_pred_pct_median': [0.6343],
                        'f0_pred_pct_stddev': [None],
                    },
                    '2': {
                        'mistuned_harm': 2,
                        'mistuned_pct': [3.0],
                        'f0_pred_pct_mean': [0.8418],
                        'f0_pred_pct_median': [0.8418],
                        'f0_pred_pct_stddev': [None],
                    },
                    '3': {
                        'mistuned_harm': 3,
                        'mistuned_pct': [3.0],
                        'f0_pred_pct_mean': [0.6319],
                        'f0_pred_pct_median': [0.6319],
                        'f0_pred_pct_stddev': [None],
                    },
                    '4': {
                        'mistuned_harm': 4,
                        'mistuned_pct': [3.0],
                        'f0_pred_pct_mean': [0.3913],
                        'f0_pred_pct_median': [0.3913],
                        'f0_pred_pct_stddev': [None],
                    },
                    '5': {
                        'mistuned_harm': 5,
                        'mistuned_pct': [3.0],
                        'f0_pred_pct_mean': [0.1475],
                        'f0_pred_pct_median': [0.1475],
                        'f0_pred_pct_stddev': [None],
                    },
                    '6': {
                        'mistuned_harm': 6,
                        'mistuned_pct': [3.0],
                        'f0_pred_pct_mean': [0.0274],
                        'f0_pred_pct_median': [0.0274],
                        'f0_pred_pct_stddev': [None],
                    },
                },
            },
        },
        'f0_ref_list': [100.0, 200.0, 400.0],
        'f0_ref_width': 0.0,
    }
    return results_dict


def get_human_results_dict_altphasecomplexes(average_conditions=True):
    '''
    Returns psychophysical results dictionary of Shackleton and Carlyon (1994, JASA)
    human data (Fig 3)
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
    if average_conditions:
        return results_dict
    else:
        return [results_dict_subjectTS, results_dict_subjectJS, results_dict_subjectSD, results_dict_subjectRB]


def get_mistuned_harmonics_bar_graph_results_dict(results_dict, mistuned_pct=3.0,
                                                  pitch_shift_key='f0_pred_pct_median',
                                                  harmonic_list=[1,2,3,4,5,6]):
    '''
    This helper function parses a results_dict from the Moore et al. (1985, JASA)
    mistuned harmonics experiment into a smaller bar_graph_results_dict, which
    allows for easier plotting of the Meddis and O'Mard (1997, JASA) summary bar
    graph (Fig 8B)
    '''
    f0_ref_list = results_dict['f0_ref_list']
    bar_graph_results_dict = {}
    for harm in harmonic_list:
        harm_key = str(harm)
        bar_graph_results_dict[harm_key] = {
            'f0_ref': [],
            pitch_shift_key: [],
        }
        for f0_ref in f0_ref_list:
            f0_ref_key = str(f0_ref)
            sub_results_dict = results_dict['f0_ref'][f0_ref_key]['mistuned_harm'][harm_key]
            mp_idx = sub_results_dict['mistuned_pct'].index(mistuned_pct)
            pitch_shift = sub_results_dict[pitch_shift_key][mp_idx]
            bar_graph_results_dict[harm_key]['f0_ref'].append(f0_ref)
            bar_graph_results_dict[harm_key][pitch_shift_key].append(pitch_shift)
    return bar_graph_results_dict


def compare_bernox2005(results_dict, human_results_dict):
    '''
    '''
    return 0


def compare_transposedtones(results_dict, human_results_dict):
    '''
    '''
    return 0


def compare_freqshiftedcomplexes(results_dict, human_results_dict):
    '''
    '''
    return 0


def compare_mistunedharmonics(results_dict, human_results_dict):
    '''
    '''
    return 0


def compare_altphasecomplexes(results_dict, human_results_dict):
    '''
    '''
    return 0
