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
    pass


def get_human_results_dict_freqshiftedcomplexes():
    '''
    Returns psychophysical results dictionary of Moore and Moore (2003, JASA)
    human data (Fig 3)
    '''
    pass


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
