from multiprocessing import Pool
from functools import partial
import sys
import json
import glob
import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import norm


def merge_result_dicts(result_dict_list, label = ''):
    ''' Helper function that only gets called from Jupyter notebook '''
    ''' Used to merge psychophysics results across random initializations '''
    merged_dict = {}
    for key in result_dict_list[0].keys():
        merged_dict[key] = []
    for rd in result_dict_list:
        for key in rd.keys():
            merged_dict[key].append(rd[key])
    merged_dict['lharm_list'] = np.array(merged_dict['lharm_list'])
    merged_dict['lharm_list'] = np.mean(merged_dict['lharm_list'], axis=0)
    merged_dict['phase_list'] = np.array(merged_dict['phase_list'])
    merged_dict['phase_list'] = np.mean(merged_dict['phase_list'], axis=0)
    merged_dict['threshold_means'] = np.array(merged_dict['threshold_means'])
    merged_dict['threshold_stdev'] = np.std(merged_dict['threshold_means'], axis=0)
    merged_dict['threshold_means'] = np.mean(merged_dict['threshold_means'], axis=0)
    merged_dict['label'] = label
    return merged_dict


def append_bernox_result_dict(result_dict_list):
    ''' Helper function that only gets called from Jupyter notebook '''
    ''' Creates psychopsycs result dictionary from Bernstein & Oxenham (2005) data '''
    bernox_sine_low = [6.011, 4.6697, 4.519, 1.4067, 0.66728, 0.40106]
    bernox_rand_low = [12.77, 10.0439, 8.8367, 1.6546, 0.68939, 0.515]
    bernox_sd_sine_low = [2.2342, 0.90965, 1.5508, 0.68285, 0.247, 0.13572]
    bernox_sd_rand_low = [4.7473, 3.2346, 2.477, 0.85577, 0.2126, 0.19908]
    lharm_low = 1560 / np.array([50, 75, 100, 150, 200, 300])
    bernox_lowspec = {}
    bernox_lowspec['lharm_list'] = lharm_low
    bernox_lowspec['phase_list'] = [0, 1]
    bernox_lowspec['threshold_means'] = np.array([bernox_sine_low, bernox_rand_low])
    bernox_lowspec['threshold_stdev'] = np.array([bernox_sd_sine_low, bernox_sd_rand_low])
    bernox_lowspec['label'] = 'bernox_lowspec'

    bernox_sine_high = [5.5257, 5.7834, 4.0372, 1.7769, 0.88999, 0.585]
    bernox_rand_high = [13.4933, 12.0717, 11.5717, 6.1242, 0.94167, 0.53161]
    bernox_sd_sine_high = [2.0004, 1.4445, 1.1155, 1.0503, 0.26636, 0.16206]
    bernox_sd_rand_high = [3.4807, 2.3967, 2.3512, 3.2997, 0.37501, 0.24618]
    lharm_high = 3280 / np.array([100, 150, 200, 300, 400, 600])
    bernox_highspec = {}
    bernox_highspec['lharm_list'] = lharm_high
    bernox_highspec['phase_list'] = [0, 1]
    bernox_highspec['threshold_means'] = np.array([bernox_sine_high, bernox_rand_high])
    bernox_highspec['threshold_stdev'] = np.array([bernox_sd_sine_high, bernox_sd_rand_high])
    bernox_highspec['label'] = 'bernox_highspec'

    bernox_sine = np.mean(np.array([bernox_sine_low, bernox_sine_high]), axis=0)
    bernox_rand = np.mean(np.array([bernox_rand_low, bernox_rand_high]), axis=0)
    bernox_sd_sine = np.mean(np.array([bernox_sd_sine_low, bernox_sd_sine_high]), axis=0)
    bernox_sd_rand = np.mean(np.array([bernox_sd_rand_low, bernox_sd_rand_high]), axis=0)
    lharm = np.mean(np.concatenate((lharm_low[:, np.newaxis], lharm_high[:, np.newaxis]), axis=1), axis=1)
    bernox = {}
    bernox['lharm_list'] = lharm
    bernox['phase_list'] = [0, 1]
    bernox['threshold_means'] = np.array([bernox_sine, bernox_rand])
    bernox['threshold_stdev'] = np.array([bernox_sd_sine, bernox_sd_rand])
    bernox['label'] = 'Humans'
    
    result_dict_list.extend([bernox])


def f0s_to_labels(f0s, mult=8):
    ''' mult=1 --> half-semitone bins; mult=8 --> 1/16 semitone bins '''
    f0_min = 100.
    bins = np.arange(0, mult * 32 + 1)
    f0_support = f0_min * 2 ** (bins / (mult * 24))
    f0_support = f0_support.reshape((1, -1))
    support_length = f0_support.shape[1]
    bin_labels = np.tile(np.reshape(f0s, (-1, 1)), (1, support_length))
    bin_labels = np.argmin(np.abs(bin_labels - f0_support), axis = 1)
    return bin_labels


def get_val_acc_from_json(fn):
    ''' Calculate validation statistics from specified output json file'''
    with open(fn, 'r') as outfile:
        output_dict = json.load(outfile)
    f0s = np.array(output_dict['f0'])
    val_idx = f0s > 0
    if not np.sum(val_idx) == f0s.shape[0]:
        print('WARNING: JSON FILE INCLUDES INVALID EXAMPLES')
    f0s = f0s[val_idx]
    f0s_predicted = np.array(output_dict['f0_predicted'])[val_idx]
    try:
        labels = np.array(output_dict['labels'])[val_idx]
    except:
        labels = f0s_to_labels(f0s)
    labels_predicted = np.array(output_dict['labels_predicted'])[val_idx]
    acc_pct = 100*np.sum(labels == labels_predicted) / labels.shape[0]
    mean_pct_error = np.mean(np.abs(100 * (f0s_predicted - f0s) / f0s))
    median_pct_error = np.median(np.abs(100 * (f0s_predicted - f0s) / f0s))
    print(acc_pct, mean_pct_error, median_pct_error)
    return acc_pct, mean_pct_error, median_pct_error


def generate_validation_lists_from_regex(json_regex):
    ''' Returns list of checkpoint numnbers and validation statistics
        Inputs
            json_regex: regex for validation set output files
        Returns
            ckpt_num_list: list of checkpoint numbers
            val_acc_list: validation accuracy list (%)
            mean_err_list: mean f0 error list (%)
            median_err_list: median f0 error list (%)
    '''
    fn_list = sorted(glob.glob(json_regex))
    ckpt_num_list = []
    for fn in fn_list:
        idx = fn.rfind('.ckpt-')
        idx_end = fn.rfind('_EVAL')
        cn_str = fn[(idx+6):idx_end]#fn[(idx+6):(idx+8)]
        if '_' in cn_str: cn_str = cn_str[0]
        ckpt_num_list.append(int(cn_str))
    ckpt_num_list = np.unique(ckpt_num_list)
    val_acc_list = np.zeros(ckpt_num_list.shape)
    mean_err_list = np.zeros(ckpt_num_list.shape)
    median_err_list = np.zeros(ckpt_num_list.shape)
    for fn in fn_list:
        idx = fn.rfind('.ckpt-')
        idx_end = fn.rfind('_EVAL')
        cn_str = fn[(idx+6):idx_end]#fn[(idx+6):(idx+8)]
        if '_' in cn_str: cn_str = cn_str[0]
        cn = int(cn_str)
        cni = np.where(ckpt_num_list == cn)
        val_acc, mean_err, median_err = get_val_acc_from_json(fn)
        val_acc_list[cni] = val_acc
        mean_err_list[cni] = mean_err
        median_err_list[cni] = median_err
    return ckpt_num_list, val_acc_list, mean_err_list, median_err_list


def identify_best_checkpoint(json_regex, metric='val_acc'):
    ckpt_list, val_list, mean_list, median_list = generate_validation_lists_from_regex(json_regex)
    try:
        if metric == 'val_acc':
            idx = np.argmax(val_list)
        elif metric == 'mean_err':
            idx = np.argmin(mean_list)
        else:
            idx = np.argmin(median_list)
        best_ckpt = ckpt_list[idx]
        val_acc = val_list[idx]
        mean_err = mean_list[idx]
        median_err = median_list[idx]
    except:
        print("------> ERROR OCCURRED <------")
        best_ckpt = None
        val_acc = None
        mean_err = None
        median_err = None
    return best_ckpt, val_acc, mean_err, median_err


def calculate_thresholds_for_best_checkpoint(valid_regex, formattable_diag_fn):
    best_ckpt, val_acc, mean_err, median_err = identify_best_checkpoint(valid_regex)
    assert(not best_ckpt == None)
    
    diag_fn = formattable_diag_fn.format(best_ckpt)
    min_f0 = 0; max_f0 = 250; min_snr = -np.inf
    params = {}
    
    with open(diag_fn, 'r') as outfile:
        output_dict = json.load(outfile)
        
    phase_list, lharm_list, threshold_means, threshold_stdev = main_experiment(
        output_dict, min_f0=min_f0, max_f0=max_f0, params=params, min_snr=min_snr)
    
    result_dict = {}
    result_dict['phase_list'] = phase_list.tolist()
    result_dict['lharm_list'] = lharm_list.tolist()
    result_dict['threshold_means'] = threshold_means.tolist()
    result_dict['threshold_stdev'] = threshold_stdev.tolist()
    result_dict['diag_fn'] = diag_fn
    result_dict['best_ckpt'] = int(best_ckpt)
    result_dict['val_acc'] = float(val_acc)
    result_dict['mean_err'] = float(mean_err)
    result_dict['median_err'] = float(median_err)
    
    result_fn = formattable_diag_fn.format('BEST')
    with open(result_fn, 'w') as outfile:
        print('Writing:', result_fn, 'acc: {:0.4}, median_err: {:0.4}'.format(val_acc, median_err))
        json.dump(result_dict, outfile)
    
    return phase_list, lharm_list, threshold_means, threshold_stdev
        

def get_paramdf_from_dict(output_dict, phase_cap = np.inf):
    d = {}
    phase_array = np.array(output_dict['phase_mode'][:])
    keep_idx = phase_array <= phase_cap
    d['f0'] = np.array(output_dict['f0'][:])[keep_idx]
    d['f0_pred'] = np.array(output_dict['f0_predicted'][:])[keep_idx]
    d['low_harm'] = np.array(output_dict['low_harm'][:])[keep_idx]
    d['phase_mode'] = phase_array[keep_idx]
    return pd.DataFrame.from_dict(d), keep_idx


def get_pyschoacoustic_results(f0_true, f0_pred, max_pct_diff = 10, decision_noise = 1e-9):
    pct_diffs = [] # List of percent differences between f01 and f02
    judgments = [] # List of booleans indicating if model predicts f01 > f02
    for itr1 in range(f0_true.shape[0]):
        for itr2 in range(itr1 + 1, f0_true.shape[0]):
            pct = 100. * (f0_true[itr1] - f0_true[itr2]) / f0_true[itr1]
            if np.abs(pct) <= max_pct_diff:
                perceived_pct = 100*(f0_pred[itr1] - f0_pred[itr2]) / f0_pred[itr1]
                pct_diffs.append(pct)
                judgments.append(perceived_pct > decision_noise * np.random.randn())
    return np.array(pct_diffs), np.array(judgments)


def get_threhsold_from_results(pct_diffs, judgments, threshold_value):
    d = fit_norm_cdf(pct_diffs, judgments)
    return norm(0, *d['popt']).ppf(threshold_value)


def normcdf(x, sigma):
    return norm(0, sigma).cdf(x) ### NOTE THAT mu is fixed at 0


def fit_norm_cdf(pct_diffs, judgments):
    # Bin the semitone_diffs and calculate judgment accuracies
    bins = np.arange(np.min(pct_diffs), np.max(pct_diffs), 1e-4)
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
    # Fit norm cdf function to psychometric curve
    popt, pcov = optimize.curve_fit(normcdf, bins, bin_means)
    output = {'popt':popt, 'pcov':pcov,
              'bins':bins, 'bin_means':bin_means, 'bin_counts':bin_counts}
    return output


def bootstrap(xvals, yvals):
    idx = np.random.randint(0, xvals.shape[0], (xvals.shape[0],))
    return xvals[idx], yvals[idx]


def get_bootstrapped_threhsold(pct_diffs, judgments, threshold_value, seed=None):
    np.random.seed(seed)
    p, j = (pct_diffs, judgments)#bootstrap(pct_diffs, judgments)
    return get_threhsold_from_results(p, j, threshold_value)


def get_lharm_threshold(paramdf_phase, max_pct_diff, decision_noise, threshold_value,
                        bootstrap_repeats, lharm_list, par_index):
    lharm = lharm_list[par_index]
    dfx = paramdf_phase.loc[(paramdf_phase['low_harm'] == lharm)]
    f0_true = dfx['f0'].values
    f0_pred = dfx['f0_pred'].values
    pct_diffs, judgments = get_pyschoacoustic_results(f0_true, f0_pred, max_pct_diff, decision_noise)
    th = []
    for itr0 in range(bootstrap_repeats):
        th.append(get_bootstrapped_threhsold(pct_diffs, judgments, threshold_value))#, seed=itr0))
    return (par_index, np.mean(th), np.std(th))


def main_experiment(output_dict, min_f0 = 0, max_f0 = 250, params = {}, min_snr = -np.inf):
    ''' Parallelize over low harmonic numbers '''
    
    # Extract parameters
    threshold_value = params.get('threshold_value', 0.707)
    bootstrap_repeats = params.get('bootstrap_repeats', 1)
    max_pct_diff = params.get('max_pct_diff', 6)
    decision_noise = params.get('decision_noise', 1e-9)
    
    # Collect metadata and true / model-predicted F0 values
    df, phase_idx = get_paramdf_from_dict(output_dict, phase_cap = 1)
    dfn = df.loc[(df['f0'] >= min_f0) & (df['f0'] < max_f0)]
    lharm_list = np.unique(dfn['low_harm'].values)
    phase_list = np.array([0, 1])
    
    # Set up output arrays and pool of parallel workers
    threshold_means = np.zeros((len(phase_list), len(lharm_list)))
    threshold_stdev = np.zeros((len(phase_list), len(lharm_list)))
    with Pool(processes = len(lharm_list)) as pool:    
        # Use F0 predictions to calculate discrimination thresholds
        # for different phase / low_harm number conditions
        for phase_idx, phase in enumerate(phase_list):
            paramdf_phase = dfn.loc[(dfn['phase_mode'] == phase)]

            func = partial(get_lharm_threshold, paramdf_phase, max_pct_diff, decision_noise, threshold_value,
                           bootstrap_repeats, lharm_list)
            results = pool.map(func, range(0, len(lharm_list)))
            for (lharm_idx, mn, sd) in results:
                threshold_means[phase_idx, lharm_idx] = mn
                threshold_stdev[phase_idx, lharm_idx] = sd
    
    return phase_list, lharm_list, threshold_means, threshold_stdev


def main_multi(json_regex, min_f0 = 0, max_f0 = 250, params = {}, min_snr = -np.inf):
    ''' Call main_experiment function for each file that matches json_regex '''
    json_filenames = sorted(glob.glob(json_regex))
    
    threshold_means_list = []
    threshold_stdev_list = []
    
    for itr0, json_fn in enumerate(json_filenames):
        with open(json_fn, 'r') as outfile:
            print('### Processing file {} of {} ###'.format(itr0+1, len(json_filenames)))
            output_dict = json.load(outfile)
            print(json_fn)
        
            phase_list, lharm_list, threshold_means, threshold_stdev = main_experiment(
                output_dict, min_f0, max_f0, params, min_snr)

            threshold_means_list.append(threshold_means.copy())
            threshold_stdev_list.append(threshold_stdev.copy())
    
    threshold_means_stack = np.stack(threshold_means_list)
    threshold_stdev_stack = np.stack(threshold_stdev_list)
    thresh_means = np.mean(threshold_means_stack, axis=0)
    thresh_stdev = np.std(threshold_means_stack, axis=0)

    return phase_list, lharm_list, thresh_means, thresh_stdev, threshold_means_stack, threshold_stdev_stack


if __name__ == "__main__":
    # SET UP SCRIPT TO RUN FROM COMMAND LINE

    if not len(sys.argv) == 2:
        print('COMMAND LINE USAGE: run <script_name.py> <job_id>')
        assert(False)

    # Use job_id from command line argument to select filenames, dataset, and inputs
    job_id = int(sys.argv[1])
    
    list_ckpt_fn = [
        # 0-2 Species 2
        '/om/user/msaddler/models_pitch50ms_bez2018/arch160/NRTW-jwss_train_CF50-SR70-sp2_filt00_30-90dB_meanrates0.ckpt',
        '/om/user/msaddler/models_pitch50ms_bez2018/arch160/NRTW-jwss_train_CF50-SR70-sp2_filt00_30-90dB_meanrates1.ckpt',
        '/om/user/msaddler/models_pitch50ms_bez2018/arch160/NRTW-jwss_train_CF50-SR70-sp2_filt00_30-90dB_meanrates2.ckpt',
        # 3-5 Species 1
        '/om/user/msaddler/models_pitch50ms_bez2018/arch160/NRTW-jwss_train_CF50-SR70-sp1_filt00_30-90dB_meanrates0.ckpt',
        '/om/user/msaddler/models_pitch50ms_bez2018/arch160/NRTW-jwss_train_CF50-SR70-sp1_filt00_30-90dB_meanrates1.ckpt',
        '/om/user/msaddler/models_pitch50ms_bez2018/arch160/NRTW-jwss_train_CF50-SR70-sp1_filt00_30-90dB_meanrates2.ckpt',
        # 6-8 Species 2, cohc00
        '/om/user/msaddler/models_pitch50ms_bez2018/arch160/NRTW-jwss_train_CF50-SR70-sp2-cohc00_filt00_30-90dB_meanrates0.ckpt',
        '/om/user/msaddler/models_pitch50ms_bez2018/arch160/NRTW-jwss_train_CF50-SR70-sp2-cohc00_filt00_30-90dB_meanrates1.ckpt',
        '/om/user/msaddler/models_pitch50ms_bez2018/arch160/NRTW-jwss_train_CF50-SR70-sp2-cohc00_filt00_30-90dB_meanrates2.ckpt',
    ]
    valid_regex = list_ckpt_fn[job_id] + '-*_EVAL_VALID.json'
    formattable_diag_fn = list_ckpt_fn[job_id] + '-{}_EVAL_DIAGNOSTIC_bernox2005stim_2018-11-29-1930_thresh33dB.json'
    
    print(valid_regex)
    print(formattable_diag_fn)
    phase_list, lharm_list, threshold_means, threshold_stdev = calculate_thresholds_for_best_checkpoint(valid_regex, formattable_diag_fn)
