########### RUN f0dl_bernox.py ###########

import sys
import os
import json
import numpy as np
import glob
import f0dl_bernox
%matplotlib inline
import matplotlib.pyplot as plt
import scipy.stats

json_regex = '/om2/user/msaddler/pitchnet/saved_models/PND_v04_JWSS*_classification*/EVAL_bernox2005_FixedFilter_bestckpt.json'


json_fn_list = sorted(glob.glob(json_regex))
model_name_list = []
for json_fn in json_fn_list:
    model_name = json_fn.replace('/om2/user/msaddler/pitchnet/saved_models/', '')
    model_name = model_name[:model_name.rfind('/')]
    model_name_list.append(model_name)
print('found {} files'.format(len(json_fn_list)))

results_dict_list = []
confmat_dict_list = []
for fn, mn in zip(json_fn_list, model_name_list):
    print(fn)
    if 'classification' in mn:
        f0_label_pred_key='f0_label:labels_pred'
        f0_label_true_key='f0_label:labels_true'
    else:
        f0_label_pred_key = 'f0_lognormal:labels_pred'
        f0_label_true_key = 'f0_lognormal:labels_true'
    
    metadata_key_list=['low_harm', 'phase_mode', 'f0']
    if 'FixedFilter' in fn: metadata_key_list = metadata_key_list + ['base_f0']
    results_dict = f0dl_bernox.run_f0dl_experiment(fn, max_pct_diff=1, bin_width=5e-2,
                                                   f0_label_pred_key=f0_label_pred_key,
                                                   f0_label_true_key=f0_label_true_key,
                                                   use_empirical_f0dl_if_possible=False,
                                                   metadata_key_list=metadata_key_list,
                                                   f0_min=-np.inf, f0_max=np.inf)
    confmat_dict = f0dl_bernox.compute_confusion_matrices(json_fn,
                                                          f0_label_pred_key=f0_label_pred_key,
                                                          f0_label_true_key=f0_label_true_key)
    results_dict_list.append(results_dict)
    confmat_dict_list.append(confmat_dict)
    
human_results_dict = f0dl_bernox.bernox2005_human_results_dict()


########### PLOT THRESHOLDS f0dl_bernox.py ###########

def make_threshold_plot(ax, results_dict, title_str=None, legend_on=True,
                        sine_plot_kwargs={}, rand_plot_kwargs={}):
    phase_mode_list = np.array(results_dict['phase_mode'])
    low_harm_list = np.array(results_dict['low_harm'])
    f0dl_list = np.array(results_dict['f0dl'])
    unique_phase_modes = np.unique(phase_mode_list)
    for phase_mode in unique_phase_modes:
        x = low_harm_list[phase_mode_list == phase_mode]
        y = f0dl_list[phase_mode_list == phase_mode]
        
        if phase_mode == 0:
            plot_kwargs = {'label': 'sine', 'color': 'b', 'ls':'-', 'lw':2, 'ms':8, 'marker':''}
            plot_kwargs.update(sine_plot_kwargs)
        else:
            plot_kwargs = {'label': 'rand', 'color': 'b', 'ls':'--', 'lw':2, 'ms':8, 'marker':''}
            plot_kwargs.update(rand_plot_kwargs)
        
        if not legend_on: plot_kwargs['label'] = None
        ax.plot(x, y, **plot_kwargs)

    ax.set_yscale('log')
    ax.set_ylim([1e-1, 3e2])
    ax.set_xlim([0, 32])
    ax.set_xlabel('Lowest harmonic number', fontsize=10)
    ax.set_ylabel('F0 discrimination threshold (%F0)', fontsize=10)
    if title_str is not None: ax.set_title(model_name, fontsize=10)
    if legend_on: ax.legend(loc='lower right', frameon=False, fontsize=10)


NCOLS = 3
NROWS = int(np.ceil(len(results_dict_list) / NCOLS))
fig, ax_arr = plt.subplots(nrows=NROWS, ncols=NCOLS, figsize=(4*NCOLS, 3*NROWS))
ax_arr = ax_arr.flatten()

for idx, (results_dict, model_name) in enumerate(zip(results_dict_list, model_name_list)):
    ax = ax_arr[idx]
    make_threshold_plot(ax, human_results_dict, title_str=None, legend_on=False,
                        sine_plot_kwargs={'color':'r', 'lw':0.5}, rand_plot_kwargs={'color':'r', 'lw':0.5})
    make_threshold_plot(ax, results_dict, title_str=model_name, legend_on=True,
                        sine_plot_kwargs={'color':'k', 'lw':2}, rand_plot_kwargs={'color':'b', 'lw':2})

for idx in range(len(results_dict_list), len(ax_arr)): ax_arr[idx].axis('off')

plt.tight_layout()
plt.show()


########### PLOT PSYCHOMETRIC FUNCTIONS f0dl_bernox.py ###########

IDX = 0
results_dict = results_dict_list[IDX]
model_name = model_name_list[IDX]

phase_mode_list = np.array(results_dict['phase_mode'])
low_harm_list = np.array(results_dict['low_harm'])
unique_phase_mode_list = np.unique(phase_mode_list)
unique_low_harm_list = np.unique(low_harm_list)

NCOLS = 4
NROWS = int(np.ceil(len(unique_low_harm_list) / NCOLS))
fig, ax_arr = plt.subplots(nrows=NROWS, ncols=NCOLS, figsize=(5*NCOLS, 2.5*NROWS))
ax_arr = ax_arr.flatten()
for idx, low_harm in enumerate(unique_low_harm_list):
    ax = ax_arr[idx]
    
    for phase_mode in unique_phase_mode_list:
        
        rdi = np.squeeze(np.argwhere(np.logical_and(phase_mode_list==phase_mode, low_harm_list==low_harm)))
        x = results_dict['psychometric_function'][rdi]['bins']
        y = results_dict['psychometric_function'][rdi]['bin_means']
        
        normcdf = lambda x, sigma: scipy.stats.norm(results_dict['psychometric_function'][rdi]['mu'], sigma).cdf(x)
        
        if phase_mode == 0:
            color = 'k'
            label = 'sine'
            ls = '-'
        else:
            color = 'b'
            label = 'rand'
            ls = '-'
        
        ax.plot(x, y, color=color, ls=ls, label=label)
        ax.plot(x, normcdf(x, results_dict['psychometric_function'][rdi]['sigma']), color=color, ls='--')
    
    ax.plot(x, np.ones_like(x)*results_dict['psychometric_function'][rdi]['threshold_value'], 'r--')
    ax.set_ylim([0, 1])
    ax.set_xlim([np.min(x), np.max(x)])
    ax.set_xlabel('%F0 difference')
    ax.set_ylabel('Proportion correct')
    
    ax.legend(loc='lower right', frameon=False, handlelength=0.5)
    ax.text(np.min(x), 1-.05, '  Low harm = {}'.format(low_harm), horizontalalignment='left', verticalalignment='top')

for idx in range(len(unique_low_harm_list), len(ax_arr)): ax_arr[idx].axis('off')
    
plt.tight_layout()
plt.show()


############ PLOT CONFUSION MATRICES f0dl_bernox.py ############

def make_confusion_matrix_plot(ax, confmat_dict, confmat_idx, fontsize=8,
                               title_str=None, log_scale=True,
                               sine_plot_kwargs={}, rand_plot_kwargs={}):
    
    f0_true = confmat_dict['f0_true'][confmat_idx]
    f0_pred = confmat_dict['f0_pred'][confmat_idx]
    
    if confmat_dict['phase_mode'][confmat_idx] == 0:
        plot_kwargs = {'color': 'k', 'ls':'', 'ms':2, 'marker':'.'}
        plot_kwargs.update(sine_plot_kwargs)
    else:
        plot_kwargs = {'color': 'b', 'ls':'', 'ms':2, 'marker':'.'}
        plot_kwargs.update(rand_plot_kwargs)
    
    if title_str is None:
        title_str = 'low_harm={}, phase_mode={}'.format(confmat_dict['low_harm'][confmat_idx],
                                                        confmat_dict['phase_mode'][confmat_idx])
    
    ax.plot(f0_true, f0_pred, **plot_kwargs)
    ax.set_xlim(confmat_dict['f0_min'], confmat_dict['f0_max'])
    ax.set_ylim(confmat_dict['f0_min'], confmat_dict['f0_max'])
    ax.set_xlabel('f0_true', fontsize=fontsize)
    ax.set_ylabel('f0_pred', fontsize=fontsize)
    ax.set_title(title_str, fontsize=fontsize)
    if log_scale:
        ax.set_xscale('log')
        ax.set_yscale('log')


IDX = 0
cfmd = confmat_dict_list[IDX]
model_name = model_name_list[IDX]

print(model_name)

phase_mode_list = np.array(confmat_dict['phase_mode'])
low_harm_list = np.array(confmat_dict['low_harm'])
unique_phase_mode_list = np.unique(phase_mode_list)
unique_low_harm_list = np.unique(low_harm_list)

NAXES = len(unique_low_harm_list) * len(unique_phase_mode_list)
NCOLS = 8
NROWS = int(np.ceil(NAXES / NCOLS))
fig, ax_arr = plt.subplots(nrows=NROWS, ncols=NCOLS, figsize=(2.5*NCOLS, 2.5*NROWS))
ax_arr = ax_arr.flatten()
ax_idx = 0
for low_harm in unique_low_harm_list:
    for phase_mode in unique_phase_mode_list:
        ax = ax_arr[ax_idx]
        ax_idx = ax_idx + 1
        rdi = np.squeeze(np.argwhere(np.logical_and(phase_mode_list==phase_mode, low_harm_list==low_harm)))
        make_confusion_matrix_plot(ax, cfmd, rdi)
        
for idx in range(NAXES, len(ax_arr)): ax_arr[idx].axis('off')

plt.tight_layout()
plt.show()


############ RUN AND PLOT FREQUENCY-SHIFTED COMPLEXES (Moore & Moore 2003) ############

def compute_f0_shifts(expt_dict):
    f0_true = expt_dict['f0']
    f0_pred = expt_dict['f0_pred']
    expt_dict['f0_pred_shift'] = (f0_pred - f0_true) / f0_true
    return expt_dict


def compute_shift_curve(expt_dict, filter_key, filter_value, f0_min=80.0, f0_max=1e3):
    indexes = expt_dict[filter_key] == filter_value
    indexes = np.logical_and(indexes, np.logical_and(expt_dict['f0'] >= f0_min, expt_dict['f0'] <= f0_max))
    
    f0_shift = expt_dict['f0_shift'][indexes]
    f0_pred_shift = expt_dict['f0_pred_shift'][indexes]
    
    f0_shift_unique = np.unique(f0_shift)
    f0_pred_shift_mean = np.zeros_like(f0_shift_unique)
    f0_pred_shift_median = np.zeros_like(f0_shift_unique)
    f0_pred_shift_stddev = np.zeros_like(f0_shift_unique)
    
    for idx, f0_shift_value in enumerate(f0_shift_unique):
        current_value_indexes = f0_shift == f0_shift_value
        f0_pred_shift_mean[idx] = np.mean(f0_pred_shift[current_value_indexes])
        f0_pred_shift_median[idx] = np.median(f0_pred_shift[current_value_indexes])
        f0_pred_shift_stddev[idx] = np.std(f0_pred_shift[current_value_indexes])
    
    return f0_shift_unique, f0_pred_shift_mean, f0_pred_shift_median, f0_pred_shift_stddev


def get_mooremoore2003_results_dict(expt_dict, filter_key='spectral_envelope_centered_harmonic',
                                    key_to_label_map={5:'RES', 11:'INT', 16:'UNRES'},
                                    f0_min=-np.inf, f0_max=np.inf):
    '''
    '''
    expt_dict = compute_f0_shifts(expt_dict)
    results_dict = {}
    for condition in np.unique(expt_dict[filter_key]):
        f0_shift, f0_pred_shift_mean, f0_pred_shift_median, f0_pred_shift_stddev = compute_shift_curve(
            expt_dict, filter_key, condition, f0_min=f0_min, f0_max=f0_max)
        if condition not in key_to_label_map.keys():
            key_to_label_map[condition] = condition
        results_dict[key_to_label_map[condition]] = {
            'f0_shift': 100.0 * f0_shift,
            'f0_pred_shift_mean': 100.0 * f0_pred_shift_mean,
            'f0_pred_shift_median': 100.0 * f0_pred_shift_median,
            'f0_pred_shift_stddev': 100.0 * f0_pred_shift_stddev,
        }
    return results_dict


NCOLS = 3
NROWS = int(np.ceil(len(expt_dict_list) / NCOLS))
fig, ax_arr = plt.subplots(nrows=NROWS, ncols=NCOLS, figsize=(5*NCOLS, 3*NROWS))
ax_arr = ax_arr.flatten()

for idx, (expt_dict, model_name) in enumerate(zip(expt_dict_list, model_name_list)):
    ax = ax_arr[idx]
    
    results_dict = get_mooremoore2003_results_dict(expt_dict)
    
    for key in results_dict.keys(): #['RES', 'INT', 'UNRES']:
        xval = results_dict[key]['f0_shift']
        yval = results_dict[key]['f0_pred_shift_median']
        yerr = results_dict[key]['f0_pred_shift_stddev']
        ax.plot(xval, yval, '.-', label=key)

    ax.legend(loc=2, frameon=False)
    ax.set_title(model_name)
    ax.set_xlabel('Component shift (%F0)')
    ax.set_ylabel('Shift in predicted F0 (%F0)')
    ax.set_xlim([-0.5, 24.5])
    ax.set_ylim([-4, 12])


for idx in range(len(expt_dict_list), len(ax_arr)): ax_arr[idx].axis('off')

plt.tight_layout()
plt.show()


############ RUN AND PLOT TRANSPOSED TONES (Oxenham et al. 2004) ############

import sys
import os
import json
import numpy as np
import glob
import f0dl_transposed_tones
%matplotlib inline
import matplotlib.pyplot as plt

json_regex = '/om2/user/msaddler/pitchnet/saved_models/PND_v*/EVAL_oxenham2004_080to320Hz_bestckpt.json'
json_regex = '/om2/user/msaddler/pitchnet/saved_models/PND_v04*/EVAL_oxenham2004_*_bestckpt.json'


json_fn_list = sorted(glob.glob(json_regex))
model_name_list = []
for json_fn in json_fn_list:
    model_name = json_fn.replace('/om2/user/msaddler/pitchnet/saved_models/', '')
    model_name = model_name.replace('_bestckpt.json', '')
    model_name = model_name[:model_name.rfind('/')] + '\n' + model_name[model_name.rfind('/'):]
    model_name_list.append(model_name)

metadata_key_list = [
    'f0',
    'f_carrier',
    'f_envelope',
]

results_dict_list = []
for json_fn, model_name in zip(json_fn_list, model_name_list):
    print(json_fn)
    results_dict = f0dl_transposed_tones.run_f0dl_experiment(json_fn, max_pct_diff=6, noise_stdev=1e-12, bin_width=5e-2, mu=0.0,
                        threshold_value=0.707, use_empirical_f0dl_if_possible=False,
                        f0_label_true_key='f0_label:labels_true', f0_label_pred_key='f0_label:labels_pred',
                        kwargs_f0_bins={}, kwargs_f0_octave={}, kwargs_f0_normalization={},
                        f0_ref_min=100.0, f0_ref_max=320.0, f0_ref_n_step=5,
                        metadata_key_list=['f_carrier', 'f_envelope', 'f0'])
    results_dict_list.append(results_dict)


def make_TT_threshold_plot(ax, results_dict, title_str=None, legend_on=True):
    f0_ref = np.array(results_dict['f0_ref'])
    f_carrier_list = np.array(results_dict['f_carrier'])
    f0dl_list = np.array(results_dict['f0dl'])
    unique_f_carrier_list = np.unique(f_carrier_list)
    for f_carrier in unique_f_carrier_list:
        x = f0_ref[f_carrier_list == f_carrier]
        y = f0dl_list[f_carrier_list == f_carrier]
        
        if f_carrier > 0:
            label = '{}-Hz TT'.format(int(f_carrier))
            plot_kwargs = {'label': label, 'color': 'k', 'ls':'-', 'lw':2, 'ms':8,
                           'marker':'o', 'markerfacecolor': 'w'}
            if int(f_carrier) == 10080: plot_kwargs['marker'] = 'D'
            if int(f_carrier) == 6350: plot_kwargs['marker'] = '^'
            if int(f_carrier) == 4000: plot_kwargs['marker'] = 's'
        else:
            label = 'Pure tone'
            plot_kwargs = {'label': label, 'color': 'k', 'ls':'-', 'lw':2, 'ms':8,
                           'marker':'o', 'markerfacecolor': 'k'}
            
        if not legend_on: plot_kwargs['label'] = None
        ax.plot(x, y, **plot_kwargs)

    ax.set_yscale('log')
    ax.set_ylim([1e-1, 3e1])
    ax.set_xscale('log')
    ax.set_xlim([40, 500])
    ax.set_xlabel('Frequency (Hz)', fontsize=10)
    ax.set_ylabel('Frequency difference (%)', fontsize=10)
    if title_str is not None: ax.set_title(model_name, fontsize=10)
    if legend_on: ax.legend(loc='lower right', frameon=False, fontsize=10)


NCOLS = 3
NROWS = int(np.ceil(len(results_dict_list) / NCOLS))
fig, ax_arr = plt.subplots(nrows=NROWS, ncols=NCOLS, figsize=(3.5*NCOLS, 3*NROWS))
ax_arr = ax_arr.flatten()

for idx, (results_dict, model_name) in enumerate(zip(results_dict_list, model_name_list)):
    ax = ax_arr[idx]
    make_TT_threshold_plot(ax, results_dict, title_str=model_name, legend_on=False)

for idx in range(len(results_dict_list), len(ax_arr)): ax_arr[idx].axis('off')

plt.tight_layout()
plt.show()


############ RUN AND PLOT ALT PHASE HARMONICS (Shackleton and Carlyon 1994) ############

import sys
import os
import json
import numpy as np
import glob
import f0dl_bernox
%matplotlib inline
import matplotlib.pyplot as plt


def compute_f0_pred_ratio(expt_dict):
    f0_true = expt_dict['f0']
    f0_pred = expt_dict['f0_pred']
    expt_dict['f0_pred_ratio'] = f0_pred / f0_true
    return expt_dict


def compute_histograms(expt_dict_list, phase_mode=4, f0_bin_centers=[125, 250], f0_bin_width=0.04):
    if not isinstance(expt_dict_list, list):
        expt_dict_list = [expt_dict_list]
    
    expt_dict = expt_dict_list[0]
    filter_conditions = np.unique(expt_dict['filter_fl'])
    f0 = np.unique(expt_dict['phase_mode'])
    
    f0_pred_ratio_list = []
    f0_condition_list = []
    filter_condition_list = []
    
    for filt_cond in filter_conditions:
        for f0_center in f0_bin_centers:
            f0_range = [f0_center*(1-f0_bin_width), f0_center*(1+f0_bin_width)]
            f0_pred_ratio_sublist = []
            for expt_dict in expt_dict_list:
                sub_expt_dict = f0dl_bernox.filter_expt_dict(expt_dict,
                                filter_dict={'filter_fl': filt_cond, 'f0': f0_range, 'phase_mode': phase_mode})
                sub_expt_dict = compute_f0_pred_ratio(sub_expt_dict)
                f0_pred_ratio_sublist.extend(sub_expt_dict['f0_pred_ratio'].tolist())
            
            f0_pred_ratio_list.append(f0_pred_ratio_sublist)
            filter_condition_list.append(filt_cond)
            f0_condition_list.append(f0_center)
    
    return filter_condition_list, f0_condition_list, f0_pred_ratio_list


# json_regex = '/om2/user/msaddler/pitchnet/saved_models/PND_v04_JWSS_classification*/EVAL_AltPhase_v01_bestckpt.json'
json_regex = '/om2/user/msaddler/pitchnet/saved_models/PND_v04_JWSS_halfbandpass_classification*/EVAL_AltPhase_v01_bestckpt.json'

json_fn_list = sorted(glob.glob(json_regex))
model_name_list = []
for json_fn in json_fn_list:
    model_name = json_fn.replace('/om2/user/msaddler/pitchnet/saved_models/', '')
    model_name = model_name.replace('_bestckpt.json', '')
    model_name = model_name[:model_name.rfind('/')] + '\n' + model_name[model_name.rfind('/'):]
    model_name_list.append(model_name)

metadata_key_list = [
    'f0',
    'phase_mode',
    'filter_fl',
    'filter_fh',
]

expt_dict_list = []
for json_fn, model_name in zip(json_fn_list, model_name_list):
    print(json_fn)
    if 'regress' in json_fn:
        f0_label_pred_key = 'f0_lognormal:labels_pred'
        f0_label_true_key = 'f0_lognormal:labels_true'
    else:
        f0_label_pred_key = 'f0_label:labels_pred'
        f0_label_true_key = 'f0_label:labels_true'
    expt_dict = f0dl_bernox.load_f0_expt_dict_from_json(json_fn,
                                                        f0_label_true_key=f0_label_true_key,
                                                        f0_label_pred_key=f0_label_pred_key,
                                                        metadata_key_list=metadata_key_list)
    expt_dict = f0dl_bernox.add_f0_estimates_to_expt_dict(expt_dict,
                                                          f0_label_true_key=f0_label_true_key,
                                                          f0_label_pred_key=f0_label_pred_key)
    expt_dict_list.append(expt_dict)



filter_condition_list, f0_condition_list, f0_pred_ratio_list = compute_histograms(
    expt_dict_list, phase_mode=4, f0_bin_centers=[80, 125, 250], f0_bin_width=0.04)

NCOLS = 3
NROWS = 3
fig, ax_arr = plt.subplots(nrows=NROWS, ncols=NCOLS, figsize=(3*NCOLS, 2*NROWS), sharex=True, sharey=True)
ax_arr = ax_arr.flatten()
for itr0 in range(len(f0_pred_ratio_list)):
    ax = ax_arr[itr0]
    ax.set_xscale('log')
    label = 'filter={}, f0={}'.format(filter_condition_list[itr0], f0_condition_list[itr0])

    # Create bins for the ratio histogram (log-scale)
    bin_step = 0.02
    bins = [0.9]
    while bins[-1] < 2.3: bins.append(bins[-1] * (1.0+bin_step))
    # Manually compute histogram and convert to percentage
    bin_counts, bin_edges = np.histogram(f0_pred_ratio_list[itr0], bins=bins)
    bin_percentages = 100.0 * bin_counts / np.sum(bin_counts)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_widths = bin_edges[:-1] - bin_edges[1:]
    ax.bar(bin_centers, bin_percentages, width=bin_widths, align='center', label=label, color='k')
    ax.legend(loc=0, frameon=False, markerscale=0, handlelength=0)
    
    from matplotlib.ticker import ScalarFormatter, NullFormatter
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks([1.0, 1.5, 2.0])
    ax.set_xticks(np.arange(0.9, 2.4, 0.1), minor=True)

ax_arr[3].set_ylabel('Percentage of pitch matches in {:.1f}% wide bins'.format(bin_step*100.0))
ax_arr[7].set_xlabel('Ratio of predicted f0 to target f0')

plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)
plt.show()


############ RUN AND PLOT MISTUNED HARMONICS (Moore et al 1985) ############

import sys
import os
import json
import numpy as np
import glob
import f0dl_bernox
%matplotlib inline
import matplotlib.pyplot as plt


def compute_f0_pred_percent_shift(expt_dict):
    f0_true = expt_dict['f0']
    f0_pred = expt_dict['f0_pred']
    expt_dict['f0_pred_pct'] = 100.0 * (f0_pred - f0_true) / f0_true
    return expt_dict


def compute_mistuning_shifts(expt_dict, key_mistuned_harm='mistuned_harm', key_mistuned_pct='mistuned_pct',
                             f0_min=-np.inf, f0_max=np.inf):
    expt_dict = compute_f0_pred_percent_shift(expt_dict)
    unique_harm = np.unique(expt_dict[key_mistuned_harm])
    unique_pct = np.unique(expt_dict[key_mistuned_pct])
    results_dict = {}
    for harm in unique_harm:
        results_dict[harm] = {
            'f0_pred_pct_median': [],
            'f0_pred_pct_mean': [],
            'f0_pred_pct_stddev': [],
            'mistuned_pct': [],
            'mistuned_harm': harm
        }
        for pct in unique_pct:
            filter_dict = {
                key_mistuned_harm: harm,
                key_mistuned_pct: pct,
                'f0': [f0_min, f0_max],
            }
            sub_expt_dict = f0dl_bernox.filter_expt_dict(expt_dict, filter_dict=filter_dict)
            results_dict[harm]['f0_pred_pct_median'].append(np.median(sub_expt_dict['f0_pred_pct']))
            results_dict[harm]['f0_pred_pct_mean'].append(np.mean(sub_expt_dict['f0_pred_pct']))
            results_dict[harm]['f0_pred_pct_stddev'].append(np.std(sub_expt_dict['f0_pred_pct']))
            results_dict[harm]['mistuned_pct'].append(pct)
    return results_dict



json_regex = '/om2/user/msaddler/pitchnet/saved_models/PND_v04_JWSS*_classification*/EVAL_MistunedHarm_v00_bestckpt.json'

json_fn_list = sorted(glob.glob(json_regex))
model_name_list = []
for json_fn in json_fn_list:
    model_name = json_fn.replace('/om2/user/msaddler/pitchnet/saved_models/', '')
    model_name = model_name.replace('_bestckpt.json', '')
    model_name = model_name[:model_name.rfind('/')] + '\n' + model_name[model_name.rfind('/'):]
    model_name_list.append(model_name)

metadata_key_list = [
    'f0',
    'mistuned_harm',
    'mistuned_pct',
]

results_dict_list = []
for json_fn, model_name in zip(json_fn_list, model_name_list):
    print(json_fn)
    if 'regress' in json_fn:
        f0_label_pred_key = 'f0_lognormal:labels_pred'
        f0_label_true_key = 'f0_lognormal:labels_true'
    else:
        f0_label_pred_key = 'f0_label:labels_pred'
        f0_label_true_key = 'f0_label:labels_true'
    expt_dict = f0dl_bernox.load_f0_expt_dict_from_json(json_fn,
                                                        f0_label_true_key=f0_label_true_key,
                                                        f0_label_pred_key=f0_label_pred_key,
                                                        metadata_key_list=metadata_key_list)
    expt_dict = f0dl_bernox.add_f0_estimates_to_expt_dict(expt_dict,
                                                          f0_label_true_key=f0_label_true_key,
                                                          f0_label_pred_key=f0_label_pred_key)
    results_dict = compute_mistuning_shifts(expt_dict,
                                            f0_min=60,#-np.inf,
                                            f0_max=140)#np.inf)
    results_dict_list.append(results_dict)


NCOLS = 3
NROWS = int(np.ceil(len(expt_dict_list) / NCOLS))
fig, ax_arr = plt.subplots(nrows=NROWS, ncols=NCOLS, figsize=(5*NCOLS, 3*NROWS))
ax_arr = ax_arr.flatten()

for idx, (results_dict, model_name) in enumerate(zip(results_dict_list, model_name_list)):
    ax = ax_arr[idx]
        
    for key in sorted(results_dict.keys()):
        xval = results_dict[key]['mistuned_pct']
        yval = results_dict[key]['f0_pred_pct_median']#['f0_pred_pct_mean']
        yerr = results_dict[key]['f0_pred_pct_stddev']
        ax.plot(xval, yval, '.-', label=key, markersize=8)

    ax.legend(loc=4, frameon=False, ncol=2, handlelength=2, markerscale=0, fontsize=8)
    ax.set_title(model_name)
    ax.set_xlabel('Shift in harmonic (%)')
    ax.set_ylabel('Shift in predicted F0 (%)')
#     ax.set_xlim([-0.5, 24.5])
    ax.set_ylim([-1.0, 0.25])


for idx in range(len(expt_dict_list), len(ax_arr)): ax_arr[idx].axis('off')

plt.tight_layout()
plt.show()
