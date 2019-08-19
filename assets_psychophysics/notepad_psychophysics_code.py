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
