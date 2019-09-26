import sys
import os
import json
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.ticker

import util_human_model_comparison


def make_bernox_threshold_plot(ax, results_dict_input,
                               title_str=None,
                               legend_on=True,
                               include_yerr=False,
                               sine_plot_kwargs={},
                               rand_plot_kwargs={},
                               fontsize_title=12,
                               fontsize_labels=12,
                               fontsize_legend=12,
                               fontsize_ticks=12,
                               xlimits=[0, 33],
                               ylimits=[1e-1, 1e2]):
    '''
    Function for plotting Bernstein & Oxenham (2005) experiment results:
    F0 discrimination thresholds as a function of lowest harmonic number.
    '''
    if isinstance(results_dict_input, dict):
        results_dict = results_dict_input
        if 'f0dl_stddev' not in results_dict.keys():
            results_dict['f0dl_stddev'] = [0] * len(results_dict['f0dl'])
    elif isinstance(results_dict_input, list):
        f0dls = np.array([rd['f0dl'] for rd in results_dict_input])
        results_dict = {
            'phase_mode': results_dict_input[0]['phase_mode'],
            'low_harm': results_dict_input[0]['low_harm'],
            'f0dl': np.mean(f0dls, axis=0),
            'f0dl_stddev': np.std(f0dls, axis=0),
        }
    else:
        raise ValueError("INVALID results_dict_input")
    
    phase_mode_list = np.array(results_dict['phase_mode'])
    low_harm_list = np.array(results_dict['low_harm'])
    f0dl_list = np.array(results_dict['f0dl'])
    f0dl_stddev_list = np.array(results_dict['f0dl_stddev'])
    unique_phase_modes = np.flip(np.unique(phase_mode_list))
    for phase_mode in unique_phase_modes:
        xval = low_harm_list[phase_mode_list == phase_mode]
        yval = f0dl_list[phase_mode_list == phase_mode]
        yerr = f0dl_stddev_list[phase_mode_list == phase_mode]
        if phase_mode == 0:
            plot_kwargs = {'label': 'sine', 'color': 'k',
                           'ls':'-', 'lw':2, 'marker':''}
            plot_kwargs.update(sine_plot_kwargs)
        else:
            plot_kwargs = {'label': 'rand', 'color': 'k',
                           'ls':'--', 'lw':2, 'marker':''}
            plot_kwargs.update(rand_plot_kwargs)
        if not legend_on: plot_kwargs['label'] = None
        if include_yerr:
            ax.fill_between(xval, yval-yerr, yval+yerr, alpha=0.15,
                            facecolor=plot_kwargs.get('color', 'k'))
        ax.plot(xval, yval, **plot_kwargs)
    
    ax.set_xlim(xlimits)
    ax.set_xticks(np.arange(xlimits[0], xlimits[1], 5))
    ax.set_ylim(ylimits)
    ax.set_yscale('log')
    ax.tick_params(axis='both', labelsize=fontsize_ticks)
    ax.set_xlabel('Lowest harmonic number', fontsize=fontsize_labels)
    ax.set_ylabel('F0 discrimination\nthreshold (%F0)', fontsize=fontsize_labels)
    if title_str is not None: ax.set_title(title_str, fontsize=fontsize_title)
    if legend_on: ax.legend(loc='lower right', frameon=False, fontsize=fontsize_legend)
    return results_dict


def make_TT_threshold_plot(ax, results_dict_input,
                           title_str=None,
                           legend_on=True,
                           include_yerr=False,
                           plot_kwargs_update={},
                           threshold_cap=100.0,
                           fontsize_title=12,
                           fontsize_labels=12,
                           fontsize_legend=9,
                           fontsize_ticks=12,
                           xlimits=[40, 400],
                           ylimits=[1e-1, 1e2]):
    '''
    Function for plotting transposed tones discrimination experiment results:
    F0 discrimination thresholds as a function of frequency.
    '''
    if isinstance(results_dict_input, dict):
        results_dict = results_dict_input
        if 'f0dl_stddev' not in results_dict.keys():
            results_dict['f0dl_stddev'] = [0] * len(results_dict['f0dl'])
        f0dls = np.array(results_dict['f0dl'])
        if threshold_cap is not None:
            f0dls[f0dls > threshold_cap] = threshold_cap
        results_dict['f0dl'] = f0dls
    elif isinstance(results_dict_input, list):
        f0dls = np.array([rd['f0dl'] for rd in results_dict_input])
        if threshold_cap is not None:
            f0dls[f0dls > threshold_cap] = threshold_cap
        results_dict = {
            'f0_ref': results_dict_input[0]['f0_ref'],
            'f_carrier': results_dict_input[0]['f_carrier'],
            'f0dl': np.mean(f0dls, axis=0),
            'f0dl_stddev': np.std(f0dls, axis=0),
        }
    else:
        raise ValueError("INVALID results_dict_input")
    
    f0_ref = np.array(results_dict['f0_ref'])
    f_carrier_list = np.array(results_dict['f_carrier'])
    f0dl_list = np.array(results_dict['f0dl'])
    f0dl_stddev_list = np.array(results_dict['f0dl_stddev'])
    unique_f_carrier_list = np.unique(f_carrier_list)
    for f_carrier in unique_f_carrier_list:
        xval = f0_ref[f_carrier_list == f_carrier]
        yval = f0dl_list[f_carrier_list == f_carrier]
        yerr = f0dl_stddev_list[f_carrier_list == f_carrier]
        if f_carrier > 0:
            label = '{}-Hz TT'.format(int(f_carrier))
            plot_kwargs = {'label': label, 'color': 'k', 'ls':'-', 'lw':2, 'ms':6,
                           'marker':'o', 'markerfacecolor': 'w'}
            if int(f_carrier) == 10080: plot_kwargs['marker'] = 'D'
            if int(f_carrier) == 6350: plot_kwargs['marker'] = '^'
            if int(f_carrier) == 4000: plot_kwargs['marker'] = 's'
        else:
            label = 'Pure tone'
            plot_kwargs = {'label': label, 'color': 'k', 'ls':'-', 'lw':2, 'ms':6,
                           'marker':'o', 'markerfacecolor': 'k'}
        if not legend_on: plot_kwargs['label'] = None
        plot_kwargs.update(plot_kwargs_update)
        if include_yerr:
            ax.fill_between(xval, yval-yerr, yval+yerr, alpha=0.15,
                            facecolor=plot_kwargs.get('color', 'k'))
        ax.plot(xval, yval, **plot_kwargs)
    
    ax.set_xlim(xlimits)
    ax.set_xscale('log')
    xmin, xmax = (int(xlimits[0]), int(xlimits[1]))
    xticks_major = [t for t in np.arange(xmin, xmax + 1) if t%100==0]
    xtick_labels_major = [None] * len(xticks_major)
    for xtl in [100, 200, 300]: xtick_labels_major[xticks_major.index(xtl)] = xtl
    x_formatter = matplotlib.ticker.FixedFormatter(xtick_labels_major)
    x_locator = matplotlib.ticker.FixedLocator(xticks_major)
    ax.xaxis.set_major_formatter(x_formatter)
    ax.xaxis.set_major_locator(x_locator)
    ax.set_xticks([t for t in np.arange(xmin, xmax + 1) if t%10==0], minor=True)
    ax.set_xticklabels([], minor=True)
    ax.set_ylim(ylimits)
    ax.set_yscale('log')
    ax.tick_params(axis='both', labelsize=fontsize_ticks)
    ax.set_xlabel('Frequency (Hz)', fontsize=fontsize_labels)
    ax.set_ylabel('Frequency difference (%)', fontsize=fontsize_labels)
    if title_str is not None: ax.set_title(title_str, fontsize=fontsize_title)
    if legend_on: ax.legend(loc='lower left', frameon=False,
                            fontsize=fontsize_legend, handlelength=0)
    return results_dict
