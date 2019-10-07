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
                               restrict_conditions=None,
                               sine_plot_kwargs={},
                               rand_plot_kwargs={},
                               threshold_cap=None,
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
        f0dls = np.array(results_dict['f0dl'])
        if threshold_cap is not None:
            f0dls[f0dls > threshold_cap] = threshold_cap
        results_dict['f0dl'] = f0dls
        if 'f0dl_stddev' not in results_dict.keys():
            results_dict['f0dl_stddev'] = [0] * len(results_dict['f0dl'])
    elif isinstance(results_dict_input, list):
        f0dls = np.array([rd['f0dl'] for rd in results_dict_input])
        if threshold_cap is not None:
            f0dls[f0dls > threshold_cap] = threshold_cap
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
    if restrict_conditions is not None:
        unique_phase_modes = restrict_conditions
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
    if legend_on:
        ax.legend(loc='lower right', frameon=False, fontsize=fontsize_legend)
    return results_dict


def make_TT_threshold_plot(ax, results_dict_input,
                           title_str=None,
                           legend_on=True,
                           include_yerr=False,
                           restrict_conditions=None,
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
        f0dls = np.array(results_dict['f0dl'])
        if threshold_cap is not None:
            f0dls[f0dls > threshold_cap] = threshold_cap
        results_dict['f0dl'] = f0dls
        if 'f0dl_stddev' not in results_dict.keys():
            results_dict['f0dl_stddev'] = [0] * len(results_dict['f0dl'])
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
    if restrict_conditions is not None:
        unique_f_carrier_list = restrict_conditions
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
    ax.set_ylabel('Discrimination\nthreshold (%)', fontsize=fontsize_labels)
    if title_str is not None: ax.set_title(title_str, fontsize=fontsize_title)
    if legend_on:
        ax.legend(loc='lower left', frameon=False,
                  fontsize=fontsize_legend, handlelength=0)
    return results_dict


def make_freqshiftedcomplexes_plot(ax, results_dict_input,
                                   expt_key='spectral_envelope_centered_harmonic',
                                   pitch_shift_key='f0_pred_shift_median',
                                   pitch_shift_key_stddev=None,
                                   condition_plot_kwargs={},
                                   plot_kwargs_update={},
                                   title_str=None,
                                   legend_on=True,
                                   include_yerr=False,
                                   restrict_conditions=None,
                                   fontsize_title=12,
                                   fontsize_labels=12,
                                   fontsize_legend=12,
                                   fontsize_ticks=12,
                                   xlimits=[-1, 25],
                                   ylimits=[-4, 12]):
    '''
    Function for plotting frequency-shifted complexes experiment results:
    F0 shift as a function of frequency shift.
    '''
    if pitch_shift_key_stddev is None: pitch_shift_key_stddev = pitch_shift_key + '_stddev'
    if isinstance(results_dict_input, dict):
        results_dict = results_dict_input
        for condition in results_dict[expt_key].keys():
            if pitch_shift_key_stddev not in results_dict[expt_key][condition].keys():
                dummy_vals = [0] * len(results_dict[expt_key][condition][pitch_shift_key])
                results_dict[expt_key][condition][pitch_shift_key_stddev] = dummy_vals
    
    elif isinstance(results_dict_input, list):
        rd0 = results_dict_input[0]
        results_dict = {expt_key: {}}
        for key in rd0.keys():
            if not key == expt_key: results_dict[key] = rd0[key]
        
        for condition in rd0[expt_key].keys():
            plot_vals = np.array([rd[expt_key][condition][pitch_shift_key] for rd in results_dict_input])
            results_dict[expt_key][condition] = {
                'f0_shift': rd0[expt_key][condition]['f0_shift'],
                pitch_shift_key: np.mean(plot_vals, axis=0),
                pitch_shift_key_stddev: np.std(plot_vals, axis=0),
            }
    else:
        raise ValueError("INVALID results_dict_input")
    
    if not condition_plot_kwargs:
        condition_plot_kwargs = {
            '5': {'label': 'RES', 'color': 'r', 'marker': '.', 'ms':8, 'ls':'-', 'lw': 2},
            '11': {'label': 'INT', 'color': 'b', 'marker': '.', 'ms':8, 'ls':'-', 'lw': 2},
            '16': {'label': 'UNRES', 'color': 'k', 'marker': '.', 'ms':8, 'ls':'-', 'lw': 2},
        }
    
    assert set(results_dict[expt_key].keys()) == set(condition_plot_kwargs.keys())
    condition_list = sorted(results_dict[expt_key].keys())
    if restrict_conditions is not None:
        condition_list = restrict_conditions
    for condition in condition_list:
        xval = np.array(results_dict[expt_key][condition]['f0_shift'])
        yval = np.array(results_dict[expt_key][condition][pitch_shift_key])
        yerr = np.array(results_dict[expt_key][condition][pitch_shift_key_stddev])
        plot_kwargs = condition_plot_kwargs[condition]
        plot_kwargs.update(plot_kwargs_update)
        if not legend_on: plot_kwargs['label'] = None
        if include_yerr:
            ax.fill_between(xval, yval-yerr, yval+yerr, alpha=0.15,
                            facecolor=plot_kwargs.get('color', 'k'))
        ax.plot(xval, yval, **plot_kwargs)
    
    if legend_on:
        ax.legend(loc='upper left', frameon=False,
                  fontsize=fontsize_legend, handlelength=1.5)
    if title_str: ax.set_title(title_str, fontsize=fontsize_title)
    ax.set_xlabel('Component shift (%F0)', fontsize=fontsize_labels)
    ax.set_ylabel('Shift in pred F0 (%F0)', fontsize=fontsize_labels)
    ax.set_xlim(xlimits)
    ax.set_ylim(ylimits)
    xval = np.array(results_dict[expt_key]['5']['f0_shift'])
    ax.set_xticks(np.arange(xval[0], xval[-1]+1, 8), minor=False)
    ax.set_xticks(np.arange(xval[0], xval[-1]+1, 4), minor=True)
    ax.set_yticks(np.arange(ylimits[0], ylimits[-1]+1, 4), minor=False)
    ax.set_yticks(np.arange(ylimits[0], ylimits[-1]+1, 2), minor=True)
    ax.tick_params(axis='both', labelsize=fontsize_ticks)
    return results_dict


def make_mistuned_harmonics_bar_graph(ax, results_dict_input,
                                      mistuned_pct=3.0,
                                      use_relative_shift=True,
                                      pitch_shift_key='f0_pred_pct_median',
                                      pitch_shift_key_stddev=None,
                                      title_str=None,
                                      legend_on=True,
                                      include_yerr=False,
                                      barwidth=0.12,
                                      harmonic_list=[1,2,3,4,5,6],
                                      fontsize_title=12,
                                      fontsize_labels=12,
                                      fontsize_legend=12,
                                      fontsize_ticks=12,
                                      xlimits=[2, 8],
                                      ylimits=[-0.1, 1.1]):
    '''
    Function for plotting mistuned harmonics experiment results:
    F0 shift bar graph for a given mistuning percent.
    '''
    if pitch_shift_key_stddev is None: pitch_shift_key_stddev = pitch_shift_key + '_stddev'
    if isinstance(results_dict_input, dict):
        results_dict = results_dict_input
        bg_results_dict = util_human_model_comparison.get_mistuned_harmonics_bar_graph_results_dict(
            results_dict,
            mistuned_pct=mistuned_pct,
            pitch_shift_key=pitch_shift_key,
            harmonic_list=harmonic_list,
            use_relative_shift=use_relative_shift)
        for group_key in bg_results_dict.keys():
            if pitch_shift_key_stddev not in bg_results_dict[group_key].keys():
                dummy_vals = [0] * len(bg_results_dict[group_key][pitch_shift_key])
                bg_results_dict[group_key][pitch_shift_key_stddev] = dummy_vals
    elif isinstance(results_dict_input, list):
        bg_results_dict_list = []
        for results_dict in results_dict_input:
            bg_results_dict_list.append(
                util_human_model_comparison.get_mistuned_harmonics_bar_graph_results_dict(
                    results_dict,
                    mistuned_pct=mistuned_pct,
                    pitch_shift_key=pitch_shift_key,
                    harmonic_list=harmonic_list,
                    use_relative_shift=use_relative_shift)
            )
        bg_results_dict = {}
        for group_key in bg_results_dict_list[0].keys():
            plot_vals = np.array([bgrd[group_key][pitch_shift_key]
                                  for bgrd in bg_results_dict_list])
            bg_results_dict[group_key] = {
                'f0_ref': bg_results_dict_list[0][group_key]['f0_ref'],
                pitch_shift_key: np.mean(plot_vals, axis=0),
                pitch_shift_key_stddev: np.std(plot_vals, axis=0),
            }
    else:
        raise ValueError("INVALID results_dict_input")
    
    num_groups = len(bg_results_dict.keys())
    group_xoffsets = np.arange(num_groups) - np.mean(np.arange(num_groups))
    for group_idx, group_key in enumerate(sorted(bg_results_dict.keys())):
        bars_per_group = len(bg_results_dict[group_key]['f0_ref'])
        xval = np.arange(bars_per_group)
        xval = xval + barwidth*group_xoffsets[group_idx]
        yval = np.array(bg_results_dict[group_key][pitch_shift_key])
        yerr = np.array(bg_results_dict[group_key][pitch_shift_key_stddev])
        plot_kwargs = {'width': barwidth, 'edgecolor':'w', 'label':group_key}
        if not legend_on: plot_kwargs['label'] = None
        if include_yerr:
            plot_kwargs['yerr'] = yerr
            plot_kwargs['error_kw'] = {'ecolor': 'k', 'elinewidth':1, 'capsize':1}
        ax.bar(xval, yval, **plot_kwargs)
    
    base_xvals = np.arange(bars_per_group)
    f0_ref_values = bg_results_dict[group_key]['f0_ref']
    
    ax.axhline(y=0, xmin=0, xmax=1, color='k', lw=1)
    if title_str: ax.set_title(title_str, fontsize=fontsize_title)
    if legend_on:
        ax.legend(loc='upper right', frameon=False,
                  fontsize=fontsize_legend, handlelength=1)
    ax.set_xlim([barwidth*group_xoffsets[0]-xlimits[0]*barwidth,
                 np.max(base_xvals) + barwidth*group_xoffsets[-1] + xlimits[1]*barwidth])
    ax.set_xlabel('F0 (Hz)', fontsize=fontsize_labels)
    ax.set_xticks(base_xvals, minor=True)
    ax.set_xticklabels(f0_ref_values, minor=True)
    ax.tick_params(which='minor', length=0)
    ax.set_xticks(base_xvals[:-1]+0.5, minor=False)
    ax.set_xticklabels([], minor=False)
    ax.tick_params(which='major', length=6)
    ax.set_ylim(ylimits)
    ax.set_ylabel('% pitch shift', fontsize=fontsize_labels)
    return bg_results_dict


def make_altphase_plot(ax, results_dict_input,
                       expt_key='filter_fl_bin_means',
                       expt_key_stddev=None,
                       condition_plot_kwargs={},
                       plot_kwargs_update={},
                       title_str=None,
                       legend_on=True,
                       include_yerr=False,
                       fontsize_title=12,
                       fontsize_labels=12,
                       fontsize_legend=12,
                       fontsize_ticks=12,
                       xlimits=[62.5*0.9, 250*1.1],
                       ylimits=[-1.1, 1.1]):
    '''
    '''
    if expt_key_stddev is None: expt_key_stddev = expt_key + '_stddev'
    if isinstance(results_dict_input, dict):
        results_dict = results_dict_input
        if expt_key_stddev not in results_dict.keys():
            results_dict[expt_key_stddev] = {}
            for condition in results_dict[expt_key].keys():
                results_dict[expt_key_stddev][condition] = [0] * len(results_dict[expt_key][condition])
    elif isinstance(results_dict_input, list):
        results_dict = {
            'f0_bin_centers': results_dict_input[0]['f0_bin_centers'],
            expt_key: {},
            expt_key_stddev: {},
        }
        for condition in results_dict_input[0][expt_key].keys():
            yvals = np.array([rd[expt_key][condition] for rd in results_dict_input])
            results_dict[expt_key][condition] = np.mean(yvals, axis=0)
            results_dict[expt_key_stddev][condition] = np.std(yvals, axis=0)
    else:
        raise ValueError("INVALID results_dict_input")
    
    if not condition_plot_kwargs:
        condition_plot_kwargs = {
            '125.0': {'label': 'LOW', 'color': 'r', 'marker': '.', 'ms':8, 'ls':'-', 'lw': 2},
            '1375.0': {'label': 'MID', 'color': 'b', 'marker': '.', 'ms':8, 'ls':'-', 'lw': 2},
            '3900.0': {'label': 'HIGH', 'color': 'k', 'marker': '.', 'ms':8, 'ls':'-', 'lw': 2},
        }
    
    assert set(results_dict[expt_key].keys()) == set(condition_plot_kwargs.keys())
    for condition in sorted(results_dict[expt_key].keys()):
        xval = np.array(results_dict['f0_bin_centers'])
        yval = np.array(results_dict[expt_key][condition])
        yerr = np.array(results_dict[expt_key_stddev][condition])
        plot_kwargs = condition_plot_kwargs[condition]
        plot_kwargs.update(plot_kwargs_update)
        if not legend_on: plot_kwargs['label'] = None
        if include_yerr:
            ax.fill_between(xval, yval-yerr, yval+yerr, alpha=0.15,
                            facecolor=plot_kwargs.get('color', 'k'))
        ax.plot(xval, yval, **plot_kwargs)
    
    if legend_on:
        ax.legend(loc=0, frameon=False, fontsize=fontsize_legend,
                  handlelength=1, markerscale=1)
    if title_str: ax.set_title(title_str, fontsize=fontsize_title)
    ax.set_xlabel('F0 (Hz)', fontsize=fontsize_labels)
    ax.set_ylabel('Fraction judged 2*F0 -\n Fraction judged 1*F0',
                  fontsize=fontsize_labels)
    ax.set_xlim(xlimits)
    ax.set_xscale('log')
    xmin, xmax = (int(xlimits[0]), int(xlimits[1]))
    xticks_major = [t for t in np.arange(xmin, xmax + 1) if t%100==0]
    xtick_labels_major = [None] * len(xticks_major)
    for xtl in [100, 200]: xtick_labels_major[xticks_major.index(xtl)] = xtl
    x_formatter = matplotlib.ticker.FixedFormatter(xtick_labels_major)
    x_locator = matplotlib.ticker.FixedLocator(xticks_major)
    ax.xaxis.set_major_formatter(x_formatter)
    ax.xaxis.set_major_locator(x_locator)
    ax.set_xticks([t for t in np.arange(xmin, xmax + 1) if t%10==0], minor=True)
    ax.set_xticklabels([], minor=True)
    ax.set_ylim(ylimits)
    ax.set_yticks(np.arange(-1, 1.1, 0.5), minor=False)
    ax.set_yticks(np.arange(-1, 1.1, 0.1), minor=True)
    return results_dict
