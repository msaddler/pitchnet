import sys
import os
import json
import numpy as np
import glob
import copy
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.cm
import matplotlib.colors

import util_human_model_comparison

sys.path.append('/om2/user/msaddler/python-packages/msutil')
import util_figures


def bootstrap(data,
              bootstrap_repeats=1000,
              sample_size=None,
              metric_function=None):
    '''
    Computes bootstrap mean and standard deviation of data
    across the primary axis (axis=0).
    
    Args
    ----
    data (np.array): bootstrap samples will be drawn from the primary axis
    bootstrap_repeats (int): number of times to sample bootstrap distribution
    sample_size (int): number of samples drawn (with replacement) each repeat
        (default is the number of data points: data.shape[0])
    metric_function (str or function): specifies statistic to be computed for
        each bootstrap repeat (default is 'mean'; 'median' is also supported)
    
    Returns
    -------
    bootstrap_mean (np.array): mean of bootstrapped distribution
    bootstrap_std (np.array): standard deviation of bootstrapped distribution
    '''
    if sample_size is None:
        sample_size = data.shape[0]
    if metric_function is None:
        metric_function = np.mean
    elif isinstance(metric_function, str):
        if metric_function.lower() == 'mean':
            metric_function = np.mean
        elif metric_function.lower() == 'median':
            metric_function = np.median
        else:
            msg = "metric function `{}` is not recognized"
            raise ValueError(msg.format(metric_function))
    
    bootstrap_data_shape = list(data.shape)
    bootstrap_data_shape[0] = bootstrap_repeats
    bootstrap_data = np.zeros(bootstrap_data_shape,
                              dtype=data.dtype)
    
    subject_indexes = np.arange(0, data.shape[0])
    for index in np.arange(0, bootstrap_repeats):
        sampled_indexes = np.random.choice(subject_indexes,
                                           size=[sample_size],
                                           replace=True)
        sample_data = data[sampled_indexes]
        sample_metric = metric_function(sample_data, axis=0)
        bootstrap_data[index] = sample_metric
    
    bootstrap_mean = np.mean(bootstrap_data, axis=0)
    bootstrap_std = np.std(bootstrap_data, axis=0)
    return bootstrap_mean, bootstrap_std


def combine_subjects(subject_data, kwargs_bootstrap={}):
    '''
    Helper function for combining data across subjects and computing
    error bars. Default behavior is to compute mean and standard error
    of the mean across subjets. If `kwargs_bootstrap` is specified,
    the bootstrap mean and standard deviation will be returned.
    
    Args
    ----
    subject_data (np.array): 2D data array with subjects along primary axis
    kwargs_bootstrap (dict): keyword arguments for `bootstrap` function
    
    Returns
    -------
    yval (np.array): y-values to plot (combined across subjects)
    yerr (np.array): y-value errorbars to plot
    '''
    if kwargs_bootstrap:
        # If kwargs_bootstrap is specified, compute mean and
        # standard deviation of bootstrapped subject data
        yval, yerr = bootstrap(subject_data, **kwargs_bootstrap)
    else:
        # Default behavior is to compute mean and standard error
        n = subject_data.shape[0]
        yval = np.mean(subject_data, axis=0)
        yerr = np.std(subject_data, axis=0) / np.sqrt(n)
    return yval, yerr


def make_bernox_threshold_plot(ax, results_dict_input,
                               str_title=None,
                               legend_on=True,
                               include_yerr=False,
                               restrict_conditions=None,
                               sine_plot_kwargs={},
                               rand_plot_kwargs={},
                               threshold_cap=100.0,
                               fontsize_title=12,
                               fontsize_labels=12,
                               fontsize_legend=12,
                               fontsize_ticks=12,
                               xlimits=[0, 33],
                               ylimits=[1e-1, 1e2],
                               kwargs_legend={},
                               kwargs_bootstrap={'bootstrap_repeats': 1000, 'metric_function': 'mean'}):
    '''
    Function for plotting Bernstein & Oxenham (2005) experiment results:
    F0 discrimination thresholds as a function of lowest harmonic number.
    '''
    if isinstance(results_dict_input, dict):
        results_dict = copy.deepcopy(results_dict_input)
        f0dls = np.array(results_dict['f0dl'])
        if threshold_cap is not None:
            f0dls[f0dls > threshold_cap] = threshold_cap
        results_dict['log10_f0dl'] = np.log10(f0dls)
        if 'f0dl_err' in results_dict.keys():
            results_dict['log10_f0dl_err'] = np.log10(np.array(results_dict['f0dl_err']))
        else:
            results_dict['log10_f0dl_err'] = np.zeros_like(results_dict['log10_f0dl'])
    elif isinstance(results_dict_input, list):
        f0dls = np.array([rd['f0dl'] for rd in results_dict_input])
        if threshold_cap is not None:
            f0dls[f0dls > threshold_cap] = threshold_cap
        yval, yerr = combine_subjects(np.log10(f0dls), kwargs_bootstrap=kwargs_bootstrap)
        results_dict = {
            'phase_mode': results_dict_input[0]['phase_mode'],
            'low_harm': results_dict_input[0]['low_harm'],
            'log10_f0dl': yval,
            'log10_f0dl_err': yerr,
        }
    else:
        raise ValueError("INVALID results_dict_input")
    
    phase_mode_list = np.array(results_dict['phase_mode'])
    low_harm_list = np.array(results_dict['low_harm'])
    log10_f0dl_list = np.array(results_dict['log10_f0dl'])
    log10_f0dl_err_list = np.array(results_dict['log10_f0dl_err'])
    unique_phase_modes = np.flip(np.unique(phase_mode_list))
    if restrict_conditions is not None:
        unique_phase_modes = restrict_conditions
    for phase_mode in unique_phase_modes:
        xval = low_harm_list[phase_mode_list == phase_mode]
        log10_yval = log10_f0dl_list[phase_mode_list == phase_mode]
        log10_yerr = log10_f0dl_err_list[phase_mode_list == phase_mode]
        if phase_mode == 0:
            plot_kwargs = {'label': 'SINE', 'color': 'k',
                           'ls':'-', 'lw':2, 'marker':''}
            if len(xval) <= 8:
                plot_kwargs.update({'ms':10, 'marker':'.'})
            plot_kwargs.update(sine_plot_kwargs)
        else:
            plot_kwargs = {'label': 'RAND', 'color': 'k',
                           'ls':'--', 'lw':2, 'marker':'', 'dashes': (1,1)}
            if len(xval) <= 8:
                plot_kwargs.update({'ms':10, 'marker':'.'})
            plot_kwargs.update(rand_plot_kwargs)
        if not legend_on: plot_kwargs['label'] = None
        if include_yerr:
            ax.fill_between(xval,
                            np.power(10.0, log10_yval-2*log10_yerr),
                            np.power(10.0, log10_yval+2*log10_yerr),
                            alpha=0.15,
                            facecolor=plot_kwargs.get('color', 'k'))
        ax.plot(xval, np.power(10.0, log10_yval), **plot_kwargs)
    
    ax = util_figures.format_axes(ax,
                                  str_xlabel='Lowest harmonic number',
                                  str_ylabel='F0 discrimination\nthreshold (%F0)',
                                  fontsize_labels=fontsize_labels,
                                  fontsize_ticks=fontsize_ticks,
                                  fontweight_labels=None,
                                  xscale='linear',
                                  yscale='log',
                                  xlimits=xlimits,
                                  ylimits=ylimits,
                                  xticks=np.arange(xlimits[0], xlimits[1], 5),
                                  yticks=None,
                                  xticks_minor=np.arange(xlimits[0], xlimits[1], 1),
                                  yticks_minor=None,
                                  xticklabels=None,
                                  yticklabels=None,
                                  spines_to_hide=[],
                                  major_tick_params_kwargs_update={},
                                  minor_tick_params_kwargs_update={})
    
    if str_title is not None:
        ax.set_title(str_title, fontsize=fontsize_title)
    if legend_on:
        legend_plot_kwargs = {
            'loc': 'lower right',
            'borderpad': 0.4,
            'borderaxespad': 0.5,
            'handletextpad': 0.8,
            'frameon': False,
            'handlelength': 1.5,
            'markerscale': 0.0,
            'fontsize': fontsize_legend,
        }
        legend_plot_kwargs.update(kwargs_legend)
        ax.legend(**legend_plot_kwargs)
    return results_dict


def make_TT_threshold_plot(ax, results_dict_input,
                           str_title=None,
                           legend_on=True,
                           include_yerr=False,
                           restrict_conditions=None,
                           plot_kwargs_update={},
                           threshold_cap=100.0,
                           fontsize_title=12,
                           fontsize_labels=12,
                           fontsize_legend=11,
                           fontsize_ticks=12,
                           xlimits=[40, 360],
                           ylimits=[1e-1, 1e2],
                           kwargs_legend={},
                           kwargs_bootstrap={'bootstrap_repeats': 1000, 'metric_function': 'mean'},
                           combine_transposedtones=True):
    '''
    Function for plotting transposed tones discrimination experiment results:
    F0 discrimination thresholds as a function of frequency.
    '''
    if combine_transposedtones:
        # F0 discrimination thresholds will be averaged across transposed tones
        # with different carrier frequencies
        tmp_results_dict_input = []
        if isinstance(results_dict_input, dict):
            results_dict_input = [results_dict_input]
        for results_dict in results_dict_input:
            tmp_rd = util_human_model_comparison.combine_transposedtones_thresholds(
                results_dict,
                threshold_cap=threshold_cap)
            tmp_results_dict_input.append(tmp_rd)
        results_dict_input = tmp_results_dict_input
    
    if isinstance(results_dict_input, dict):
        results_dict = copy.deepcopy(results_dict_input)
        f0dls = np.array(results_dict['f0dl'])
        if threshold_cap is not None:
            f0dls[f0dls > threshold_cap] = threshold_cap
        results_dict['log10_f0dl'] = np.log10(f0dls)
        if 'f0dl_err' in results_dict.keys():
            results_dict['log10_f0dl_err'] = np.log10(np.array(results_dict['f0dl_err']))
        else:
            results_dict['log10_f0dl_err'] = np.zeros_like(results_dict['log10_f0dl'])
    elif isinstance(results_dict_input, list):
        f0dls = np.array([rd['f0dl'] for rd in results_dict_input])
        if threshold_cap is not None:
            f0dls[f0dls > threshold_cap] = threshold_cap
        yval, yerr = combine_subjects(np.log10(f0dls), kwargs_bootstrap=kwargs_bootstrap)
        results_dict = {
            'f0_ref': results_dict_input[0]['f0_ref'],
            'f_carrier': results_dict_input[0]['f_carrier'],
            'log10_f0dl': yval,
            'log10_f0dl_err': yerr,
        }
    else:
        raise ValueError("INVALID results_dict_input")
    
    f0_ref = np.array(results_dict['f0_ref'])
    f_carrier_list = np.array(results_dict['f_carrier'])
    log10_f0dl_list = np.array(results_dict['log10_f0dl'])
    log10_f0dl_err_list = np.array(results_dict['log10_f0dl_err'])
    unique_f_carrier_list = np.flip(np.unique(f_carrier_list))
    if restrict_conditions is not None:
        unique_f_carrier_list = restrict_conditions
    for f_carrier in unique_f_carrier_list:
        xval = f0_ref[f_carrier_list == f_carrier]
        log10_yval = log10_f0dl_list[f_carrier_list == f_carrier]
        log10_yerr = log10_f0dl_err_list[f_carrier_list == f_carrier]
        if f_carrier == 0.0:
            label = 'Pure tones'
            plot_kwargs = {'label': label, 'color': 'k', 'ls':'-', 'lw':2, 'ms':8,
                           'marker':'o', 'markerfacecolor': 'k'}
        else:
            label = '{}-Hz TT'.format(int(f_carrier))
            plot_kwargs = {'label': label, 'color': 'k', 'ls':'-', 'lw':2, 'ms':8,
                           'marker':'o', 'markerfacecolor': 'w'}
            if int(f_carrier) == 10080: plot_kwargs['marker'] = 'D'
            if int(f_carrier) == 6350: plot_kwargs['marker'] = '^'
            if int(f_carrier) == 4000: plot_kwargs['marker'] = 's'
            if int(f_carrier) == 1: plot_kwargs['label'] = 'Transposed tones'
        if not legend_on: plot_kwargs['label'] = None
        plot_kwargs.update(plot_kwargs_update)
        if include_yerr:
            ax.fill_between(xval,
                            np.power(10.0, log10_yval-2*log10_yerr),
                            np.power(10.0, log10_yval+2*log10_yerr),
                            alpha=0.15,
                            facecolor=plot_kwargs.get('color', 'k'))
        ax.plot(xval, np.power(10.0, log10_yval), **plot_kwargs)
    
    ax = util_figures.format_axes(ax,
                                  str_xlabel='Frequency (Hz)',
                                  str_ylabel='Discrimination\nthreshold (%)',
                                  fontsize_labels=fontsize_labels,
                                  fontsize_ticks=fontsize_ticks,
                                  fontweight_labels=None,
                                  xscale='log',
                                  yscale='log',
                                  xlimits=xlimits,
                                  ylimits=ylimits,
                                  xticks=None,
                                  yticks=None,
                                  xticks_minor=None,
                                  yticks_minor=None,
                                  xticklabels=None,
                                  yticklabels=None,
                                  spines_to_hide=[],
                                  major_tick_params_kwargs_update={},
                                  minor_tick_params_kwargs_update={})
    
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
    
    if str_title is not None:
        ax.set_title(str_title, fontsize=fontsize_title)
    if legend_on:
        legend_plot_kwargs = {
            'loc': 'lower left',
            'borderpad': 0.4,
            'borderaxespad': 0.5,
            'handletextpad': 0.8,
            'frameon': False,
            'handlelength': 0.0,
            'markerscale': 1.0,
            'fontsize': fontsize_legend,
        }
        legend_plot_kwargs.update(kwargs_legend)
        ax.legend(**legend_plot_kwargs)
    return results_dict


def make_freqshiftedcomplexes_plot(ax, results_dict_input,
                                   use_relative_shift=True,
                                   expt_key='spectral_envelope_centered_harmonic',
                                   pitch_shift_key='f0_pred_shift_median',
                                   pitch_shift_err_key=None,
                                   condition_plot_kwargs={},
                                   plot_kwargs_update={},
                                   str_title=None,
                                   legend_on=True,
                                   include_yerr=False,
                                   restrict_conditions=['5', '16'],#['5', '11', '16'],
                                   fontsize_title=12,
                                   fontsize_labels=12,
                                   fontsize_legend=12,
                                   fontsize_ticks=12,
                                   xlimits=[-1, 25],
                                   ylimits=[-4, 12],
                                   kwargs_legend={},
                                   kwargs_bootstrap={'bootstrap_repeats': 1000, 'metric_function': 'mean'}):
    '''
    Function for plotting frequency-shifted complexes experiment results:
    F0 shift as a function of frequency shift.
    '''
    if pitch_shift_err_key is None: pitch_shift_err_key = pitch_shift_key + '_err'
    if isinstance(results_dict_input, dict):
        results_dict = copy.deepcopy(results_dict_input)
        for condition in results_dict[expt_key].keys():
            condition_dict = results_dict[expt_key][condition]
            if (use_relative_shift) and (0.0 in condition_dict['f0_shift']):
                unshifted_idx = condition_dict['f0_shift'].index(0.0)
                pitch_shifts = np.array(condition_dict[pitch_shift_key])
                condition_dict[pitch_shift_key] = pitch_shifts - pitch_shifts[unshifted_idx]
            if pitch_shift_err_key not in results_dict[expt_key][condition].keys():
                dummy_vals = [0] * len(results_dict[expt_key][condition][pitch_shift_key])
                results_dict[expt_key][condition][pitch_shift_err_key] = dummy_vals
    elif isinstance(results_dict_input, list):
        rd0 = results_dict_input[0]
        results_dict = {expt_key: {}}
        for condition in rd0[expt_key].keys():
            plot_vals = np.array([rd[expt_key][condition][pitch_shift_key] for rd in results_dict_input])
            if (use_relative_shift) and (0.0 in rd0[expt_key][condition]['f0_shift']):
                unshifted_idx = rd0[expt_key][condition]['f0_shift'].index(0.0)
                plot_vals = plot_vals - plot_vals[:, unshifted_idx:unshifted_idx+1]
            yval, yerr = combine_subjects(plot_vals, kwargs_bootstrap=kwargs_bootstrap)
            results_dict[expt_key][condition] = {
                'f0_shift': rd0[expt_key][condition]['f0_shift'],
                pitch_shift_key: yval,
                pitch_shift_err_key: yerr,
            }
    else:
        raise ValueError("INVALID results_dict_input")
    
    if not condition_plot_kwargs:
        color_list = util_figures.get_color_list(3, cmap_name='coolwarm')
        condition_plot_kwargs = {
            '5': {'label': 'Low harmonics', 'marker': '.', 'ms':12, 'ls':'-', 'lw': 2, 'color':color_list[0]},
            '11': {'label': 'Mid harmonics', 'marker': '.', 'ms':12, 'ls':'-', 'lw': 2, 'color':color_list[1]},
            '16': {'label': 'High harmonics', 'marker': '.', 'ms':12, 'ls':'-', 'lw': 2, 'color':color_list[2]},
        }
    
    assert set(results_dict[expt_key].keys()) == set(condition_plot_kwargs.keys())
    condition_list = sorted([int(x) for x in results_dict[expt_key].keys()])
    if restrict_conditions is not None:
        condition_list = restrict_conditions
    for condition in condition_list:
        xval = np.array(results_dict[expt_key][condition]['f0_shift'])
        yval = np.array(results_dict[expt_key][condition][pitch_shift_key])
        yerr = np.array(results_dict[expt_key][condition][pitch_shift_err_key])
        plot_kwargs = condition_plot_kwargs[condition]
        plot_kwargs.update(plot_kwargs_update)
        if not legend_on: plot_kwargs['label'] = None
        if include_yerr:
            ax.fill_between(xval,
                            yval-2*yerr,
                            yval+2*yerr,
                            alpha=0.15,
                            facecolor=plot_kwargs['color'])
        ax.plot(xval, yval, **plot_kwargs)
    
    xval = np.array(results_dict[expt_key]['5']['f0_shift'])
    ax = util_figures.format_axes(ax,
                                  str_xlabel='Component shift (%F0)',
                                  str_ylabel='Shift in reported F0\n(%F0)',
                                  fontsize_labels=fontsize_labels,
                                  fontsize_ticks=fontsize_ticks,
                                  fontweight_labels=None,
                                  xscale='linear',
                                  yscale='linear',
                                  xlimits=xlimits,
                                  ylimits=ylimits,
                                  xticks=np.arange(xval[0], xval[-1]+1, 8),
                                  yticks=np.arange(ylimits[0], ylimits[-1]+1, 4),
                                  xticks_minor=np.arange(xval[0], xval[-1]+1, 1),
                                  yticks_minor=np.arange(ylimits[0], ylimits[-1]+1, 1),
                                  xticklabels=None,
                                  yticklabels=None,
                                  spines_to_hide=[],
                                  major_tick_params_kwargs_update={},
                                  minor_tick_params_kwargs_update={})
    
    if str_title is not None:
        ax.set_title(str_title, fontsize=fontsize_title)
    if legend_on:
        legend_plot_kwargs = {
            'loc': 'upper left',
            'borderpad': 0.4,
            'borderaxespad': 0.5,
            'handletextpad': 0.8,
            'frameon': False,
            'handlelength': 0.4,
            'markerscale': 0.0,
            'fontsize': fontsize_legend,
        }
        legend_plot_kwargs.update(kwargs_legend)
        leg = ax.legend(**legend_plot_kwargs)
        for legobj in leg.legendHandles:
            legobj.set_linewidth(8.0)
    return results_dict


def make_mistuned_harmonics_bar_graph(ax, results_dict_input,
                                      mistuned_pct=3.0,
                                      use_relative_shift=True,
                                      pitch_shift_key='f0_pred_pct_median',
                                      pitch_shift_err_key=None,
                                      str_title=None,
                                      legend_on=True,
                                      include_yerr=False,
                                      barwidth=0.12,
                                      harmonic_list=[1,2,3,4,5,6],
                                      fontsize_title=12,
                                      fontsize_labels=12,
                                      fontsize_legend=12,
                                      fontsize_ticks=12,
                                      xlimits=[2, 8],
                                      ylimits=[-0.1, 1.1],
                                      cmap_name='coolwarm',
                                      kwargs_legend={},
                                      kwargs_bootstrap={'bootstrap_repeats': 1000, 'metric_function': 'mean'}):
    '''
    Function for plotting mistuned harmonics experiment results:
    F0 shift bar graph for a given mistuning percent.
    '''
    if pitch_shift_err_key is None: pitch_shift_err_key = pitch_shift_key + '_err'
    if isinstance(results_dict_input, dict):
        results_dict = copy.deepcopy(results_dict_input)
        bg_results_dict = util_human_model_comparison.get_mistuned_harmonics_bar_graph_results_dict(
            results_dict,
            mistuned_pct=mistuned_pct,
            pitch_shift_key=pitch_shift_key,
            harmonic_list=harmonic_list,
            use_relative_shift=use_relative_shift)
        for group_key in bg_results_dict.keys():
            if pitch_shift_err_key not in bg_results_dict[group_key].keys():
                dummy_vals = [0] * len(bg_results_dict[group_key][pitch_shift_key])
                bg_results_dict[group_key][pitch_shift_err_key] = dummy_vals
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
            yval, yerr = combine_subjects(plot_vals, kwargs_bootstrap=kwargs_bootstrap)
            bg_results_dict[group_key] = {
                'f0_ref': bg_results_dict_list[0][group_key]['f0_ref'],
                pitch_shift_key: yval,
                pitch_shift_err_key: yerr,
            }
    else:
        raise ValueError("INVALID results_dict_input")
    
    if cmap_name == 'tab10': num_colors = max(10, len(harmonic_list))
    else: num_colors = len(harmonic_list)
    color_list = util_figures.get_color_list(num_colors, cmap_name=cmap_name)
    num_groups = len(bg_results_dict.keys())
    group_xoffsets = np.arange(num_groups) - np.mean(np.arange(num_groups))
    for group_idx, group_key in enumerate(sorted(bg_results_dict.keys())):
        bars_per_group = len(bg_results_dict[group_key]['f0_ref'])
        xval = np.arange(bars_per_group)
        xval = xval + barwidth*group_xoffsets[group_idx]
        yval = np.array(bg_results_dict[group_key][pitch_shift_key])
        yerr = np.array(bg_results_dict[group_key][pitch_shift_err_key])
        plot_kwargs = {
            'width': barwidth,
            'color': color_list[group_idx],
            'edgecolor':'k',
            'label':group_key
        }
        if not legend_on: plot_kwargs['label'] = None
        if include_yerr:
            plot_kwargs['yerr'] = 2*yerr
            plot_kwargs['error_kw'] = {'ecolor': 'k', 'elinewidth':1, 'capsize':1}
        ax.bar(xval, yval, **plot_kwargs)
    
    base_xvals = np.arange(bars_per_group)
    f0_ref_values = bg_results_dict[group_key]['f0_ref']
    xlimits_adjusted = [
        barwidth*group_xoffsets[0]-xlimits[0]*barwidth,
        np.max(base_xvals) + barwidth*group_xoffsets[-1] + xlimits[1]*barwidth
    ]
    ax = util_figures.format_axes(ax,
                                  str_xlabel='F0 (Hz)',
                                  str_ylabel='Shift in reported F0\n(%F0)',
                                  fontsize_labels=fontsize_labels,
                                  fontsize_ticks=fontsize_ticks,
                                  fontweight_labels=None,
                                  xscale='linear',
                                  yscale='linear',
                                  xlimits=xlimits_adjusted,
                                  ylimits=ylimits,
                                  xticks=base_xvals[:-1]+0.5,
                                  yticks=np.arange(0, ylimits[-1], 0.2),
                                  xticks_minor=base_xvals,
                                  yticks_minor=np.arange(0, ylimits[-1], 0.1),
                                  xticklabels=[],
                                  yticklabels=None,
                                  spines_to_hide=[],
                                  major_tick_params_kwargs_update={},
                                  minor_tick_params_kwargs_update={})
    ax.tick_params(axis='x', which='major', length=fontsize_ticks, direction='inout')
    ax.tick_params(axis='x', which='minor', length=0, direction='inout')
    ax.set_xticklabels([int(x) for x in f0_ref_values], minor=True, fontsize=fontsize_ticks)
    ax.axhline(y=0, xmin=0, xmax=1, color='k', lw=1)
    
    if str_title is not None:
        ax.set_title(str_title, fontsize=fontsize_title)
    if legend_on:
        legend_plot_kwargs = {
            'loc': 'upper right',
            'borderpad': 0.4,
            'borderaxespad': 0.1,
            'handletextpad': 0.8,
            'frameon': False,
            'handlelength': 1.0,
            'markerscale': 1.0,
            'fontsize': fontsize_legend,
        }
        legend_plot_kwargs.update(kwargs_legend)
        ax.legend(**legend_plot_kwargs)
    return bg_results_dict


def make_mistuned_harmonics_line_graph(ax, results_dict_input,
                                       mistuned_pct=3.0,
                                       use_relative_shift=True,
                                       pitch_shift_key='f0_pred_pct_median',
                                       pitch_shift_err_key=None,
                                       harmonic_list=None,
                                       str_title=None,
                                       legend_on=True,
                                       include_yerr=False,
                                       fontsize_title=12,
                                       fontsize_labels=12,
                                       fontsize_legend=12,
                                       fontsize_ticks=12,
                                       xlimits=None,
                                       ylimits=[-0.05, 1.05],
                                       cmap_name='coolwarm',
                                       kwargs_legend={},
                                       kwargs_bootstrap={'bootstrap_repeats': 1000, 'metric_function': 'mean'}):
    '''
    '''
    if pitch_shift_err_key is None: pitch_shift_err_key = pitch_shift_key + '_err'
    if isinstance(results_dict_input, dict):
        results_dict = copy.deepcopy(results_dict_input)
        bg_results_dict = util_human_model_comparison.get_mistuned_harmonics_bar_graph_results_dict(
            results_dict,
            mistuned_pct=mistuned_pct,
            pitch_shift_key=pitch_shift_key,
            harmonic_list=None,
            use_relative_shift=use_relative_shift)
        for group_key in bg_results_dict.keys():
            if pitch_shift_err_key not in bg_results_dict[group_key].keys():
                dummy_vals = [0] * len(bg_results_dict[group_key][pitch_shift_key])
                bg_results_dict[group_key][pitch_shift_err_key] = dummy_vals
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
            yval, yerr = combine_subjects(plot_vals, kwargs_bootstrap=kwargs_bootstrap)
            bg_results_dict[group_key] = {
                'f0_ref': bg_results_dict_list[0][group_key]['f0_ref'],
                pitch_shift_key: yval,
                pitch_shift_err_key: yerr,
            }
    else:
        raise ValueError("INVALID results_dict_input")
    
    list_harmonic_numbers = []
    list_f0 = None
    array_pitch_shift = []
    array_pitch_shift_err = []
    for harm_key in bg_results_dict.keys():
        list_harmonic_numbers.append(int(harm_key))
        if list_f0 is None:
            list_f0 = bg_results_dict[harm_key]['f0_ref']
        else:
            assert list_f0 == bg_results_dict[harm_key]['f0_ref']
        array_pitch_shift.append(bg_results_dict[harm_key][pitch_shift_key])
        array_pitch_shift_err.append(bg_results_dict[harm_key][pitch_shift_err_key])
    sort_idx = np.argsort(list_harmonic_numbers)
    list_harmonic_numbers = np.array(list_harmonic_numbers)[sort_idx]
    list_f0 = np.array(list_f0)
    array_pitch_shift = np.array(array_pitch_shift)[sort_idx]
    array_pitch_shift_err = np.array(array_pitch_shift_err)[sort_idx]
    
    if cmap_name == 'tab10': num_colors = max(10, len(list_f0))
    else: num_colors = len(list_f0)
    color_list = util_figures.get_color_list(num_colors, cmap_name=cmap_name)
    
    for f0_idx, f0 in enumerate(list_f0):
        xval = list_harmonic_numbers
        yval = array_pitch_shift[:, f0_idx]
        yerr = array_pitch_shift_err[:, f0_idx]
        plot_kwargs = {
            'label': '{:.0f}Hz'.format(f0),
            'marker': 'o',
            'ms': 8,
            'ls': '-',
            'lw': 2,
            'markeredgecolor': 'k',
            'markerfacecolor': color_list[f0_idx],
            'color': color_list[f0_idx],
        }
        if include_yerr:
            errorbar_kwargs = {
                'yerr': 2*yerr,
                'fmt': 'none',
                'ecolor': 'k',
                'elinewidth': 1,
                'capsize': 3,
            }
            ax.errorbar(xval, yval, **errorbar_kwargs)
        ax.plot(xval, yval, **plot_kwargs)
    if ylimits is None:
        buffer_ylim = 0.1
        [xb, yb, dxb, dyb] = ax.dataLim.bounds
        ylimits = [yb - buffer_ylim * dyb, yb + dyb + buffer_ylim * dyb]
    if xlimits is None:
        xlimits = [list_harmonic_numbers[0]-0.5, list_harmonic_numbers[-1]+0.5]
    ax = util_figures.format_axes(ax,
                                  str_xlabel='Mistuned harmonic ({:+.0f}%)'.format(mistuned_pct),
                                  str_ylabel='Shift in reported F0\n(%F0)',
                                  fontsize_labels=fontsize_labels,
                                  fontsize_ticks=fontsize_ticks,
                                  fontweight_labels=None,
                                  xscale='linear',
                                  yscale='linear',
                                  xlimits=xlimits,
                                  ylimits=ylimits,
                                  xticks=list_harmonic_numbers,
                                  yticks=np.arange(0, ylimits[-1]+.01, 0.2),
                                  xticks_minor=[],
                                  yticks_minor=np.arange(0, ylimits[-1]+.01, 0.1),
                                  xticklabels=list_harmonic_numbers,
                                  yticklabels=None,
                                  spines_to_hide=[],
                                  major_tick_params_kwargs_update={},
                                  minor_tick_params_kwargs_update={})
    
    if str_title is not None:
        ax.set_title(str_title, fontsize=fontsize_title)
    if legend_on:
        legend_plot_kwargs = {
            'loc': 'upper right',
            'borderpad': 0.4,
            'borderaxespad': 0.5,
            'handletextpad': 0.8,
            'frameon': False,
            'handlelength': 0.0,
            'markerscale': 1.0,
            'fontsize': fontsize_legend,
        }
        legend_plot_kwargs.update(kwargs_legend)
        ax.legend(**legend_plot_kwargs)
    return bg_results_dict


def make_mistuned_harmonics_line_plot(ax, results_dict_input,
                                      use_relative_shift=True,
                                      expt_key='mistuned_harm',
                                      pitch_shift_key='f0_pred_pct_median',
                                      pitch_shift_err_key=None,
                                      str_title=None,
                                      legend_on=True,
                                      include_yerr=False,
                                      f0_ref=200.0,
                                      restrict_conditions=[1,2,3,4,5,6,12],
                                      plot_kwargs_update={},
                                      fontsize_title=12,
                                      fontsize_labels=12,
                                      fontsize_legend=12,
                                      fontsize_ticks=12,
                                      xlimits=[0, 8],
                                      ylimits=None,
                                      cmap_name='coolwarm',
                                      kwargs_legend={},
                                      kwargs_bootstrap={'bootstrap_repeats': 1000, 'metric_function': 'mean'}):
    '''
    Function for plotting mistuned harmonics experiment results:
    F0 shift as a function of percent mistuning.
    '''
    if pitch_shift_err_key is None: pitch_shift_err_key = pitch_shift_key + '_err'
    if isinstance(results_dict_input, dict):
        results_dict = copy.deepcopy(results_dict_input['f0_ref'][str(f0_ref)])
        for condition in results_dict[expt_key].keys():
            condition_dict = results_dict[expt_key][condition]
            if pitch_shift_err_key not in condition_dict.keys():
                condition_dict[pitch_shift_err_key] = [0] * len(condition_dict[pitch_shift_key])
            if (use_relative_shift) and (0.0 in condition_dict['mistuned_pct']):
                unshifted_idx = condition_dict['mistuned_pct'].index(0.0)
                pitch_shifts = np.array(condition_dict[pitch_shift_key])
                condition_dict[pitch_shift_key] = pitch_shifts - pitch_shifts[unshifted_idx]
    elif isinstance(results_dict_input, list):
        results_dict = {expt_key: {}}
        rd0 = results_dict_input[0]['f0_ref'][str(f0_ref)]
        for condition in rd0[expt_key].keys():
            cd0 = rd0[expt_key][condition]
            if (use_relative_shift) and (0.0 in cd0['mistuned_pct']):
                unshifted_idx = cd0['mistuned_pct'].index(0.0)
                for rdi in results_dict_input:
                    cdi = rdi['f0_ref'][str(f0_ref)][expt_key][condition]
                    pitch_shifts = np.array(cdi[pitch_shift_key])
                    cdi[pitch_shift_key] = pitch_shifts - pitch_shifts[unshifted_idx]
            plot_vals = np.array([rd['f0_ref'][str(f0_ref)][expt_key][condition][pitch_shift_key]
                                  for rd in results_dict_input])
            yval, yerr = combine_subjects(plot_vals, kwargs_bootstrap=kwargs_bootstrap)
            results_dict[expt_key][condition] = {
                'mistuned_pct': rd0[expt_key][condition]['mistuned_pct'],
                pitch_shift_key: yval,
                pitch_shift_err_key: yerr,
            }
    else:
        raise ValueError("INVALID results_dict_input")
    
    condition_list = sorted([int(c) for c in results_dict[expt_key].keys()])
    condition_list = [str(c) for c in condition_list]
    if restrict_conditions is not None:
        condition_list = [str(c) for c in restrict_conditions]
    if cmap_name == 'tab10': num_colors = max(10, len(condition_list))
    else: num_colors = len(condition_list)
    color_list = util_figures.get_color_list(num_colors, cmap_name=cmap_name)
    for cidx, condition in enumerate(condition_list):
        xval = np.array(results_dict[expt_key][condition]['mistuned_pct'])
        xval_idx = np.logical_and(xval >= xlimits[0], xval <= xlimits[1])
        xval = xval[xval_idx]
        yval = np.array(results_dict[expt_key][condition][pitch_shift_key])[xval_idx]
        yerr = np.array(results_dict[expt_key][condition][pitch_shift_err_key])[xval_idx]
        plot_kwargs = {
            'label': condition, 'color': color_list[cidx],
            'marker': '.', 'ms':10, 'ls':'-', 'lw': 2,
        }
        plot_kwargs.update(plot_kwargs_update)
        if not legend_on: plot_kwargs['label'] = None
        if include_yerr:
            ax.fill_between(xval,
                            yval-2*yerr,
                            yval+2*yerr,
                            alpha=0.15,
                            facecolor=plot_kwargs.get('color', color_list[cidx]))
        ax.plot(xval, yval, **plot_kwargs)
    
    if ylimits is None:
        buffer_ylim = 0.3
        [xb, yb, dxb, dyb] = ax.dataLim.bounds
        ylimits = [-0.1, yb + dyb * (1 + buffer_ylim)]
        if ylimits[1] < 0.4: ylimits[1] = 0.4
        if ylimits[1] > 4.0: ylimits[1] = 4.0
    yticks = np.arange(0, 10, 0.3)
    yticks = yticks[np.logical_and(yticks>=ylimits[0], yticks<=ylimits[1])]
    yticks_minor = np.arange(-0.1, 10, 0.1)
    yticks_minor = yticks_minor[np.logical_and(yticks_minor>=ylimits[0], yticks_minor<=ylimits[1])]
    ax = util_figures.format_axes(ax,
                                  str_xlabel='Harmonic mistuning (%)',
                                  str_ylabel='Shift in reported F0\n(%F0)',
                                  fontsize_labels=fontsize_labels,
                                  fontsize_ticks=fontsize_ticks,
                                  fontweight_labels=None,
                                  xscale='linear',
                                  yscale='linear',
                                  xlimits=xlimits,
                                  ylimits=ylimits,
                                  xticks=np.arange(xlimits[0], xlimits[-1]+1, 1),
                                  yticks=yticks,
                                  xticks_minor=None,
                                  yticks_minor=yticks_minor,
                                  xticklabels=None,
                                  yticklabels=None,
                                  spines_to_hide=[],
                                  major_tick_params_kwargs_update={},
                                  minor_tick_params_kwargs_update={})
    
    if str_title is not None:
        ax.set_title(str_title, fontsize=fontsize_title)    
    if legend_on:
        legend_plot_kwargs = {
            'loc': 'upper right',
            'ncol': len(condition_list),
            'columnspacing': 0.8,
            'borderpad': 0.0,
            'borderaxespad': 0.2,
            'handletextpad': 0.4,
            'frameon': False,
            'handlelength': 0.0,
            'markerscale': 1.0,
            'fontsize': fontsize_legend,
        }
        legend_plot_kwargs.update(kwargs_legend)
        leg = ax.legend(**legend_plot_kwargs)
        for line, text in zip(leg.get_lines(), leg.get_texts()):
            text.set_color(line.get_color())
    return results_dict


def make_altphase_line_plot(ax, results_dict_input,
                            expt_key='filter_fl_bin_means',
                            expt_err_key=None,
                            condition_plot_kwargs={},
                            restrict_conditions=[125.0, 3900.0],#[125.0, 1375.0, 3900.0],
                            plot_kwargs_update={},
                            str_title=None,
                            legend_on=True,
                            include_yerr=False,
                            fontsize_title=12,
                            fontsize_labels=12,
                            fontsize_legend=12,
                            fontsize_ticks=12,
                            xlimits=[62.5*0.9, 250*1.1],
                            ylimits=[-1.1, 1.1],
                            kwargs_legend={},
                            kwargs_bootstrap={'bootstrap_repeats': 1000, 'metric_function': 'mean'}):
    '''
    Function for plotting alternating phase experiment results:
    fraction of 2*F0 - 1*F0 judgments as a function of F0 for
    different filter conditions.
    '''
    if expt_err_key is None: expt_err_key = expt_key + '_err'
    if isinstance(results_dict_input, dict):
        results_dict = copy.deepcopy(results_dict_input)
        if expt_err_key not in results_dict.keys():
            results_dict[expt_err_key] = {}
            for condition in results_dict[expt_key].keys():
                results_dict[expt_err_key][condition] = [0] * len(results_dict[expt_key][condition])
    elif isinstance(results_dict_input, list):
        results_dict = {
            'f0_bin_centers': results_dict_input[0]['f0_bin_centers'],
            expt_key: {},
            expt_err_key: {},
        }
        for condition in results_dict_input[0][expt_key].keys():
            plot_vals = np.array([rd[expt_key][condition] for rd in results_dict_input])
            yval, yerr = combine_subjects(plot_vals, kwargs_bootstrap=kwargs_bootstrap)
            results_dict[expt_key][condition] = yval
            results_dict[expt_err_key][condition] = yerr
    else:
        raise ValueError("INVALID results_dict_input")
    
    if not condition_plot_kwargs:
        color_list = util_figures.get_color_list(3, cmap_name='coolwarm')
        condition_plot_kwargs = {
            '125.0': {'label': 'Low harmonics', 'marker': '.', 'ms':10, 'ls':'-', 'lw': 2, 'color': color_list[0]},
            '1375.0': {'label': 'Mid harmonics', 'marker': '.', 'ms':10, 'ls':'-', 'lw': 2, 'color': color_list[1]},
            '3900.0': {'label': 'High harmonics', 'marker': '.', 'ms':10, 'ls':'-', 'lw': 2, 'color': color_list[2]},
        }
    
    condition_list = sorted(results_dict[expt_key].keys())
    if restrict_conditions is not None:
        condition_list = [str(x) for x in restrict_conditions]
    assert set(condition_list).issubset(set(condition_plot_kwargs.keys()))
    for condition in condition_list:
        xval = np.array(results_dict['f0_bin_centers'])
        yval = np.array(results_dict[expt_key][condition])
        yerr = np.array(results_dict[expt_err_key][condition])
        plot_kwargs = condition_plot_kwargs[condition]
        plot_kwargs.update(plot_kwargs_update)
        if not legend_on: plot_kwargs['label'] = None
        if include_yerr:
            ax.fill_between(xval,
                            yval-2*yerr,
                            yval+2*yerr,
                            alpha=0.15,
                            facecolor=plot_kwargs['color'])
        ax.plot(xval, yval, **plot_kwargs)
    
    ax = util_figures.format_axes(ax,
                                  str_xlabel='F0 (Hz)',
                                  str_ylabel='2F0 preferences -\nF0 preferences',
                                  fontsize_labels=fontsize_labels,
                                  fontsize_ticks=fontsize_ticks,
                                  fontweight_labels=None,
                                  xscale='log',
                                  yscale='linear',
                                  xlimits=xlimits,
                                  ylimits=ylimits,
                                  xticks=None,
                                  yticks=np.arange(-1, 1.1, 0.5),
                                  xticks_minor=None,
                                  yticks_minor=np.arange(-1, 1.1, 0.1),
                                  xticklabels=None,
                                  yticklabels=None,
                                  spines_to_hide=[],
                                  major_tick_params_kwargs_update={},
                                  minor_tick_params_kwargs_update={})
    
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
    
    if str_title is not None:
        ax.set_title(str_title, fontsize=fontsize_title)    
    if legend_on:
        legend_plot_kwargs = {
            'loc': 'best',
            'borderpad': 0.4,
            'borderaxespad': 0.5,
            'handletextpad': 0.8,
            'frameon': False,
            'handlelength': 0.0,
            'markerscale': 1.25,
            'fontsize': fontsize_legend,
        }
        legend_plot_kwargs.update(kwargs_legend)
        ax.legend(**legend_plot_kwargs)
    return results_dict


def make_altphase_histograms(results_dict_input,
                             bin_step=0.01,
                             figsize=(9,6),
                             fontsize_labels=16,
                             fontsize_legend=16,
                             fontsize_ticks=14,
                             xticks=[1.0, 1.5, 2.0],
                             xlimits=[0.9, 2.3],
                             yticks=5,
                             ylimits=[0, 25],
                             condition_plot_labels={}):
    '''
    Function for plotting alternating phase experiment results:
    histograms of ratio between reported F0s and target F0s
    for different F0 and filter conditions.
    '''
    if isinstance(results_dict_input, dict):
        results_dict_list = [results_dict_input]
    elif isinstance(results_dict_input, list):
        results_dict_list = results_dict_input
    else:
        raise ValueError("INVALID results_dict_input")
    # Convert list of input result dictionaries to histogram result dictionaries
    hist_results_dict_list = []
    for results_dict in results_dict_list:
        tmp = util_human_model_comparison.get_altphase_histogram_results_dict(results_dict,
                                                                              bin_step=bin_step,
                                                                              bin_limits=xlimits)
        hist_results_dict_list.append(tmp)
    # Pool data across subjects in hist_results_dict_list
    filter_conditions = hist_results_dict_list[0]['filter_conditions']
    f0_conditions = hist_results_dict_list[0]['f0_conditions']
    bin_centers = hist_results_dict_list[0]['bin_centers']
    bin_widths = hist_results_dict_list[0]['bin_widths']
    bin_heights_array = np.zeros_like(hist_results_dict_list[0]['bin_heights_array'])
    for hist_results_dict in hist_results_dict_list:
        assert np.all(filter_conditions == hist_results_dict['filter_conditions'])
        assert np.all(f0_conditions == hist_results_dict['f0_conditions'])
        assert np.all(bin_centers == hist_results_dict['bin_centers'])
        assert np.all(bin_widths == hist_results_dict['bin_widths'])
        bin_heights_array += (hist_results_dict['bin_heights_array'] / len(hist_results_dict_list))
    # Construct the grid of histograms
    NROWS = len(np.unique(filter_conditions))
    NCOLS = len(np.unique(f0_conditions))
    fig, ax_arr = plt.subplots(nrows=NROWS, ncols=NCOLS, figsize=figsize, sharex=True, sharey=True)
    ax_arr = ax_arr.flatten()
    xlabel_idx = NCOLS * NROWS - int(NCOLS/2) - 1
    ylabel_idx = NCOLS
    if not condition_plot_labels:
        condition_plot_labels = {
            '125.0': 'Low',
            '1375.0': 'Mid',
            '3900.0': 'High',
        }
    for idx in range(bin_heights_array.shape[0]):
        ax = ax_arr[idx]
        ax.set_xscale('log')
        label = '{}, {} Hz'.format(condition_plot_labels[str(filter_conditions[idx])],
                                   f0_conditions[idx])
        ax.bar(bin_centers, bin_heights_array[idx], width=bin_widths, align='center', label=label, color='k')
        ax.legend(loc=0, frameon=False, markerscale=0, handlelength=0, fontsize=fontsize_legend)
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))
        ax.set_xlim(xlimits)
        ax.set_xticks(xticks)
        ax.set_xticks(np.arange(xlimits[0], xlimits[1], 0.1), minor=True)
        ax.set_ylim(ylimits)
        ax.set_yticks(np.arange(ylimits[0], ylimits[1], yticks))
        ax.tick_params(axis='y', which='both', labelsize=fontsize_ticks, length=5,
                       direction='inout', right=True, left=True)
        ax.tick_params(axis='x', which='major', labelsize=fontsize_ticks, length=9,
                       direction='inout', top=True, bottom=True)
        ax.tick_params(axis='x', which='minor', length=5,
                       direction='inout', top=True, bottom=True)
    
    ax_arr[xlabel_idx].set_xlabel('Ratio of reported F0 to target F0',
                                  fontsize=fontsize_labels)
    ax_arr[ylabel_idx].set_ylabel('Pitch matches in {:.0f}% wide bins (%)'.format(bin_step*100.0),
                                  fontsize=fontsize_labels)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    return fig, ax_arr


def make_altphase_histogram_plot(ax, results_dict_input,
                                 bin_step=0.0201,
                                 xticks=[1.0, 1.5, 2.0],
                                 xlimits=[0.9, 2.3],
                                 ylimits=[0, 35],
                                 condition_plot_kwargs_filter={},
                                 condition_plot_kwargs_f0={},
                                 restrict_conditions_filter=[125.0, 3900.0],#[125.0, 1375.0, 3900.0],
                                 restrict_conditions_f0=[125.0],
                                 str_title=None,
                                 legend_on=True,
                                 fontsize_title=12,
                                 fontsize_labels=12,
                                 fontsize_legend=12,
                                 fontsize_ticks=12,
                                 include_yerr=False,
                                 kwargs_legend={}):
    '''
    Function for plotting alternating phase experiment results:
    histograms of ratio between reported F0s and target F0s
    for limited number of F0 and spectral conditions.
    '''
    if isinstance(results_dict_input, dict):
        results_dict_list = [results_dict_input]
    elif isinstance(results_dict_input, list):
        results_dict_list = results_dict_input
    else:
        raise ValueError("INVALID results_dict_input")
    # Convert list of input result dictionaries to histogram result dictionaries
    hist_results_dict_list = []
    for results_dict in results_dict_list:
        tmp = util_human_model_comparison.get_altphase_histogram_results_dict(results_dict,
                                                                              bin_step=bin_step,
                                                                              bin_limits=xlimits)
        hist_results_dict_list.append(tmp)
    # Pool data across subjects in hist_results_dict_list
    filter_conditions = hist_results_dict_list[0]['filter_conditions']
    f0_conditions = hist_results_dict_list[0]['f0_conditions']
    bin_centers = hist_results_dict_list[0]['bin_centers']
    bin_widths = hist_results_dict_list[0]['bin_widths']
    bin_heights_array = np.zeros_like(hist_results_dict_list[0]['bin_heights_array'])

    bin_xvals = np.ravel(np.column_stack((bin_centers-bin_widths/2, bin_centers+bin_widths/2)))
    for hist_results_dict in hist_results_dict_list:
        assert np.all(filter_conditions == hist_results_dict['filter_conditions'])
        assert np.all(f0_conditions == hist_results_dict['f0_conditions'])
        assert np.all(bin_centers == hist_results_dict['bin_centers'])
        assert np.all(bin_widths == hist_results_dict['bin_widths'])
        bin_heights_array += (hist_results_dict['bin_heights_array'] / len(hist_results_dict_list))
    
    # Plot pitch match histograms for specified conditions
    if not condition_plot_kwargs_filter:
        color_list = util_figures.get_color_list(3, cmap_name='coolwarm')
        condition_plot_kwargs_filter = {
            '125.0': {'label': 'Low harmonics', 'color': color_list[0], 'alpha':0.3, 'lw':1.3},
            '1375.0': {'label': 'Mid harmonics', 'color': color_list[1], 'alpha':0.3, 'lw':1.3}, 
            '3900.0': {'label': 'High harmonics', 'color': color_list[2], 'alpha':0.3, 'lw':1.3},
        }
    
    for f0_val in restrict_conditions_f0:
        for filter_val in restrict_conditions_filter:
            idx = np.logical_and(f0_conditions==f0_val, filter_conditions==filter_val)
            assert np.sum(idx) == 1
            idx = list(idx).index(True)
            plot_kwargs = condition_plot_kwargs_filter[str(filter_val)]
            label = plot_kwargs.pop('label')
            bin_yvals = np.ravel(np.column_stack((bin_heights_array[idx], bin_heights_array[idx])))
            ax.fill_between(bin_xvals, bin_yvals, **plot_kwargs)
            plot_kwargs.pop('alpha')
            ax.plot(bin_xvals, bin_yvals, label=label, **plot_kwargs)
    
    ax = util_figures.format_axes(ax,
                                  str_xlabel='Reported F0 / stimulus F0',
                                  str_ylabel='Percentage of stimuli',
                                  fontsize_labels=fontsize_labels,
                                  fontsize_ticks=fontsize_ticks,
                                  fontweight_labels=None,
                                  xscale='log',
                                  yscale='linear',
                                  xlimits=xlimits,
                                  ylimits=ylimits,
                                  xticks=xticks,
                                  yticks=np.arange(ylimits[0], ylimits[1]+1, 10),
                                  xticks_minor=np.arange(xlimits[0], xlimits[1], 0.1),
                                  yticks_minor=np.arange(ylimits[0], ylimits[1]+1, 1),
                                  xticklabels=xticks,
                                  yticklabels=None,
                                  spines_to_hide=[],
                                  major_tick_params_kwargs_update={},
                                  minor_tick_params_kwargs_update={})
    
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))
    
    if str_title is not None:
        ax.set_title(str_title, fontsize=fontsize_title)
    if legend_on:
        legend_plot_kwargs = {
            'loc':'best',
            'bbox_to_anchor': (0.0, 0.6, 1.0, 0.4),
            'borderpad': 0.4,
            'borderaxespad': 0.5,
            'handletextpad': 0.8,
            'frameon': False,
            'handlelength': 0.4,
            'markerscale': 0.0,
            'fontsize': fontsize_legend,
        }
        legend_plot_kwargs.update(kwargs_legend)
        leg = ax.legend(**legend_plot_kwargs)
        for legobj in leg.legendHandles:
            legobj.set_linewidth(8.0)
    return hist_results_dict_list


def make_f0dl_threshold_plot(ax, results_dict_input,
                             key_xval=None,
                             str_title=None,
                             str_xlabel=None,
                             str_ylabel='F0 discrimination\nthreshold (%F0)',
                             legend_on=True,
                             include_yerr=True,
                             plot_kwargs_update={},
                             threshold_cap=100.0,
                             fontsize_title=12,
                             fontsize_labels=12,
                             fontsize_legend=12,
                             fontsize_ticks=12,
                             xlimits=None,
                             ylimits=[1e-1, 1e2],
                             xticks=None,
                             xticks_minor=None,
                             kwargs_legend={},
                             kwargs_bootstrap={'bootstrap_repeats': 1000, 'metric_function': 'mean'}):
    '''
    Function for plotting generic F0 discrimination experiment results:
    F0 discrimination thresholds as a function of value specified by `key_xval`.
    '''
    if isinstance(results_dict_input, dict):
        results_dict = copy.deepcopy(results_dict_input)
        f0dls = np.array(results_dict['f0dl'])
        if threshold_cap is not None:
            f0dls[f0dls > threshold_cap] = threshold_cap
        results_dict['log10_f0dl'] = np.log10(f0dls)
        if 'f0dl_err' in results_dict.keys():
            results_dict['log10_f0dl_err'] = np.log10(np.array(results_dict['f0dl_err']))
        else:
            results_dict['log10_f0dl_err'] = np.zeros_like(results_dict['log10_f0dl'])
    elif isinstance(results_dict_input, list):
        f0dls = np.array([rd['f0dl'] for rd in results_dict_input])
        if threshold_cap is not None:
            f0dls[f0dls > threshold_cap] = threshold_cap
        yval, yerr = combine_subjects(np.log10(f0dls), kwargs_bootstrap=kwargs_bootstrap)
        results_dict = {
            key_xval: results_dict_input[0][key_xval],
            'log10_f0dl': yval,
            'log10_f0dl_err': yerr,
        }
    else:
        raise ValueError("INVALID results_dict_input")
    
    xval = np.array(results_dict[key_xval])
    log10_yval = np.array(results_dict['log10_f0dl'])
    log10_yerr = np.array(results_dict['log10_f0dl_err'])
    
    if xlimits is not None:
        IDX = np.logical_and(xval > xlimits[0], xval < xlimits[1])
        xval = xval[IDX]
        log10_yval = log10_yval[IDX]
        log10_yerr = log10_yerr[IDX]
    
    plot_kwargs = {
        'color': 'k',
        'ls': '-',
        'lw': 2,
        'marker': '',
    }
    plot_kwargs.update(plot_kwargs_update)
    if not legend_on:
        plot_kwargs['label'] = None
    if include_yerr:
#         errorbar_kwargs = {
#             'yerr': [
#                 np.power(10.0, log10_yval) - np.power(10.0, log10_yval-2*log10_yerr),
#                 np.power(10.0, log10_yval+2*log10_yerr) - np.power(10.0, log10_yval)
#             ],
#             'fmt': 'none',
#             'ecolor': plot_kwargs.get('color', 'k'),
#             'elinewidth': plot_kwargs.get('lw', 2),
#             'capsize': 1.2*plot_kwargs.get('lw', 2),
#         }
#         ax.errorbar(xval, np.power(10.0, log10_yval), **errorbar_kwargs)
        
        ax.fill_between(xval,
                        np.power(10.0, log10_yval-2*log10_yerr),
                        np.power(10.0, log10_yval+2*log10_yerr),
                        alpha=0.15,
                        facecolor=plot_kwargs.get('color', 'k'))
    ax.plot(xval, np.power(10.0, log10_yval), **plot_kwargs)
    
    ax = util_figures.format_axes(ax,
                                  str_xlabel=str_xlabel,
                                  str_ylabel=str_ylabel,
                                  fontsize_labels=fontsize_labels,
                                  fontsize_ticks=fontsize_ticks,
                                  fontweight_labels=None,
                                  xscale='linear',
                                  yscale='log',
                                  xlimits=xlimits,
                                  ylimits=ylimits,
                                  xticks=None,
                                  yticks=None,
                                  xticks_minor=None,
                                  yticks_minor=None,
                                  xticklabels=None,
                                  yticklabels=None,
                                  spines_to_hide=[],
                                  major_tick_params_kwargs_update={},
                                  minor_tick_params_kwargs_update={})
    if isinstance(xticks, (int, float)):
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(xticks))
    if isinstance(xticks_minor, (int, float)):
        ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(xticks_minor))
    
    if str_title is not None:
        ax.set_title(str_title, fontsize=fontsize_title)
    if legend_on:
        legend_plot_kwargs = {
            'loc': 'lower right',
            'borderpad': 0.4,
            'borderaxespad': 0.5,
            'handletextpad': 0.8,
            'frameon': False,
            'handlelength': 1.5,
            'markerscale': 0.0,
            'fontsize': fontsize_legend,
        }
        legend_plot_kwargs.update(kwargs_legend)
        ax.legend(**legend_plot_kwargs)
    return results_dict
