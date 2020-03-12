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

import util_figures

sys.path.append('/om2/user/msaddler/pitchnet/assets_datasets')
import stimuli_util


def make_nervegram_plot(ax, nervegram,
                        sr=20000,
                        cfs=[],
                        fontsize_title=12,
                        fontsize_labels=12,
                        fontsize_legend=12,
                        fontsize_ticks=12,
                        fontweight_labels=None,
                        nxticks=6,
                        nyticks=5,
                        tmin=None,
                        tmax=None,
                        treset=True,
                        vmin=None,
                        vmax=None,
                        vticks=None,
                        str_title=None,
                        str_xlabel='Time (s)',
                        str_ylabel='Characteristic Frequency (Hz)',
                        str_clabel=None):
    '''
    '''
    nervegram = np.squeeze(nervegram)
    assert len(nervegram.shape) == 2, "nervegram must be 2D array"
    cfs = np.array(cfs)
    t = np.arange(0, nervegram.shape[1]) / sr
    
    if (tmin is not None) and (tmax is not None):
        t_IDX = np.logical_and(t >= tmin, t < tmax)
        t = t[t_IDX]
        nervegram = nervegram[:, t_IDX]
    if treset:
        t = t - t[0]
    
    time_idx = np.linspace(0, t.shape[0]-1, nxticks, dtype=int)
    time_labels = ['{:.3f}'.format(t[itr0]) for itr0 in time_idx]
    if not len(cfs) == nervegram.shape[0]:
        cfs = np.arange(0, nervegram.shape[0])
    freq_idx = np.linspace(0, cfs.shape[0]-1, nyticks, dtype=int)
    freq_labels = ['{:.0f}'.format(cfs[itr0]) for itr0 in freq_idx]
    
    cmap = ax.imshow(nervegram, origin='lower', aspect='auto',
                     extent=[0, nervegram.shape[1], 0, nervegram.shape[0]],
                     cmap=matplotlib.cm.gray, vmin=vmin, vmax=vmax)
    
    if str_clabel is not None:
        cbar = plt.colorbar(cmap, ax=ax, pad=0.02)
        cbar.ax.set_ylabel(str_clabel, fontsize=fontsize_labels)
        if vticks is not None:
            cbar.set_ticks(vticks)
        else:
            cbar.ax.yaxis.set_major_locator(plt.MaxNLocator(nyticks, integer=True))
        cbar.ax.tick_params(direction='out',
                            axis='both',
                            which='both',
                            labelsize=fontsize_ticks,
                            length=fontsize_ticks/2)
        from matplotlib.ticker import FormatStrFormatter
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%03d'))
    
    ax = util_figures.format_axes(ax,
                                  str_xlabel=str_xlabel,
                                  str_ylabel=str_ylabel,
                                  fontsize_labels=fontsize_labels,
                                  fontsize_ticks=fontsize_ticks,
                                  fontweight_labels=fontweight_labels,
                                  xlimits=None,
                                  ylimits=None,
                                  xticks=time_idx,
                                  xticklabels=time_labels,
                                  yticks=freq_idx,
                                  yticklabels=freq_labels,
                                  spines_to_hide=[],
                                  major_tick_params_kwargs_update={},
                                  minor_tick_params_kwargs_update={})
    if str_title is not None:
        ax.set_title(str_title, fontsize=fontsize_title)
    return ax


def make_line_plot(ax, x, y,
                   plot_kwargs={},
                   fontsize_title=12,
                   fontsize_labels=12,
                   fontsize_legend=12,
                   fontsize_ticks=12,
                   fontweight_labels=None,
                   str_title=None,
                   str_xlabel=None,
                   str_ylabel=None,
                   xlimits=None,
                   ylimits=None,
                   xticks=[],
                   xticklabels=[],
                   yticks=[],
                   yticklabels=[],
                   legend_on=False,
                   legend_kwargs={},
                   spines_to_hide=['left', 'right', 'bottom', 'top']):
    '''
    '''
    tmp_plot_kwargs = {
        'marker': '',
        'ls': '-',
        'color': [0, 0, 0],
        'lw': 1,
    }
    tmp_plot_kwargs.update(plot_kwargs)
    ax.plot(x, y, **tmp_plot_kwargs)
    ax = util_figures.format_axes(ax,
                                  str_xlabel=str_xlabel,
                                  str_ylabel=str_ylabel,
                                  fontsize_labels=fontsize_labels,
                                  fontsize_ticks=fontsize_ticks,
                                  fontweight_labels=fontweight_labels,
                                  xlimits=xlimits,
                                  ylimits=ylimits,
                                  xticks=xticks,
                                  xticklabels=xticklabels,
                                  yticks=yticks,
                                  yticklabels=yticklabels,
                                  spines_to_hide=spines_to_hide,
                                  major_tick_params_kwargs_update={},
                                  minor_tick_params_kwargs_update={})
    if str_title is not None:
        ax.set_title(str_title, fontsize=fontsize_title)
    if legend_on:
        legend_plot_kwargs = {
            'loc': 'lower right',
            'frameon': False,
            'handlelength': 1.0,
            'markerscale': 1.0,
            'fontsize': fontsize_legend,
        }
        legend_plot_kwargs.update(legend_kwargs)
        ax.legend(**legend_plot_kwargs)
    return ax


def figure_wrapper_nervegram_stimulus(ax_arr,
                                      ax_idx_nervegram=None,
                                      ax_idx_spectrum=None,
                                      ax_idx_excitation=None,
                                      ax_idx_waveform=None,
                                      nervegram=None,
                                      nervegram_sr=None,
                                      waveform=None,
                                      waveform_sr=None,
                                      cfs=[],
                                      tmin=None,
                                      tmax=None,
                                      treset=True,
                                      fontsize_title=12,
                                      fontsize_labels=12,
                                      fontsize_legend=12,
                                      fontsize_ticks=12,
                                      fontweight_labels=None,
                                      spines_to_hide_spectrum=['top', 'bottom', 'left', 'right'],
                                      spines_to_hide_excitation=['top', 'bottom', 'left', 'right'],
                                      spines_to_hide_waveform=['top', 'bottom', 'left', 'right'],
                                      nxticks=6,
                                      nyticks=6,
                                      plot_kwargs={},
                                      limits_buffer=0.1,
                                      ax_arr_clear_leftover=True):
    '''
    '''
    # KEEP TRACK OF AXES IN 1D ARRAY
    ax_arr = np.array([ax_arr]).reshape([-1])
    assert len(ax_arr.shape) == 1
    ax_idx_list = []
    # PLOT AUDITORY NERVEGRAM
    if ax_idx_nervegram is not None:
        ax_idx_list.append(ax_idx_nervegram)
        if ax_idx_spectrum is not None:
            nervegram_nxticks = nxticks
            nervegram_nyticks = 0
            nervegram_str_xlabel = 'Time (s)'
            nervegram_str_ylabel = None
        else:
            nervegram_nxticks = nxticks
            nervegram_nyticks = nyticks
            nervegram_str_xlabel = 'Time (s)'
            nervegram_str_ylabel = 'Characteristic frequency (Hz)'
        make_nervegram_plot(ax_arr[ax_idx_nervegram],
                            nervegram,
                            sr=nervegram_sr,
                            cfs=cfs,
                            fontsize_title=fontsize_title,
                            fontsize_labels=fontsize_labels,
                            fontsize_legend=fontsize_legend,
                            fontsize_ticks=fontsize_ticks,
                            fontweight_labels=fontweight_labels,
                            nxticks=nervegram_nxticks,
                            nyticks=nervegram_nyticks,
                            tmin=tmin,
                            tmax=tmax,
                            treset=treset,
                            str_title=None,
                            str_xlabel=nervegram_str_xlabel,
                            str_ylabel=nervegram_str_ylabel,
                            str_clabel=None)
    # PLOT POWER SPECTRUM
    if ax_idx_spectrum is not None:
        ax_idx_list.append(ax_idx_spectrum)
        fxx, pxx = stimuli_util.power_spectrum(waveform, waveform_sr)
        IDX = np.logical_and(fxx >= np.min(cfs), fxx <= np.max(cfs))
        x_pxx = pxx[IDX]
        y_pxx = stimuli_util.freq2erb(fxx[IDX])
        xlimits_buffer_pxx = limits_buffer * np.max(x_pxx)
        ylimits_pxx = [np.min(y_pxx), np.max(y_pxx)]
        xlimits_pxx = [np.max(x_pxx) + xlimits_buffer_pxx, np.min(x_pxx) - xlimits_buffer_pxx]
        xlimits_pxx[-1] = 0
        yticks = np.linspace(stimuli_util.freq2erb(cfs[0]), stimuli_util.freq2erb(cfs[-1]), nyticks)
        yticklabels = ['{:.0f}'.format(yt) for yt in stimuli_util.erb2freq(yticks)]
        make_line_plot(ax_arr[ax_idx_spectrum], x_pxx, y_pxx,
                       plot_kwargs=plot_kwargs,
                       fontsize_title=fontsize_title,
                       fontsize_labels=fontsize_labels,
                       fontsize_legend=fontsize_legend,
                       fontsize_ticks=fontsize_ticks,
                       fontweight_labels=None,
                       str_title=None,
                       str_xlabel=None,
                       str_ylabel='Frequency (Hz)',
                       xlimits=xlimits_pxx,
                       ylimits=ylimits_pxx,
                       xticks=[],
                       xticklabels=[],
                       yticks=yticks,
                       yticklabels=yticklabels,
                       legend_on=False,
                       legend_kwargs={},
                       spines_to_hide=spines_to_hide_spectrum)
    # PLOT EXCITATION PATTERN
    if ax_idx_excitation is not None:
        ax_idx_list.append(ax_idx_excitation)
        x_exc = np.mean(nervegram, axis=1)
        y_exc = np.arange(0, nervegram.shape[0])
        xlimits_exc_buffer = limits_buffer * np.max(x_exc)
        xlimits_exc = [np.min(x_exc) - xlimits_exc_buffer, np.max(x_exc) + xlimits_exc_buffer]
        ylimits_exc = [np.min(y_exc), np.max(y_exc)]
        make_line_plot(ax_arr[ax_idx_excitation], x_exc, y_exc,
                       plot_kwargs=plot_kwargs,
                       fontsize_title=fontsize_title,
                       fontsize_labels=fontsize_labels,
                       fontsize_legend=fontsize_legend,
                       fontsize_ticks=fontsize_ticks,
                       fontweight_labels=fontweight_labels,
                       str_title=None,
                       str_xlabel=None,
                       str_ylabel=None,
                       xlimits=xlimits_exc,
                       ylimits=ylimits_exc,
                       xticks=[],
                       xticklabels=[],
                       yticks=[],
                       yticklabels=[],
                       legend_on=False,
                       legend_kwargs={},
                       spines_to_hide=spines_to_hide_excitation)
    # PLOT WAVEFORM
    if ax_idx_waveform is not None:
        ax_idx_list.append(ax_idx_waveform)
        y_wav = np.squeeze(waveform)
        assert len(y_wav.shape) == 1, "waveform must be 1D array"
        x_wav = np.arange(0, y_wav.shape[0]) / waveform_sr
        if (tmin is not None) and (tmax is not None):
            IDX = np.logical_and(x_wav >= tmin, x_wav < tmax)
            x_wav = x_wav[IDX]
            y_wav = y_wav[IDX]
        if treset:
            x_wav = x_wav - x_wav[0]
        xlimits_wav = [x_wav[0], x_wav[-1]]
        ylimits_wav = [np.max(np.abs(y_wav)), -np.max(np.abs(y_wav))]
        ylimits_wav = np.array(ylimits_wav) * (1 + limits_buffer)
        make_line_plot(ax_arr[ax_idx_waveform], x_wav, y_wav,
                       plot_kwargs=plot_kwargs,
                       fontsize_title=fontsize_title,
                       fontsize_labels=fontsize_labels,
                       fontsize_legend=fontsize_legend,
                       fontsize_ticks=fontsize_ticks,
                       fontweight_labels=fontweight_labels,
                       str_title=None,
                       str_xlabel=None,
                       str_ylabel=None,
                       xlimits=xlimits_wav,
                       ylimits=ylimits_wav,
                       xticks=[],
                       xticklabels=[],
                       yticks=[],
                       yticklabels=[],
                       legend_on=False,
                       legend_kwargs={},
                       spines_to_hide=spines_to_hide_waveform)
    # CLEAR UNUSED AXES
    if ax_arr_clear_leftover:
        for ax_idx in range(ax_arr.shape[0]):
            if ax_idx not in ax_idx_list:
                ax_arr[ax_idx].axis('off')
    return ax_arr
