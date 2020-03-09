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


def make_nervegram_plot(ax, nervegram,
                        sr=20000,
                        cfs=[],
                        fontsize_labels=12,
                        fontsize_ticks=12,
                        nxticks=6,
                        nyticks=5,
                        tmin=None,
                        tmax=None,
                        treset=True,
                        vmin=None,
                        vmax=None,
                        vticks=None,
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
                                  fontweight_labels=None,
                                  xlimits=None,
                                  ylimits=None,
                                  xticks=time_idx,
                                  xticklabels=time_labels,
                                  yticks=freq_idx,
                                  yticklabels=freq_labels,
                                  spines_to_hide=[],
                                  major_tick_params_kwargs_update={},
                                  minor_tick_params_kwargs_update={})
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
    ax.plot(x, y, **plot_kwargs)
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


def make_waveform_plot(ax,
                       waveform,
                       sr=32000,
                       fontsize_labels=12,
                       fontsize_ticks=12,
                       lw=4.0,
                       tmin=None,
                       tmax=None,
                       treset=True):
    '''
    '''
    waveform = np.squeeze(waveform)
    assert len(waveform.shape) == 1, "waveform must be 1D array"
    t = np.arange(0, waveform.shape[0]) / sr
    
    if (tmin is not None) and (tmax is not None):
        t_IDX = np.logical_and(t >= tmin, t < tmax)
        t = t[t_IDX]
        waveform = waveform[t_IDX]
    if treset:
        t = t - t[0]
    
    ax.plot(t, waveform, color='k', lw=lw)
    
    ax = util_figures.format_axes(ax,
                                  str_xlabel=None,
                                  str_ylabel=None,
                                  fontsize_labels=fontsize_labels,
                                  fontsize_ticks=fontsize_ticks,
                                  fontweight_labels=None,
                                  xlimits=[t[0], t[-1]],
                                  ylimits=None,
                                  xticks=[],
                                  xticklabels=None,
                                  yticks=[],
                                  yticklabels=None,
                                  spines_to_hide=['left', 'right', 'bottom', 'top'],
                                  major_tick_params_kwargs_update={},
                                  minor_tick_params_kwargs_update={})
    return ax
