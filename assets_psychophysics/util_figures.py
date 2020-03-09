import sys
import os
import json
import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.cm
import matplotlib.colors


def get_color_list(num_colors, cmap_name='Accent'):
    '''
    Helper function returns list of colors for plotting.
    '''
    if isinstance(cmap_name, list): return cmap_name
    cmap = plt.get_cmap(cmap_name)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=num_colors)
    scalar_map = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    color_list = [scalar_map.to_rgba(x) for x in range(num_colors)]
    return color_list


def format_axes(ax,
                str_xlabel=None,
                str_ylabel=None,
                fontsize_labels=12,
                fontsize_ticks=12,
                fontweight_labels=None,
                xscale='linear',
                yscale='linear',
                xlimits=None,
                ylimits=None,
                xticks=None,
                yticks=None,
                xticks_minor=None,
                yticks_minor=None,
                xticklabels=None,
                yticklabels=None,
                spines_to_hide=[],
                major_tick_params_kwargs_update={},
                minor_tick_params_kwargs_update={}):
    '''
    Helper function for setting axes-related formatting parameters.
    '''
    ax.set_xlabel(str_xlabel, fontsize=fontsize_labels, fontweight=fontweight_labels)
    ax.set_ylabel(str_ylabel, fontsize=fontsize_labels, fontweight=fontweight_labels)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlim(xlimits)
    ax.set_ylim(ylimits)
    
    if xticks_minor is not None:
        ax.set_xticks(xticks_minor, minor=True)
    if yticks_minor is not None:
        ax.set_yticks(yticks_minor, minor=True)
    if xticks is not None:
        ax.set_xticks(xticks, minor=False)
    if yticks is not None:
        ax.set_yticks(yticks, minor=False)
    if xticklabels is not None:
        ax.set_xticklabels([], minor=True)
        ax.set_xticklabels(xticklabels, minor=False)
    if yticklabels is not None:
        ax.set_yticklabels([], minor=True)
        ax.set_yticklabels(yticklabels, minor=False)
    
    major_tick_params_kwargs = {
        'axis': 'both',
        'which': 'major',
        'labelsize': fontsize_ticks,
        'length': fontsize_ticks/2,
        'direction': 'out',
    }
    major_tick_params_kwargs.update(major_tick_params_kwargs_update)
    ax.tick_params(**major_tick_params_kwargs)
    
    minor_tick_params_kwargs = {
        'axis': 'both',
        'which': 'minor',
        'labelsize': fontsize_ticks,
        'length': fontsize_ticks/4,
        'direction': 'out',
    }
    minor_tick_params_kwargs.update(minor_tick_params_kwargs_update)
    ax.tick_params(**minor_tick_params_kwargs)
    
    for spine_key in spines_to_hide:
        ax.spines[spine_key].set_visible(False)
    
    return ax


def make_nervegram_plot(ax, subbands,
                        sr_subbands=10000,
                        cfs=[],
                        fontsize_labels=12,
                        fontsize_ticks=10,
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
    subbands = np.squeeze(subbands)
    assert len(subbands.shape) == 2, "subbands must be 2D array"
    cfs = np.array(cfs)
    t_subbands = np.arange(0, subbands.shape[1]) / sr_subbands
    
    if (tmin is not None) and (tmax is not None):
        TIDX_subbands = np.logical_and(t_subbands >= tmin, t_subbands < tmax)
        t_subbands = t_subbands[TIDX_subbands]
        subbands = subbands[:, TIDX_subbands]
    if treset: t_subbands = t_subbands - t_subbands[0]
    
    time_idx = np.linspace(0, t_subbands.shape[0]-1, nxticks, dtype=int)
    time_labels = ['{:.3f}'.format(t_subbands[itr0]) for itr0 in time_idx]
    if not len(cfs) == subbands.shape[0]: cfs = np.arange(0, subbands.shape[0])
    freq_idx = np.linspace(0, cfs.shape[0]-1, nyticks, dtype=int)
    freq_labels = ['{:.0f}'.format(cfs[itr0]) for itr0 in freq_idx]
    
    cmap = ax.imshow(subbands, origin='lower', aspect='auto',
                     extent=[0, subbands.shape[1], 0, subbands.shape[0]],
                     cmap=cm.gray, vmin=vmin, vmax=vmax)
    
    if str_clabel is not None:
        cbar = plt.colorbar(cmap, ax=ax, pad=0.02)
        cbar.ax.set_ylabel(str_clabel, fontsize=fontsize_labels)
        if vticks is not None:
            cbar.set_ticks(vticks)
        else:
            cbar.ax.yaxis.set_major_locator(plt.MaxNLocator(nyticks,
#                                                             min_n_ticks=nyticks,
                                                            integer=True))
        cbar.ax.tick_params(direction='out',
                            axis='both',
                            which='both',
                            labelsize=fontsize_ticks,
                            length=fontsize_ticks/2)
        from matplotlib.ticker import FormatStrFormatter
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%03d'))
    
    ax.set_yticks(freq_idx)
    ax.set_yticklabels(freq_labels)
    ax.set_xticks(time_idx)
    ax.set_xticklabels(time_labels)
    ax.set_xlabel(str_xlabel, fontsize=fontsize_labels)
    ax.set_ylabel(str_ylabel, fontsize=fontsize_labels)
    ax.tick_params(direction='out',
                   axis='both',
                   which='both',
                   labelsize=fontsize_ticks,
                   length=fontsize_ticks/2)
    
    return ax


def make_waveform_plot(ax, waveform, sr_waveform=20000,
                       fontsize_labels=12, fontsize_ticks=10,
                       lw=4.0, tmin=None, tmax=None, treset=True):
    '''
    '''
    waveform = np.squeeze(waveform)
    assert len(waveform.shape) == 1, "waveform must be 1D array"
    t_waveform = np.arange(0, waveform.shape[0]) / sr_waveform
    
    if (tmin is not None) and (tmax is not None):
        TIDX_waveform = np.logical_and(t_waveform >= tmin, t_waveform < tmax)
        t_waveform = t_waveform[TIDX_waveform]
        waveform = waveform[TIDX_waveform]
    if treset: t_waveform = t_waveform - t_waveform[0]
    
#     nxticks = 6
#     time_idx = np.linspace(0, t_waveform.shape[0]-1, nxticks, dtype=int)
#     time_labels = ['{:.3f}'.format(t_waveform[itr0]) for itr0 in time_idx]
    
    ax.plot(t_waveform, waveform, color='k', lw=lw)
    
    ax = util_figures.format_axes(ax,
                                  str_xlabel=None,
                                  str_ylabel=None,
                                  fontsize_labels=fontsize_labels,
                                  fontsize_ticks=fontsize_ticks,
                                  fontweight_labels=None,
                                  xlimits=[t_waveform[0], t_waveform[-1]],
                                  ylimits=None,
                                  xticks=[],
                                  xticklabels=None,
                                  yticks=[],
                                  yticklabels=None,
                                  spines_to_hide=['left', 'right', 'bottom', 'top'],
                                  tick_params_kwargs_update={})
    return ax
