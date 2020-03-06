import sys
import os
import json
import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.cm
import matplotlib.colors


def format_axes(ax,
                str_xlabel=None,
                str_ylabel=None,
                fontsize_labels=12,
                fontsize_ticks=12,
                fontweight_labels=None,
                xlimits=None,
                ylimits=None,
                xticks=None,
                xticklabels=None,
                yticks=None,
                yticklabels=None,
                spines_to_hide=[],
                tick_params_kwargs_update={}):
    '''
    Helper function for setting axes-related formatting parameters.
    '''
    ax.set_xlabel(str_xlabel, fontsize=fontsize_labels, fontweight=fontweight_labels)
    ax.set_ylabel(str_ylabel, fontsize=fontsize_labels, fontweight=fontweight_labels)
    ax.set_xlim(xlimits)
    ax.set_ylim(ylimits)
    
    if xticks is not None:
        ax.set_xticks([], minor=True)
        ax.set_xticks(xticks, minor=False)
    if yticks is not None:
        ax.set_yticks([], minor=True)
        ax.set_yticks(yticks, minor=False)
    if xticklabels is not None:
        ax.set_xticklabels([], minor=True)
        ax.set_xticklabels(xticklabels, minor=False)
    if yticklabels is not None:
        ax.set_yticklabels([], minor=True)
        ax.set_yticklabels(yticklabels, minor=False)
    
    tick_params_kwargs = {
        'axis': 'both',
        'which': 'both',
        'labelsize': fontsize_ticks,
        'length': fontsize_ticks/2,
        'direction': 'out',
    }
    tick_params_kwargs.update(tick_params_kwargs_update)
    ax.tick_params(**tick_params_kwargs)
    
    for spine_key in spines_to_hide:
        ax.spines[spine_key].set_visible(False)
    
    return ax