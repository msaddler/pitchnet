import sys
import os
import numpy as np
import h5py
import json
import argparse
import pdb

import stimuli_util


def compute_running_mean_power_spectrum(signal_list,
                                        sr=32000,
                                        rescaled_dBSPL=60.0,
                                        nopad_start=None,
                                        nopad_end=None,
                                        running_freqs=None,
                                        running_mean_spectrum=None,
                                        running_n=None,
                                        kwargs_power_spectrum={}):
    '''
    '''
    assert len(signal_list.shape) == 2
    for signal_idx in range(signal_list.shape[0]):
        x = signal_list[signal_idx]
        if (nopad_start is not None) and (nopad_end is not None):
            x = x[nopad_start:nopad_end]
        if rescaled_dBSPL is not None:
            x = stimuli_util.set_dBSPL(x, rescaled_dBSPL)
        fxx, pxx = stimuli_util.power_spectrum(x, sr, **kwargs_power_spectrum)
        if running_mean is None:
            running_freqs = fxx
            running_mean_spectrum = np.zeros_like(pxx)
            running_n = 0
        running_mean_spectrum = (pxx + (running_n * running_mean_spectrum)) / (running_n + 1)
        running_n = running_n + 1
    return running_freqs, running_mean_spectrum, running_n


def serial_compute_mean_power_spectrum(source_fn_regex,
                                       output_fn,
                                       key_signal='/stimuli/signal',
                                       key_sr='/sr',
                                       buffer_start_dur=0.070,
                                       buffer_end_dur=0.010,
                                       rescaled_dBSPL=60.0,
                                       kwargs_power_spectrum={}):
    '''
    '''
    source_fn_list = sorted(glob.glob(source_fn_regex))
    running_freqs = None
    running_mean_spectrum = None
    running_n = None
    print('Processing {} files'.format(len(source_fn_list)), flush=True)
    source_fn_idx in range(0, len(source_fn_list)):
        source_fn = source_fn_list[source_fn_idx]
        source_f = h5py.File(source_fn, 'r')
        sr = source_f[key_sr][0]
        signal_list = source_f[key_signal]
        nopad_start = int(buffer_start_dur * sr)
        nopad_end = int(signal_list.shape[1] - buffer_end_dur * sr)
        running_freqs, running_mean_spectrum, running_n = compute_running_mean_power_spectrum(
            signal_list,
            sr=sr,
            rescaled_dBSPL=rescaled_dBSPL,
            nopad_start=nopad_start,
            nopad_end=nopad_end,
            running_freqs=running_freqs,
            running_mean_spectrum=running_mean_spectrum,
            running_n=running_n,
            kwargs_power_spectrum=kwargs_power_spectrum)
        source_f.close()
        progress_str = 'Processed file {} of {} (running_n = {})'
        print(progress_str.format(source_fn_idx+1, len(source_fn_list), running_n), flush=True)
    
    return running_freqs, running_mean_spectrum, running_n

