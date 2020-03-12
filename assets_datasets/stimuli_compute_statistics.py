import sys
import os
import numpy as np
import glob
import h5py
import json
import copy
import argparse
import pdb

import util_stimuli


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
            x = util_stimuli.set_dBSPL(x, rescaled_dBSPL)
        fxx, pxx = util_stimuli.power_spectrum(x, sr, **kwargs_power_spectrum)
        if running_freqs is None:
            running_freqs = fxx
            running_mean_spectrum = np.zeros_like(pxx)
            running_n = 0
        running_mean_spectrum = (pxx + (running_n * running_mean_spectrum)) / (running_n + 1)
        running_n = running_n + 1
    return running_freqs, running_mean_spectrum, running_n


def serial_compute_mean_power_spectrum(source_fn_regex,
                                       output_fn=None,
                                       key_signal='/stimuli/signal',
                                       key_sr='/sr',
                                       buffer_start_dur=0.070,
                                       buffer_end_dur=0.010,
                                       rescaled_dBSPL=60.0,
                                       kwargs_power_spectrum={}):
    '''
    '''
    CONFIG = copy.deepcopy(locals())
    source_fn_list = sorted(glob.glob(source_fn_regex))
    initial_source_fn_idx = 0
    running_freqs = None
    running_mean_spectrum = None
    running_n = None
    print('Total files to process: {}'.format(len(source_fn_list)), flush=True)
    for key in sorted(CONFIG.keys()):
        print('CONFIG', key, CONFIG[key], flush=True)
    
    if (output_fn is not None) and (os.path.isfile(output_fn)):
        print('Loading data from existing output_fn: {}'.format(output_fn), flush=True)
        with open(output_fn, 'r') as output_f:
            results_dict = json.load(output_f)
        initial_source_fn_idx = results_dict['source_fn_idx'] + 1
        running_freqs = np.array(results_dict['freqs'])
        running_mean_spectrum = np.array(results_dict['mean_spectrum'])
        running_n = results_dict['n']
        assert results_dict['key_signal'] == key_signal
        assert results_dict['key_sr'] == key_sr
        print('Setting initial_source_fn_idx={} (running_n={})'.format(
            initial_source_fn_idx, running_n), flush=True)
    
    for source_fn_idx in range(initial_source_fn_idx, len(source_fn_list)):
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
        progress_str = 'Processed file {} of {} (running_n={})'
        print(progress_str.format(source_fn_idx+1, len(source_fn_list), running_n), flush=True)
        if output_fn is not None:
            results_dict = {
                'CONFIG': CONFIG,
                'source_fn_regex': source_fn_regex,
                'source_fn_idx': source_fn_idx,
                'key_signal': key_signal,
                'key_sr': key_sr,
                'freqs': running_freqs,
                'mean_spectrum': running_mean_spectrum,
                'n': running_n,
            }
            # Write results_dict to json_results_dict_fn
            with open(output_fn, 'w') as output_f:
                json.dump(results_dict, output_f, cls=NumpyEncoder)
            save_str = 'Updated output file: {}'
            print(save_str.format(output_fn), flush=True)
    
    return running_freqs, running_mean_spectrum, running_n


class NumpyEncoder(json.JSONEncoder):
    ''' Helper class to JSON serialize the results_dict '''
    def default(self, obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.int64): return int(obj)  
        return json.JSONEncoder.default(self, obj)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="compute mean power spectrum of hdf5 dataset")
    parser.add_argument('-r', '--source_fn_regex', type=str, default=None)
    parser.add_argument('-o', '--output_fn', type=str, default=None)
    parser.add_argument('-k', '--key_signal', type=str, default='/stimuli/signal')
    parser.add_argument('-ksr', '--key_sr', type=str, default='/sr')
    parsed_args_dict = vars(parser.parse_args())
    
    source_fn_regex = parsed_args_dict['source_fn_regex']
    output_fn = parsed_args_dict['output_fn']
    key_signal = parsed_args_dict['key_signal']
    key_sr = parsed_args_dict['key_sr']
    assert source_fn_regex is not None, "source_fn_regex is a required argument"
    if output_fn is None:
        output_fn_dirname = os.path.dirname(source_fn_regex)
        output_fn_basename = 'TMP_mean_spectrum_KEY{}.json'.format(key_signal.replace('/', '_'))
        output_fn = os.path.join(output_fn_dirname, output_fn_basename)
    
    serial_compute_mean_power_spectrum(source_fn_regex,
                                       output_fn=output_fn,
                                       key_signal=key_signal,
                                       key_sr=key_sr,
                                       buffer_start_dur=0.070,
                                       buffer_end_dur=0.010,
                                       rescaled_dBSPL=60.0,
                                       kwargs_power_spectrum={})
