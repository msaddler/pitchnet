import sys
import os
import numpy as np
import librosa
import glob
import h5py
import json
import copy
import argparse
import pdb

sys.path.append('/om2/user/msaddler/python-packages/msutil')
import util_stimuli
import util_misc

sys.path.append('/om4/group/mcdermott/user/msaddler/pitchnet_dataset/pitchnetDataset/pitchnetDataset')
import dataset_util


def compute_spectral_statistics(fn_input,
                                fn_output,
                                key_sr='sr',
                                key_f0='nopad_f0_mean',
                                key_signal_list=['stimuli/signal', 'stimuli/noise'],
                                buffer_start_dur=0.070,
                                buffer_end_dur=0.010,
                                rescaled_dBSPL=60.0,
                                kwargs_power_spectrum={},
                                kwargs_spectral_envelope={'M':12},
                                n_mels=40,
                                disp_step=100):
    '''
    '''
    assert not fn_output == fn_input, "input and output hdf5 filenames must be different"
    f_input = h5py.File(fn_input, 'r')
    
    sr = f_input[key_sr][0]
    N = f_input[key_signal_list[0]].shape[0]
    nopad_start = int(buffer_start_dur * sr)
    nopad_end = int(f_input[key_signal_list[0]].shape[1] - buffer_end_dur * sr)
    
    n_fft = nopad_end - nopad_start
    mel_filterbank = librosa.filters.mel(sr, n_fft, n_mels=n_mels)
    
    for itrN in range(N):
        data_dict = {key_f0: f_input[key_f0][itrN]}
        for key_signal in key_signal_list:
            x = f_input[key_signal][itrN, nopad_start:nopad_end]
            if rescaled_dBSPL is not None:
                x = util_stimuli.set_dBSPL(x, rescaled_dBSPL)
                data_dict[key_signal + '_dBSPL'] = rescaled_dBSPL
            else:
                x = x - np.mean(x)
                data_dict[key_signal + '_dBSPL'] = util_stimuli.get_dBSPL(x)
            fxx, pxx = util_stimuli.power_spectrum(x, sr, **kwargs_power_spectrum)
            data_dict[key_signal + '_power_spectrum'] = pxx
            b_lp, a_lp = util_stimuli.get_spectral_envelope_lp_coefficients(x, **kwargs_spectral_envelope)
            data_dict[key_signal + '_spectral_envelope_b_lp'] = b_lp
            data_dict[key_signal + '_spectral_envelope_a_lp'] = a_lp
            data_dict[key_signal + '_mfcc'] = util_stimuli.get_mfcc(x, mel_filterbank)
        
        if itrN == 0:
            print('[INITIALIZING]: {}'.format(fn_output))
            data_key_pair_list = [(k, k) for k in sorted(data_dict.keys())]
            config_dict = {
                key_sr: sr,
                'freqs': fxx,
                'buffer_start_dur': buffer_start_dur,
                'buffer_end_dur': buffer_end_dur,
                'nopad_start': nopad_start,
                'nopad_end': nopad_end,
                'n_fft': n_fft,
                'n_mels': n_mels,
                'mel_filterbank': mel_filterbank,
            }
            config_dict = util_misc.recursive_dict_merge(config_dict, kwargs_power_spectrum)
            config_dict = util_misc.recursive_dict_merge(config_dict, kwargs_spectral_envelope)
            config_key_pair_list = [(k, k) for k in sorted(config_dict.keys())]
            data_dict = util_misc.recursive_dict_merge(data_dict, config_dict)
            dataset_util.initialize_hdf5_file(fn_output,
                                              N,
                                              data_dict,
                                              file_mode='w',
                                              data_key_pair_list=data_key_pair_list,
                                              config_key_pair_list=config_key_pair_list,
                                              fillvalue=-1)
            f_output = h5py.File(fn_output, 'r+')
            for k in sorted(data_dict.keys()):
                print('[___', f_output[k])
        
        # Write each stimulus' data_dict to output hdf5 file
        dataset_util.write_example_to_hdf5(f_output,
                                           data_dict,
                                           itrN,
                                           data_key_pair_list=data_key_pair_list)
        if itrN % disp_step == 0:
            print('... signal {} of {}'.format(itrN, N))
    
    f_input.close()
    f_output.close()
    print('[END]: {}'.format(fn_output))
    return


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
        if (np.isfinite(pxx).all()) and (not np.isnan(pxx).any()):
            if running_freqs is None:
                running_freqs = fxx
                running_mean_spectrum = np.zeros_like(pxx)
                running_n = 0
            running_mean_spectrum = (pxx + (running_n * running_mean_spectrum)) / (running_n + 1)
            running_n = running_n + 1
        else:
            print('Excluding spectrum with nan/inf value(s): {}'.format(signal_idx))
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
                json.dump(results_dict, output_f, cls=util_misc.NumpyEncoder)
            save_str = 'Updated output file: {}'
            print(save_str.format(output_fn), flush=True)
    
    return running_freqs, running_mean_spectrum, running_n


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="compute spectral features of hdf5 dataset")
    parser.add_argument('-r', '--source_fn_regex', type=str, default=None)
    parser.add_argument('-d', '--dest_dir', type=str, default=None)
    parser.add_argument('-skf', '--source_f0_key', type=str, help='source path for f0 values')
    parser.add_argument('-j', '--job_idx', type=int, default=None,
                        help='index of current job')
    parsed_args_dict = vars(parser.parse_args())
    
    source_fn_regex = parsed_args_dict['source_fn_regex']
    source_fn_list = sorted(glob.glob(source_fn_regex))
    
    fn_input = source_fn_list[parsed_args_dict['job_idx']]
    
    dirname_input = os.path.dirname(fn_input)
    dirname_output = parsed_args_dict['dest_dir']
    if os.path.basename(dirname_output) == dirname_output:
        dirname_output = os.path.join(dirname_input, dirname_output)
    if not os.path.exists(dirname_output):
        os.mkdir(dirname_output)
    fn_output = os.path.join(dirname_output, os.path.basename(fn_input))
    
    print('job_idx = {} of {}'.format(parsed_args_dict['job_idx'], len(source_fn_list)))
    print('fn_input = {}'.format(fn_input))
    print('fn_output = {}'.format(fn_output))
    
    compute_spectral_statistics(fn_input,
                                fn_output,
                                key_sr='sr',
                                key_f0=parsed_args_dict['source_f0_key'],
                                key_signal_list=['stimuli/signal', 'stimuli/noise'],
                                buffer_start_dur=0.070,
                                buffer_end_dur=0.010,
                                rescaled_dBSPL=60.0,
                                kwargs_power_spectrum={},
                                kwargs_spectral_envelope={'M':12},
                                disp_step=100)
