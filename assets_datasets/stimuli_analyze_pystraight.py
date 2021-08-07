import sys
import os
import h5py
import json
import glob
import time
import numpy as np
import scipy.fftpack
import argparse
import pdb
import warnings
warnings.filterwarnings('ignore')

import dataset_util

# Import pystraight and librosa for `run_pystraight_analysis`
try:
    sys.path.append('/om2/user/msaddler/python-packages/')
    import matlab.engine # matlab.engine MUST BE IMPORTED BEFORE librosa
    import pystraight # pystraight MUST BE IMPORTED BEFORE librosa
    import librosa
except ImportError as e:
    print('[FAILED] `import pystraight; import matlab.engine; import librosa`')

sys.path.append('/packages/msutil')
import util_stimuli # note this package also imports librosa
import util_misc


def summarize_pystraight_statistics(regex_fn,
                                    fn_results='results_dict.json',
                                    key_sr='sr',
                                    key_signal_list=['stimuli/signal']):
    '''
    '''
    list_fn = sorted(glob.glob(regex_fn))
    dict_mean_filter_spectrum = {}
    dict_mfcc = {key: [] for key in key_signal_list}
    
    for itr_fn, fn in enumerate(list_fn):
        with h5py.File(fn, 'r') as f:
            for key in key_signal_list:
                dict_mfcc[key].append(f[key + '_FILTER_spectrumSTRAIGHT_mfcc'][:])
                
                if itr_fn == 0:
                    sr = f[key_sr][0]
                    n_fft = f['n_fft'][0]
                    freqs = f['freqs'][0]
                    dict_mean_filter_spectrum[key] = {
                        'freqs': freqs,
                        'summed_power_spectrum': np.zeros_like(freqs),
                        'count': 0,
                        'n_fft': n_fft,
                    }
                all_filter_spectra = f[key_signal_list[0] + '_FILTER_spectrumSTRAIGHT'][:]
                all_filter_spectra = 10*np.log10(all_filter_spectra)
                for itr_stim in range(all_filter_spectra.shape[0]):
                    filter_spectrum = all_filter_spectra[itr_stim]
                    if np.isfinite(np.sum(filter_spectrum)):
                        dict_mean_filter_spectrum[key]['summed_power_spectrum'] += filter_spectrum
                        dict_mean_filter_spectrum[key]['count'] += 1
            
            print('Processed file {} of {} ({} stim)'.format(
                itr_fn, len(list_fn), dict_mean_filter_spectrum[key]['count']))
    
    for key in key_signal_list:
        print('concatenating {} mfcc arrays'.format(key))
        dict_mfcc[key] = np.concatenate(dict_mfcc[key], axis=0)
    
    results_dict = {}
    for key in sorted(dict_mean_filter_spectrum.keys()):
        mfcc_cov = np.cov(dict_mfcc[key], rowvar=False)
        mfcc_mean = np.mean(dict_mfcc[key], axis=0)
        results_dict[key] = {
            'mfcc_mean': mfcc_mean,
            'mfcc_cov': mfcc_cov,
            'sr': sr,
            'mean_filter_spectrum': dict_mean_filter_spectrum[key]['summed_power_spectrum'],
            'mean_filter_spectrum_freqs': dict_mean_filter_spectrum[key]['freqs'],
            'mean_filter_spectrum_count': dict_mean_filter_spectrum[key]['count'],
            'mean_filter_spectrum_n_fft': dict_mean_filter_spectrum[key]['n_fft'],
        }
        results_dict[key]['mean_filter_spectrum'] /= results_dict[key]['mean_filter_spectrum_count']
    
    if os.path.basename(fn_results) == fn_results:
        fn_results = os.path.join(os.path.dirname(fn), fn_results)
    with open(fn_results, 'w') as f:
        json.dump(results_dict, f, sort_keys=True, cls=util_misc.NumpyEncoder)
    print('[END]: {}'.format(fn_results))
    return


def run_pystraight_analysis(hdf5_filename_input,
                            hdf5_filename_output,
                            key_sr='sr',
                            key_f0=None,
                            key_signal='stimuli/signal',
                            n_mels=40,
                            signal_dBSPL=60.0,
                            buffer_start_dur=0.070,
                            buffer_end_dur=0.010,
                            disp_step=10,
                            kwargs_continuation={'check_key':'pystraight_success', 'check_key_fill_value':-1},
                            kwargs_initialization={},
                            kwargs_matlab_engine={'verbose':0},
                            kwargs_pystraight={'verbose':0}):
    '''
    '''
    # Check if the hdf5 output dataset can be continued and get correct itrN_start
    continuation_flag, itrN_start = dataset_util.check_hdf5_continuation(hdf5_filename_output,
                                                                         **kwargs_continuation)
    if itrN_start is None:
        print('>>> [END] No indexes remain in {}'.format(hdf5_filename_output))
        return
    
    # Open input hdf5 file
    f_input = h5py.File(hdf5_filename_input, 'r+')
    sr = f_input[key_sr][0]
    N = f_input[key_signal].shape[0]
    nopad_start = int(buffer_start_dur * sr)
    nopad_end = int(f_input[key_signal].shape[1] - buffer_end_dur * sr)
    if key_f0 is None:
        if 'f0' in f_input:
            key_f0 = 'f0'
        elif 'nopad_f0_mean' in f_input:
            key_f0 = 'nopad_f0_mean'
        else:
            raise ValueError("`key_f0` must be specified")
    assert key_f0 in f_input, "`key_f0` not found in input hdf5 file"
    
    # Compute n_fft used by PYSTRAIGHT
    n_fft_tmp = nopad_end - nopad_start
    pwr_of_two = 0
    while n_fft_tmp > 2:
        n_fft_tmp /= 2
        pwr_of_two += 1
    n_fft = int(2 ** pwr_of_two)
    freqs = np.fft.rfftfreq(n_fft, d=1/sr)
    
    # Define Mel-scale filterbank and inverse filterbank
    M = librosa.filters.mel(sr, n_fft, n_mels=n_mels)
    Minv = np.linalg.pinv(M)
    
    # Open output hdf5 file if continuing existing dataset
    if continuation_flag:
        print('>>> [CONTINUING] {} from index {} of {}'.format(hdf5_filename_output, itrN_start, N))
        f_output = h5py.File(hdf5_filename_output, 'r+')
    
    # Start MATLAB engine
    eng = pystraight.setup_default_engine(**kwargs_matlab_engine)
    
    # Main loop: iterate over all signals that have not been processed yet
    t_start = time.time()
    first_success = False
    count_failure = 0
    for itrN in range(itrN_start, N):
        # Run pystraight analysis
        data_dict = {
            'pystraight_success': 1,
            key_f0: f_input[key_f0][itrN]
        }
        
        try:
            y = f_input[key_signal][itrN, nopad_start:nopad_end]
            y = util_stimuli.set_dBSPL(y, signal_dBSPL)
            source_params, filter_params, interp_params, did_fail = pystraight.backend_straight_analysis(
                y, sr, matlab_engine=eng, straight_params=kwargs_pystraight)
            
            for k in sorted(filter_params.keys()):
                value = np.array(filter_params[k])
                if np.issubdtype(value.dtype, np.number):
                    # Down-cast floats to np.float32
                    if np.issubdtype(value.dtype, np.floating):
                        value = value.astype(np.float32)
                    output_k = '{}_{}_{}'.format(key_signal, 'FILTER', k)
                    if 'spectrogram' in output_k:
                        output_k = output_k.replace('spectrogram', 'spectrum')
                        value = np.mean(value, axis=-1)
                        mfcc = scipy.fftpack.dct(np.log(np.matmul(M, value)), norm='ortho')
                        data_dict['{}_mfcc'.format(output_k)] = mfcc
                    data_dict[output_k] = value
            for k in sorted(interp_params.keys()):
                value = np.array(interp_params[k])
                if np.issubdtype(value.dtype, np.number):
                    output_k = '{}_{}_{}'.format(key_signal, 'INTERP', k)
                    if np.issubdtype(value.dtype, np.floating):
                        value = value.astype(np.float32)
                    data_dict[output_k] = value
            if did_fail:
                data_dict['pystraight_success'] = 0
                count_failure += 1
            else:
                first_success = True
                data_key_pair_list = [(k, k) for k in sorted(data_dict.keys())]
        except matlab.engine.MatlabExecutionError as e:
            data_dict['pystraight_success'] = 0
            count_failure += 1
            pass
        
        # If output hdf5 file dataset has not been initialized, do so on first successful iteration
        if first_success and (not continuation_flag):
            print('>>> [INITIALIZING] {}'.format(hdf5_filename_output))
            config_dict = {
                key_sr: sr,
                'buffer_start_dur': buffer_start_dur,
                'buffer_end_dur': buffer_end_dur,
                'nopad_start': nopad_start,
                'nopad_end': nopad_end,
                'n_fft': n_fft,
                'freqs': freqs,
            }
            config_key_pair_list = [(k, k) for k in sorted(config_dict.keys())]
            data_dict = util_misc.recursive_dict_merge(data_dict, config_dict)
            dataset_util.initialize_hdf5_file(hdf5_filename_output,
                                              N,
                                              data_dict,
                                              data_key_pair_list=data_key_pair_list,
                                              config_key_pair_list=config_key_pair_list,
                                              **kwargs_initialization)
            continuation_flag = True
            f_output = h5py.File(hdf5_filename_output, 'r+')
            for k in sorted(data_dict.keys()):
                print('[___', k, f_output[k].shape, f_output[k].dtype)
        
        # Write each stimulus' data_dict to output hdf5 file
        if first_success and continuation_flag:
            dataset_util.write_example_to_hdf5(f_output,
                                               data_dict,
                                               itrN,
                                               data_key_pair_list=data_key_pair_list)
        if itrN % disp_step == 0:
            t_mean_per_signal = (time.time() - t_start) / (itrN - itrN_start + 1) # Seconds per signal
            t_est_remaining = (N - itrN - 1) * t_mean_per_signal / 60.0 # Estimated minutes remaining
            disp_str = ('### signal {:06d} of {:06d} | {:06d} failures |'
                        'time_per_signal: {:02.2f} sec | time_est_remaining: {:06.0f} min ###')
            print(disp_str.format(itrN, N, count_failure, t_mean_per_signal, t_est_remaining))
    
    f_input.close()
    f_output.close()
    print('[END]: {}'.format(fn_output))
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run pystraight analysis on stimuli")
    parser.add_argument('-r', '--source_fn_regex', type=str, default=None)
    parser.add_argument('-d', '--dest_dir', type=str, default=None)
    parser.add_argument('-sks', '--source_key_signal', type=str, help='source path for signals')
    parser.add_argument('-j', '--job_idx', type=int, default=None, help='index of current job')
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
    
    run_pystraight_analysis(fn_input,
                            fn_output,
                            key_sr='sr',
                            key_signal=parsed_args_dict['source_key_signal'])
