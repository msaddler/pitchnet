import sys
import os
import numpy as np
import h5py
import glob
import time
import pdb
import argparse
import warnings
warnings.filterwarnings('ignore')

sys.path.append('/om2/user/msaddler/python-packages/')
import pystraight
import matlab.engine

sys.path.append('/om2/user/msaddler/python-packages/msutil')
import util_stimuli
import util_misc

sys.path.append('/om4/group/mcdermott/user/msaddler/pitchnet_dataset/pitchnetDataset/pitchnetDataset')
import dataset_util


def run_pystraight_analysis(hdf5_filename_input,
                            hdf5_filename_output,
                            key_sr='sr',
                            key_f0='nopad_f0_mean',
                            key_signal_list=['stimuli/signal'],
                            signal_dBSPL=60.0,
                            buffer_start_dur=0.070,
                            buffer_end_dur=0.010,
                            disp_step=10,
                            kwargs_continuation={'check_key':'pystraight_did_fail', 'check_key_fill_value':-1},
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
    N = f_input[key_signal_list[0]].shape[0]
    nopad_start = int(buffer_start_dur * sr)
    nopad_end = int(f_input[key_signal_list[0]].shape[1] - buffer_end_dur * sr)
    
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
        data_dict = {'pystraight_did_fail': np.array(0)}
        if key_f0 is not None:
            data_dict[key_f0] = f_input[key_f0][itrN]
        for key_signal in key_signal_list:
            try:
                y = f_input[key_signal][itrN, nopad_start:nopad_end]
                y = util_stimuli.set_dBSPL(y, signal_dBSPL)
                source_params, filter_params, interp_params, did_fail = pystraight.backend_straight_analysis(
                    y, sr, matlab_engine=eng, straight_params=kwargs_pystraight)
#                 for k in sorted(source_params.keys()):
#                     source_params[k] = np.array(source_params[k])
#                     if np.issubdtype(source_params[k].dtype, np.number):
#                         data_dict['{}_{}_{}'.format(key_signal, 'SOURCE', k)] = source_params[k]
                for k in sorted(filter_params.keys()):
                    filter_params[k] = np.array(filter_params[k])
                    if np.issubdtype(filter_params[k].dtype, np.number):
                        data_dict['{}_{}_{}'.format(key_signal, 'FILTER', k)] = filter_params[k]
                for k in sorted(interp_params.keys()):
                    interp_params[k] = np.array(interp_params[k])
                    if np.issubdtype(interp_params[k].dtype, np.number):
                        data_dict['{}_{}_{}'.format(key_signal, 'INTERP', k)] = interp_params[k]
                data_dict['{}_{}'.format(key_signal, 'pystraight_did_fail')] = np.array(int(did_fail))
                if not did_fail:
                    first_success = True
                    data_key_pair_list = [(k, k) for k in sorted(data_dict.keys())]
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                count_failure += 1
                data_dict['pystraight_did_fail'] = np.array(1)
                print('------> pystraight failed with itrN={} <------'.format(itrN))
        
        for k in sorted(data_dict.keys()):
            if np.issubdtype(data_dict[k].dtype, np.floating):
                data_dict[k] = data_dict[k].astype(np.float32)
        
        # If output hdf5 file dataset has not been initialized, do so on first successful iteration
        if first_success and (not continuation_flag):
            print('>>> [INITIALIZING] {}'.format(hdf5_filename_output))
            
            config_dict = {
                key_sr: sr,
                'buffer_start_dur': buffer_start_dur,
                'buffer_end_dur': buffer_end_dur,
                'nopad_start': nopad_start,
                'nopad_end': nopad_end,
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
    parser.add_argument('-skf', '--source_key_f0', type=str, help='source path for f0 values')
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
                            key_f0=parsed_args_dict['source_key_f0'],
                            key_signal_list=[parsed_args_dict['source_key_signal']],
                            signal_dBSPL=60.0,
                            buffer_start_dur=0.070,
                            buffer_end_dur=0.010,
                            disp_step=10,
                            kwargs_continuation={'check_key':'pystraight_did_fail', 'check_key_fill_value':-1},
                            kwargs_initialization={},
                            kwargs_matlab_engine={'verbose':0},
                            kwargs_pystraight={'verbose':0})
