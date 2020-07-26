import sys
import os
import io
import contextlib
import numpy as np
import h5py
import glob
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
        print('>>> [EXITING] No indexes remain in {}'.format(hdf5_filename_output))
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
    data_key_pair_list = None
    for itrN in range(itrN_start, N):
        # Run pystraight analysis
        data_dict = {'pystraight_did_fail': np.array(0)}
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
                if key_f0 is not None:
                    data_dict[key_f0] = f_input[key_f0][itrN]
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                data_dict['pystraight_did_fail'] = np.array(1)
                print('------> pystraight failed with itrN={} <------'.format(itrN))
            if (itrN == 0) and data_dict['pystraight_did_fail']:
                raise ValueError("pystraight failed with itrN=0 (cannot initialize output file)")
        
        if data_key_pair_list is None:
            data_key_pair_list = [(k, k) for k in sorted(data_dict.keys())]
        for k in sorted(data_dict.keys()):
            if np.issubdtype(data_dict[k].dtype, np.floating):
                data_dict[k] = data_dict[k].astype(np.float32)
        
        # If output hdf5 file dataset has not been initialized, do so on first iteration
        if not continuation_flag:
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
            assert itrN == 0, "hdf5_filename_output can only be initialized when itrN=0"
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


if __name__ == "__main__":
    regex_hdf5_filename_input = '/om/scratch/*/msaddler/data_pitchnet/PND_v08/noise_TLAS_snr_neg10pos10/*.hdf5'
    list_hdf5_filename_input = sorted(glob.glob(regex_hdf5_filename_input))
    hdf5_filename_input = list_hdf5_filename_input[0]
    
    hdf5_filename_output = 'tmp.hdf5'
    
    run_pystraight_analysis(hdf5_filename_input,
                            hdf5_filename_output)
