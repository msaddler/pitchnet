import sys
import os
import numpy as np
import h5py
import scipy.signal
import itertools
import pdb

sys.path.append('/om2/user/msaddler/python-packages/msutil')
import util_stimuli

sys.path.append('/om4/group/mcdermott/user/msaddler/pitchnet_dataset/pitchnetDataset/pitchnetDataset')
from dataset_util import initialize_hdf5_file, write_example_to_hdf5


def complex_tone_in_TENoise(f0,
                            fs,
                            dur,
                            kwargs_complex_tone={},
                            kwargs_TENoise={}):
    '''
    '''
    signal = util_stimuli.complex_tone(f0, fs, dur, **kwargs_complex_tone)
    if kwargs_TENoise:
        signal = signal + util_stimuli.TENoise(fs, dur, **kwargs_TENoise)
    return signal


def main(hdf5_filename,
         fs=32e3,
         dur=0.150,
         phase_modes=['sine'],
         low_harm_min=1,
         low_harm_max=30,
         num_harm=12,
         f0_min=80.0,
         f0_max=320.0,
         f0_step_in_octaves=1/384,
         TENoise_dBSPL_per_ERB=10.0,
         component_dBSPL=45.0,
         disp_step=100):
    '''
    '''
    # Define encoding / decoding dictionaries for phase_mode
    phase_mode_encoding = {'sine':0, 'rand':1, 'sch':2, 'cos':3, 'alt':4}
    phase_mode_decoding = {0:'sine', 1:'rand', 2:'sch', 3:'cos', 4:'alt'}
    # Define lists of unique phase modes and low harm numbers
    unique_ph_list = np.array([phase_mode_encoding[p] for p in phase_modes])
    unique_lh_list = np.arange(low_harm_min, low_harm_max + 1)
    # Define list of f0 values
    list_f0 = np.arange(0, np.log2(f0_max / f0_min), f0_step_in_octaves)
    list_f0 = f0_min * (np.power(2, list_f0))
    # Compute number of stimuli (all combinations of phase_mode, low_harm, and f0)
    N = len(unique_ph_list) * len(unique_lh_list) * len(list_f0)
    
    # Prepare data_dict and config_key_pair_list for hdf5 filewriting
    data_dict = {
        'sr': fs,
        'config_tone/fs': fs,
        'config_tone/dur': dur,
        'config_tone/low_harm_min': low_harm_min,
        'config_tone/low_harm_max': low_harm_max,
        'config_tone/num_harm': num_harm,
        'config_tone/f0_min': f0_min,
        'config_tone/f0_max': f0_max,
        'config_tone/f0_step_in_octaves': f0_step_in_octaves,
        'config_tone/component_dBSPL': component_dBSPL,
    }
    if TENoise_dBSPL_per_ERB is not None:
        data_dict['config_tone/TENoise_dBSPL_per_ERB'] = TENoise_dBSPL_per_ERB
    else:
        data_dict['config_tone/TENoise_dBSPL_per_ERB'] = -np.inf
    config_key_pair_list = [(k, k) for k in data_dict.keys()]
    data_key_pair_list = [] # Will be populated right before initializing hdf5 file
    # Main loop to generate the harmonic tones
    itrN = 0
    for ph in unique_ph_list:
        for lh in unique_lh_list:
            for f0 in list_f0:
                    # Construct tone in noise with specified parameters
                    harmonic_numbers = np.arange(lh, lh + num_harm)
                    amplitudes = 20e-6 * np.power(10, (component_dBSPL/20)) * np.ones_like(harmonic_numbers)
                    kwargs_complex_tone = {
                        'phase_mode': phase_mode_decoding[ph],
                        'harmonic_numbers': harmonic_numbers,
                        'amplitudes': amplitudes,
                    }
                    if TENoise_dBSPL_per_ERB is not None:
                        kwargs_TENoise = {
                            'dBSPL_per_ERB': TENoise_dBSPL_per_ERB,
                        }
                    else:
                        kwargs_TENoise = {}
                    tone_in_noise = complex_tone_in_TENoise(f0,
                                                            fs,
                                                            dur,
                                                            kwargs_complex_tone=kwargs_complex_tone,
                                                            kwargs_TENoise=kwargs_TENoise)
                    data_dict['tone_in_noise'] = tone_in_noise.astype(np.float32)
                    data_dict['f0'] = f0
                    data_dict['phase_mode'] = int(ph)
                    data_dict['low_harm'] = int(lh)
                    data_dict['min_audible_harm'] = int(np.min(harmonic_numbers))
                    data_dict['max_audible_harm'] = int(np.max(harmonic_numbers))
                    # Initialize the hdf5 file on the first iteration
                    if itrN == 0:
                        print('[INITIALIZING]: {}'.format(hdf5_filename))
                        for k in data_dict.keys():
                            if not (k, k) in config_key_pair_list:
                                data_key_pair_list.append((k, k))
                        initialize_hdf5_file(hdf5_filename, N, data_dict, file_mode='w',
                                             data_key_pair_list=data_key_pair_list,
                                             config_key_pair_list=config_key_pair_list)
                        hdf5_f = h5py.File(hdf5_filename, 'r+')
                    # Write each data_dict to hdf5 file
                    write_example_to_hdf5(hdf5_f, data_dict, itrN,
                                          data_key_pair_list=data_key_pair_list)
                    if itrN % disp_step == 0:
                        print('... signal {} of {} ({:.3f} dBSPL)'.format(itrN, N, util_stimuli.get_dBSPL(tone_in_noise)))
                    itrN += 1
    # Close hdf5 file
    hdf5_f.close()
    print('[END]: {}'.format(hdf5_filename))


if __name__ == "__main__":
    ''' TEMPORARY COMMAND-LINE USAGE '''
    assert len(sys.argv) == 2, "scipt usage: python <script_name> <hdf5_filename>"
    hdf5_filename = str(sys.argv[1])
    main(hdf5_filename,
         fs=48e3,
         dur=0.150,
         phase_modes=['sine'],
         low_harm_min=1,
         low_harm_max=30,
         num_harm=1,
         f0_min=80.0,
         f0_max=320.0,
         f0_step_in_octaves=1/768,
         TENoise_dBSPL_per_ERB=10.0,
         component_dBSPL=45.0,
         disp_step=100)
