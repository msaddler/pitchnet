import sys
import os
import numpy as np
import h5py
import scipy.signal
import itertools
import pdb

sys.path.append('/packages/msutil')
import util_stimuli

from dataset_util import initialize_hdf5_file, write_example_to_hdf5


def generate_MistunedHarmonics_dataset(hdf5_filename,
                                       fs=32000,
                                       dur=0.150,
                                       f0_ref_list=[100.0, 200.0, 400.0],
                                       f0_ref_width=0.04,
                                       step_size_in_octaves=1/(192*8),
                                       phase_mode='sine', 
                                       low_harm=1,
                                       upp_harm=12,
                                       harmonic_dBSPL=60.0,
                                       list_mistuned_pct=[-8, -6, -4, -3, -2, -1, 0, 1, 2, 3, 4, 6, 8],
                                       list_mistuned_harm=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                                       noise_params={},
                                       disp_step=100):
    '''
    Main routine for generating Moore et al. (1985, JASA) mistuned harmonics dataset.
    
    Args
    ----
    '''
    # Define encoding / decoding dictionaries for phase_mode
    phase_mode_encoding = {'sine':0, 'rand':1, 'sch':2, 'cos':3, 'alt':4}
    phase_mode_decoding = {0:'sine', 1:'rand', 2:'sch', 3:'cos', 4:'alt'}
    
    # Define stimulus-specific parameters
    list_f0 = []
    for f0_ref in f0_ref_list:
        f0_min = f0_ref * (1-f0_ref_width)
        f0_max = f0_ref * (1+f0_ref_width)
        list_f0.extend(list(f0_min * (np.power(2, np.arange(0, np.log2(f0_max / f0_min), step_size_in_octaves)))))
    list_mistuned_pct = np.array(list_mistuned_pct).astype(float)
    list_mistuned_harm = np.array(list_mistuned_harm).astype(int)
    N = len(list_f0) * len(list_mistuned_pct) * len(list_mistuned_harm)
    
    # Define stimulus-shared parameters
    phase = phase_mode_encoding[phase_mode]
    harmonic_numbers = np.arange(low_harm, upp_harm+1, dtype=int)
    amplitudes = 20e-6 * np.power(10, (harmonic_dBSPL/20)) * np.ones_like(harmonic_numbers)
    
    # Prepare config_dict with config values
    config_dict = {
        'sr': fs,
        'config_tone/fs': fs,
        'config_tone/dur': dur,
        'config_tone/harmonic_dBSPL': harmonic_dBSPL,
        'config_tone/phase_mode': phase,
        'config_tone/low_harm': low_harm,
        'config_tone/upp_harm': upp_harm,
    }
    config_key_pair_list = [(k, k) for k in config_dict.keys()]
    
    # Iterate over all combinations of stimulus-specific parameters
    itrN = 0
    for mistuned_harm in list_mistuned_harm:
        for mistuned_pct in list_mistuned_pct:
            for f0 in list_f0:
                # Build signal with specified harmonic mistuning and f0
                harmonic_freqs = f0 * harmonic_numbers
                mistuned_index = harmonic_numbers == mistuned_harm
                harmonic_freqs[mistuned_index] = (1.0 + mistuned_pct/100.0) * harmonic_freqs[mistuned_index]
                signal = util_stimuli.complex_tone(f0, fs, dur, harmonic_numbers=None,
                                                   frequencies=harmonic_freqs,
                                                   amplitudes=amplitudes,
                                                   phase_mode=phase_mode)
                # Add signal and metadata to data_dict for hdf5 filewriting
                data_dict = {
                    'stimuli/signal': signal.astype(np.float32),
                    'f0': np.float32(f0),
                    'mistuned_harm': int(mistuned_harm),
                    'mistuned_pct': np.float32(mistuned_pct),
                }
                # If noise_params is specified, add UMNm
                if noise_params:
                    noise = util_stimuli.modified_uniform_masking_noise(fs, dur, **noise_params)
                    signal_in_noise = signal + noise
                    data_dict['stimuli/signal_in_noise'] = signal_in_noise.astype(np.float32)
                # Initialize output hdf5 dataset on first iteration
                if itrN == 0:
                    print('[INITIALIZING]: {}'.format(hdf5_filename))
                    data_dict.update(config_dict)
                    data_key_pair_list = [(k, k) for k in set(data_dict.keys()).difference(config_dict.keys())]
                    initialize_hdf5_file(hdf5_filename, N, data_dict, file_mode='w',
                                         data_key_pair_list=data_key_pair_list,
                                         config_key_pair_list=config_key_pair_list)
                    hdf5_f = h5py.File(hdf5_filename, 'r+')
                # Write each data_dict to hdf5 file
                write_example_to_hdf5(hdf5_f, data_dict, itrN, data_key_pair_list=data_key_pair_list)
                if itrN % disp_step == 0: print('... signal {} of {}, (f0={})'.format(itrN, N, f0))
                itrN += 1
    
    # Close hdf5 file
    hdf5_f.close()
    print('[END]: {}'.format(hdf5_filename))


if __name__ == "__main__":
    ''' TEMPORARY COMMAND-LINE USAGE '''
    assert len(sys.argv) == 2, "scipt usage: python <script_name> <hdf5_filename>"
    hdf5_filename = str(sys.argv[1])
    
    generate_MistunedHarmonics_dataset(hdf5_filename,
                                       noise_params={'dBHzSPL':15.0, 'attenuation_start':600.0, 'attenuation_slope':2.0})
