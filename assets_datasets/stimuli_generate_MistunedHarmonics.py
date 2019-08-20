import sys
import os
import numpy as np
import h5py
import scipy.signal
import itertools
import pdb

import stimuli_util

sys.path.append('/om4/group/mcdermott/user/msaddler/pitchnet_dataset/pitchnetDataset/pitchnetDataset')
from dataset_util import initialize_hdf5_file, write_example_to_hdf5


def generate_MistunedHarmonics_dataset(hdf5_filename, fs=32000, dur=0.150,
                                       f0_ref_list=[100.0, 200.0, 400.0], f0_ref_width=0.04,
                                       step_size_in_octaves=1/(192*8), phase_mode='sine', 
                                       low_harm=1, upp_harm=12, harmonic_dBSPL=60.0,
                                       mistuned_harm_min=1, mistuned_harm_max=6, mistuned_harm_step=1,
                                       mistuned_pct_min=-8.0, mistuned_pct_max=8.0, mistuned_pct_step=1.0,
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
    list_mistuned_pct = list(np.arange(mistuned_pct_min, mistuned_pct_max+mistuned_pct_step, mistuned_pct_step))
    list_mistuned_harm = list(range(mistuned_harm_min, mistuned_harm_max+1, mistuned_harm_step))
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
                signal = stimuli_util.complex_tone(f0, fs, dur, harmonic_numbers=None,
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
    
    generate_MistunedHarmonics_dataset(hdf5_filename)
    