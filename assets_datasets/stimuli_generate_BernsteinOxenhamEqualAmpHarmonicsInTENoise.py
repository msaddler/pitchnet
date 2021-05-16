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
    noise = util_stimuli.TENoise(fs, dur, **kwargs_TENoise)
    return signal + noise


def main(hdf5_filename,
         fs=32e3,
         dur=0.150,
         phase_modes=['sine'],
         low_harm_min=1,
         low_harm_max=15,
         num_harm=9,
         base_f0_min=80.0,
         base_f0_max=640.0,
         base_f0_n=192*4*3,
         delta_f0_min=1,
         delta_f0_max=1,
         delta_f0_n=1,
         TENoise_dBSPL_per_ERB=10.0,
         threshold_dBSPL=0.0,
         component_dBSL=45.0,
         include_pure_tones=True,
         inharmonic_jitter=None,
         inharmonic_jitter_fixed=True,
         inharmonic_min_freq_diff=30.0,
         disp_step=100,
         random_seed=858):
    '''
    '''
    np.random.seed(random_seed)
    
    # Define encoding / decoding dictionaries for phase_mode
    phase_mode_encoding = {'sine':0, 'rand':1, 'sch':2, 'cos':3, 'alt':4}
    phase_mode_decoding = {0:'sine', 1:'rand', 2:'sch', 3:'cos', 4:'alt'}
    # Define lists of unique phase modes and low harm numbers
    unique_ph_list = np.array([phase_mode_encoding[p] for p in phase_modes])
    unique_lh_list = np.arange(low_harm_min, low_harm_max + 1)
    # Define list of "base f0" values (Hz), which are used to set reference points
    base_f0_list = np.power(2, np.linspace(np.log2(base_f0_min), np.log2(base_f0_max), base_f0_n))
    # Define list of "delta f0" values (fraction of f0)
    delta_f0_list = np.linspace(delta_f0_min, delta_f0_max, delta_f0_n)
    # Compute number of stimuli (all combinations of low_harm, base_f0, delta_f0, and phase_mode)
    N = len(unique_ph_list) * len(unique_lh_list) * len(base_f0_list) * len(delta_f0_list)
    if include_pure_tones:
        N = N + (len(base_f0_list) * len(delta_f0_list))
    
    if (inharmonic_jitter is not None) and inharmonic_jitter_fixed:
        all_possible_harmonic_numbers = np.arange(1, low_harm_max + num_harm)
        jittered_frequencies = np.zeros_like(all_possible_harmonic_numbers)
        while np.min(np.diff(jittered_frequencies)) < 30.0:
            frequencies = base_f0_min * all_possible_harmonic_numbers
            fixed_jitter_values = np.random.uniform(
                low=-inharmonic_jitter,
                high=inharmonic_jitter,
                size=frequencies.shape)
            jittered_frequencies = frequencies + base_f0_min * fixed_jitter_values
    
    # Prepare data_dict and config_key_pair_list for hdf5 filewriting
    data_dict = {
        'sr': fs,
        'config_tone/fs': fs,
        'config_tone/dur': dur,
        'config_tone/low_harm_min': low_harm_min,
        'config_tone/low_harm_max': low_harm_max,
        'config_tone/num_harm': num_harm,
        'config_tone/base_f0_min': base_f0_min,
        'config_tone/base_f0_max': base_f0_max,
        'config_tone/base_f0_n': base_f0_n,
        'config_tone/delta_f0_min': delta_f0_min,
        'config_tone/delta_f0_max': delta_f0_max,
        'config_tone/delta_f0_n': delta_f0_n,
        'config_tone/TENoise_dBSPL_per_ERB': TENoise_dBSPL_per_ERB,
        'config_tone/threshold_dBSPL': threshold_dBSPL,
        'config_tone/component_dBSL': component_dBSL,
    }
    if inharmonic_jitter is not None:
        data_dict['config_tone/inharmonic_jitter'] = inharmonic_jitter
        if inharmonic_jitter_fixed:
            data_dict['config_tone/inharmonic_jitter_values_fixed'] = fixed_jitter_values
    config_key_pair_list = [(k, k) for k in data_dict.keys()]
    data_key_pair_list = [] # Will be populated right before initializing hdf5 file
    # Main loop to generate the harmonic tones
    itrN = 0
    for lh in unique_lh_list:
        for base_f0 in base_f0_list:
            for delta_f0 in delta_f0_list:
                for ph in unique_ph_list:
                    # Construct tone in noise with specified parameters
                    f0 = base_f0 * delta_f0
                    harmonic_numbers = np.arange(lh, lh + num_harm)
                    harmonic_dBSPL = threshold_dBSPL + component_dBSL
                    amplitudes = 20e-6 * np.power(10, (harmonic_dBSPL/20)) * np.ones_like(harmonic_numbers)
                    kwargs_complex_tone = {
                        'phase_mode': phase_mode_decoding[ph],
                        'harmonic_numbers': harmonic_numbers,
                        'amplitudes': amplitudes,
                    }
                    if inharmonic_jitter is not None:
                        frequencies = f0 * harmonic_numbers
                        if inharmonic_jitter_fixed:
                            _, _, HARM_IDX = np.intersect1d(
                                harmonic_numbers,
                                all_possible_harmonic_numbers,
                                return_indices=True)
                            assert HARM_IDX.shape == harmonic_numbers.shape
                            jittered_frequencies = frequencies + f0 * fixed_jitter_values[HARM_IDX]
                        else:
                            jittered_frequencies = np.zeros_like(harmonic_numbers)
                            while np.min(np.diff(jittered_frequencies)) < inharmonic_min_freq_diff:
                                jitter_values = np.random.uniform(
                                    low=-inharmonic_jitter,
                                    high=inharmonic_jitter,
                                    size=frequencies.shape)
                                jittered_frequencies = frequencies + f0 * jitter_values
                        kwargs_complex_tone = {
                            'phase_mode': phase_mode_decoding[ph],
                            'harmonic_numbers': None,
                            'frequencies': jittered_frequencies,
                            'amplitudes': amplitudes,
                        }
                    kwargs_TENoise = {
                        'dBSPL_per_ERB': TENoise_dBSPL_per_ERB,
                    }
                    tone_in_noise = complex_tone_in_TENoise(f0,
                                                            fs,
                                                            dur,
                                                            kwargs_complex_tone=kwargs_complex_tone,
                                                            kwargs_TENoise=kwargs_TENoise)
                    data_dict['tone_in_noise'] = tone_in_noise.astype(np.float32)
                    data_dict['f0'] = f0
                    data_dict['base_f0'] = base_f0
                    data_dict['delta_f0'] = delta_f0
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
                        print('... signal {} of {}'.format(itrN, N))
                    itrN += 1
    if include_pure_tones:
        for base_f0 in base_f0_list:
            for delta_f0 in delta_f0_list:
                # Construct tone in noise with specified parameters
                f0 = base_f0 * delta_f0
                harmonic_numbers = np.array([1])
                harmonic_dBSPL = threshold_dBSPL + component_dBSL
                amplitudes = 20e-6 * np.power(10, (harmonic_dBSPL/20)) * np.ones_like(harmonic_numbers)
                kwargs_complex_tone = {
                    'phase_mode': 'sine',
                    'harmonic_numbers': harmonic_numbers,
                    'amplitudes': amplitudes,
                }
                kwargs_TENoise = {
                    'dBSPL_per_ERB': TENoise_dBSPL_per_ERB,
                }
                tone_in_noise = complex_tone_in_TENoise(f0,
                                                        fs,
                                                        dur,
                                                        kwargs_complex_tone=kwargs_complex_tone,
                                                        kwargs_TENoise=kwargs_TENoise)
                data_dict['tone_in_noise'] = tone_in_noise.astype(np.float32)
                data_dict['f0'] = f0
                data_dict['base_f0'] = base_f0
                data_dict['delta_f0'] = delta_f0
                data_dict['phase_mode'] = int(ph)
                data_dict['low_harm'] = int(lh)
                data_dict['min_audible_harm'] = int(np.min(harmonic_numbers))
                data_dict['max_audible_harm'] = int(np.max(harmonic_numbers))
                # Write each data_dict to hdf5 file
                write_example_to_hdf5(hdf5_f, data_dict, itrN,
                                      data_key_pair_list=data_key_pair_list)
                if itrN % disp_step == 0:
                    print('... signal {} of {}'.format(itrN, N))
                itrN += 1
    # Close hdf5 file
    hdf5_f.close()
    print('[END]: {}'.format(hdf5_filename))


if __name__ == "__main__":
    ''' TEMPORARY COMMAND-LINE USAGE '''
    assert len(sys.argv) == 2, "scipt usage: python <script_name> <hdf5_filename>"
    hdf5_filename = str(sys.argv[1])
    
    main(hdf5_filename,
         fs=32e3,
         dur=0.150,
         phase_modes=['sine'],
         low_harm_min=1,
         low_harm_max=15,
         num_harm=9,
         base_f0_min=80.0,
         base_f0_max=640.0,
         base_f0_n=192*4*2,
         delta_f0_min=1,
         delta_f0_max=1,
         delta_f0_n=1,
         TENoise_dBSPL_per_ERB=10.0,
         threshold_dBSPL=0.0,
         component_dBSL=45.0,
         include_pure_tones=True,
         inharmonic_jitter=0.5,
         inharmonic_jitter_fixed=True,
         inharmonic_min_freq_diff=30.0,
         disp_step=100,
         random_seed=862)
