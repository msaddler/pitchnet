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


def main(hdf5_filename,
         fs=32e3,
         dur=0.150,
         f0_min=80.0,
         f0_max=10240.0,
         f0_n=50,
         dbspl_min=20.0,
         dbspl_max=60.0,
         dbspl_step=0.25,
         noise_dBHzSPL=10.0,
         noise_attenuation_start=600.0,
         noise_attenuation_slope=2,
         disp_step=100):
    '''
    '''
    # Define list of f0 values (Hz)
    f0_list = np.power(2, np.linspace(np.log2(f0_min), np.log2(f0_max), f0_n))
    # Define list of dbspl values (dB SPL)
    dbspl_list = np.arange(dbspl_min, dbspl_max + dbspl_step/2, dbspl_step)
    # Compute number of stimuli (combinations of f0 and dbspl)
    N = len(f0_list) * len(dbspl_list)
    # Prepare data_dict and config_key_pair_list for hdf5 filewriting
    data_dict = {
        'config_tone/fs': fs,
        'config_tone/dur': dur,
        'config_tone/f0_min': f0_min,
        'config_tone/f0_max': f0_max,
        'config_tone/f0_n': f0_n,
        'config_tone/dbspl_min': dbspl_min,
        'config_tone/dbspl_max': dbspl_max,
        'config_tone/dbspl_step': dbspl_step,
        'config_tone/noise_dBHzSPL': noise_dBHzSPL,
        'config_tone/noise_attenuation_start': noise_attenuation_start,
        'config_tone/noise_attenuation_slope': noise_attenuation_slope,
    }
    config_key_pair_list = [(k, k) for k in data_dict.keys()]
    data_key_pair_list = [] # Will be populated right before initializing hdf5 file
    # Main loop to generate the bandpass filtered tones
    itrN = 0
    for f0 in f0_list:
        for dbspl in dbspl_list:
            noise = util_stimuli.modified_uniform_masking_noise(
                fs,
                dur,
                dBHzSPL=noise_dBHzSPL,
                attenuation_start=noise_attenuation_start,
                attenuation_slope=noise_attenuation_slope)
            signal = util_stimuli.complex_tone(
                f0,
                fs,
                dur,
                harmonic_numbers=[1],
                frequencies=None,
                amplitudes=[20e-6 * np.power(10, (dbspl/20))],
                phase_mode='sine',
                offset_start=True,
                strict_nyquist=True)
            # Add signal + noise and metadata to data_dict for hdf5 filewriting
            tone_in_noise = signal + noise
            data_dict['tone_in_noise'] = tone_in_noise.astype(np.float32)
            data_dict['f0'] = f0
            data_dict['dbspl'] = dbspl
            # Initialize the hdf5 file on the first iteration
            if itrN == 0:
                print('[INITIALIZING]: {}'.format(hdf5_filename))
                for k in data_dict.keys():
                    if not (k, k) in config_key_pair_list:
                        data_key_pair_list.append((k, k))
                initialize_hdf5_file(
                    hdf5_filename,
                    N,
                    data_dict,
                    file_mode='w',
                    data_key_pair_list=data_key_pair_list,
                    config_key_pair_list=config_key_pair_list)
                hdf5_f = h5py.File(hdf5_filename, 'r+')
            # Write each data_dict to hdf5 file
            write_example_to_hdf5(
                hdf5_f,
                data_dict,
                itrN,
                data_key_pair_list=data_key_pair_list)
            if itrN % disp_step == 0:
                print('... signal {:06d} of {:06d}, f0={:04.2f}, dbspl={:04.2f}'.format(
                    itrN,
                    N,
                    f0,
                    dbspl))
            itrN += 1
    # Close hdf5 file
    hdf5_f.close()
    print('[END]: {}'.format(hdf5_filename))
