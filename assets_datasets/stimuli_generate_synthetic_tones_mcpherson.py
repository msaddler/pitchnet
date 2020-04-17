import sys
import os
import numpy as np
import h5py
import scipy.signal
import pdb

sys.path.append('/om2/user/msaddler/python-packages/msutil')
import util_stimuli

sys.path.append('/om4/group/mcdermott/user/msaddler/pitchnet_dataset/pitchnetDataset/pitchnetDataset')
from dataset_util import initialize_hdf5_file, write_example_to_hdf5


def mcpherson_noisy_tone(f0,
                         fs,
                         dur,
                         dbsnr_component=0.0,
                         dbspl_overall=60.0,
                         harmonic_numbers=[1,2,3,4,5,6,7,8,9,10],
                         kwargs_complex_tone={},
                         kwargs_modified_uniform_masking_noise={},
                         noise_filter=None):
    '''
    Generates a harmonic complex tone in modified_uniform_masking_noise with specified
    harmonic numbers, component signal-to-noise ratio, and overall sound pressure level.
    Based on tones used by McPherson et al. ("Harmonicity aids hearing in noise").
    
    Args
    ----
    f0 (float): fundamental frequency (Hz)
    fs (int): sampling rate (Hz)
    dur (float): duration of tone (s)
    dbsnr_component (float): signal-to-noise ratio of each frequency component in complex tone
    dbspl_overall (float): overall stimulus sound pressure level in dB re 20e-6 Pa
    harmonic_numbers (list): list of harmonic numbers to include in tone (equal amplitude)
    kwargs_complex_tone (dict): keyword arguments passed to `util_stimuli.complex_tone()`
    kwargs_modified_uniform_masking_noise (dict): keyword arguments passed to
        `util_stimuli.kwargs_modified_uniform_masking_noise()`
    noise_filter (function): filter function applied to modified_uniform_masking_noise
    
    Returns
    -------
    tone_in_noise (np.ndarray): sound waveform (Pa)
    tone_in_noise_params (dict): dictionary containing signal metadata (snr, snr_per_component)
    '''
    noise = util_stimuli.modified_uniform_masking_noise(fs, dur, **kwargs_modified_uniform_masking_noise)
    if noise_filter is not None:
        noise = noise_filter(noise)
    dbspl_component = util_stimuli.get_dBSPL(noise) + dbsnr_component
    amplitudes = [20e-6 * np.power(10, (dbspl_component/20))] * len(harmonic_numbers)
    signal = util_stimuli.complex_tone(f0, fs, dur,
                                       harmonic_numbers=harmonic_numbers,
                                       amplitudes=amplitudes,
                                       **kwargs_complex_tone)
    tone_in_noise = signal + noise
    tone_in_noise = util_stimuli.set_dBSPL(tone_in_noise, dbspl_overall)
    dbsnr_overall = util_stimuli.get_dBSPL(signal) - util_stimuli.get_dBSPL(noise)
    tone_in_noise_params = {
        'snr': dbsnr_overall,
        'snr_per_component': dbsnr_component,
        'dbspl': dbspl_overall,
    }
    return tone_in_noise, tone_in_noise_params


def generate_mcpherson_noisy_tone_dataset(hdf5_filename,
                                          fs=32e3,
                                          dur=0.150,
                                          phase_modes=['sine'],
                                          harmonic_numbers=[1,2,3,4,5,6,7,8,9,10],
                                          f0_min=80.0,
                                          f0_max=320.0,
                                          step_size_in_octaves=1/(12*16*16),
                                          list_dbsnr_component=np.arange(-32, -10, 1.5),
                                          list_dbspl_overall=[60.0],
                                          kwargs_complex_tone={},
                                          kwargs_modified_uniform_masking_noise={},
                                          noise_filter_cutoff=6e3,
                                          noise_filter_order=6,
                                          noise_filter_type='lowpass',
                                          disp_step=100):
    '''
    '''
    # Define encoding / decoding dictionaries for phase_mode
    phase_mode_encoding = {'sine':0, 'rand':1, 'sch':2, 'cos':3, 'alt':4}
    phase_mode_decoding = {0:'sine', 1:'rand', 2:'sch', 3:'cos', 4:'alt'}
    # Define lists of unique phase modes, SNR values, dBSPL values, and F0 values
    unique_ph_list = np.array([phase_mode_encoding[p] for p in phase_modes])
    unique_snr_list = np.array(list_dbsnr_component)
    unique_dbspl_list = np.array(list_dbspl_overall)
    unique_f0_list = np.arange(0, np.log2(f0_max / f0_min), step_size_in_octaves)
    unique_f0_list = f0_min * (np.power(2, unique_f0_list))
    # Compute number of stimuli (all combinations of phase_mode, SNR, dBSPL, and F0)
    N = len(unique_ph_list) * len(unique_snr_list) * len(unique_dbspl_list) * len(unique_f0_list)
    # Define filter to apply to noise
    if noise_filter_type is None:
        noise_filter = None
    else:
        b, a = scipy.signal.butter(noise_filter_order,
                                   np.array(noise_filter_cutoff)/(fs/2),
                                   btype=noise_filter_type)
        noise_filter = lambda x: scipy.signal.filtfilt(b, a, x)
    
    # Prepare data_dict and config_key_pair_list for hdf5 filewriting
    data_dict = {
        'sr': fs,
        'config_tone/fs': fs,
        'config_tone/dur': dur,
        'config_tone/f0_min': f0_min,
        'config_tone/f0_max': f0_max,
        'config_tone/harmonic_numbers': np.array(harmonic_numbers),
        'config_tone/noise_filter_order': noise_filter_order,
        'config_tone/noise_filter_cutoff': noise_filter_cutoff,
        'config_tone/noise_filter_type': noise_filter_type,
    }
    config_key_pair_list = [(k, k) for k in data_dict.keys()]
    data_key_pair_list = [] # Will be populated right before initializing hdf5 file
    # Main loop to generate the bandpass filtered tones
    itrN = 0
    for ph in unique_ph_list:
        for snr in unique_snr_list:
            for dbspl in unique_dbspl_list:
                for f0 in unique_f0_list:
                    # Construct signal with specified phase_mode, snr, dbspl, f0
                    kwargs_complex_tone.update({'phase_mode': phase_mode_decoding[ph]})
                    tone_in_noise, tone_in_noise_params = mcpherson_noisy_tone(
                        f0,
                        fs,
                        dur,
                        dbsnr_component=snr,
                        dbspl_overall=dbspl,
                        harmonic_numbers=harmonic_numbers,
                        kwargs_complex_tone=kwargs_complex_tone,
                        kwargs_modified_uniform_masking_noise=kwargs_modified_uniform_masking_noise,
                        noise_filter=noise_filter)
                    # Add signal + noise and metadata to data_dict for hdf5 filewriting
                    data_dict['tone_in_noise'] = tone_in_noise.astype(np.float32)
                    data_dict['f0'] = f0
                    data_dict['phase_mode'] = int(ph)
                    data_dict.update(tone_in_noise_params)
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
    # Close hdf5 file
    hdf5_f.close()
    print('[END]: {}'.format(hdf5_filename))


if __name__ == "__main__":
    ''' TEMPORARY COMMAND-LINE USAGE '''
    assert len(sys.argv) == 2, "scipt usage: python <script_name> <hdf5_filename>"
    hdf5_filename = str(sys.argv[1])
    
    generate_mcpherson_noisy_tone_dataset(hdf5_filename)
