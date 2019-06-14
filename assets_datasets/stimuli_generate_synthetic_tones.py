import sys
import os
import numpy as np
import h5py
import scipy.signal

sys.path.append('/om2/user/msaddler/python-packages/bez2018model')
from bez2018model_run_hdf5_dataset import initialize_hdf5_file, write_example_to_hdf5


def complex_tone(f0, fs, dur, harmonic_numbers=[1], amplitudes=None, phase_mode='sine',
                 offset_start=True, strict_nyquist=True):
    '''
    Function generates a complex harmonic tone with specified relative phase
    and component amplitudes.
    
    Args
    ----
    f0 (float): fundamental frequency (Hz)
    fs (int): sampling rate (Hz)
    dur (float): duration of tone (s)
    harmonic_numbers (list): harmonic numbers to include in complex tone (sorted lowest to highest)
    amplitudes (list): amplitudes of individual harmonics (None = equal amplitude harmonics)
    phase_mode (str): specify relative phases (`sch` and `alt` assume contiguous harmonics)
    offset_start (bool): if True, starting phase is offset by np.random.rand()/f0
    strict_nyquist (bool): if True, function will raise ValueError if Nyquist is exceeded;
        if False, frequencies above the Nyquist will be silently ignored
    
    Returns
    -------
    signal (np array): complex tone
    '''
    # Time vector has step size 1/fs and is of length int(dur*fs)
    t = np.arange(0, dur, 1/fs)[0:int(dur*fs)]
    if offset_start: t = t + (1/f0) * np.random.rand()
    # Create array of harmonic_numbers and set default amplitudes if not provided
    harmonic_numbers = np.array(harmonic_numbers).reshape([-1])
    if amplitudes is None:
        amplitudes = 1/len(harmonic_numbers) * np.ones_like(harmonic_numbers)
    else:
        assert_msg = "provided `amplitudes` must be same length as `harmonic_numbers`"
        assert len(amplitudes) == len(harmonic_numbers), assert_msg
    # Create array of harmonic phases using phase_mode
    if phase_mode.lower() == 'sine':
        phase_list = np.zeros(len(harmonic_numbers))
    elif (phase_mode.lower() == 'rand') or (phase_mode.lower() == 'random'):
        phase_list = 2*np.pi * np.random.rand(len(harmonic_numbers))
    elif (phase_mode.lower() == 'sch') or (phase_mode.lower() == 'schroeder'):
        phase_list = np.pi/2 + (np.pi * np.square(harmonic_numbers) / len(harmonic_numbers))
    elif (phase_mode.lower() == 'cos') or (phase_mode.lower() == 'cosine'):
        phase_list = np.pi/2 * np.ones(len(harmonic_numbers))
    elif (phase_mode.lower() == 'alt') or (phase_mode.lower() == 'alternating'):
        phase_list = np.pi/2 * np.ones(len(harmonic_numbers))
        phase_list[::2] = 0
    else:
        raise ValueError('Unsupported phase_mode: {}'.format(phase_mode))
    # Build and return the complex tone
    signal = np.zeros_like(t)
    for harm_num, amp, phase in zip(harmonic_numbers, amplitudes, phase_list):
        f = f0 * harm_num
        if f > fs/2:
            if strict_nyquist: raise ValueError('Nyquist frequency exceeded')
            else: break
        signal += amp * np.sin(2*np.pi*f*t + phase)
    return signal


def lowpass_complex_tone(f0, fs, dur, attenuation_start=1000., attenuation_slope=0.,
                         phase_mode='sine', amp_noise=0.):
    '''
    Function generates a lowpass complex tone with harmonics attenuated
    linearly on a log scale above specified cutoff.
    
    Args
    ----
    f0 (float): fundamental frequency (Hz)
    fs (int): sampling rate (Hz)
    dur (float): duration of tone (s)
    attenuation_start (float): cutoff frequency for start of attenuation (Hz)
    attenuation_slope (float): slope in units of dB/octave above attenuation_start
    phase_mode (str): specifies relative phase between harmonics
    amp_noise (float): if > 0, scales noise multiplied with harmonic amplitudes
    
    Returns
    -------
    '''
    # Compute harmonic numbers and relative amplitude of each harmonic
    harmonics = np.arange(f0, fs/2, f0)
    harmonic_numbers = harmonics / f0
    dB_attenuation = -attenuation_slope * np.log2(harmonics / attenuation_start)
    dB_attenuation[dB_attenuation > 0] = 0
    amplitudes = np.power(10, (dB_attenuation/20))
    # Jitter the harmonic amplitudes with multiplicative noise
    if amp_noise > 0:
        amplitude_noise = np.random.uniform(1-amp_noise, 1+amp_noise, amplitudes.shape)
        amplitudes = amplitude_noise * amplitudes
    # Build and return the complex tone
    signal = complex_tone(f0, fs, dur, harmonic_numbers=harmonic_numbers,
                          amplitudes=amplitudes,
                          phase_mode=phase_mode,
                          offset_start=True, strict_nyquist=True)
    return signal


def power_spectrum(x, fs, rfft=True, dBSPL=True):
    '''
    Helper function for computing power spectrum of sound wave.
    
    Args
    ----
    x (np array): input waveform (units Pa)
    fs (int): sampling rate (Hz)
    rfft (bool): if True, only positive half of power spectra is returned
    dBSPL (bool): if True, power spectrum is rescaled to dB re 20e-6 Pa
    
    Returns
    -------
    freqs (np array): frequency vector (Hz)
    power_spectrum (np array): power spectrum (Pa^2 or dB SPL)
    '''
    if rfft:
        power_spectrum = np.square(np.abs(np.fft.rfft(x) / len(x)))
        freqs = np.fft.rfftfreq(len(x), d=1/fs)
    else:
        power_spectrum = np.square(np.abs(np.fft.fft(x) / len(x)))
        freqs = np.fft.fftfreq(len(x), d=1/fs)
    if dBSPL:
        power_spectrum = 10. * np.log10(power_spectrum / np.square(20e-6)) 
    return freqs, power_spectrum


def generate_lowpass_complex_tone_dataset(hdf5_filename, N, fs=32e3, dur=0.150,
                                          disp_step=100, f0_min=80., f0_max=1e3,
                                          amp_noise=0., phase_mode_list=['sine'],
                                          atten_start_min=10., atten_start_max=4e3,
                                          atten_slope_min=10., atten_slope_max=60.):
    '''
    Main routine for generating a dataset of randomly-sampled lowpass complex tones.
    
    Args
    ----
    hdf5_filename (str): 
    N (int):
    fs (int): sampling rate (Hz)
    dur (float): duration of tones (s)
    disp_step (int): every disp_step, progress is displayed
    f0_min (float):
    f0_max (float):
    amp_noise (float):
    phase_mode_list (list):
    atten_start_min (float):
    atten_start_max (float):
    atten_slope_min (float):
    atten_slope_max (float):
    '''
    # Randomly sample f0s, phase_modes, and attenuation parameters
    f0_list = np.exp(np.random.uniform(np.log(f0_min), np.log(f0_max), size=[N]))
    phase_mode_list = np.random.choice(phase_mode_list, size=[N])
    attenuation_start_list = np.exp(np.random.uniform(np.log(atten_start_min), np.log(atten_start_max), size=[N]))
    attenuation_slope_list = np.random.uniform(atten_slope_min, atten_slope_max, size=[N])
    # Define encoding / decoding dictionaries for phase_mode
    phase_mode_encoding = {'sine':0, 'rand':1, 'sch':2, 'cos':3, 'alt':4}
    phase_mode_decoding = {0:'sine', 1:'rand', 2:'sch', 3:'cos', 4:'alt'}
    # Prepare data_dict and config_key_pair_list for hdf5 filewriting
    data_dict = {
        'config_tone/fs': fs,
        'config_tone/dur': dur,
        'config_tone/f0_min': f0_min,
        'config_tone/f0_max': f0_max,
        'config_tone/atten_start_min': atten_start_min,
        'config_tone/atten_start_max': atten_start_max,
        'config_tone/atten_slope_min': atten_slope_min,
        'config_tone/atten_slope_max': atten_slope_max,
        'config_tone/amp_noise': amp_noise,
    }
    config_key_pair_list = [(k, k) for k in data_dict.keys()]
    data_key_pair_list = [] # Will be populated right before initializing hdf5 file
    # Main loop to generate lowpass complex tones
    for itrN in range(0, N):
        signal = lowpass_complex_tone(f0_list[itrN], fs, dur,
                                      attenuation_start=attenuation_start_list[itrN],
                                      attenuation_slope=attenuation_slope_list[itrN],
                                      phase_mode=phase_mode_list[itrN],
                                      amp_noise=amp_noise)
        data_dict['tone'] = signal
        data_dict['f0'] = f0_list[itrN]
        data_dict['attenuation_start'] = attenuation_start_list[itrN]
        data_dict['attenuation_slope'] = attenuation_slope_list[itrN]
        data_dict['phase_mode'] = phase_mode_encoding[phase_mode_list[itrN]]
        # Initialize the hdf5 file on the first iteration
        if itrN == 0:
            print('[INITIALIZING]: {}'.format(hdf5_filename))
            for k in data_dict.keys():
                if not (k, k) in config_key_pair_list:
                    data_key_pair_list.append((k, k))
            initialize_hdf5_file(hdf5_filename, N, data_dict, file_mode='w',
                                 data_key_pair_list=data_key_pair_list,
                                 config_key_pair_list=config_key_pair_list,
                                 dtype=np.float32, cast_data=True, cast_config=False)
            hdf5_f = h5py.File(hdf5_filename, 'r+')
        # Write each data_dict to hdf5 file
        write_example_to_hdf5(hdf5_f, data_dict, itrN, data_key_pair_list=data_key_pair_list)
        if itrN % disp_step == 0:
            print('... signal {} of {}'.format(itrN, N))
    # Close hdf5 file
    hdf5_f.close()
    print('[END]: {}'.format(hdf5_filename))


if __name__ == "__main__":
    ''' TEMPORARY COMMAND-LINE USAGE '''
    assert len(sys.argv) == 3, "scipt usage: python <script_name> <hdf5_filename> <N>"
    hdf5_filename = str(sys.argv[1])
    N = int(sys.argv[2])
    generate_lowpass_complex_tone_dataset(hdf5_filename, N)