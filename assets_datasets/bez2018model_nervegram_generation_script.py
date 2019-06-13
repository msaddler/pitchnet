import sys
import numpy as np
import scipy.signal
import h5py
import dask.array as da
import bez2018model.core



def write_example_to_hdf5(f, out, idx,
                          main_key_list=['meanrates','pin','f0'],
                          augmentation_key_list=['snr','meanrates_clip_indexes','pin_clip_indexes','pin_dBSPL']):
    '''
    Function writes individual examples to the hdf5 file.
    
    Args:
        f (h5py object): open and writeable hdf5 file
        out (dict): dict containing auditory nerve model output
        idx (int): specifies row of hdf5 file to write to
        main_key_list (list): list of keys in `out` to write with no group
        augmentation_key_list (list): list of keys in `out` to write to group 'augmentation/'
    Returns:
        None
    '''
    for key in main_key_list:
        f[key][idx] = np.array(out[key])
    for key in augmentation_key_list:
        f['augmentation'][key][idx] = np.array(out[key])


def initialize_hdf5_file(filename, N, out,
                         main_key_list=['meanrates','pin','f0'],
                         augmentation_key_list=['snr','meanrates_clip_indexes','pin_clip_indexes','pin_dBSPL']):
    '''
    Function creates a new hdf5 file.
    
    Args:
        filename (str): filename for new hdf5 file
        N (int): number of examples that will be written to file
        out (dict): dict containing auditory nerve model output
        main_key_list (list): list of keys in `out` to write with no group
        augmentation_key_list (list): list of keys in `out` to write to group 'augmentation/'
    Returns:
        None
    '''
    f = h5py.File(filename, 'w') # Will overwrite if file exists
    print('... initializing file:', filename)
    # Generate datasets for all keys in out dict
    for key in out.keys():
        key_value = np.squeeze(np.array(out[key]))
        if key == 'meanrates': key_value = key_value.astype(np.float32)
        key_shape = list(key_value.shape)
        if key in main_key_list:
            f.create_dataset(key, [N]+key_shape, dtype=key_value.dtype)
        elif key in augmentation_key_list:
            f.create_dataset('augmentation/'+key, [N]+key_shape, dtype=key_value.dtype)
        else:
            f.create_dataset('config/'+key, [1]+key_shape, dtype=key_value.dtype)
            f['config/'+key][0] = key_value
    # Print out the hdf5 file structure
    for val in f.values():
        print(val)
    for val in f['augmentation'].values():
        print('augmentation/', val)
    for val in f['config'].values():
        print('config/', val)
    # Close the initialized hdf5 file
    f.close()


def get_ERB_CF_list(num_CFs, min_CF=125., max_CF=10e3):
    '''
    Get array of num_CFs ERB-scaled CFs between min_CF and max_CF
    '''
    E_start = 21.4 * np.log10(0.00437 * min_CF + 1.0)
    E_end = 21.4 * np.log10(0.00437 * max_CF + 1.0)
    CF_list = np.linspace(E_start, E_end, num = num_CFs)
    CF_list = (1.0/0.00437) * (10.0 ** (CF_list / 21.4) - 1.0)
    return list(CF_list)


def flat_noise(samples, samplerate):
    '''
    Helper function to generate white noise with very flat spectrum
    https://stackoverflow.com/questions/33933842/how-to-generate-noise-in-frequency-range-with-numpy
    '''
    freqs = np.fft.fftfreq(int(samples), 1/samplerate)
    f = np.ones(freqs.shape)
    f = np.array(f, dtype='complex')
    Np = (len(f) - 1) // 2
    phases = np.random.rand(Np) * 2 * np.pi
    phases = np.cos(phases) + 1j * np.sin(phases)
    f[1:Np+1] *= phases
    f[-1:-1-Np:-1] = np.conj(f[1:Np+1])
    return np.fft.ifft(f).real


def rms(a, strict=True):
    '''FROM RAY email 2018.11.05'''
    out = np.sqrt(np.mean(a * a))
    if strict and np.isnan(out):
        raise ValueError('rms calculation resulted in a nan: this will affect ' +
                         'later computation. Ignore with `strict`=False')
    return out


def combine_signal_and_noise(signal, noise, snr, rms_mask=None):
    '''FROM RAY email 2018.11.05'''
    rms_mask = np.full(len(signal), True, dtype=bool) if rms_mask is None else rms_mask
    signal = signal / rms(signal[rms_mask])
    sf = np.power(10, snr / 20)
    signal_rms = rms(signal[rms_mask])
    noise = noise * ((signal_rms / rms(noise[rms_mask])) / sf)
    signal_and_noise = signal + noise
    return signal_and_noise


def apply_butterworth_filtfilt(signal, fs, fc, kind, order):
    if fc <= 0: return signal # If cutoff below 0, do not filter
    if fc >= fs/2: return signal # If cutoff above Nyquist, do not filter
    b, a = scipy.signal.butter(order, fc / (fs/2), kind)
    return scipy.signal.filtfilt(b, a, signal)


def apply_random_bpfilter(signal, fs):
    filt_order = np.random.randint(1, 6) # <-- specify range of filter orders
    log_low_range = np.log([100, 1000]) # <-- specify min, max low cutoff (Hz)
    log_high_range = np.log([4000, 8000]) # <-- specify min, max high cutoff (Hz)
    bp_low = np.exp(np.random.uniform(log_low_range[0], log_low_range[1]))
    bp_high = np.exp(np.random.uniform(log_high_range[0], log_high_range[1]))
    signal = apply_butterworth_filtfilt(signal, fs, bp_low, 'highpass', filt_order)
    signal = apply_butterworth_filtfilt(signal, fs, bp_high, 'lowpass', filt_order)
    return signal


def apply_random_hpfilter(signal, fs):
    filt_order = np.random.randint(1, 6) # <-- specify range of filter orders
    log_low_range = np.log([250, 6000]) # <-- specify min, max low cutoff (Hz)
    bp_low = np.exp(np.random.uniform(log_low_range[0], log_low_range[1]))
    signal = apply_butterworth_filtfilt(signal, fs, bp_low, 'highpass', filt_order)
    return signal


def apply_random_lpfilter(signal, fs):
    filt_order = np.random.randint(1, 6) # <-- specify range of filter orders
    log_high_range = np.log([100, 6000]) # <-- specify min, max high cutoff (Hz)
    bp_high = np.exp(np.random.uniform(log_high_range[0], log_high_range[1]))
    signal = apply_butterworth_filtfilt(signal, fs, bp_high, 'lowpass', filt_order)
    return signal


def harmonic_stack(f0, fs, dur, phase_mode):
    t_offset = np.random.rand() * dur
    t = np.arange(0, dur, 1 / fs) + t_offset
    t = t[0:np.int(dur * fs)]
    signal = np.zeros(t.shape)
    harm_list = np.arange(1, np.floor((fs/2) / f0) + 1)
    for f in f0 * harm_list:
        if phase_mode == 0: phase = 0
        else: phase = 2 * np.pi * np.random.rand()
        signal += np.sin(2 * np.pi * f * t + phase)
    return signal


def modify_signal_and_recombine_noise(signal, noise, snr, fs, filter_params={}):
    '''
    === DEFINE MODIFICATION APPLIED TO SIGNAL HERE ===
    '''
    modified_signal = signal

    #f0 = filter_params['f0']
    #phase_mode = np.random.randint(low=0, high=2) # Replace with synth tone and apply random lpfilter
    #modified_signal = harmonic_stack(f0, fs, modified_signal.shape[0]/fs, phase_mode)
    #modified_signal = apply_random_lpfilter(modified_signal, fs)
    
    #if np.random.rand() > 0.5:
    #    modified_signal = apply_random_bpfilter(modified_signal, fs)

    #noise = flat_noise(len(noise), fs) ### REPLACE THE TEXTURE NOISE WITH FLAT WHITE NOISE
    ##if np.random.rand() >= 0.5: ### APPLY RANDOM BANDPASS FILTER TO HALF
    ##    modified_signal = apply_random_bpfilter(modified_signal, fs)
    ###modified_signal = apply_random_hpfilter(modified_signal, fs) ### APPLY RANDOM HPFILTER TO ALL
    return combine_signal_and_noise(modified_signal, noise, snr)


def generate_training_data(filename, inputs, output_params,
                           ANmodel_params, manipulation_params={}):
    '''
    Generates ANmodel nervegrams for each signal in inputs dict.
    Writes training examples to hdf5 file specified by filename.
    '''
    signal_Fs = inputs['signal_Fs'] # Input signal sampling rate
    N = inputs['signal_list'].shape[0] # Number of input signals

    ### Determine if hdf5 file needs to be initialized and get start_index ###
    try: # Attempt to open the file and start at first unfinished index
        f = h5py.File(filename, 'r+')
        start_index = np.reshape(np.argwhere(f['augmentation/pin_dBSPL'][:] == 0), (-1,))
        if len(start_index) == 0: # Quit if file is complete
            f.close(); print('>>> FILE FOUND: no indexes remain'); return
        else:
            start_index = start_index[0]
            start_index = np.max([0, start_index-1]) # Start 1 signal before to be safe
            print('>>> FILE FOUND: starting at index {}'.format(start_index))
        init_flag = False
    except: # Initialize new hdf5 file if the specified one does not exist
        init_flag = True
        start_index = 0

    ### Start MATLAB engine and iterate through signals ###
    eng = bez2018model.core.start_matlab_engine() # Start matlab engine
    for idx in range(start_index, N):
        ### === APPLY MODIFICATION TO SIGNAL ===
        signal = inputs['signal_list'][idx]
        snr = inputs['snr'][idx]
        if 'noise_list' in inputs.keys():
            noise = inputs['noise_list'][idx]
            signal = modify_signal_and_recombine_noise(signal, noise, snr, signal_Fs)
        ### === APPLY MODIFICATION TO SIGNAL ===
        
        ### Set individual output_params and run the auditory nerve model ###
        if 'pin_dBSPL' in inputs.keys():
            output_params['pin_dBSPL'] = inputs['pin_dBSPL'][idx]
        out = bez2018model.core.generate_nervegram(eng, signal, signal_Fs, output_params,
                                                   ANmodel_params, manipulation_params)
        out['snr'] = inputs['snr'][idx]
        out['f0'] = inputs['f0'][idx] # <--- modify f0 computation here
        
        ### Write auditory nerve model output to hdf5 file ###
        if init_flag:
            print('INITIALIZING:', filename)
            initialize_hdf5_file(filename, N, out)
            f = h5py.File(filename, 'r+')
            init_flag = False
        write_example_to_hdf5(f, out, idx)

        if idx % 5 == 0:
            f.close()
            f = h5py.File(filename, 'r+')
            print('... signal {} of {}'.format(idx, N), '| dB_SPL = {}'.format(out['pin_dBSPL']), '| f0 = {}'.format(out['f0']))

    f.close()
    bez2018model.core.quit_matlab_engine(eng)



if __name__ == "__main__":
    # SET UP SCRIPT TO RUN FROM COMMAND LINE

    error_string = 'COMMAND LINE USAGE: run <script_name.py> <job_id> <num_examples> <dset_id>'
    assert len(sys.argv)==4, error_string
    
    # + + + Get job_id, max number of examples per dataset, and dset_id from command line + + +
    job_id = int(sys.argv[1])
    N = int(sys.argv[2])
    dset_id = int(sys.argv[3])

    # + + + Source signals filename + + +
    source_filename_list = [
        '/om/scratch/Wed/msaddler/bernox2005_stimuli/bernox2005stim_2018-11-29-1930.hdf5',
    ]
    filename_list = [
        '/om/user/msaddler/data_tmp/bernox2005stimset_2018-11-29-1930_CF50-SR70-sp2-cohc00_filt00_thresh33dB_{:06}-{:06}.hdf5',
    ]
    source_filename = source_filename_list[dset_id]
    filename = filename_list[dset_id]
    print('### SOURCE_FILENAME:', source_filename)
    print('### OUTPUT_FILENAME:', filename)
    
    # + + + Open the source hdf5 file in read mode + + +
    source_f = h5py.File(source_filename, 'r')
    source_N = source_f['/f0'].shape[0]
    print('___ Total signals to generate = {} ___'.format(source_N))
    
    # + + + Use job_id to assign indexes and format output filename + + +
    idx_start = job_id * N
    idx_end = min(idx_start + N, source_N)
    assert(idx_start < idx_end)
    filename = filename.format(idx_start, idx_end)
    
    # + + + Specify input signals dictionary: inputs + + +
    inputs = {}
    inputs['signal_Fs'] = source_f['/config/tone_sr'][0]
    inputs['snr'] = source_f['/snr'][idx_start : idx_end]
    inputs['f0'] = source_f['/f0'][idx_start : idx_end]
    if 'nsynth-rwc-timit-wsj' in source_filename:
        inputs['signal_list'] = source_f['/extra/raw_source_signal'][idx_start : idx_end]
        inputs['noise_list'] = source_f['/extra/unsnipped_jwss'][idx_start : idx_end]
        (spl_min, spl_max) = (30, 90) ### SET STIMULUS SPL HERE (in dB SPL)
        spl_list = np.random.rand(inputs['signal_list'].shape[0]) * (spl_max - spl_min) + spl_min
        inputs['pin_dBSPL'] = spl_list
    else:
        inputs['signal_list'] = source_f['/unsnipped_signal'][idx_start : idx_end]

    # + + + Specify output parameters: output_params + + +
    output_params = {}
    output_params['meanrates_dur'] = 0.050 # s (default is 50ms) -- set to 100ms for 100ms
    output_params['meanrates_Fs'] = 10e3 # Hz (10kHz puts Nyquist above phase-locking limit)
    output_params['buffer_front_dur'] = 0.070 # s (default is 70ms) -- set to 30ms for 100ms
    output_params['buffer_end_dur'] = 0.010 # s (default is 10ms) -- set to 10ms for 100ms
    if 'pin_dBSPL' in inputs.keys(): output_params['set_dBSPL_flag'] = 1 # Set stimulus SPLs
    else: output_params['set_dBSPL_flag'] = 0 # Do not set stimulus SPLs

    # + + + Specify ANmodel parameters: ANmodel_params + + +
    ANmodel_params = {}
    ANmodel_params['CF_list'] = get_ERB_CF_list(50, min_CF=125., max_CF=10e3)
    ANmodel_params['spont_list'] = [70.0] #[70.0, 4.0, 0.1] # HSR, MSR, LSR fibers according to BEZ2018
    ANmodel_params['cohc'] = 1.
    ANmodel_params['cihc'] = 1.
    if 'sp1' in filename:
        print('>>> SPECIES 1 <<<')
        ANmodel_params['species'] = 1. # (2 = Shera BW humans, 1 = cat)
    else:
        print('>>> SPECIES 2 <<<')
        ANmodel_params['species'] = 2. # (2 = Shera BW humans, 1 = cat)

    # + + + Specify cochlear manipulation parameters: manipulation_params + + +
    manipulation_params = {}
    manipulation_params['manipulation_flag'] = 0 # Must be 1 to apply cochlear manipulation
    manipulation_params['filt_cutoff'] = 0 # Low-pass filter cutoff (Hz)
    manipulation_params['filt_order'] = 6 # Default low-pass filter order

    # + + + Run dataset generation routine + + +
    print('### BEGIN GENERATING:', filename)
    generate_training_data(filename, inputs, output_params, ANmodel_params, manipulation_params)
    
    # + + + Add diagnostic information from source file + + +
    if 'diagnostic' in source_f.keys():
        print('### ADDING DIAGNOSTIC INFORMATION:', filename)
        DIAG = source_f['diagnostic'] # group containing diagnostic information
        diag_arrays = {}
        for key in DIAG.keys():
            key_str = 'diagnostic/' + key # Create .hdf5 path for output filename
            key_dset = DIAG[key][idx_start : idx_end] # Select diagnostic info for desired signals
            key_arr = da.from_array(key_dset, chunks=key_dset.shape) # Store as dask array
            diag_arrays[key_str] = key_arr # Create a dictionary of dask arrays for conversion back to hdf5
            print('... including:', key_str)
        da.to_hdf5(filename, diag_arrays)

    # + + + End of dataset generation routine + + +
    print('### END GENERATING:', filename)
    source_f.close()
    