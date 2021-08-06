import sys
import os
import tensorflow as tf
import numpy as np
import functions_erbfilter as erb
sys.path.append('/code_location/WaveNet-Enhancement') #TODO: move to more appropriate file
import utilsIBM #TODO: move to more appropriate file


def peripheral_model_graph(batch_input_signal, signal_rate=20000,
                           cochlear_model='tfcochleagram', **COCH_PARAMS):
    '''
    Function for building the peripheral auditory model. This function
    acts as a switch to select which peripheral model to build. Future
    peripheral models can be added here as elif blocks.
    
    Args
    ----
    batch_input_signal (tensor): input signal for peripheral model
    signal_rate (int): sampling rate of input signal (Hz)
    cochlear_model (str): specifies which peripheral auditory model to use
    COCH_PARAMS (dict): peripheral model configuration parameters
    
    Returns
    -------
    batch_subbands (tensor): output tensor of peripheral model
    coch_container (dict): dictionary of peripheral model tensors
    '''
    if cochlear_model.lower() == 'tfcochleagram':
        batch_subbands, coch_container = tfcochleagram_wrapper(batch_input_signal,
                                                               signal_rate=signal_rate,
                                                               **COCH_PARAMS)
    elif cochlear_model.lower() == 'tfcochlearn':
        sys.path.append('/python-packages/tfcochlearn')
        import tfcochlearn # peripheral model modules are imported only as needed
        msg = "tfcochlearn: ambiguous values specified for sr_signal and signal_rate"
        assert COCH_PARAMS.get('sr_signal', signal_rate) == signal_rate, msg
        COCH_PARAMS['sr_signal'] = signal_rate
        batch_subbands, coch_container = tfcochlearn.tfcochlearn_graph(batch_input_signal,
                                                                       **COCH_PARAMS)
    elif cochlear_model.lower() == 'connear':
        sys.path.append('/python-packages/CoNNear_cochlea')
        import connear_wrapper # peripheral model modules are imported only as needed
        msg = "connear: ambiguous values specified for sr_input and signal_rate"
        assert COCH_PARAMS.get('sr_input', signal_rate) == signal_rate, msg
        COCH_PARAMS['sr_input'] = signal_rate
        batch_subbands, coch_container = connear_wrapper.connear_wrapper(batch_input_signal,
                                                                         **COCH_PARAMS)
    else:
        raise ValueError('cochlear_model `{}` is not supported'.format(cochlear_model))
    return batch_subbands, coch_container


def tfcochleagram_wrapper(wavs, signal_rate=20000, N=40, filter_type='half-cosine',
                          LOW_LIM=20, HIGH_LIM=8000, SAMPLE_FACTOR=1, include_lowpass=True, include_highpass=True,
                          min_cf=None, max_cf=None, bandwidth_scale_factor=1.0, dc_ramp_cutoff=30, 
                          filter_spacing='erb', **kwargs):
    '''
    Wrapper function for interacting with tfcochleagram. This function will import
    tfcochleagram module and use it to build the cochlear model graph.
    Args:
        wavs (tensor): input signal waveform
        signal_rate (int): sampling rate of signal waveform
        N (int): number of cochlear bandpass filters
        filter_type (str): type of cochlear filters to build ('half-cosine' or 'roex')
        LOW_LIM (float): low frequency cutoff of filterbank (only used for 'half-cosine')
        HIGH_LIM (float): high frequency cutoff of filterbank (only used for 'half-cosine')
        SAMPLE_FACTOR (int): specifies how densely to sample cochlea (only used for 'half-cosine')
        include_lowpass (bool): determines if filterbank includes lowpass filter(s) (only used for 'half-cosine')
        include_highpass (bool): determines if filterbank includes highpass filter(s) (only used for 'half-cosine')
        min_cf (float): center frequency of lowest roex cochlear filter (only used for 'roex')
        max_cf (float): center frequency of highest roex cochlear filter (only used for 'roex')
        bandwidth_scale_factor (float): factor by which to symmetrically scale the filter bandwidths
            bandwidth_scale_factor=2.0 means filters will be twice as wide.
            Note that values < 1 will cause frequency gaps between the filters.
        dc_ramp_cutoff (float): roex filterbank is multiplied by a ramp that goes from 0 to 1 between
            0 and `dc_ramp_cutoff` Hz to remove DC component (only applied if dc_ramp_cutoff > 0)
        kwargs (dict): dictionary of parameters passed directly to tfcochleagram
    Returns:
        wav_subbands (tensor): subbands lowpass filtered in time
    Raises:
        ValueError if filter_type is not supported (currently 'half-cosine' and 'roex' are supported)
    '''
    sys.path.append('/python-packages/tfcochleagram')
    import tfcochleagram # peripheral model modules are imported only as needed

    signal_length = wavs.get_shape().as_list()[-1]

    if filter_type == 'half-cosine':
        assert HIGH_LIM <= signal_rate/2, "cochlear filterbank high_lim is above Nyquist frequency"
        filts, center_freqs, freqs = erb.make_cos_filters_nx(signal_length, signal_rate, N, LOW_LIM, HIGH_LIM, SAMPLE_FACTOR,
                                                             padding_size=None, full_filter=True, strict=True,
                                                             bandwidth_scale_factor=bandwidth_scale_factor,
                                                             include_lowpass=include_lowpass,
                                                             include_highpass=include_highpass,
                                                             filter_spacing=filter_spacing)
        assert filts.shape[1] == signal_length, "custom filter array shape must match signal length"

    elif filter_type == 'roex':
        assert not min_cf == None, "min_cf must be specified to use roex filterbank"
        assert not max_cf == None, "max_cf must be specified to use roex filterbank"
        assert max_cf <= signal_rate/2, "max_cf is above Nyquist frequency"
        filts, center_freqs, freqs = erb.make_roex_filters(signal_length, signal_rate, N, min_cf=min_cf, max_cf=max_cf,
                                                           bandwidth_scale_factor=bandwidth_scale_factor, padding_size=None,
                                                           full_filter=True, dc_ramp_cutoff=dc_ramp_cutoff)
        assert filts.shape[1] == signal_length, "custom filter array shape must match signal length"

    elif filter_type == 'with_tfcoch_params':
        # Use the tfcochleagram parameters to generate the filters
        filts = None

    else:
        raise ValueError('Specified filter_type {} is not supported'.format(filter_type))
    
    coch_container = {'input_signal':wavs}
    coch_container = tfcochleagram.cochleagram_graph(coch_container, signal_length, signal_rate,
                                                    LOW_LIM=LOW_LIM, HIGH_LIM=HIGH_LIM, N=N,
                                                    SAMPLE_FACTOR=SAMPLE_FACTOR, custom_filts=filts,
                                                    **kwargs)
    wav_subbands = coch_container['output_tfcoch_graph']
    return wav_subbands, coch_container


def wavenet_logits_to_waveform(logits, signal_rate=20000):
    '''
    Calculate expected value of waveform from wavenet logits
    Args:
        logits (tensor): wavenet logits
        signal_rate (int): sampling rate
    Returns:
        exp_val (tensor): expected value of wavenet output waveform
    #TODO: move this function to a more appropriate file
    '''
    bins, center_bins = utilsIBM.mu_law_bins(256)
    bin_values = tf.constant(center_bins[None,:,None], dtype=tf.float32) 
    exp_val = tf.reduce_mean(tf.multiply(bin_values, logits),axis=1)
    return exp_val
