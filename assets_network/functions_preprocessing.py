import os
import sys
import tensorflow as tf


def tf_demean(x, axis=1):
    '''
    Helper function to mean-subtract tensor.
    
    Args
    ----
    x (tensor): tensor to be mean-subtracted
    axis (int): kwarg for tf.reduce_mean (axis along which to compute mean)
    
    Returns
    -------
    x_demean (tensor): mean-subtracted tensor
    '''
    x_demean = tf.math.subtract(x, tf.reduce_mean(x, axis=1, keepdims=True))
    return x_demean


def tf_rms(x, axis=1, keepdims=True):
    '''
    Helper function to compute RMS amplitude of a tensor.
    
    Args
    ----
    x (tensor): tensor for which RMS amplitude should be computed
    axis (int): kwarg for tf.reduce_mean (axis along which to compute mean)
    keepdims (bool): kwarg for tf.reduce_mean (specify if mean should keep collapsed dimension) 
    
    Returns
    -------
    rms_x (tensor): root-mean-square amplitude of x
    '''
    rms_x = tf.sqrt(tf.reduce_mean(tf.math.square(x), axis=axis, keepdims=keepdims))
    return rms_x


def tf_set_snr(signal, noise, snr):
    '''
    Helper function to combine signal and noise tensors with specified SNR.
    
    Args
    ----
    signal (tensor): signal tensor
    noise (tensor): noise tensor
    snr (tensor): desired signal-to-noise ratio in dB
    
    Returns
    -------
    signal_in_noise (tensor): equal to signal + noise_scaled
    signal (tensor): mean-subtracted version of input signal tensor
    noise_scaled (tensor): mean-subtracted and scaled version of input noise tensor
    
    Raises
    ------
    InvalidArgumentError: Raised when rms(signal) == 0 or rms(noise) == 0.
        Occurs if noise or signal input are all zeros, which is incompatible with set_snr implementation.
    '''
    # Mean-subtract the provided signal and noise
    signal = tf_demean(signal, axis=1)
    noise = tf_demean(noise, axis=1)
    # Compute RMS amplitudes of provided signal and noise
    rms_signal = tf_rms(signal, axis=1, keepdims=True)
    rms_noise = tf_rms(noise, axis=1, keepdims=True)
    # Ensure neither signal nor noise has an RMS amplitude of zero
    msg = 'The rms({:s}) == 0. Results from {:s} input values all equal to zero'
    tf.debugging.assert_none_equal(rms_signal, tf.zeros_like(rms_signal),
                                   message=msg.format('signal','signal')).mark_used()
    tf.debugging.assert_none_equal(rms_noise, tf.zeros_like(rms_noise),
                                   message=msg.format('noise','noise')).mark_used()
    # Convert snr from dB to desired ratio of RMS(signal) to RMS(noise)
    rms_ratio = tf.math.pow(10.0, snr / 20.0)
    # Re-scale RMS of the noise such that signal + noise will have desired SNR
    noise_scale_factor = tf.math.divide(rms_signal, tf.math.multiply(rms_noise, rms_ratio))
    noise_scaled = tf.math.multiply(noise_scale_factor, noise)
    signal_in_noise = tf.math.add(signal, noise_scaled)
    return signal_in_noise, signal, noise_scaled


def tf_set_dbspl(x, dbspl):
    '''
    Helper function to scale tensor to a specified sound pressure level
    in dB re 20e-6 Pa (dB SPL).
    
    Args
    ----
    x (tensor): tensor to be scaled
    dbspl (tensor): desired sound pressure level in dB re 20e-6 Pa
    
    Returns
    -------
    x (tensor): mean-subtracted and scaled tensor
    scale_factor (tensor): constant x is multiplied by to set dB SPL 
    '''
    x = tf_demean(x, axis=1)
    rms_new = 20e-6 * tf.math.pow(10.0, dbspl / 20.0)
    rms_old = tf_rms(x, axis=1, keepdims=True)
    scale_factor = rms_new / rms_old
    x = tf.math.multiply(scale_factor, x)
    return x, scale_factor


def build_dataset_preprocess_graph(input_tensor_dict, path_preprocess=None,
                                   params_snr={}, params_dbspl={}):
    '''
    This function builds the graph for pre-processing model inputs (currently limited
    to setting SNR in dB and sound presentation level in dB re 20e-6 Pa).
    
    Args
    ----
    input_tensor_dict (dict): dictionary of tensors that contains output of iterator.get_next()
    path_preprocess (string): key in `input_tensor_dict` that points to preprocessed input
        input_tensor_dict[`path_preprocess`] will be created if SNR is set
        input_tensor_dict[`path_preprocess`] will be modified in place if dBSPL is set
    params_snr (dict): parameters for setting SNR of input stimuli in background noise
    params_dbspl (dict): parameters for setting sound presentation level of input stimuli
    
    Returns
    -------
    input_tensor_dict (dict): dictionary of tensors that contains output of iterator.get_next()
    '''
    if params_snr:
        msg = "path_preprocess is a required argument to set SNR"
        assert path_preprocess is not None, msg
        range_snr = params_snr.get('range_snr', None)
        path_snr = params_snr.get('path_snr', None)
        path_signal = params_snr.get('path_signal', None)
        path_noise = params_snr.get('path_noise', None)
        rms_signal = params_snr.get('rms_signal', None)
        if (range_snr is not None) and (path_snr is not None):
            raise ValueError("Ambiguous dataset_preprocess_params[params_snr]")
        if (path_signal is not None) and (path_noise is not None):
            # SNR can only be set if path_signal and path_noise are both specified
            if range_snr is not None:
                if range_snr[0]!=range_snr[1]:
                    # Randomly sample SNR if `range_snr` is specified with a min and max that differ
                    snr = tf.random.uniform([tf.shape(input_tensor_dict[path_signal])[0], 1],
                                            minval=range_snr[0], maxval=range_snr[1],
                                            dtype=tf.dtypes.float32, name='snr_uniformly_sampled')
                else:
                    snr = range_snr[0]
            elif path_snr is not None:
                # Use SNR from tfrecords if `path_snr` is specified
                snr = input_tensor_dict[path_snr]
                if len(snr.shape) < 2: snr = tf.expand_dims(snr, 1)
            else:
                raise ValueError("range_snr or path_snr must be specified")
            # If rms_signal is specified, set RMS of input_tensor_dict[path_signal] accordingly 
            if rms_signal is not None:
                signal = tf_demean(input_tensor_dict[path_signal], axis=1)
                rms_signal_old = tf_rms(signal, axis=1, keepdims=True)
                signal = tf.multiply(rms_signal / rms_signal_old, signal)
                input_tensor_dict[path_signal] = signal
            # Combine signal and noise with the specified SNR (noise is re-scaled to set SNR)
            signal_in_noise, signal, noise_scaled = tf_set_snr(input_tensor_dict[path_signal],
                                                               input_tensor_dict[path_noise],
                                                               snr)
            # Update values of input_tensor_dict (NOTE: we replace `input_tensor_dict[path_noise]`
            # with `noise_scaled` to ensure: input_tensor_dict[path_preprocess] =
            #     input_tensor_dict[path_signal] + input_tensor_dict[path_noise])
            input_tensor_dict[path_preprocess] = signal_in_noise
            input_tensor_dict[path_signal] = signal
            input_tensor_dict[path_noise] = noise_scaled
    
    if params_dbspl:
        msg = "path_preprocess must exist in `input_tensor_dict` to change dBSPL"
        assert path_preprocess in input_tensor_dict.keys(), msg
        range_dbspl = params_dbspl.get('range_dbspl', None)
        path_dbspl = params_dbspl.get('path_dbspl', None)
        if (range_dbspl is not None) and (path_dbspl is not None):
            raise ValueError("Ambiguous dataset_preprocess_params[params_dbspl]")
        if (range_dbspl is not None) or (path_dbspl is not None):
            # dBSPL can only be set if range_dbspl or path_dbspl is set
            if range_dbspl is not None:
                if range_dbspl[0]!=range_dbspl[1]:
                    # Randomly sample dB SPL if `range_dbspl` is specified with a min and max that differ
                    dbspl = tf.random.uniform([tf.shape(input_tensor_dict[path_preprocess])[0], 1],
                                              minval=range_dbspl[0], maxval=range_dbspl[1],
                                              dtype=tf.dtypes.float32, name='dbspl_uniformly_sampled')
                else:
                    dbspl = range_dbspl[0]
            elif path_dbspl is not None:
                # Use dB SPL from tfrecords if `path_dbspl` is specified
                dbspl = input_tensor_dict[path_dbspl]
                if len(dbspl.shape) < 2: dbspl = tf.expand_dims(dbspl, 1)
            else:
                raise ValueError("range_dbspl or path_dbspl must be specified")
            # Re-scale `input_tensor_dict[path_preprocess]` to the specified dBSPL
            input_tensor_dict[path_preprocess], scale_factor = tf_set_dbspl(
                input_tensor_dict[path_preprocess], dbspl)
            # If params_snr was specified, rescale RMS of signal and RMS of noise by the same
            # factor to to preserve the equality: input_tensor_dict[path_preprocess] =
            #     input_tensor_dict[path_signal] + input_tensor_dict[path_noise])
            path_signal = params_snr.get('path_signal', None)
            path_noise = params_snr.get('path_noise', None)
            if (path_signal is not None) and (path_noise is not None):
                input_tensor_dict[path_signal] = tf.math.multiply(scale_factor,
                                                                  input_tensor_dict[path_signal])
                input_tensor_dict[path_noise] = tf.math.multiply(scale_factor,
                                                                 input_tensor_dict[path_noise])
    
    return input_tensor_dict
