from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import warnings
import numpy as np


def freq2erb(freq_hz):
    """Converts Hz to human-defined ERBs, using the formula of Glasberg and Moore.
    
    Args:
        freq_hz (array_like): frequency to use for ERB.
    
    Returns:
        ndarray: **n_erb** -- Human-defined ERB representation of input.
    """
    return 9.265 * np.log(1 + freq_hz / (24.7 * 9.265))


def erb2freq(n_erb):
    """Converts human ERBs to Hz, using the formula of Glasberg and Moore.

    Args:
        n_erb (array_like): Human-defined ERB to convert to frequency.

    Returns:
        ndarray: **freq_hz** -- Frequency representation of input.
    """
    return 24.7 * 9.265 * (np.exp(n_erb / 9.265) - 1)


def get_freq_rand_conversions(xp, seed=0, minval=0.0, maxval=1.0):
    """Generates freq2rand and rand2freq conversion functions.
    
    Args:
        xp (array_like): xvals for freq2rand linear interpolation.
        seed (int): numpy seed to generate yvals for linear interpolation.
        minval (float): yvals for linear interpolation are scaled to [minval, maxval].
        maxval (float): yvals for linear interpolation are scaled to [minval, maxval].
    
    Returns:
        freq2rand (function): converts Hz to random frequency scale
        rand2freq (function): converts random frequency scale to Hz
    """
    np.random.seed(seed)
    yp = np.cumsum(np.random.poisson(size=xp.shape))
    yp = ((maxval - minval) * (yp - yp.min())) / (yp.max() - yp.min()) + minval
    freq2rand = lambda x : np.interp(x, xp, yp)
    rand2freq = lambda y : np.interp(y, yp, xp)
    return freq2rand, rand2freq


def make_cosine_filter(freqs, l, h, convert_to_erb=True):
    """Generate a half-cosine filter. Represents one subband of the cochleagram.

    A half-cosine filter is created using the values of freqs that are within the
    interval [l, h]. The half-cosine filter is centered at the center of this
    interval, i.e., (h - l) / 2. Values outside the valid interval [l, h] are
    discarded. So, if freqs = [1, 2, 3, ... 10], l = 4.5, h = 8, the cosine filter
    will only be defined on the domain [5, 6, 7] and the returned output will only
    contain 3 elements.

    Args:
        freqs (array_like): Array containing the domain of the filter, in ERB space;
            see convert_to_erb parameter below.. A single half-cosine
            filter will be defined only on the valid section of these values;
            specifically, the values between cutoffs ``l`` and ``h``. A half-cosine filter
            centered at (h - l ) / 2 is created on the interval [l, h].
        l (float): The lower cutoff of the half-cosine filter in ERB space; see
            convert_to_erb parameter below.
        h (float): The upper cutoff of the half-cosine filter in ERB space; see
            convert_to_erb parameter below.
        convert_to_erb (bool, default=True): If this is True, the values in
            input arguments ``freqs``, ``l``, and ``h`` will be transformed from Hz to ERB
            space before creating the half-cosine filter. If this is False, the
            input arguments are assumed to be in ERB space.

    Returns:
        ndarray: **half_cos_filter** -- A half-cosine filter defined using elements of
            freqs within [l, h].
    """
    if convert_to_erb:
        freqs_erb = freq2erb(freqs)
        l_erb = freq2erb(l)
        h_erb = freq2erb(h)
    else:
        freqs_erb = freqs
        l_erb = l
        h_erb = h

    avg_in_erb = (l_erb + h_erb) / 2  # center of filter
    rnge_in_erb = h_erb - l_erb  # width of filter
    # return np.cos((freq2erb(freqs[a_l_ind:a_h_ind+1]) - avg)/rnge * np.pi)  # h_ind+1 to include endpoint
    # return np.cos((freqs_erb[(freqs_erb >= l_erb) & (freqs_erb <= h_erb)]- avg_in_erb) / rnge_in_erb * np.pi)  # map cutoffs to -pi/2, pi/2 interval
    return np.cos((freqs_erb[(freqs_erb > l_erb) & (freqs_erb < h_erb)]- avg_in_erb) / rnge_in_erb * np.pi)  # map cutoffs to -pi/2, pi/2 interval


def make_full_filter_set(filts, signal_length=None):
    """Create the full set of filters by extending the filterbank to negative FFT
    frequencies.

    Args:
        filts (array_like): Array containing the cochlear filterbank in frequency space,
            i.e., the output of make_cos_filters_nx. Each row of ``filts`` is a
            single filter, with columns indexing frequency.
        signal_length (int, optional): Length of the signal to be filtered with this filterbank.
            This should be equal to filter length * 2 - 1, i.e., 2*filts.shape[1] - 1, and if
            signal_length is None, this value will be computed with the above formula.
            This parameter might be deprecated later.
            
    Returns:
        ndarray: **full_filter_set** -- Array containing the complete filterbank in
            frequency space. This output can be directly applied to the frequency
            representation of a signal.
    """
    if signal_length is None:
        signal_length = 2 * filts.shape[1] - 1

    # note that filters are currently such that each ROW is a filter and COLUMN idxs freq
    if np.remainder(signal_length, 2) == 0:  # even -- don't take the DC & don't double sample nyquist
        neg_filts = np.flipud(filts[1:filts.shape[0] - 1, :])
    else:  # odd -- don't take the DC
        neg_filts = np.flipud(filts[1:filts.shape[0], :])
    fft_filts = np.vstack((filts, neg_filts))
    # we need to switch representation to apply filters to fft of the signal, not sure why, but do it here
    return fft_filts.T


def make_cos_filters_nx(signal_length, sr, n, low_lim, hi_lim, sample_factor,
                        padding_size=None, full_filter=True, strict=True,
                        bandwidth_scale_factor=1.0, include_lowpass=True, 
                        include_highpass=True, filter_spacing='erb'):
    """Create cosine filters, oversampled by a factor provided by "sample_factor"

    Args:
        signal_length (int): Length of signal to be filtered with the generated
            filterbank. The signal length determines the length of the filters.
        sr (int): Sampling rate associated with the signal waveform.
        n (int): Number of filters (subbands) to be generated with standard
            sampling (i.e., using a sampling factor of 1). Note, the actual number of
            filters in the generated filterbank depends on the sampling factor, and
            may optionally include lowpass and highpass filters that allow for
            perfect reconstruction of the input signal (the exact number of lowpass
            and highpass filters is determined by the sampling factor). The
            number of filters in the generated filterbank is given below:

        +---------------+---------------+-+------------+---+---------------------+
        | sample factor |     n_out     |=|  bandpass  |\ +|  highpass + lowpass |
        +===============+===============+=+============+===+=====================+
        |      1        |     n+2       |=|     n      |\ +|     1     +    1    |
        +---------------+---------------+-+------------+---+---------------------+
        |      2        |   2*n+1+4     |=|   2*n+1    |\ +|     2     +    2    |
        +---------------+---------------+-+------------+---+---------------------+
        |      4        |   4*n+3+8     |=|   4*n+3    |\ +|     4     +    4    |
        +---------------+---------------+-+------------+---+---------------------+
        |      s        | s*(n+1)-1+2*s |=|  s*(n+1)-1 |\ +|     s     +    s    |
        +---------------+---------------+-+------------+---+---------------------+

        low_lim (int): Lower limit of frequency range. Filters will not be defined
            below this limit.
        hi_lim (int): Upper limit of frequency range. Filters will not be defined
            above this limit.
        sample_factor (int): Positive integer that determines how densely ERB function
            will be sampled to create bandpass filters. 1 represents standard sampling;
            adjacent bandpass filters will overlap by 50%. 2 represents 2x overcomplete sampling;
            adjacent bandpass filters will overlap by 75%. 4 represents 4x overcomplete sampling;
            adjacent bandpass filters will overlap by 87.5%.
        padding_size (int, optional): If None (default), the signal will not be padded
            before filtering. Otherwise, the filters will be created assuming the
            waveform signal will be padded to length padding_size*signal_length.
        full_filter (bool, default=True): If True (default), the complete filter that
            is ready to apply to the signal is returned. If False, only the first
            half of the filter is returned (likely positive terms of FFT).
        strict (bool, default=True): If True (default), will throw an error if
            sample_factor is not a power of two. This facilitates comparison across
            sample_factors. Also, if True, will throw an error if provided hi_lim
            is greater than the Nyquist rate.
        bandwidth_scale_factor (float, default=1.0): scales the bandpass filter bandwidths.
            bandwidth_scale_factor=2.0 means half-cosine filters will be twice as wide.
            Note that values < 1 will cause frequency gaps between the filters.
            bandwidth_scale_factor requires sample_factor=1, include_lowpass=False, include_highpass=False.
        include_lowpass (bool, default=True): if set to False, lowpass filter will be discarded.
        include_highpass (bool, default=True): if set to False, highpass filter will be discarded.
        filter_spacing (str, default='erb'): Specifies the type of reference spacing for the 
            half-cosine filters. Options include 'erb' and 'linear'.

    Returns:
        tuple:
            A tuple containing the output:

            * **filts** (*array*)-- The filterbank consisting of filters have
                cosine-shaped frequency responses, with center frequencies equally
                spaced from low_lim to hi_lim on a scale specified by filter_spacing
            * **center_freqs** (*array*) -- center frequencies of filterbank in filts
            * **freqs** (*array*) -- freq vector in Hz, same frequency dimension as filts

    Raises:
        ValueError: Various value errors for bad choices of sample_factor or frequency
        limits; see description for strict parameter.
        UserWarning: Raises warning if cochlear filters exceed the Nyquist
        limit or go below 0.
        NotImplementedError: Raises error if specified filter_spacing is not implemented
    """

    # Specifiy the type of filter spacing, if using linear filters instead
    if filter_spacing == 'erb':
        _freq2ref = freq2erb
        _ref2freq = erb2freq
    elif filter_spacing == 'erb_r':
        _freq2ref = lambda x: freq2erb(hi_lim) - freq2erb(hi_lim - x)
        _ref2freq = lambda x: hi_lim - erb2freq(freq2erb(hi_lim) - x)
    elif (filter_spacing == 'lin') or (filter_spacing == 'linear'):
        _freq2ref = lambda x: x
        _ref2freq = lambda x: x
    elif 'random' in filter_spacing:
        _freq2ref, _ref2freq = get_freq_rand_conversions(
            np.linspace(low_lim, hi_lim, n),
            seed=int(filter_spacing.split('-')[1].replace('seed', '')),
            minval=freq2erb(low_lim),
            maxval=freq2erb(hi_lim))
    else:
        raise NotImplementedError('unrecognized spacing mode: %s' % filter_spacing)
    print('[functions_erbfilter] using filter_spacing=`{}`'.format(filter_spacing))

    if not bandwidth_scale_factor == 1.0:
        assert sample_factor == 1, "bandwidth_scale_factor only supports sample_factor=1"
        assert include_lowpass == False, "bandwidth_scale_factor only supports include_lowpass=False"
        assert include_highpass == False, "bandwidth_scale_factor only supports include_highpass=False"
    
    if not isinstance(sample_factor, int):
        raise ValueError('sample_factor must be an integer, not %s' % type(sample_factor))
    if sample_factor <= 0:
        raise ValueError('sample_factor must be positive')

    if sample_factor != 1 and np.remainder(sample_factor, 2) != 0:
        msg = 'sample_factor odd, and will change filter widths. Use even sample factors for comparison.'
        if strict:
            raise ValueError(msg)
        else:
            warnings.warn(msg, RuntimeWarning, stacklevel=2)

    if padding_size is not None and padding_size >= 1:
        signal_length += padding_size

    if np.remainder(signal_length, 2) == 0:  # even length
        n_freqs = signal_length // 2  # .0 does not include DC, likely the sampling grid
        max_freq = sr / 2  # go all the way to nyquist
    else:  # odd length
        n_freqs = (signal_length - 1) // 2  # .0
        max_freq = sr * (signal_length - 1) / 2 / signal_length  # just under nyquist

    # verify the high limit is allowed by the sampling rate
    if hi_lim > sr / 2:
        hi_lim = max_freq
        msg = 'input arg "hi_lim" exceeds nyquist limit for max frequency; ignore with "strict=False"'
        if strict:
              raise ValueError(msg)
        else:
              warnings.warn(msg, RuntimeWarning, stacklevel=2)

    # changing the sampling density without changing the filter locations
    # (and, thereby changing their widths) requires that a certain number of filters
    # be used.
    n_filters = sample_factor * (n + 1) - 1
    n_lp_hp = 2 * sample_factor
    freqs = np.linspace(0, max_freq, n_freqs + 1)
    filts = np.zeros((n_freqs + 1, n_filters + n_lp_hp))

    # cutoffs are evenly spaced on the scale specified by filter_spacing; for ERB scale,
    # interpolate linearly in erb space then convert back.
    # Also return the actual spacing used to generate the sequence (in case numpy does 
    # something weird)
    center_freqs, step_spacing = np.linspace(_freq2ref(low_lim), _freq2ref(hi_lim), n_filters + 2, retstep=True)  # +2 for bin endpoints
    # we need to exclude the endpoints
    center_freqs = center_freqs[1:-1]

    freqs_ref = _freq2ref(freqs)
    for i in range(n_filters):
        i_offset = i + sample_factor
        l = center_freqs[i] - sample_factor * bandwidth_scale_factor * step_spacing
        h = center_freqs[i] + sample_factor * bandwidth_scale_factor * step_spacing
        if _ref2freq(h) > sr/2:
            cf = _ref2freq(center_freqs[i])
            msg = "High ERB cutoff of filter with cf={:.2f}Hz exceeds {:.2f}Hz (Nyquist frequency)"
            warnings.warn(msg.format(cf, sr/2))
        if _ref2freq(l) < 0:
            cf = _ref2freq(center_freqs[i])
            msg = 'Low ERB cutoff of filter with cf={:.2f}Hz is not strictly positive'
            warnings.warn(msg.format(cf))
        # the first sample_factor # of rows in filts will be lowpass filters
        filts[(freqs_ref > l) & (freqs_ref < h), i_offset] = make_cosine_filter(freqs_ref, l, h, convert_to_erb=False)

    # add lowpass and highpass filters (there will be sample_factor number of each)
    for i in range(sample_factor):
        # account for the fact that the first sample_factor # of filts are lowpass
        i_offset = i + sample_factor
        lp_h_ind = max(np.where(freqs < _ref2freq(center_freqs[i]))[0]) # lowpass filter goes up to peak of first cos filter
        lp_filt = np.sqrt(1 - np.power(filts[:lp_h_ind+1, i_offset], 2))

        hp_l_ind = min(np.where(freqs > _ref2freq(center_freqs[-1-i]))[0])  # highpass filter goes down to peak of last cos filter
        hp_filt = np.sqrt(1 - np.power(filts[hp_l_ind:, -1-i_offset], 2))

        filts[:lp_h_ind+1, i] = lp_filt
        filts[hp_l_ind:, -1-i] = hp_filt

    # get center freqs for lowpass and highpass filters
    cfs_low = np.copy(center_freqs[:sample_factor]) - sample_factor * step_spacing
    cfs_hi = np.copy(center_freqs[-sample_factor:]) + sample_factor * step_spacing
    center_freqs = np.concatenate((cfs_low, center_freqs, cfs_hi))

    # ensure that squared freq response adds to one
    filts = filts / np.sqrt(sample_factor)

    # convert center freqs from ERB numbers to Hz
    center_freqs = _ref2freq(center_freqs)
    
    # rectify
    center_freqs[center_freqs < 0] = 1

    # discard highpass and lowpass filters, if requested
    if include_lowpass == False:
        filts = filts[:, sample_factor:]
        center_freqs = center_freqs[sample_factor:]
    if include_highpass == False:
        filts = filts[:, :-sample_factor]
        center_freqs = center_freqs[:-sample_factor]

    # make the full filter by adding negative components
    if full_filter:
        filts = make_full_filter_set(filts, signal_length)

    return filts, center_freqs, freqs


def make_roex_filters(signal_length, sr, n,  min_cf=50, max_cf=8e3, bandwidth_scale_factor=1.0,
                      padding_size=None, full_filter=True, dc_ramp_cutoff=30):
    ''' Create roex filterbank with center frequencies spaced linearly on ERB scale.

    Args:
        signal_length (int): length of signal (in samples) that filters will be applied to
        sr (int): Sampling rate associated with the signal waveform.
        n (int): Number of cochlear filters (number of CFs)
        min_cf (float): center frequency of lowest frequency cochlear filter
        max_cf (float): center frequency of highest frequency cochlear filter
        bandwidth_scale_factor (float): factor by which to symmetrically scale the filter bandwidths
        padding_size (int, optional): If None (default), the signal will not be padded
            before filtering. Otherwise, the filters will be created assuming the
            waveform signal will be padded to length padding_size*signal_length.
        full_filter (bool, default=True): If True (default), the complete filter that
            is ready to apply to the signal is returned. If False, only the first
            half of the filter is returned (likely positive terms of FFT).
        dc_ramp_cutoff (float): filterbank is multiplied by a ramp that goes from 0 to 1 between
            0 and `dc_ramp_cutoff` Hz to remove DC component (only applied if dc_ramp_cutoff > 0)
    
    Returns:
         filts (array of floats): bank of roex filters with center frequencies equally
            spaced on an ERB scale from min_cf to max_cf.
        cfs (array of floats): center frequencies of filterbank in filts
        freqs (array of floats): freq vector in Hz
    '''
    
    ### Prepare list of CFs (n CFs spaced linearly on ERB scale)
    min_erb = freq2erb(min_cf)
    max_erb = freq2erb(max_cf)
    cfs_erb = np.linspace(min_erb, max_erb, n)
    cfs = erb2freq(cfs_erb)
    
    ### Setup the frequency vector and initialize filter array
    if padding_size is not None and padding_size >= 1:
        signal_length += padding_size
    if np.remainder(signal_length, 2) == 0: # even length
        n_freqs = (signal_length // 2) + 1
        max_freq = sr / 2
    else:  # odd length
        n_freqs = ((signal_length - 1) // 2) + 1
        max_freq = sr * (signal_length - 1) / 2 / signal_length
    freqs = np.linspace(0, max_freq, n_freqs)
    filts = np.zeros((n_freqs, n))
        
    ### Verify the max cf is reasonable for the sampling rate
    assert max_cf <= sr/2, 'Highest cf cannot be above the Nyquist frequency'
    
    ### Build the roex filterbank
    for cf_idx, cf in enumerate(cfs):
        roex, B_ER = build_roex_transfer_function(cf, r=0, bandwidth_scale_factor=bandwidth_scale_factor)
        filts[:, cf_idx] = roex(freqs)
        if cf + B_ER/2 > sr/2:
            msg = 'High ERB cutoff of filter with cf={:.2f}Hz exceeds {:.2f}Hz (Nyquist frequency)'
            warnings.warn(msg.format(cf, sr/2))
        if cf - B_ER/2 <= 0:
            msg = 'Low ERB cutoff of filter with cf={:.2f}Hz is not strictly positive'
            warnings.warn(msg.format(cf))
    
    ### Multiply the filterbank by a ramp to eliminate gain at DC component
    if dc_ramp_cutoff > 0:
        print('Including ramp below {} Hz to eliminate DC offset in roex filterbank'.format(dc_ramp_cutoff))
        dc_ramp = np.ones(freqs.shape)
        ramped_indexes = np.argwhere(freqs < dc_ramp_cutoff)
        dc_ramp[ramped_indexes] = freqs[ramped_indexes] / dc_ramp_cutoff
        for cf_idx in range(filts.shape[1]): filts[:, cf_idx] = filts[:, cf_idx] * dc_ramp
    
    ### Make the full filters by adding negative frequency components
    if full_filter:
        filts = make_full_filter_set(filts, signal_length)

    return filts, cfs, freqs


def get_roex_transfer_function(p, r, cf):
    ''' Returns a roex transfer function (Patterson et al. 1982) '''
    g = lambda f: np.abs((f - cf) / cf)
    roex = lambda f: r + (1. - r) * (1. + p * g(f)) * np.exp(-p * g(f));
    return roex


def get_asymmetric_roex_transfer_function(p_l, p_u, r, cf):
    ''' This is where the asymmetric roex transfer function will be implemented later.
    '''
    # Equivalent rectangular bandwidth will be: B_ER = (2/p_l + 2/p_u) / f_c.
    pass


def get_equivalent_rectangular_bandwidth_from_cf(cf):
    ''' This function returns the equivalent rectangular bandwidth of a
    cochlear filter centered at `cf`. Currently this function only
    implements the Glasberg & Moore (Hear. Res. 1990) bandwidths.
    '''
    return 24.7 * (1 + 4.37*(cf/1000)) # Human ERBs from Glasberg & Moore (Hear. Res. 1990)


def build_roex_transfer_function(cf, r=0, bandwidth_scale_factor=1.0):
    ''' This function returns a roex transfer function with the specified parameters.
    Reference: Signals, Sound, and Senation / Hartmann (1998) pages 247-248.
    
    #TODO: add options to set other filter bandwidths (i.e. Shera et al. 2002)
    #TODO: implement asymmetric broadening of the roex filters (bandwidth_scale_factor is very crude)
    
    Args:
        cf (float): center frequency of roex filter
        r (float): positive values for this parameter will add thicker tails to the filters
        bandwidth_scale_factor (float): factor by which to symmetrically scale the filter bandwidths
    Returns:
        roex (function): transfer function of the roex filter
        B_ER (float): equivalent rectangular bandwidth of the roex filter (in Hz)
    ''' 
    B_ER = get_equivalent_rectangular_bandwidth_from_cf(cf)
    B_ER = B_ER * bandwidth_scale_factor
    p = (4 * cf) / B_ER # Convert B_ER to dimensionless p parameter for roex transfer function
    roex = get_roex_transfer_function(p, r, cf)
    return roex, B_ER
