
import tensorflow as tf
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
import sys
import logging
import functools
from pkg_resources import parse_version

from koogu.data.raw import Filters, Settings


class Linear2dB(tf.keras.layers.Layer):
    """Layer for converting time-frequency (tf) representations from linear
    to decibel scale.

    Arguments:
      eps: Epsilon value to add, for avoiding divide-by-zero errors.
      full_scale: Whether to convert to dB full-scale (default: False)
      name: A string, the name of the layer.
    """

    def __init__(self, eps, full_scale, **kwargs):

        # Throw away data_format from kwargs if given
        if 'data_format' in kwargs:
            kwargs.pop('data_format')

        super(Linear2dB, self).__init__(trainable=False, **kwargs)

        self.eps = eps
        self.full_scale = full_scale

    def call(self, inputs):

        # The below value includes the "10 * " (the d part of dB) and the
        # "log10" part. Because TF doesn't have log10, I'm  log10(val) as
        # tf.log(val) / np.log(10).
        db_conversion_constant = \
            (10. / np.log(10.)).astype(inputs.dtype.as_numpy_dtype)
        convr_const = tf.constant(db_conversion_constant,
                                  dtype=inputs.dtype)

        outputs = convr_const * tf.math.log(inputs + self.eps)

        if self.full_scale:
            # Normalize the dB values to bring them to the range [0.0, 1.0].
            # Considering [eps, 1.0] to be the range of possible values in
            # linear scale; this range translates to [10log10(eps), 0.0] in
            # dB scale. Hence, subtracting the dB values by 10log10(eps)
            # first and then dividing by the max possible value.
            eps_db = tf.math.log(self.eps) * convr_const
            outputs = (outputs - eps_db) / ((10 * np.log10(1.)) - eps_db)

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'eps': self.eps,
            'full_scale': self.full_scale
        }

        base_config = super(Linear2dB, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Audio2Spectral(tf.keras.layers.Layer):
    """Layer for converting waveforms into time-frequency (tf) representations.

    Arguments:
      fs: sampling frequency of the data in the last dimension of 'inputs'.
      spec_settings: an dict.
      eps: (optional) will override the eps value in spec_settings.
      name: A string, the name of the layer.
    """

    def __init__(self, fs, spec_settings, **kwargs):

        self.config = {
            'fs': fs,
            'spec_settings': spec_settings
        }
        if 'trainable' not in kwargs:
            kwargs['trainable'] = False
        for key, val in kwargs.items():
            self.config[key] = val

        spec_settings = Settings.Spectral(fs, **spec_settings)

        if spec_settings.tf_rep_type not in \
            ['spec', 'spec_db', 'spec_dbfs',
             'melspec', 'melspec_db', 'melspec_dbfs']:
            raise NotImplementedError('tf_rep_type {} not implemented'.format(
                    repr(spec_settings.tf_rep_type)))

        self.eps = kwargs.pop('eps') if 'eps' in kwargs else spec_settings.eps

        super(Audio2Spectral, self).__init__(**kwargs)

        self.fs = fs
        self.spec_settings = spec_settings

        # Values from numpy
        sp_hann = signal.windows.hann(spec_settings.win_len_samples)
        f = np.fft.rfftfreq(spec_settings.nfft, 1 / fs)

        self.scale = 1 / (fs * (sp_hann * sp_hann).sum())

        # Find out the indices of where to clip the spectral representation
        valid_f_idx_start = f.searchsorted(
            spec_settings.bandwidth_clip[0], side='left')
        valid_f_idx_end = f.searchsorted(
            spec_settings.bandwidth_clip[1], side='right') - 1

        # Pre-compute segmentation boundaries (see _to_psd() for more info).
        # Store (seg. start idx, seg. size) pairs.
        self.psd_segs = np.zeros((3, 2), dtype=np.int)
        if valid_f_idx_start == 0:
            self.psd_segs[0, 1] = 1  # Include the 0 Hz bin
            self.psd_segs[1, 0] = 1  # For next chunk
        else:
            self.psd_segs[1, 0] = valid_f_idx_start
        if spec_settings.nfft % 2 == 0 and valid_f_idx_end == len(f) - 1:
            self.psd_segs[1, 1] = valid_f_idx_end - self.psd_segs[1, 0]
            # Handle unpaired Nyquist
            self.psd_segs[2, :] = [len(f) - 1, 1]
        else:
            self.psd_segs[1, 1] = valid_f_idx_end - self.psd_segs[1, 0] + 1

        self.mel_filterbank = None
        if spec_settings.tf_rep_type.startswith('mel'):
            # Mel scale is requested, prepare the filterbank
            self.mel_filterbank, _ = Filters.mel_filterbanks2(
                spec_settings.num_mels,
                np.asarray(spec_settings.bandwidth_clip,
                           dtype=self.dtype().as_numpy_dtype),
                f, dtype=self.dtype().as_numpy_dtype)

            # Clip to non-zero range & avoid unnecessary multiplications
            self.mel_filterbank = \
                self.mel_filterbank[valid_f_idx_start:
                                    (valid_f_idx_end + 1), :]

        #self.input_spec = tf.keras.layers.InputSpec(ndim=2)

    # A constant for converting to dB scale. This includes the "10 * " (the d
    # part of dB) and the "log10" part. Because TF doesn't have log10, I
    # obtain log10(val) as tf.log(val) / np.log(10).
    _db_conversion_constant = (10. / np.log(10.))

    @tf.function
    def call(self, inputs, **kwargs):

        # Compute STFT
        outputs = tf.signal.stft(
            inputs, name='STFT',
            frame_length=self.spec_settings.win_len_samples,
            frame_step=self.spec_settings.win_len_samples -
                       self.spec_settings.win_overlap_samples,
            fft_length=self.spec_settings.nfft,
            window_fn=functools.partial(tf.signal.hann_window, periodic=False),
            pad_end=False)

        # Convert to PSD
        outputs = self._to_psd(outputs)

        # Convert to mel scale if requested
        if self.mel_filterbank is not None:
            # Apply the mel_filterbank.
            outputs = tf.tensordot(outputs, self.mel_filterbank, 1)

        if self.spec_settings.tf_rep_type.endswith('_db') or \
           self.spec_settings.tf_rep_type.endswith('_dbfs'):

            convr_const = tf.constant(self._db_conversion_constant,
                                      dtype=outputs.dtype)

            # Convert to dB scale
            outputs = tf.math.log(outputs + self.eps) * convr_const

            if self.spec_settings.tf_rep_type.endswith('fs'):  # full-scale
                # Normalize the dB values to bring them to the range [0.0, 1.0].
                # Considering [eps, 1.0] to be the range of possible values in
                # linear scale; this range translates to [10log10(eps), 0.0] in
                # dB scale. Hence, subtracting the dB values by 10log10(eps)
                # first and then dividing by the max possible value.
                eps_db = tf.math.log(self.eps) * convr_const
                outputs = (outputs - eps_db) / ((10 * np.log10(1.)) - eps_db)

        # Transpose t and f
        axes = np.arange(outputs.get_shape().ndims)
        axes[-2:] = np.flip(axes[-2:])
        return tf.transpose(outputs, perm=axes)

    @tf.function
    def _to_psd(self, stft):
        """This function provides a graph-optimized implementation of the
        following operations

          psd = scale * tf.real((tf.conj(stft) * stft))

          if nfft % 2:
              psd = tf.concat([psd[..., 0:1], psd[..., 1:] * 2], axis=-1)
          else:
              # Last point is unpaired Nyquist freq point, don't double
              psd = tf.concat(
                    [psd[..., 0:1], psd[..., 1:-1] * 2, psd[..., -1:]], axis=-1)

          # Clip along the frequency axis for the requested bandwidth
          psd = psd[..., valid_f_idx_start:(valid_f_idx_end + 1)]
        """

        # a container to collect the frequency bins that fall within the
        # requested bandwidth
        psd_segments = list()
        slice_starts = [0] * (stft.get_shape().ndims - 1)
        slice_sizes = [-1] * (stft.get_shape().ndims - 1)

        if self.psd_segs[0, 1] > 0:
            first_bin = tf.slice(stft,
                                 slice_starts + [self.psd_segs[0, 0]],
                                 slice_sizes + [self.psd_segs[0, 1]])
            psd_segments.append(
                self.scale * tf.math.real(tf.math.conj(first_bin) * first_bin))

        if self.psd_segs[1, 1] > 0:
            mid_bins = tf.slice(stft,
                                slice_starts + [self.psd_segs[1, 0]],
                                slice_sizes + [self.psd_segs[1, 1]])
            psd_segments.append((2 * self.scale) *
                                tf.math.real(tf.math.conj(mid_bins) * mid_bins))

        if self.psd_segs[2, 1] > 0:  # Include the unpaired Nyquist bin
            last_bin = tf.slice(stft,
                                slice_starts + [self.psd_segs[2, 0]],
                                slice_sizes + [self.psd_segs[2, 1]])
            psd_segments.append(
                self.scale * tf.math.real(tf.math.conj(last_bin) * last_bin))

        return psd_segments[0] if len(psd_segments) == 0 \
            else tf.concat(psd_segments, axis=-1)

    def compute_output_shape(self, input_shape):
        num_samples = input_shape[-1]

        frame_step = self.spec_settings.win_len_samples - \
                     self.spec_settings.win_overlap_samples
        cols = np.maximum(0, 1 +
            (num_samples - self.spec_settings.win_len_samples) // frame_step)

        rows = self.psd_segs[:, 1].sum() if self.mel_filterbank is None \
            else self.mel_filterbank.shape[1]

        return input_shape[:-1] + [rows, cols]

    def get_config(self):
        base_config = super(Audio2Spectral, self).get_config()
        return dict(list(base_config.items()) + list(self.config.items()))


class LoG(tf.keras.layers.Layer):
    """
    Layer for applying Laplacian of Gaussian operator(s) to
    time-frequency (tf) representations.

    Arguments:
      scales_sigmas: Must be a tuple or list of sigma values at different
        (usually, geometrically progressing) scales. You may use this formula
        to determine the possible set of sigma values beyond the lowest_sigma:
            lowest_sigma * (2 ^ (range(2, floor(
                log2((max_len - 1) / ((2 x 3) x lowest_sigma)) + 1) + 1) - 1))
        For example, if lowest_sigma is 4 & max_len is 243, the resulting set
        of sigmas should be (4, 8, 16, 32).
      add_offsets: If True (default is False), add a trainable offset value to
        LoG responses.
      conv_filters: If not None, must be either a single integer (applicable
        to outputs of all scales) or a list-like group of integers (one per
        scale, applicable to outputs of respective scales). As many 3x3
        filters (trainable) will be created and they will be applied to the
        final outputs of this layer.
      retain_LoG: If True, and if conv_filters is enabled, the LoG outputs
        will be included in the outputs.
    """

    def __init__(self, scales_sigmas=(4,),
                 add_offsets=False,
                 conv_filters=None,
                 retain_LoG=None,
                 **kwargs):

        assert len(scales_sigmas) > 0
        assert isinstance(add_offsets, bool)
        assert conv_filters is None or isinstance(conv_filters, int) or \
            (isinstance(conv_filters, (list, tuple)) and
             len(conv_filters) == len(scales_sigmas))

        data_format = kwargs.pop('data_format') if 'data_format' in kwargs \
            else 'channels_last'

        assert data_format in ['channels_first', 'channels_last'], \
            'Only 2 formats supported'

        kwargs['trainable'] = \
            True if (add_offsets or conv_filters is not None) else False

        super(LoG, self).__init__(
            name=kwargs.pop('name') if 'name' in kwargs else 'LoG',
            **kwargs)

#        scales_sigmas = np.asarray(scales_sigmas)
#        temp = ((2 * 3) * scales_sigmas) + 1  # compute 6 x sigma width
#        if any(temp > height):
#            logging.warning(
#               'Ignoring blob scales_sigmas larger than {:.2f}'.format(
#                   (height - 1) / (2 * 3)))

        self.sigmas = sorted(scales_sigmas)
        self.add_offsets = add_offsets
        self.conv_filters = conv_filters
        self.data_format = data_format

        self.kernels = [tf.constant(
                            np.reshape(k, [len(k), 1, 1, 1]),
                            dtype=self.dtype)
                        for k, _ in (Filters.LoG_kernel_1d(sigma)
                                     for sigma in self.sigmas)]

        prev_scale_padding = int(0)
        f_padding_vec = list()
        f_axis = 1 if data_format == 'channels_last' else 2
        for s_idx, curr_sigma in enumerate(scales_sigmas):
            # Add padding (incrementally) prior to convolutions so that values
            # at boundaries are not very unrealistic.
            curr_scale_padding = int(round(3 * curr_sigma))
            incr_padding = curr_scale_padding - prev_scale_padding

            base_padding_vec = [[0, 0], [0, 0], [0, 0], [0, 0]]
            base_padding_vec[f_axis] = [incr_padding, incr_padding]
            f_padding_vec.append(base_padding_vec)

            # Update for next iteration
            prev_scale_padding = curr_scale_padding
        self.f_padding_vec = [tf.constant(pv, dtype=tf.int32)
                              for pv in f_padding_vec]

        self.offsets = None
        if add_offsets:
            self.offsets = [
                self.add_weight(
                    name='offset{:d}'.format(sc_idx),
                    shape=[],
                    initializer=tf.keras.initializers.zeros(),
                    regularizer=None,
                    constraint=tf.keras.constraints.non_neg(),
                    trainable=True,
                    dtype=self.kernels[0].dtype)
                for sc_idx in range(len(scales_sigmas))]

        if conv_filters is None:
            self.conv_ops = None
            self.activation = None
            self.retain_LoG = None  # Force this to be unset
        else:
            if isinstance(conv_filters, int):   # One for all
                self.conv_ops = [tf.keras.layers.Conv2D(
                    filters=conv_filters,
                    kernel_size=(3, 3), strides=(1, 1),
                    padding='same', use_bias=False, data_format=data_format,
                    kernel_initializer=tf.keras.initializers.VarianceScaling(),
                    name='LoG_Conv2D')]
            else:
                self.conv_ops = [tf.keras.layers.Conv2D(
                    filters=num_filters,
                    kernel_size=(3, 3), strides=(1, 1),
                    padding='same', use_bias=False, data_format=data_format,
                    kernel_initializer=tf.keras.initializers.VarianceScaling(),
                    name='LoG{:d}_Conv2D'.format(sc_idx+1))
                    for sc_idx, num_filters in enumerate(conv_filters)]

            self.activation = tf.keras.layers.Activation('relu', name='LoG_ReLu')
            self.retain_LoG = (retain_LoG is not None and retain_LoG is True)

        self.input_spec = tf.keras.layers.InputSpec(ndim=4)

    @tf.function
    def call(self, inputs, **kwargs):

        if self.data_format == 'channels_last':
            data_format_other = 'NHWC'
            channel_axis = 3
        else:
            data_format_other = 'NCHW'
            channel_axis = 1

        # Process at all scales
        blob_det_inputs = inputs
        conv_op_idxs = ([0] * len(self.kernels)) if len(self.conv_ops) == 1 \
            else np.arange(len(self.kernels))
        all_scale_LoGs = list()
        conv_outputs = list()
        for sc_idx in range(len(self.kernels)):
            # Add padding (incrementally) prior to convolutions so that values
            # at boundaries are not very unrealistic.
            blob_det_inputs = tf.pad(blob_det_inputs,
                                     self.f_padding_vec[sc_idx], 'SYMMETRIC')

            # Apply LoG filter
            curr_scale_LoG = tf.nn.conv2d(
                blob_det_inputs,
                self.kernels[sc_idx],
                strides=[1, 1, 1, 1],
                padding='VALID',
                data_format=data_format_other)

            # Add offset, if enabled
            if self.offsets is not None:
                curr_scale_LoG = curr_scale_LoG + self.offsets[sc_idx]

            all_scale_LoGs.append(curr_scale_LoG)

            # Apply post-conv, if enabled
            if self.conv_ops is not None:
                # Add offset and suppress values below zero. Then apply conv.
                conv_outputs.append(
                    self.conv_ops[conv_op_idxs[sc_idx]](
                        self.activation(curr_scale_LoG)))

        outputs = (all_scale_LoGs + conv_outputs) if self.retain_LoG else conv_outputs

        if len(outputs) > 1:
            outputs = tf.concat(outputs, axis=channel_axis)
        else:
            outputs = outputs[0]

        return outputs

    def compute_output_shape(self, input_shape):
        output_shape = input_shape
        output_shape[3 if self.data_format == 'channels_last' else 1] = \
            len(self.sigmas)
        return output_shape

    def get_config(self):
        config = {
            'scales_sigmas': self.sigmas,
            'add_offsets': self.add_offsets,
            'conv_filters': self.conv_filters,
            'retain_LoG': self.retain_LoG,
            'data_format': self.data_format
        }

        base_config = super(LoG, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def apply_gaussian_blur(surface, apply_2d=True, sigma=1.0, padding='SAME', data_format='NHWC'):
    """Apply Gaussian blurring in 1-dimension (if apply_2d is False) or
    in 2-dimensions (as separable 1-dimensional convolutions).
    Note: If apply_2d is False, blurring is applied only along the x-dimension."""

    kernel = Filters.gauss_kernel_1d(sigma)
    kernel_len = np.int32(len(kernel))

    scale_t = sigma ** 2

    with tf.variable_scope('GaussBlur_%dd_t%.2f' % (apply_2d + 1, scale_t)):

        # Transform Gaussian kernel as appropriate for convolutions in x & y directions.
        # Reshaping as [H, W, in_channels, out_channels]
        kernel_x = tf.constant(np.reshape(kernel, [1, kernel_len, 1, 1]), name='kr_gauss_x', dtype=surface.dtype)
        if apply_2d:
            kernel_y = tf.constant(np.reshape(kernel, [kernel_len, 1, 1, 1]), name='kr_gauss_y', dtype=surface.dtype)

        # Perform spatial(Sp) smoothing(Sm)
        if apply_2d:
            retval = tf.nn.conv2d(
                tf.nn.conv2d(surface, kernel_x, strides=[1, 1, 1, 1], padding=padding, data_format=data_format),
                kernel_y, strides=[1, 1, 1, 1], padding=padding, data_format=data_format)
        else:
            retval = tf.nn.conv2d(surface, kernel_x, strides=[1, 1, 1, 1], padding=padding, data_format=data_format)

    return retval


def apply_1d_LoG(surface, sigma=1.0, data_format='NHWC'):
    """Apply 1-dimensional Laplacian of Gaussian along the y-axis of the input surface.
    NOTE: The 'surface' is expected to be sufficiently pre-padded along the y-axis. So, the function only returns the
    valid points following a convolution."""

    kernel, scale_factor = Filters.LoG_kernel_1d(sigma)
    kernel_len = np.int32(len(kernel))

    scale = sigma ** 2

    with tf.variable_scope('LoG_t%d' % scale):
        kernel_y = tf.constant(np.reshape(kernel, [kernel_len, 1, 1, 1]), name='ker_LoG_y', dtype=surface.dtype)

        retval = tf.nn.conv2d(surface, kernel_y, strides=[1, 1, 1, 1], padding='VALID', data_format=data_format)

    return retval


def _get_tf_constants_for_directional_derivatives(height, width, tf_dtype=tf.float32, data_format='NHWC'):
    """Supplies the TF constants needed in get_directional_derivatives().
    The convolution kernel values mentioned in Farid & Simoncelli (2004) are transformed appropriately in the
    first and second dimensions and returned, along with indices for appropriate slicing, as a python dict."""

    # Kernel values, from the paper
    pre_filt = np.array([0.030320, 0.249724, 0.439911, 0.249724, 0.030320]).astype(tf_dtype.as_numpy_dtype)
    der = np.array([0.104550, 0.292315, 0.000000, -0.292315, -0.104550]).astype(tf_dtype.as_numpy_dtype)
    der2 = np.array([0.232905, 0.002668, -0.471147, 0.002668, 0.232905]).astype(tf_dtype.as_numpy_dtype)

    ker_len = pre_filt.size
    pad_len = ker_len // 2

    assert data_format in ['NHWC', 'NCHW']   # only 2 formats supported
    if data_format == 'NHWC':
        begin_t2_l2 = [0, pad_len, pad_len, 0]
        size_t2_l2 = [-1, height, width, -1]
    else:
        begin_t2_l2 = [0, 0, pad_len, pad_len]
        size_t2_l2 = [-1, -1, height, width]

    # Build the constants that are independent of scale
    dir_der_constants = {
        # Build the kernels transforming them as appropriate for convolutions.
        # Reshaping as [H, W, in_channels, out_channels]
        'kernel_pre_filt_x': tf.constant(np.reshape(pre_filt, [1, ker_len, 1, 1]),
                                         name='kr_pre_filt_x', dtype=tf_dtype),
        'kernel_pre_filt_y': tf.constant(np.reshape(pre_filt, [ker_len, 1, 1, 1]),
                                         name='kr_pre_filt_y', dtype=tf_dtype),
        'kernel_dx': tf.constant(np.reshape(der, [1, ker_len, 1, 1]), name='kr_dx', dtype=tf_dtype),
        'kernel_dy': tf.constant(np.reshape(der, [ker_len, 1, 1, 1]), name='kr_dy', dtype=tf_dtype),
        'kernel_d2x': tf.constant(np.reshape(der2, [1, ker_len, 1, 1]), name='kr_d2x', dtype=tf_dtype),
        'kernel_d2y': tf.constant(np.reshape(der2, [ker_len, 1, 1, 1]), name='kr_d2y', dtype=tf_dtype),

        # Slicing offsets and sizes
        'begin_t2_l2': tf.constant(begin_t2_l2, dtype=tf.int32, name='begin_t2_l2'),
        'size_t2_l2': tf.constant(size_t2_l2, dtype=tf.int32, name='size_t2_l2')
    }

    return dir_der_constants


def get_directional_derivatives(in_surface, dir_der_constants=None, data_format='NHWC'):
    """Determine partial derivatives of a 2D surface.
    The method used here is faster and more accurate than conventional gradient computation.
    The method is described in:
          Farid, H., & Simoncelli, E. P. (2004). Differentiation of discrete multidimensional
          signals. Image Processing, IEEE Transactions on, 13(4), 496-508.

    If dir_der_constants is not None, then it MUST be the python dict returned from a previous call to
    _get_tf_constants_for_directional_derivatives(). Note that no checks for integrity of a not-None dir_der_constants
    is performed; make sure you pass along good parameter.
    Note that the in_surface is expected to be pre-padded with 2 points (because conv kennels are 5 points long) along
    both H- & W- dimensions. Only the "valid" interior points post convolution are returned."""

    assert data_format in ['NHWC', 'NCHW']   # only 2 formats supported
    if dir_der_constants is None:
        if data_format == 'NHWC':
            h, w = in_surface.shape[1:3]
        else:
            h, w = in_surface.shape[2:4]

        dir_der_constants = _get_tf_constants_for_directional_derivatives(h, w, in_surface.dtype, data_format)

    # Compute directional derivatives
    with tf.variable_scope('dir_ders', reuse=tf.AUTO_REUSE):

        with tf.variable_scope('Lx'):
            lx = tf.nn.conv2d(
                tf.nn.conv2d(in_surface, dir_der_constants['kernel_pre_filt_y'],
                             strides=[1, 1, 1, 1], padding='SAME', data_format=data_format),
                dir_der_constants['kernel_dx'], strides=[1, 1, 1, 1], padding='SAME', data_format=data_format)
            # Not doing 'VALID' padding here, coz I need this for computing Lxy yet. Will slice out "valid" Lx later.

        with tf.variable_scope('Lxx'):
            lxx = tf.nn.conv2d(
                tf.nn.conv2d(in_surface, dir_der_constants['kernel_pre_filt_y'],
                             strides=[1, 1, 1, 1], padding='VALID', data_format=data_format),
                dir_der_constants['kernel_d2x'], strides=[1, 1, 1, 1], padding='VALID', data_format=data_format)

        with tf.variable_scope('Lxy'):
            lxy = tf.nn.conv2d(
                tf.nn.conv2d(lx, dir_der_constants['kernel_dy'],
                             strides=[1, 1, 1, 1], padding='VALID', data_format=data_format),
                dir_der_constants['kernel_pre_filt_x'], strides=[1, 1, 1, 1], padding='VALID', data_format=data_format)

        with tf.variable_scope('Ly'):
            ly = tf.nn.conv2d(
                tf.nn.conv2d(in_surface, dir_der_constants['kernel_dy'],
                             strides=[1, 1, 1, 1], padding='VALID', data_format=data_format),
                dir_der_constants['kernel_pre_filt_x'], strides=[1, 1, 1, 1], padding='VALID', data_format=data_format)

        with tf.variable_scope('Lyy'):
            lyy = tf.nn.conv2d(
                tf.nn.conv2d(in_surface, dir_der_constants['kernel_d2y'],
                             strides=[1, 1, 1, 1], padding='VALID', data_format=data_format),
                dir_der_constants['kernel_pre_filt_x'], strides=[1, 1, 1, 1], padding='VALID', data_format=data_format)

        lx = tf.slice(lx, dir_der_constants['begin_t2_l2'], dir_der_constants['size_t2_l2'])

    return lx, ly, lxx, lyy, lxy


def _compute_ridge_measurements(lx, ly, lxx, lyy, lxy, normalization_factor=1.0):

    with tf.variable_scope('ridge_measurements', reuse=tf.AUTO_REUSE):
        dir_ders_sum = tf.add(lxx, lyy, name='Lxx_plus_Lyy')
        dir_ders_diff = tf.subtract(lxx, lyy, name='Lxx_minus_Lyy')

        with tf.variable_scope('discrmnt_sqrt'):
            discriminant_sqrt = tf.sqrt(((dir_ders_diff ** 2) + (4 * (lxy ** 2))))

        with tf.variable_scope('signed_discriminant_sqrt'):
            # Make discriminant_sqrt either negative or positive based on whether (Lxx+Lyy) is negative or positive.
            # discriminant_sqrt_with_sign = (1 - (2 * tf.cast(dir_ders_sum < 0, discriminant_sqrt.dtype))) * discriminant_sqrt
            discriminant_sqrt_with_sign = tf.sign(dir_ders_sum) * discriminant_sqrt
            # discriminant_sqrt_with_sign = tf.where(dir_ders_sum < 0, x=(-discriminant_sqrt), y=discriminant_sqrt)
            # TODO: Test out which of the above three statements run faster

        with tf.variable_scope('Beta_q'):
            beta_q = tf.atan(2 * lxy / (dir_ders_diff - discriminant_sqrt_with_sign))

        with tf.variable_scope('Lp'):
            lp = (tf.sin(beta_q) * lx) - (tf.cos(beta_q) * ly)

        with tf.variable_scope('M_norm'):
            # Note that the normalization factor and the negative sign are included here just to avoid additional TF
            # operations at model runtime. When Lpp (or non-nornalized M_norm) is needed, multiply the M_norm with
            # the normalization_factor value accordingly in order to compensate for this. The 'negative' is not part of
            # the definition of Lpp, but I have it here to turn the values positive (sharp ridges will have high
            # positive values instead of high-magnitude negative values).
            m_norm = tf.constant(normalization_factor * (-1/2), dtype=lxx.dtype) * \
                  (dir_ders_sum + discriminant_sqrt_with_sign)

    return lp, m_norm, discriminant_sqrt


def _get_tf_constants_for_zero_crossings(height, width, aux_axis=3):
    """Supplies the TF constants needed in get_zero_crossings()."""

    # Build the constants that are common across scales
    zc_constants = {
        # Slicing offsets and sizes
        'begin_t_l': tf.constant([0, 0, 0], dtype=tf.int32, name='begin_t_l'),
        'begin_t1_l': tf.constant([0, 1, 0], dtype=tf.int32, name='begin_t1_l'),
        'begin_t_l1': tf.constant([0, 0, 1], dtype=tf.int32, name='begin_t_l1'),
        'begin_t1_l1': tf.constant([0, 1, 1], dtype=tf.int32, name='begin_t1_l1'),
        'size_h1_w': tf.constant([-1, height - 1, -1], dtype=tf.int32, name='size_h1_w'),
        'size_h_w1': tf.constant([-1, -1, width - 1], dtype=tf.int32, name='size_h_w1'),
        'size_h1_w1': tf.constant([-1, height - 1, width - 1], dtype=tf.int32, name='size_h1_w1'),

        # Padding sides and sizes
        'freq_first': tf.constant([[0, 0], [1, 0], [0, 0]], dtype=tf.int32, name='freq_first'),
        'freq_last': tf.constant([[0, 0], [0, 1], [0, 0]], dtype=tf.int32, name='freq_last'),
        'time_first': tf.constant([[0, 0], [0, 0], [1, 0]], dtype=tf.int32, name='time_first'),
        'time_last': tf.constant([[0, 0], [0, 0], [0, 1]], dtype=tf.int32, name='time_last'),

        # Other scalars
        'one': tf.constant(1, dtype=tf.int32, name='zc_1'),
        'freq_axis': tf.constant(1, dtype=tf.int32, name='zc_freq_axis'),
        'time_axis': tf.constant(2, dtype=tf.int32, name='zc_time_axis'),
        'aux_axis': tf.constant(aux_axis, dtype=tf.int32, name='zc_aux_axis')
    }

    return zc_constants


def get_zero_crossings(in_surface, zc_constants=None):
    """Find positions of zero-crossings (ZC) in 'in_surface'.
    Adjusts positions to the lower of two points involved in each zero-crossing. Returns a boolean mask where there are
    zero-crossings. in_suface should only have dimensions N, H and W (in that order) and no C.
    If zc_constants is not None, then it MUST be the python dict returned from a previous call to
    _get_tf_constants_for_zero_crossings(). Note that no checks for integrity of a not-None zc_constants is
    performed."""

    #return tf.is_finite(in_surface)    # Use this instead of everything above, if ZC is to be bypassed

    # Auxiliary axis, in which per-scale combinatorial operations will be performed
    aux_axis = len(in_surface.shape)    # New & last axis

    if zc_constants is None:
        h, w = in_surface.shape[1:3]

        zc_constants = _get_tf_constants_for_zero_crossings(h, w, aux_axis)

    with tf.variable_scope('zero_crossings_mask', reuse=True):

        surface_abs = tf.abs(in_surface)
        surface_signs = tf.cast(tf.sign(in_surface), dtype=tf.int8)

        conditions = list()     # For gathering the conditionals' values

        # Check with pixel above
        with tf.variable_scope('zc_top', reuse=True):
            zcs = tf.not_equal(tf.slice(surface_signs, zc_constants['begin_t_l'], zc_constants['size_h1_w']),
                               tf.slice(surface_signs, zc_constants['begin_t1_l'], zc_constants['size_h1_w']))
            lower_vals = tf.less(tf.slice(surface_abs, zc_constants['begin_t_l'], zc_constants['size_h1_w']),
                                 tf.slice(surface_abs, zc_constants['begin_t1_l'], zc_constants['size_h1_w']))

            conditions.append(tf.pad(tf.logical_and(zcs, lower_vals), zc_constants['freq_last'],
                                     mode='CONSTANT', constant_values=False))
            conditions.append(tf.pad(tf.logical_and(zcs, tf.logical_not(lower_vals)), zc_constants['freq_first'],
                                     mode='CONSTANT', constant_values=False))

        # Check with pixel to right
        with tf.variable_scope('zc_right', reuse=True):
            zcs = tf.not_equal(tf.slice(surface_signs, zc_constants['begin_t_l'], zc_constants['size_h_w1']),
                               tf.slice(surface_signs, zc_constants['begin_t_l1'], zc_constants['size_h_w1']))
            lower_vals = tf.less(tf.slice(surface_abs, zc_constants['begin_t_l'], zc_constants['size_h_w1']),
                                 tf.slice(surface_abs, zc_constants['begin_t_l1'], zc_constants['size_h_w1']))

            conditions.append(tf.pad(tf.logical_and(zcs, lower_vals), zc_constants['time_last'],
                                     mode='CONSTANT', constant_values=False))
            conditions.append(tf.pad(tf.logical_and(zcs, tf.logical_not(lower_vals)), zc_constants['time_first'],
                                     mode='CONSTANT', constant_values=False))

        # Check with pixel to above-right
        with tf.variable_scope('zc_top_right', reuse=True):
            zcs = tf.not_equal(tf.slice(surface_signs, zc_constants['begin_t_l'], zc_constants['size_h1_w1']),
                               tf.slice(surface_signs, zc_constants['begin_t1_l1'], zc_constants['size_h1_w1']))
            lower_vals = tf.less(tf.slice(surface_abs, zc_constants['begin_t_l'], zc_constants['size_h1_w1']),
                                 tf.slice(surface_abs, zc_constants['begin_t1_l1'], zc_constants['size_h1_w1']))

            conditions.append(tf.pad(
                tf.pad(tf.logical_and(zcs, lower_vals), zc_constants['time_last'],
                       mode='CONSTANT', constant_values=False),
                zc_constants['freq_last'], mode='CONSTANT', constant_values=False))
            conditions.append(tf.pad(
                tf.pad(tf.logical_and(zcs, tf.logical_not(lower_vals)), zc_constants['time_first'],
                       mode='CONSTANT', constant_values=False),
                zc_constants['freq_first'], mode='CONSTANT', constant_values=False))

        # Check with pixel to above-left
        with tf.variable_scope('zc_top_left', reuse=True):
            zcs = tf.not_equal(tf.slice(surface_signs, zc_constants['begin_t_l1'], zc_constants['size_h1_w1']),
                               tf.slice(surface_signs, zc_constants['begin_t1_l'], zc_constants['size_h1_w1']))
            lower_vals = tf.less(tf.slice(surface_abs, zc_constants['begin_t_l1'], zc_constants['size_h1_w1']),
                                 tf.slice(surface_abs, zc_constants['begin_t1_l'], zc_constants['size_h1_w1']))

            conditions.append(tf.pad(
                tf.pad(tf.logical_and(zcs, lower_vals), zc_constants['time_first'],
                       mode='CONSTANT', constant_values=False),
                zc_constants['freq_last'], mode='CONSTANT', constant_values=False))
            conditions.append(tf.pad(
                tf.pad(tf.logical_and(zcs, tf.logical_not(lower_vals)), zc_constants['time_last'],
                       mode='CONSTANT', constant_values=False),
                zc_constants['freq_first'], mode='CONSTANT', constant_values=False))

        # Combine all the above conditions, and return ZC mask.
        return tf.reduce_any(tf.stack(conditions, axis=aux_axis), axis=zc_constants['aux_axis'])


def extract_ridges(in_surfaces, SNR_threshold, scales_sigmas=(1, 2), data_format='NHWC', eps=1e-10):
    """scales_sigmas must be a tuple or list of sigma values at different (usually, geometrically progressing) scales.
    """

    # Algorithm constants & parameters
    gamma = 3. / 4.
    ridge_narrowness_threshold = 0.2  # a threshold, 20%; for a description, see where it's used below
    #spectral_cutoff_thld_prctile = 90.

    pad_len = 2  # Amount of padding to do. Be sure to keep this an even number

    assert data_format in ['NHWC', 'NCHW']   # only 2 formats supported
    if data_format == 'NHWC':
        height_axis, width_axis, channel_axis = (1, 2, 3)
        padding_amounts = [[0, 0], [pad_len, pad_len], [pad_len, pad_len], [0, 0]]
    else:
        height_axis, width_axis, channel_axis = (2, 3, 1)
        padding_amounts = [[0, 0], [0, 0], [pad_len, pad_len], [pad_len, pad_len]]

    h, w = in_surfaces.shape[height_axis], in_surfaces.shape[width_axis]

    # Auxiliary axis, in which per-scale combinatorial operations will be performed
    aux_axis = channel_axis

    #depad_offset = tf.constant([0, pad_len, pad_len], dtype=tf.int32, name='depad_offset')
    #depad_size = tf.constant([-1, h, w], dtype=tf.int32, name='depad_size')

    with tf.variable_scope('ridge_extraction'):

        # TF constants across scales
        dir_der_kernels = _get_tf_constants_for_directional_derivatives(h, w, tf_dtype=in_surfaces.dtype,
                                                                        data_format=data_format)
        zc_constants = _get_tf_constants_for_zero_crossings(h, w,
                                                            len(in_surfaces.shape) - 1)  # -1, coz of later "tf.squeeze"
        # Setting to half the given value, because we compute double differentials here
        min_SNR = tf.constant(SNR_threshold / 2., dtype=in_surfaces.dtype, name='min_M_norm')
        db_conversion_constant = tf.constant(10. / np.log(10.), dtype=in_surfaces.dtype)
        aux_axis_tf = tf.constant(aux_axis, dtype=tf.int32, name='aux_axis')

        # Add padding prior to convolutions so that values at boundaries are not very unrealistic
        padded_inputs = tf.pad(in_surfaces, padding_amounts, mode='CONSTANT', constant_values=eps)

        m_norm_list = list()
        for curr_scale_sigma in np.asarray(scales_sigmas):
            curr_scale_t = curr_scale_sigma ** 2
            scale_normalization_factor = curr_scale_t ** gamma

            # Apply spatial smoothing
            curr_scale_space = apply_gaussian_blur(padded_inputs, apply_2d=True, sigma=curr_scale_sigma,
                                                   data_format=data_format)

            # Convert to dB scale
            curr_scale_space = Convert.linear2db(curr_scale_space, eps, db_conversion_constant=db_conversion_constant)

            # Get directional derivatives
            lx, ly, lxx, lyy, lxy = get_directional_derivatives(curr_scale_space, dir_der_constants=dir_der_kernels,
                                                                data_format=data_format)

            # Strip the channel axis. This should have ideally be done to 'inputs' itself. But, TF implementation of
            # conv1d and conv2d operations make that impossible. Hence, doing so here.
            with tf.variable_scope('strip_channel_axis'):
                #curr_scale_space = tf.squeeze(curr_scale_space, axis=channel_axis)
                lx = tf.squeeze(lx, axis=channel_axis)
                ly = tf.squeeze(ly, axis=channel_axis)
                lxx = tf.squeeze(lxx, axis=channel_axis)
                lyy = tf.squeeze(lyy, axis=channel_axis)
                lxy = tf.squeeze(lxy, axis=channel_axis)

            # Compute ridge strengths
            lp, m_norm, abs_lpp_minus_lqq = _compute_ridge_measurements(lx, ly, lxx, lyy, lxy,
                                                                        normalization_factor=scale_normalization_factor)

            ridge_point_mask_list = list()

            # Find zero-crossings in lp
            lp_zc_mask = get_zero_crossings(lp, zc_constants=zc_constants)
            ridge_point_mask_list.append(lp_zc_mask)

            # Get a mask for ( |Lpp-Lqq| / |Lpp| > threshold
            with tf.variable_scope('narrow_ridges_mask'):
                # Note that, below, the division by scale_normalization_factor is done because the function
                # compute_ridge_measurements() had already multiplied "lpp" with scale_normalization_factor to
                # produce m_norm. I'm just recovering the actual "lpp" here with this division.
                narrow_ridges_mask = (abs_lpp_minus_lqq >
                                      (m_norm * (ridge_narrowness_threshold / scale_normalization_factor)))
            ridge_point_mask_list.append(narrow_ridges_mask)

            # The check "m_norm > 0" weeds out valley points; I'm clubbing together a threshold value (> 0) along
            # with this check in order to discard very weak ridges.
            with tf.variable_scope('high_ridges_mask', reuse=True):
                high_m_norm_mask = tf.greater(m_norm,
                                              min_SNR)# * tf.slice(curr_scale_space, depad_offset, depad_size))
                ridge_point_mask_list.append(high_m_norm_mask)

            #with tf.variable_scope('spectral_thld_mask'):
            #    curr_scale_space = tf.slice(curr_scale_space, depad_offset, depad_size)  # de-padded
            #    # curr_scale_space = tf.log(curr_scale_space + eps)
            #    spectral_cutoff_thld = tf.contrib.distributions.percentile(curr_scale_space,
            #                                                               spectral_cutoff_thld_prctile,
            #                                                               axis=[1, 2], keep_dims=True)
            #    spectral_thld_mask = (curr_scale_space > spectral_cutoff_thld)
            #    ridge_point_mask_list.append(spectral_thld_mask)

            # First, combine the different masks
            with tf.variable_scope('combine_masks'):
                ridge_pts_mask = tf.reduce_all(
                    tf.stack(ridge_point_mask_list, axis=aux_axis),
                    axis=aux_axis_tf)

            # Now apply the combined mask
            with tf.variable_scope('apply_masks'):
                m_norm = m_norm * tf.cast(ridge_pts_mask, dtype=m_norm.dtype)

            m_norm_list.append(m_norm)

        if len(m_norm_list) > 1:
            all_scale_ridges = tf.stack(m_norm_list, axis=channel_axis)
        else:
            all_scale_ridges = tf.expand_dims(m_norm_list[0], axis=channel_axis)

    return all_scale_ridges


def extract_blobs(in_surfaces, SNR_threshold, scales_sigmas=(4, 8, 16), data_format='NHWC', eps=1e-10):
    """scales_sigmas must be a tuple or list of sigma values at different (usually, geometrically progressing) scales.
    You may use this formula to determine the possible set of sigma values beyond the lowest_sigma:
            lowest_sigma * (2 ^ (range(2, floor(log2((max_len - 1) / ((2 x 3) x lowest_sigma)) + 1) + 1) - 1))
    For example, if lowest_sigma is 4 & max_len is 243, the resulting set of sigmas should be (4, 8, 16, 32).
    """

    # Algorithm constants & parameters
    t_blur_sigma = 0.75
    t_blur_pad_len = np.ceil(3 * t_blur_sigma).astype(np.int)

    # Setting to half the given value, because we compute double differentials here
    min_SNR = tf.constant(SNR_threshold / 2., dtype=in_surfaces.dtype, name='min_blob_SNR')

    assert data_format in ['NHWC', 'NCHW']   # only 2 formats supported
    if data_format == 'NHWC':
        channel_axis = 3
        height = in_surfaces.shape.as_list()[1]
        t_padding_base_vec = [[0, 0], [0, 0], [1, 1], [0, 0]]
        f_padding_base_vec = [[0, 0], [1, 1], [0, 0], [0, 0]]
    else:
        channel_axis = 1
        height = in_surfaces.shape.as_list()[2]
        t_padding_base_vec = [[0, 0], [0, 0], [0, 0], [1, 1]]
        f_padding_base_vec = [[0, 0], [0, 0], [1, 1], [0, 0]]

    scales_sigmas = np.asarray(scales_sigmas)
    temp = ((2 * 3) * scales_sigmas) + 1      # compute 6 x sigma width
    if any(temp > height):
        logging.warning('Ignoring blob scales_sigmas larger than %.2f' % ((height - 1) / (2 * 3)))

        scales_sigmas = scales_sigmas[temp <= height]

    with tf.variable_scope('blobs_extraction'):

        # Temporal blurring, with a Gaussian kernel along x-axis only
        with tf.variable_scope('temporal_blur'):
            # Add padding so that values at boundaries are not very unrealistic
            blurred_inputs = tf.pad(in_surfaces, np.asarray(t_padding_base_vec) * t_blur_pad_len, 'SYMMETRIC')

            # Get only the valid points after convolution
            blurred_inputs = apply_gaussian_blur(blurred_inputs, apply_2d=False, sigma=t_blur_sigma,
                                                 padding='VALID', data_format=data_format)

        # For blob-detection we need log (dB) scale inputs.
        blob_det_inputs = Convert.linear2db(blurred_inputs, eps)

        # Process at all scales
        prev_scale_padding = np.int(0)
        all_scale_LoGs = list()
        for curr_scale_sigma in scales_sigmas:
            # Add padding (incrementally) prior to convolutions so that values at boundaries are not very
            # unrealistic.
            curr_scale_padding = np.int(3 * curr_scale_sigma)
            incr_padding = curr_scale_padding - prev_scale_padding

            blob_det_inputs = tf.pad(blob_det_inputs, np.asarray(f_padding_base_vec) * incr_padding, 'SYMMETRIC')

            # Apply LoG filter
            curr_scale_LoG = apply_1d_LoG(blob_det_inputs, curr_scale_sigma, data_format=data_format)

            all_scale_LoGs.append(curr_scale_LoG)

            # Update for next iteration
            prev_scale_padding = curr_scale_padding

        if len(all_scale_LoGs) > 1:
            all_scale_blobs = tf.concat(all_scale_LoGs, axis=channel_axis)
        elif len(all_scale_LoGs) == 1:
            all_scale_blobs = all_scale_LoGs[0]
        else:
            logging.warning('No outputs from blob detector at the chosen scales')
            return None

        # Suppress responses below threshold
        with tf.variable_scope('apply_thresholds'):
            all_scale_blobs = all_scale_blobs * tf.cast(all_scale_blobs > min_SNR, dtype=all_scale_blobs.dtype)

    return all_scale_blobs


class IntensityDifferentials:

    def __init__(self, snr_threshold, ridge_scales_sigmas=None, blob_scales_sigmas=None, data_format='NHWC', eps=1e-10):

        assert data_format in ['NHWC', 'NCHW']  # only 2 formats supported
        assert ridge_scales_sigmas is not None or blob_scales_sigmas is not None    # At least one is needed

        self._SNR_threshold = snr_threshold
        self._data_format = data_format
        if data_format == 'NHWC':
            self._channel_axis = 3
            self._spatial_axes = (1, 2)
        else:
            self._channel_axis = 1
            self._spatial_axes = (2, 3)
        self._eps = eps

        self._ridge_scales_sigmas = None    # Will remain None if a non-empty list or a scalar is not given
        if ridge_scales_sigmas is not None:
            if not isinstance(ridge_scales_sigmas, (list, tuple)):
                ridge_scales_sigmas = [ridge_scales_sigmas]
            if len(ridge_scales_sigmas) > 0:
                self._ridge_scales_sigmas = ridge_scales_sigmas

        self._blob_scales_sigmas = None    # Will remain None if a non-empty list or a scalar is not given
        if blob_scales_sigmas is not None:
            if not isinstance(blob_scales_sigmas, (list, tuple)):
                blob_scales_sigmas = [blob_scales_sigmas]
            if len(blob_scales_sigmas) > 0:
                self._blob_scales_sigmas = blob_scales_sigmas

    def __call__(self, inputs, **kwargs):

        with tf.variable_scope('intensity_differentials'):
            # Standard constants
            if 'eps_tf' in kwargs:
                eps = kwargs.get('eps_tf')
            else:
                eps = tf.constant(self._eps, dtype=inputs.dtype, name='eps')

            results = list()
            if self._ridge_scales_sigmas is not None:
                all_scale_ridges = extract_ridges(inputs, self._SNR_threshold,
                                                  scales_sigmas=self._ridge_scales_sigmas,
                                                  data_format=self._data_format,
                                                  eps=eps)
                results.append(all_scale_ridges)

            if self._blob_scales_sigmas is not None:
                all_scale_blobs = extract_blobs(inputs, self._SNR_threshold,
                                                scales_sigmas=self._blob_scales_sigmas,
                                                data_format=self._data_format,
                                                eps=self._eps)
                if all_scale_blobs is not None:
                    results.append(all_scale_blobs)

            # Normalize the intensity ridges and blobs, respectively, across scales for each surface in the input batch
#           with tf.variable_scope('normalize'):
#               norm_axes = tf.constant((1, 2, 3), dtype=tf.int32, name='norm_axes')
#
#               all_scale_ridges = all_scale_ridges / tf.reduce_max(all_scale_ridges, axis=norm_axes, keepdims=True)
#               all_scale_blobs = all_scale_blobs / tf.reduce_max(all_scale_blobs, axis=norm_axes, keepdims=True)

            if len(results) > 1:
                results = tf.concat(results, axis=self._channel_axis)
            elif len(results) == 1:
                results = results[0]
            else:
                results = None  # This is an error
            # spatial_axes = tf.constant(self._spatial_axes, dtype=tf.int32, name='sp_ax')
            # results = results / tf.reduce_max(results, axis=spatial_axes, keepdims=True)

        return results


def spatial_split(inputs, split_axis, patch_size, patch_overlap=0, data_format='NHWC'):
    """Split features into patches in either the frequency (split_axis=1) or the time (split_axis=2) dimension.
    Parameters patch_size & patch_overlap are self explanatory. If the parameter 'inputs' is given as an integer instead
    of a tensorflow tensor, it indicates the number of elements available in the splitting dimension and the function
    returns the number of remaining elements after splitting, without doing the actual splitting. When 'inputs' is a
    tensorflow tensor, it is split as requested and the resulting tensor(s) are returned as a python list."""

    assert patch_size > patch_overlap

    stride = patch_size - patch_overlap

    if isinstance(inputs, int):
        # Checking only. Return the number of remainder elements, if any
        return inputs - (range(0, inputs - patch_size + 1, stride)[-1] + patch_size)

    assert data_format in ['NHWC', 'NCHW']  # only 2 formats supported
    assert split_axis in [1, 2]  # can only split along frequency or time

    split_axis += (0 if data_format == 'NHWC' else 1)
    split_axis_len = inputs.get_shape().as_list()[split_axis]

    patch_start_offset_vec = [0, 0, 0, 0]
    patch_size_vec = [-1, -1, -1, -1]
    patch_size_vec[split_axis] = patch_size
    patches_list = []

    for start_offset in range(0, split_axis_len - patch_size + 1, stride):
        patch_start_offset_vec[split_axis] = start_offset

        patches_list.append(
            tf.slice(inputs, patch_start_offset_vec, patch_size_vec))

    return patches_list


class _Augmentations_Temporal:

    # Map out some version-specific tensorflow stuff to common names
    _tf_py_wrapper = tf.py_function

    def __init__(self, fs, input_shape, axis=-1, seed=13):

        self._fs = fs
        self._d_shp = input_shape
        self._axis = axis if axis >= 0 else (len(input_shape) + axis)   # Convert negative axis index to a valid positive quantity
        self._nsamp = input_shape[axis]

    def apply_augmentations(self, data, augmentations, normalize_output=False):
        """
        Apply a chained list of augmentation functions.

        :param data: Time-domain signal (single dimension) or a batch of time-domain signals (multidimensional).
        :param augmentations: A list of 3-tuples per augmentation function, containing the following fields (in the
            listed order) -
                  i. Augmentation method handle (one of the public non-static methods of this class)
                 ii. Probability of application (in the range [0, 1])
                iii. A 3-tuple describing minval, maxval (as applicable to the chosen method) and the datatype. These
                       will be passed as-is to tf.random_uniform().
        :return:
        """

        if len(augmentations) == 0:
            return data

        for fn, prob, args in augmentations:
            data = tf.cond(tf.random.uniform([], 0, 1) <= prob,
                           lambda: fn(self, data, tf.random.uniform([], *args)),
                           lambda: data,
                           name=fn.__name__)

        if normalize_output:
            # Normalize. Necessary, because the time-domain augmentation(s) would have altered the amplitudes
            return data / tf.reduce_max(tf.abs(data), axis=-1, keepdims=True)
        else:
            return data

    @staticmethod
    def _expand_dims_to_match(input, num_result_dims, pivot_axis):
        """
        Expand the dimensions of a 1-dimensional array 'input' to have 'num_result_dims' dimensions while keeping the
        values of 'input' in the 'pivot_axis' axis.
        """

        assert pivot_axis < num_result_dims

        output = input

        for _ in range(pivot_axis):     # Iteratively add dimensions until the pivot
            output = tf.expand_dims(output, 0)
        for _ in range(pivot_axis + 1, num_result_dims):   # Iteratively add dimensions beyond the pivot
            output = tf.expand_dims(output, -1)

        return output

    @staticmethod
    def volume_ramp(obj, data, val):
        """
        Alter the volume of signal by ramping up/down its amplitude linearly across the duration of the signal.

        :param obj: An instance of _Augmentations_Temporal.
        :param data: Time-domain signal (single dimension) or a batch of time-domain signals (multidimensional).
        :param val: A value in the range -1.0 to 1.0.
            Will ramp up from (1.0 - val) to 1.0 if val is non-negative.
            Will ramp down from 1.0 to (1.0 - abs(val)) if val is negative.
        :return:
            Signal with ramping up/down volume.
        """

        tf_one = tf.constant(1.0, dtype=val.dtype, name='1.0')
        start_amp, end_amp = tf.cond(val >= 0, lambda: (tf_one - val, tf_one), lambda: (tf_one, tf_one + val))
        amp_factor = tf.linspace(start_amp, end_amp, obj._nsamp)

        # Expand amp_factor array to match the dimensions of data array
        amp_factor = _Augmentations_Temporal._expand_dims_to_match(amp_factor, len(obj._d_shp), obj._axis)

        return data * amp_factor

    @staticmethod
    def volume_fluctuate(obj, data, val):
        """
        Alter the volume of signal by changing amplitudes by random values at random points.
        The points where (x) amplitude will change and the change values (y) are both chosen randomly. The points are
        used to construct a (mostly) smooth polynomial that will define the amplitude envelope of the resulting signal.

        :param obj: An instance of _Augmentations_Temporal.
        :param data: Time-domain signal (single dimension) or a batch of time-domain signals (multidimensional).
        :param val: The number of points, between the start and end of signal, where amplitude will change value.
            Best to set it to a value that yields a rate of a max of 3 pts/sec (roughly one point per 0.25 sec). Set in
            the range [1, a decent positive integer]. A very large value will distort the signal very much.
        :return:
            Signal with random amplitude fluctuations.
        """

        (min_amp, max_amp) = (0.3, 1.0)     # The range to restrict the new amplitudes to

        if val.dtype not in [tf.int32, tf.int64]:
            num_pts = tf.cast(tf.round(val), tf.int32)
        else:
            num_pts = val

        # positions where amplitudes will change and the amplitude factors at those positions
        interim_pts, _ = tf.unique(tf.random.uniform([num_pts, ], minval=1, maxval=obj._nsamp-1, dtype=tf.int32))
        positions = tf.concat([[0], interim_pts, [obj._nsamp-1]],
                              axis=0)     # Include the start and end of the time-series as well
        factors = tf.random.uniform(tf.shape(positions), minval=min_amp, maxval=max_amp, dtype=data.dtype)

        # Wrap scipy functionality to generate a (nearly) smooth curve
        amp_factor = _Augmentations_Temporal._tf_py_wrapper(
            lambda x, y: interp1d(np.sort(x.numpy()), y.numpy(), kind='quadratic' if len(x.numpy()) > 3 else 'linear')(np.arange(obj._nsamp)),
            inp=[positions, factors], Tout=data.dtype, name='scipy_amp_envlp_smooth')
        amp_factor.set_shape(obj._nsamp)
        amp_factor = tf.clip_by_value(amp_factor, min_amp, max_amp)     # Clip to valid range

        # Expand amp_factor array to match the dimensions of data array
        amp_factor = _Augmentations_Temporal._expand_dims_to_match(amp_factor, len(obj._d_shp), obj._axis)

        return data * amp_factor

    @staticmethod
    def add_echo(obj, data, val):
        """
        Produce echo effect by adding a dampened and delayed copy of data to data. The dampened copy is produced by
        using a random attenuation factor in the range [-15 dB, -12 dB].

        :param obj: An instance of _Augmentations_Temporal.
        :param data: Time-domain signal (single dimension) or a batch of time-domain signals (multidimensional).
        :param val: The delay (in seconds) of the echo.
        :return:
            Signal with added echo.
        """

        echo_amp = tf.pow(10.0, (tf.random.uniform([], minval=-15.0, maxval=-12.0) / 20))

        echo_offset = tf.cast(tf.round(np.float32(obj._fs) * val), tf.int32)

        slice_start_idxs = [0 for _ in obj._d_shp]  # start at zeros along all dimensions
        slice_sizes = [x for x in obj._d_shp]
        slice_sizes[obj._axis] = obj._d_shp[obj._axis] - echo_offset     # Only take samples from the head
        paddings = [[0, 0] for _ in obj._d_shp]
        paddings[obj._axis][0] = echo_offset   # To add these many zeros at the head along the _axis dimension

        echo = tf.pad(echo_amp * tf.slice(data, slice_start_idxs, slice_sizes), paddings, 'CONSTANT')

        return data + echo

    @staticmethod
    def pitch_shift(obj, data, val):
        """
        Shift pitch up/down using Fourier method (involves conversion to spectral domain and back).

        :param obj: An instance of _Augmentations_Temporal.
        :param data: Time-domain signal (single dimension) or a batch of time-domain signals (multidimensional).
        :param val: Number of Hertz to shift the pitch by. Positive value shifts pitch upward and negative value shifts
            pitch downward. A very high absolute value could distort the signal very badly. It is sensible to use small
            (positive or negative) values. The effective amount of shift is a value that is rounded to the nearest
            integral multiple of the frequency resolution resulting from the sampling rate and the number of samples
            available.
        :return:
            Pitch-shifted time-domain signal of the same dimensions as :param data:
        """

        df = float(obj._fs) / float(obj._nsamp)
        diff_amt = tf.cast(tf.abs(tf.round(val / df)), dtype=tf.int32)  # absolute number of bins to shift up/down

        num_bins = obj._nsamp // 2 + 1
        tf_zero = tf.constant(0, dtype=tf.int32, name='zero')

        def odd_crop(lower):
            # Returns crop_start, crop_end and last dimension's lower & upper padding amounts
            if lower:   # Keep lower part, for shifting up
                return tf_zero, num_bins - diff_amt, diff_amt, tf_zero
            else:   # Keep upper part, for shifting down
                return diff_amt, tf.constant(num_bins, dtype=tf.int32), tf_zero, diff_amt

        def even_crop(lower):
            # Returns crop_start, crop mid, crop_end, scale factor and last dimension's lower & upper padding amounts
            if lower:   # Keep lower part, for shifting up
                upper_idx = num_bins - diff_amt
                return tf_zero, upper_idx - 1, upper_idx, 2., diff_amt, tf_zero
            else:   # Keep upper part, for shifting down
                return diff_amt,\
                       tf.constant(num_bins - 1, dtype=tf.int32),\
                       tf.constant(num_bins, dtype=tf.int32),\
                       1/2.,\
                       tf_zero, diff_amt

        def do_shift(data):

            # Need to move *_axis* to be the last dimension for tf.signal.rfft()
            if obj._axis != len(obj._d_shp) - 1:
                data = tf.transpose(data, perm=np.concatenate([range(obj._axis),
                                                               range(obj._axis + 1, len(obj._d_shp)),
                                                               [obj._axis]]))

            X = tf.signal.rfft(data, [obj._nsamp])

            # win = tf.constant(fftpack.ifftshift(signal.get_window('hamming', self._nsamp))[:num_bins], dtype=tf.complex64, name='window')
            # win = _Augmentations_Temporal._expand_dims_to_match(win, len(self._d_shp), len(self._d_shp) - 1)
            #
            # X = X * win

            if obj._nsamp % 2:
                crop_start, crop_end, ld_l_pad, ld_u_pad = tf.cond(tf.greater(val, 0),
                                                                   lambda: odd_crop(lower=True),
                                                                   lambda: odd_crop(lower=False))

                X = X[..., crop_start:crop_end]

            else:   # Special treatment for the point at N/2
                crop_start, crop_mid, crop_end, mulval, ld_l_pad, ld_u_pad = tf.cond(tf.greater(val, 0),
                                                                                     lambda: even_crop(lower=True),
                                                                                     lambda: even_crop(lower=False))

                X = tf.concat([X[..., crop_start:crop_mid], X[..., crop_mid:crop_end] * tf.cast(mulval, X.dtype)],
                              axis=len(obj._d_shp) - 1)

            paddings = [[0, 0] for _ in range(len(obj._d_shp) - 1)]
            paddings.append([ld_l_pad, ld_u_pad])
            X = tf.pad(X, paddings, 'CONSTANT')     # Pad zeros at lower/higher frequencies as applicable

            data = tf.signal.irfft(X, [obj._nsamp])

            if obj._axis != len(obj._d_shp) - 1:  # transpose back, if applicable
                data = tf.transpose(data, perm=np.concatenate([range(obj._axis),
                                                               [len(obj._d_shp) - 1],
                                                               range(obj._axis, len(obj._d_shp) - 1)]))

            return data

        return tf.cond(tf.equal(diff_amt, tf_zero), lambda: data, lambda: do_shift(data))


class _Augmentations_SpectroTemporal:

    def __init__(self, input_shape, f_axis, t_axis, seed=13):
        """

        :param input_shape: Must be a 2, 3, or 4 element list/tuple. The contents will be interpreted as dimensions
            [height (H), width(W)] if 2 elements, [H, W, channels (C)] if 3 elements, and as [batches (B), H, W, C] if
            4 elements.
        :param f_axis: Index of the frequency axis in input_shape
        :param t_axis: Index of the time axis in input_shape
        :param seed: Seed to drive randomization.
        """

        assert len(input_shape) >= 2

        self._d_shp = input_shape
        self._f_axis = f_axis
        self._t_axis = t_axis

        self._batches = 1 if len(self._d_shp) < 4 else input_shape[0]
        self._height = self._d_shp[f_axis]
        self._width = self._d_shp[t_axis]
        self._channels = 1 if len(self._d_shp) < 3 else input_shape[3]

    def apply_augmentations(self, data, augmentations):
        """
        Apply a chained list of augmentation functions.

        :param data: Time-domain signal (single dimension) or a batch of time-domain signals (multidimensional).
        :param augmentations: A list of 3-tuples per augmentation function, containing the following fields (in the
            listed order) -
                  i. Augmentation method handle (one of the public non-static methods of this class)
                 ii. Probability of application (in the range [0, 1])
                iii. A 3-tuple describing minval, maxval (as applicable to the chosen method) and the datatype. These
                       will be passed as-is to tf.random_uniform().
        :return:
        """

        if len(augmentations) == 0:
            return data

        for fn, prob, args in augmentations:
            data = tf.cond(tf.random.uniform([], 0, 1) <= prob,
                           lambda: fn(self, data, tf.random.uniform([], *args)),
                           lambda: data,
                           name=fn.__name__)

        #data = tf.clip_by_value(data, 0.0, 1.0)  # Clip to valid range

        return data

    def _upsample_and_crop(self, data, f_incr=None, crop_out_higher_f=True, t_incr=None, crop_out_later_t=True):
        """
        At least one of the pairs (f_incr, crop_out_higher_f) or (t_incr, crop_out_later_t) MUST be valid values.
        """

        assert (f_incr is not None) or (t_incr is not None)

        tf_zero = tf.constant(0, dtype=tf.int32)

        if f_incr is not None:
            f_slice_offset = tf.cond(crop_out_higher_f, lambda: tf_zero, lambda: f_incr)
        else:
            f_incr = 0
            f_slice_offset = tf_zero

        if t_incr is not None:
            t_slice_offset = tf.cond(crop_out_later_t, lambda: tf_zero, lambda: t_incr)
        else:
            t_incr = 0
            t_slice_offset = tf_zero

        # Conversion to 4D is necessary for the below tf.image operation.
        if len(self._d_shp) < 4:  # Add a dummy batch axis
            data = tf.expand_dims(data, axis=0)
        if len(self._d_shp) < 3:  # Add a dummy channel axis
            data = tf.expand_dims(data, axis=-1)

        # Resize
        data = tf.image.resize(data, [self._height + f_incr, self._width + t_incr], method='bilinear')

        # Crop
        data = tf.slice(data,
                        tf.concat([[tf_zero], [f_slice_offset], [t_slice_offset], [tf_zero]], axis=0),
                        [self._batches, self._height, self._width, self._channels])

        # Revert the conversion to 4D, if done.
        if len(self._d_shp) < 3:  # Remove the added dummy channel axis
            data = tf.squeeze(data, axis=-1)
        if len(self._d_shp) < 4:  # Remove the added dummy batch axis
            data = tf.squeeze(data, axis=0)

        return data

    @staticmethod
    def frequency_smear(obj, data, val):
        """

        :param obj: An instance of _Augmentations_SpectroTemporal.
        :param data: Time-frequency representation (2D, 3D or 4D as per the input_shape parameter to the constructor).
        :param val: A small fractional value, negative or positive, indicating weather to smear downwards or upwards in
            frequency, respectively. The amount to smear by is determined as (abs(val) * height).
        :return:
        """

        incr = tf.cast(tf.abs(tf.round(np.float32(obj._height) * val)), tf.int32)

        return tf.cond(tf.equal(incr, 0),
                       lambda: data,
                       lambda: obj._upsample_and_crop(data, f_incr=incr, crop_out_higher_f=tf.greater(val, 0)))

    @staticmethod
    def time_smear(obj, data, val):
        """

        :param obj: An instance of _Augmentations_SpectroTemporal.
        :param data: Time-frequency representation (2D, 3D or 4D as per the input_shape parameter to the constructor).
        :param val: A small fractional value, negative or positive, indicating weather to smear backwards or forwards in
            time, respectively. The amount to smear by is determined as (abs(val) * width).
        :return:
        """

        incr = tf.cast(tf.abs(tf.round(np.float32(obj._width) * val)), tf.int32)

        return tf.cond(tf.equal(incr, 0),
                       lambda: data,
                       lambda: obj._upsample_and_crop(data, t_incr=incr, crop_out_later_t=tf.greater(val, 0)))

    @staticmethod
    def pitch_shift(obj, data, val):

        diff_amt = tf.cast(tf.abs(tf.round(np.float32(obj._height) * val)), tf.int32)

        def pad_and_downsample(input):
            # Build a padding matrix for padding either above or below along the frequency axis, and for no (zero)
            # padding along other axes.
            paddings = tf.cond(tf.less(val, 0), lambda: [0, diff_amt], lambda: [diff_amt, 0])
            paddings = tf.pad([paddings], [[obj._f_axis, len(obj._d_shp) - obj._f_axis - 1], [0, 0]], 'CONSTANT')

            input = tf.pad(input, paddings, mode='SYMMETRIC')     # Pad the inputs

            # Conversion to 4D is necessary for the below tf.image operation.
            if len(obj._d_shp) < 4:  # Add a dummy batch axis
                input = tf.expand_dims(input, axis=0)
            if len(obj._d_shp) < 3:  # Add a dummy channel axis
                input = tf.expand_dims(input, axis=-1)

            # Downsample
            input = tf.image.resize(input, [obj._height, obj._width], method='bilinear')

            # Revert the conversion to 4D, if done.
            if len(obj._d_shp) < 3:  # Remove the added dummy channel axis
                input = tf.squeeze(input, axis=-1)
            if len(obj._d_shp) < 4:  # Remove the added dummy batch axis
                input = tf.squeeze(input, axis=0)

            return input

        return tf.cond(tf.equal(diff_amt, 0),
                       lambda: data,
                       lambda: pad_and_downsample(data))

    @staticmethod
    def distance_effect(obj, data, val):
        """
        Mimic the effect of increased/reduced distance between the source and the receiver by attenuating/amplifying
        higher frequencies while keeping lower frequencies relatively unchanged.

        :param obj: An instance of _Augmentations_SpectroTemporal.
        :param data: Time-frequency representation (2D, 3D or 4D as per the input_shape parameter to the constructor).
        :param val: Attenuation/amplification amount (in dB) at the highest frequency, with negative values indicating
            attenuation and positive values indicating amplification.
        :return:
            Time-frequency representation with volume increasing/decreasing linearly along increasing frequency.
        """

        #start_stop = tf.cond(tf.less(val, 0), lambda: [0.0, val], lambda: [-val, 0.0])
        amp_scale = tf.linspace(tf.abs(val) * (-1/10), 0.0, obj._height)

        amp_scale = tf.cond(tf.greater_equal(val, 0), lambda: amp_scale, lambda: tf.reverse(amp_scale, [-1]))
        #amp_scale = (10 / np.log(10)) * tf.math.log(amp_scale)

        # Expand dims
        for _ in range(obj._f_axis):
            amp_scale = tf.expand_dims(amp_scale, axis=0)
        for _ in range(obj._f_axis + 1, len(obj._d_shp)):
            amp_scale = tf.expand_dims(amp_scale, axis=-1)

        return data + amp_scale


class _Augmentations_BackgroundInfuse:

    def __init__(self, tf_background_placeholder, prob, args):
        """

        :param tf_background_placeholder: A tensor, same shape and type as the input to apply().
        :param prob: Probability of application (in the range [0, 1]).
        :param args: A 3-tuple describing minval, maxval (dB attenuation levels, negative values) & the datatype. These
            will be passed as-is to tf.random_uniform().
        """
        self._background = tf_background_placeholder
        self._probability = prob
        self._randomizer_args = args

    def apply(self, data):

        def _internal_fn(clip):
            # Attenuate 'background' to a randomized -dB peak level and then add to input clip
            retval = clip + (
                    self._background * tf.pow(10.0, (tf.random.uniform([], *self._randomizer_args) / 20.0)))
            # Normalize
            return retval / tf.reduce_max(tf.abs(retval), axis=-1, keepdims=True)

        return tf.cond(tf.random.uniform([], 0, 1) <= self._probability,
                       lambda: _internal_fn(data),
                       lambda: data,
                       name='BackgroundInfuse')


class Augmentations:
    """Interface to implemented augmentation functionalities."""

    Temporal = _Augmentations_Temporal
    SpectroTemporal = _Augmentations_SpectroTemporal
    BackgroundInfuser = _Augmentations_BackgroundInfuse
