
import tensorflow as tf
import numpy as np
from scipy import signal
import logging
import functools

from koogu.data.raw import Filters, Settings


class Linear2dB(tf.keras.layers.Layer):
    """
    Layer for converting time-frequency (tf) representations from linear to
    decibel scale.

    :param eps: Epsilon value to add, for avoiding computing log(0.0).
    :param full_scale: (boolean) Whether to convert to dB full-scale.
    :param name: (optional; string) Name for the layer.

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
    """
    Layer for converting waveforms into time-frequency representations.

    :param  fs: sampling frequency of the data in the last dimension of inputs.
    :param spec_settings: A Python dictionary describing the settings to be used
        for producing spectrograms. Supported keys in the dictionary include:

        * win_len: (required)
          Length of the analysis window (in seconds)
        * win_overlap_prc: (required)
          Fraction of the analysis window to have as overlap between successive
          analysis windows. Commonly, a 50% (or 0.50) overlap is considered.
        * nfft_equals_win_len: (optional; boolean)
          If True (default), NFFT will equal the number of samples resulting
          from `win_len`. If False, NFFT will be set to the next power of 2 that
          is ≥ the number of samples resulting from `win_len`.
        * tf_rep_type: (optional)
          A string specifying the transformation output. 'spec' results in a
          linear scale spectrogram. 'spec_db' (default) results in a
          logarithmic scale (dB) spectrogram.
        * eps: (default: 1e-10)
          A small positive quantity added to avoid computing log(0.0).
        * bandwidth_clip: (optional; 2-element list/tuple)
          If specified, the generated spectrogram will be clipped along the
          frequency axis to only include components in the specified bandwidth.
    :param eps: (optional) If specified, will override the `eps` value in
        ``spec_settings``.
    :param name: (optional; string) Name for the layer.
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
            curr_scale_padding = int(np.ceil(curr_sigma * 4))
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


class GaussianBlur(tf.keras.layers.Layer):
    """
    Layer for applying Gaussian blur to time-frequency (tf) representations.

    :param sigma: Scalar value defining the Gaussian kernel.
    :param apply_2d: (boolean; default: True) If True, will apply smoothing
        along both time- and frequency axes. Otherwise, smoothing is only
        applied along the frequency axis.
    """

    def __init__(self, sigma=1, apply_2d=True, **kwargs):

        data_format = kwargs.pop('data_format', 'channels_last')

        assert data_format in ['channels_first', 'channels_last'], \
            'Only 2 formats supported'

        super(GaussianBlur, self).__init__(
            name=kwargs.pop('name', 'GaussianBlur'), **kwargs)

        self.sigma = sigma
        self.data_format = data_format

        f_axis, t_axis = (1, 2) if data_format == 'channels_last' else (2, 3)

        kernel = Filters.gauss_kernel_1d(sigma)
        kernel_len = np.int32(len(kernel))

        # Reshaping as [H, W, in_channels, out_channels]
        self.kernel_y = tf.constant(
            np.reshape(kernel, [kernel_len, 1, 1, 1]), dtype=self.dtype)
        if apply_2d:
            self.kernel_x = tf.constant(
                np.reshape(kernel, [1, kernel_len, 1, 1]), dtype=self.dtype)
        else:
            self.kernel_x = None

        padding_amt = int(np.ceil(sigma * 4))
        padding_vec = [[0, 0], [0, 0], [0, 0], [0, 0]]
        padding_vec[f_axis] = [padding_amt, padding_amt]
        if apply_2d:
            padding_vec[t_axis] = [padding_amt, padding_amt]
        self.padding_vec = tf.constant(padding_vec, dtype=tf.int32)

        self.input_spec = tf.keras.layers.InputSpec(ndim=4)

    @tf.function
    def call(self, inputs, **kwargs):

        if self.data_format == 'channels_last':
            data_format_other = 'NHWC'
        else:
            data_format_other = 'NCHW'

        # Add necessary padding
        outputs = tf.pad(inputs, self.padding_vec, 'SYMMETRIC')

        # Apply Gaussian kernel(s)
        outputs = tf.nn.conv2d(outputs, self.kernel_y,
                               strides=[1, 1, 1, 1],
                               padding='VALID',
                               data_format=data_format_other)
        if self.kernel_x is not None:
            outputs = tf.nn.conv2d(outputs, self.kernel_x,
                                   strides=[1, 1, 1, 1],
                                   padding='VALID',
                                   data_format=data_format_other)

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'sigma': self.sigma,
            'apply_2d': self.kernel_x is not None,
            'data_format': self.data_format
        }

        base_config = super(GaussianBlur, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Spec2Img(tf.keras.layers.Layer):
    """
    Layer for converting time-frequency representations into images. The layer's
    inputs can either be a single spectrogram (shape: H x W) or a batch of B
    spectrograms (shape: B x H x W).

    :param cmap: An Nx3 array of RGB color values. Typically, N is 256. If
        `cmap` also contains alpha values (Nx4 instead of Nx3), the last channel
        will be discarded. For example, to specify a 'jet' colorscale, you could
        use `matplotlib.cm.jet(range(256))`.
    :param vmin: (optional; default: None) If specified along with `vmax`,
        spectrogram values will be scaled to the range [`vmin`, `vmax`].
    :param vmax: (optional; default: None) If specified along with `vmin`,
        spectrogram values will be scaled to the range [`vmin`, `vmax`].
    :param img_size: (optional; default: None) If not None, must specify a
        2-element tuple (new H, new W) that indicates the shape that the output
        image must be resized to.
    :param resize_method: (optional; default: 'bilinear') If resizing of
        spectrogram(s) is enabled (via `img_size`), this parameter will define
        the method used for resizing. For available options, see TensorFlow's
        tf.image.resize().

    If only one of `vmin` or `vmax` is specified while the other isn't,
    spectrogram values will be scaled relative to the minimum and maximum values
    within each spectrogram.

    :return: If `img_size` was None, will return a tensor of shape [H x W x 3]
        or [B x H x W x 3]. If `img_size` was specified, then replace H with
        `img_size[0]` and W with `img_size[1]`.

    """

    def __init__(self, cmap, vmin=None, vmax=None, img_size=None, **kwargs):

        assert img_size is None or len(img_size) == 2, \
            '\'img_size\', if specified, must be a 2-element list/tuple'

        # Force to be non-trainable
        kwargs['trainable'] = False

        self._cmap = cmap[:, :3]    # keep only RGB; will discard alpha
        self._img_size = img_size
        self._resize_method = kwargs.pop('resize_method', 'bilinear')

        if vmin is not None and vmax is not None:
            self._vmin = vmin
            self._vmax = vmax
        else:
            self._vmin = self._vmax = None

        super(Spec2Img, self).__init__(
            name=kwargs.pop('name', 'Spec2Img'), **kwargs)

        self._colors = tf.constant(self._cmap, dtype=self.dtype)

    @tf.function
    def call(self, inputs, **kwargs):

        outputs = inputs

        # Normalize values to the range [0.0, 1.0]
        if self._vmin is None:
            # Scale to available range
            outputs = outputs - tf.reduce_min(outputs,
                                              axis=(-2, -1), keepdims=True)
            outputs = outputs / tf.reduce_max(outputs,
                                              axis=(-2, -1), keepdims=True)
        else:
            # Apply capping, if/as necessary
            outputs = tf.maximum(self._vmin, outputs)
            outputs = tf.minimum(self._vmax, outputs)
            # Now scale
            outputs = (outputs - self._vmin) / self._vmax

        # Quantize
        idxs = tf.cast(tf.round(outputs * (self._cmap.shape[0] - 1)), tf.int32)

        # Map to colors
        outputs = tf.gather(self._colors, idxs)

        # Resize, if requested
        if self._img_size is not None:
            outputs = tf.image.resize(outputs, self._img_size,
                                      method=self._resize_method)

        # If not already in the desired type, cast it
        if self.dtype != outputs.dtype:
            outputs = tf.cast(outputs, self.dtype)

        return outputs

    def compute_output_shape(self, input_shape):
        return (
            # Batch dim (if any)
            input_shape[:-2] +
            # Image dims
            (input_shape[-2:] if self._img_size is None else self._img_size) +
            # Channels
            (3, ))

    def get_config(self):
        config = {
            'cmap': self._cmap
        }
        if self._img_size is not None:
            config['img_size'] = self._img_size
            config['resize_method'] = self._resize_method
        if self._vmin is not None:
            config['vmin'] = self._vmin
            config['vmax'] = self._vmax

        base_config = super(Spec2Img, self).get_config()
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
