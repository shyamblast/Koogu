
import tensorflow as tf
import numpy as np
from scipy import signal
#import logging
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
        self.psd_segs = np.zeros((3, 2), dtype=int)
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
                np.asarray(spec_settings.bandwidth_clip, dtype=self.dtype),
                f, dtype=self.dtype)

            # Clip to non-zero range & avoid unnecessary multiplications
            self.mel_filterbank = \
                self.mel_filterbank[valid_f_idx_start:
                                    (valid_f_idx_end + 1), :]

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

    If only one of `vmin` or `vmax` is specified, it will be ignored and
    spectrogram values will be scaled relative to the minimum and maximum values
    within each spectrogram. If both `vmin` and `vmax` are specified, `vmin`
    must be < `vmax`.

    :return: If `img_size` was None, will return a tensor of shape [H x W x 3]
        or [B x H x W x 3]. If `img_size` was specified, then replace H with
        `img_size[0]` and W with `img_size[1]`.

    """

    def __init__(self, cmap, vmin=None, vmax=None, img_size=None, **kwargs):

        assert img_size is None or len(img_size) == 2, \
            '\'img_size\', if specified, must be a 2-element list/tuple'

        # Force to be non-trainable
        kwargs['trainable'] = False

        self._cmap = np.asarray(cmap)[:, :3]      # keep only RGB; discard alpha
        self._img_size = img_size
        self._resize_method = kwargs.pop('resize_method', 'bilinear')

        if vmin is not None and vmax is not None:
            assert vmin < vmax, '\'vmin\' must be < \'vmax\''
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
            outputs = (outputs - self._vmin) / (self._vmax - self._vmin)

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


class NormalizeAudio(tf.keras.layers.Layer):
    """
    Layer for applying normalization to audio. Normalization (means subtraction
    followed by scaling to the range [-1.0, 1.0]) is applied by determining the
    mean and range along the last axis of the inputs.
    """

    def __init__(self, **kwargs):
        super(NormalizeAudio, self).__init__(trainable=False, **kwargs)

    @tf.function
    def call(self, inputs):

        # Subtract mean
        outputs = inputs - tf.reduce_mean(inputs, axis=-1, keepdims=True)

        # Divide by max amplitude
        outputs = outputs / tf.reduce_max(tf.abs(outputs),
                                          axis=-1, keepdims=True)

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return super(NormalizeAudio, self).get_config()
