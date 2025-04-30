import abc
import tensorflow as tf
import logging


class _Augmentation(metaclass=abc.ABCMeta):
    """
    Abstract base class for data augmentations.
    """

    def __init__(self):
        pass

    def __call__(self, in_data, **kwargs):
        return self.build_graph(in_data)

    @abc.abstractmethod
    def build_graph(self, in_data, **kwargs):
        raise NotImplementedError(
            'build_graph() method not implemented in derived class')

    @staticmethod
    @abc.abstractmethod
    def apply_chain(in_data, augmentations, probabilities, **kwargs):
        """
        Implement in child class and invoke apply_domain_chain() from there.
        """
        raise NotImplementedError(
            'apply_chain() method not implemented in derived class')

    @staticmethod
    def _apply_domain_chain(aug_domain, in_data, augmentations, probabilities,
                            **kwargs):
        logger = logging.getLogger(__name__)

        output = in_data
        for aug, prob in zip(augmentations, probabilities):
            if not isinstance(aug, aug_domain):
                logger.warning(
                    f'{aug} is not an instance of ' +
                    f'{aug_domain.__name__}. Skipping...')
                continue

            output = tf.cond(tf.random.uniform([], 0, 1) <= prob,
                             lambda: aug(output, **kwargs),
                             lambda: output,
                             name=aug.__class__.__name__)

        return output


# Decorator for the concrete augmentation classes
def aug_register():
    def reg(cls):
        setattr(cls.__bases__[0], cls.__name__, cls)
        return cls
    return reg


class Temporal(_Augmentation):
    """
    Abstract base class for time-domain augmentations.
    """

    def __init__(self):
        super(Temporal, self).__init__()

    @staticmethod
    def apply_chain(clip, augmentations, probabilities, t_axis=-1):
        """
        Apply a chain of Temporal augmentations.

        :param clip: The audio clip to apply the augmentations to.
        :param augmentations: List of the Temporal augmentations to apply.
        :param probabilities: List of probabilities (each in the range 0-1), one
            per augmentation listed in `augmentations`.
        :param t_axis: (Defaults to -1, the last dimension) Index of the axis in
            `clip` corresponding to its time axis.

        :returns: The tensor `clip` after applying the specified augmentations.
        """
        return _Augmentation._apply_domain_chain(Temporal, clip,
                                                 augmentations, probabilities,
                                                 t_axis=t_axis)

    @staticmethod
    def _resample(clip, new_samps):
        """
        Uses tf.image operation to effect resampling in time-domain.
        """
        # Conversion to at least 3D is necessary for the tf.image operation.
        clip = tf.expand_dims(clip, axis=0)  # height axis
        clip = tf.expand_dims(clip, axis=-1)  # channel axis

        # Resize
        clip = tf.image.resize(clip, [1, new_samps], method='bilinear')

        # Strip out the added dimensions.
        clip = tf.squeeze(clip, axis=-1)
        clip = tf.squeeze(clip, axis=0)

        return clip

    @staticmethod
    def _upsample_and_crop(clip, orig_len, diff):

        # Upsample
        clip = Temporal._resample(clip, orig_len + diff)

        # Crop
        clip = tf.slice(clip, [diff // 2], [orig_len])

        return clip

    @staticmethod
    def _pad_and_downsample(clip, orig_len, diff):

        # Pad
        half_diff = diff // 2
        pad_amts = [half_diff, diff - half_diff]
        clip = tf.pad(clip, [pad_amts], mode='SYMMETRIC')  # Pad the inputs

        # Resize (downsample)
        clip = Temporal._resample(clip, orig_len)

        return clip

    @abc.abstractmethod
    def build_graph(self, in_data, t_axis):
        """
        Method which implements the desired time-domain augmentation logic as a
        Tensorflow (TF) graph.

        :param in_data: A TF array representing a single training/validation
            input (waveform).
        :param t_axis: Index, into `in_data`, of the dimension in which the
            waveform samples are.

        :return: The leaf node of the TF graph that represents the output of
            the implemented augmentation logic.
        """
        raise NotImplementedError(
            'build_graph() method not implemented in Temporal')


class SpectroTemporal(_Augmentation):
    """
    Abstract base class for spectro-temporal augmentations.
    """

    def __init__(self):
        super(SpectroTemporal, self).__init__()

    @staticmethod
    def apply_chain(spec, augmentations, probabilities, f_axis=0, t_axis=1):
        """
        Apply a chain of SpectroTemporal augmentations.

        :param spec: The spectrogram to apply the augmentations to.
        :param augmentations: List of the SpectroTemporal augmentations to
            apply.
        :param probabilities: List of probabilities (each in the range 0-1), one
            per augmentation listed in `augmentations`.
        :param f_axis: (Defaults to 0) Index of the axis in `spec` corresponding
            to its frequency axis.
        :param t_axis: (Defaults to 1) Index of the axis in `spec` corresponding
            to its time axis.

        :returns: The tensor `spec` after applying the specified augmentations.
        """
        return _Augmentation._apply_domain_chain(SpectroTemporal, spec,
                                                 augmentations, probabilities,
                                                 f_axis=f_axis, t_axis=t_axis)

    @staticmethod
    def _resize(spec, new_size):

        # Conversion to at least 3D is necessary for the tf.image operation.
        # spec = tf.expand_dims(spec, axis=0)
        spec = tf.expand_dims(spec, axis=-1)

        # Resize
        spec = tf.image.resize(spec, new_size, method='bilinear')

        # Strip out the added dimensions.
        spec = tf.squeeze(spec, axis=-1)
        # spec = tf.squeeze(spec, axis=0)

        return spec

    @staticmethod
    def _upsample_and_crop(spec,
                           f_incr=None, crop_out_higher_f=True,
                           t_incr=None, crop_out_later_t=True,
                           f_axis=0, t_axis=1):
        """
        At least one of the pairs (f_incr, crop_out_higher_f) or
        (t_incr, crop_out_later_t) MUST be valid values.
        """

        assert (f_incr is not None) or (t_incr is not None)

        tf_zero = tf.constant(0, dtype=tf.int32)

        sl_off = [None, None]   # offsets for slicing
        if f_incr is not None:
            sl_off[f_axis] = tf.cond(crop_out_higher_f,
                                     lambda: tf_zero, lambda: f_incr)
        else:
            f_incr = 0
            sl_off[f_axis] = tf_zero

        if t_incr is not None:
            sl_off[t_axis] = tf.cond(crop_out_later_t,
                                     lambda: tf_zero, lambda: t_incr)
        else:
            t_incr = 0
            sl_off[t_axis] = tf_zero

        orig_f_len = spec.shape[f_axis]
        orig_t_len = spec.shape[t_axis]

        # Resize (upsample)
        new_size = [None, None]
        new_size[f_axis] = orig_f_len + f_incr
        new_size[t_axis] = orig_t_len + t_incr
        spec = SpectroTemporal._resize(spec, new_size)

        # Crop
        new_size[f_axis] = orig_f_len
        new_size[t_axis] = orig_t_len
        spec = tf.slice(spec, sl_off, new_size)

        return spec

    @staticmethod
    def _pad_and_downsample(spec,
                            f_decr=None, pad_at_higher_f=True,
                            t_decr=None, pad_at_later_t=True,
                            f_axis=0, t_axis=1):
        """
        At least one of the pairs (f_decr, pad_at_higher_f) or
        (t_decr, pad_at_later_t) MUST be valid values.
        """

        assert (f_decr is not None) or (t_decr is not None)

        tf_zero = tf.constant(0, dtype=tf.int32)

        paddings = [None, None]   # padding amounts
        if f_decr is not None:
            paddings[f_axis] = tf.cond(pad_at_higher_f,
                                       lambda: [tf_zero, f_decr],
                                       lambda: [f_decr, tf_zero])
        else:
            paddings[f_axis] = [tf_zero, tf_zero]

        if t_decr is not None:
            paddings[t_axis] = tf.cond(pad_at_later_t,
                                       lambda: [tf_zero, t_decr],
                                       lambda: [t_decr, tf_zero])
        else:
            paddings[t_axis] = [tf_zero, tf_zero]

        orig_size = [None, None]
        orig_size[f_axis] = spec.shape[f_axis]
        orig_size[t_axis] = spec.shape[t_axis]

        # Pad
        spec = tf.pad(spec, paddings, mode='SYMMETRIC')  # Pad the inputs

        # Resize (downsample)
        spec = SpectroTemporal._resize(spec, orig_size)

        return spec

    @abc.abstractmethod
    def build_graph(self, in_data, f_axis, t_axis):
        """
        Method which implements the desired spectrotemporal augmentation logic
        as a Tensorflow (TF) graph.

        :param in_data: A TF array representing a single transformed
            training/validation input (spectrogram).
        :param f_axis: Index to the "frequency" dimension in `in_data`.
        :param t_axis: Index to the "time" dimension in `in_data`.

        :return: The leaf node of the TF graph that represents the output of
            the implemented augmentation logic.
        """
        raise NotImplementedError(
            'build_graph() method not implemented in SpectroTemporal')


@aug_register()
class RampVolume(Temporal):
    """
    Alter the volume of signal by ramping up/down its amplitude linearly across
    the duration of the signal. In a way simulates the effect of the source
    moving away or towards the receiver.

    :param val_range: A 2-element list/tuple. Ramp factor will be randomly
        chosen in the range val_range[0] dB to val_range[1] dB. If the chosen
        factor is non-negative, will ramp up from ~-val dB. If the chosen factor
        is negative, will ramp down to -abs(~val) dB.
    """

    def __init__(self, val_range):

        assert val_range[0] < val_range[1]

        self._val_range_args = val_range if len(val_range) > 2 else \
            (val_range[0], val_range[1], tf.float32)

        super(RampVolume, self).__init__()

    def build_graph(self, clip, t_axis=-1):

        val = tf.random.uniform([], *self._val_range_args)

        factor = tf.cast(tf.pow(10.0, -tf.abs(val) / 20.0), dtype=clip.dtype)

        tf_one = tf.constant(1.0, dtype=clip.dtype, name='1.0')
        start_amp, end_amp = tf.cond(val >= 0,
                                     lambda: (factor, tf_one),
                                     lambda: (tf_one, factor))
        amp_factor = tf.linspace(start_amp, end_amp, clip.shape[t_axis])

        return clip * amp_factor


@aug_register()
class AddGaussianNoise(Temporal):
    """
    Add Gaussian noise.

    :param val_range: A 2-element list/tuple. The level of the added noise will
        be randomly chosen from the range val_range[0] dB to val_range[1] dB
        (both must be non-positive). The peak noise level will approximately be
        as many dB below the peak level of the input signal.
    """

    def __init__(self, val_range):

        assert val_range[0] < val_range[1] <= 0.0

        self._val_range_args = val_range if len(val_range) > 2 else \
            (val_range[0], val_range[1], tf.float32)

        super(AddGaussianNoise, self).__init__()

    def build_graph(self, clip, t_axis=-1):

        val = tf.random.uniform([], *self._val_range_args)

        # Make the max possible noise amplitude to be around the desired dB
        std = (tf.pow(10.0, -tf.abs(val) / 20.0) / 3.0 *
               tf.reduce_max(tf.abs(clip)))          # scale relative to peak

        noise = tf.random.normal([clip.shape[t_axis]], 0.0, std,
                                 dtype=clip.dtype)

        return clip + noise


@aug_register()
class AddEcho(Temporal):
    """
    Add echo. Produce echo effect by adding a dampened and delayed copy of the
    input to the input. The dampened copy is produced by using a random
    attenuation factor, and the phase of the dampened copy is also randomly
    inverted.

    :param delay_range: A 2-element list/tuple, values specified in seconds. The
        delay amount will be randomly chosen from this range.
    :param fs: Sampling frequency of the input. The chosen delay amount will be
        converted to number of samples using this value.
    :param level_range: A 2-element list/tuple or None (default). The
        attenuation factor is derived from this range. If None, it will default
        to [-18 dB, -12 dB].
    """

    def __init__(self, delay_range, fs, level_range=None):

        assert 0 <= delay_range[0] < delay_range[1]

        self._val_range_args = delay_range if len(delay_range) > 2 else \
            (delay_range[0], delay_range[1], tf.float32)

        if level_range is not None:
            assert level_range[0] < level_range[1] < 0
            self._level_range_args = (level_range[0], level_range[1],
                                      tf.float32)
        else:
            self._level_range_args = (-18.0, -12.0, tf.float32)

        self._fs = tf.constant(float(fs), dtype=tf.float32, name='fs')

        super(AddEcho, self).__init__()

    def build_graph(self, clip, t_axis=-1):

        echo_amp = (
            tf.pow(
                10.0, tf.random.uniform([], *self._level_range_args) / 20.0) *
            tf.reduce_max(tf.abs(clip)))  # scale relative to peak
        # randomly invert phase
        echo_amp = tf.cond(tf.random.uniform([], minval=0.0, maxval=1.0) >= 0.5,
                           lambda: echo_amp, lambda: -echo_amp)

        val = tf.random.uniform([], *self._val_range_args)
        echo_offset = tf.cast(tf.round(self._fs * val), tf.int32)

        echo = tf.pad(
            echo_amp * tf.slice(clip, [0], [clip.shape[t_axis] - echo_offset]),
            [[echo_offset, 0]], 'CONSTANT')

        return clip + echo


@aug_register()
class ShiftPitch(Temporal):
    """
    Shift the pitch of the contained sound(s) up or down.

    :param val_range: A 2-element list/tuple. The factor by which the pitch will
        be shifted will be chosen randomly from the range val_range[0] to
        val_range[1]. Set the range around 1.0. If the chosen value is above
        1.0, pitch will be shifted upwards. If the chosen value is below 1.0,
        pitch will be shifted downwards. If the chosen value equals 1.0, there
        will be no change.
    """

    def __init__(self, val_range):

        assert 0 < val_range[0] < val_range[1]

        self._val_range_args = val_range if len(val_range) > 2 else \
            (val_range[0], val_range[1], tf.float32)

        super(ShiftPitch, self).__init__()

    def build_graph(self, clip, t_axis=-1):

        val = tf.random.uniform([], *self._val_range_args)

        orig_len = clip.shape[t_axis]

        diff = tf.cast(val * orig_len, tf.int32) - orig_len

        return tf.cond(
            diff == 0,
            lambda: clip,
            lambda: tf.cond(
                diff < 0,
                lambda: Temporal._upsample_and_crop(clip, orig_len, -diff),
                lambda: Temporal._pad_and_downsample(clip, orig_len, diff)
            ))


@aug_register()
class AlterDistance(SpectroTemporal):
    """
    Mimic the effect of increasing/reducing the distance between a source and
    receiver by attenuating/amplifying higher frequencies while keeping lower
    frequencies relatively unchanged.

    :param val_range: A 2-element list/tuple. The attenuation/amplification
        factor will be randomly chosen from the range val_range[0] dB to
        val_range[1] dB. A negative value chosen effects attenuation, while a
        positive value chosen effects amplification.
    """

    def __init__(self, val_range):

        assert val_range[0] < val_range[1]

        self._val_range_args = val_range if len(val_range) > 2 else \
            (val_range[0], val_range[1], tf.float32)

        super(AlterDistance, self).__init__()

    def build_graph(self, spec, f_axis=0, t_axis=1):

        val = tf.random.uniform([], *self._val_range_args)

        tf_zero = tf.constant(0.0, dtype=spec.dtype, name='0.0')
        amp_factor = tf.linspace(tf_zero, val, spec.shape[f_axis])

        return spec + tf.expand_dims(amp_factor, axis=t_axis)


@aug_register()
class SmearFrequency(SpectroTemporal):
    """
    Smear the spectrogram along the frequency axis. Can have the effect of
    shifting the pitch of the contained sounds.

    :param val_range: A 2-element integer list/tuple. The amount to smear is
        derived from a value chosen in the integer range val_range[0] to
        val_range[1]. Specify the range to reflect the number of frequency bins
        that will be involved in the smearing operation. If a positive value is
        chosen, the smearing occurs upwards. If a negative value is chosen,
        the smearing occurs downwards.
    """

    def __init__(self, val_range):

        assert val_range[0] <= val_range[1]

        self._val_range_args = val_range if len(val_range) > 2 else \
            (val_range[0], val_range[1], tf.int32)

        super(SmearFrequency, self).__init__()

    def build_graph(self, spec, f_axis=0, t_axis=1):

        val = tf.random.uniform([], *self._val_range_args)

        return tf.cond(
            tf.equal(val, 0),
            lambda: spec,
            lambda: SpectroTemporal._upsample_and_crop(
                spec,
                f_incr=tf.abs(val), crop_out_higher_f=tf.greater(val, 0),
                f_axis=f_axis, t_axis=t_axis))


@aug_register()
class SmearTime(SpectroTemporal):
    """
    Smear the spectrogram along the time axis. Can have the effect of
    elongating the duration of the contained sounds.

    :param val_range: A 2-element integer list/tuple. The amount to smear is
        derived from a value chosen in the integer range val_range[0] to
        val_range[1]. Specify the range to reflect the number of time windows
        that will be involved in the smearing operation. If a positive value is
        chosen, the smearing occurs forwards. If a negative value is chosen, the
        smearing occurs backwards.
    """

    def __init__(self, val_range):

        assert val_range[0] <= val_range[1]

        self._val_range_args = val_range if len(val_range) > 2 else \
            (val_range[0], val_range[1], tf.int32)

        super(SmearTime, self).__init__()

    def build_graph(self, spec, f_axis=0, t_axis=1):

        val = tf.random.uniform([], *self._val_range_args)

        return tf.cond(
            tf.equal(val, 0),
            lambda: spec,
            lambda: SpectroTemporal._upsample_and_crop(
                spec,
                t_incr=tf.abs(val), crop_out_later_t=tf.greater(val, 0),
                f_axis=f_axis, t_axis=t_axis))


@aug_register()
class SquishFrequency(SpectroTemporal):
    """
    Squish the spectrogram along the frequency axis. Can have the effect of
    shifting the pitch of the contained sounds.

    :param val_range: A 2-element integer list/tuple. The amount to squish is
        derived from a value chosen in the integer range val_range[0] to
        val_range[1]. Specify the range to reflect the number of frequency bins
        that will be involved in the squishing operation. If a positive value is
        chosen, the squishing occurs upwards. If a negative value is chosen, the
        squishing occurs downwards.
    """

    def __init__(self, val_range):

        assert val_range[0] <= val_range[1]

        self._val_range_args = val_range if len(val_range) > 2 else \
            (val_range[0], val_range[1], tf.int32)

        super(SquishFrequency, self).__init__()

    def build_graph(self, spec, f_axis=0, t_axis=1):

        val = tf.random.uniform([], *self._val_range_args)

        return tf.cond(
            tf.equal(val, 0),
            lambda: spec,
            lambda: SpectroTemporal._pad_and_downsample(
                spec,
                f_decr=tf.abs(val), pad_at_higher_f=tf.less(val, 0),
                f_axis=f_axis, t_axis=t_axis))


@aug_register()
class SquishTime(SpectroTemporal):
    """
    Squish the spectrogram along the time axis. Can have the effect of
    compressing the duration of the contained sounds.

    :param val_range: A 2-element integer list/tuple. The amount to squish is
        derived from a value chosen in the integer range val_range[0] to
        val_range[1]. Specify the range to reflect the number of time windows
        that will be involved in the squishing operation. If a positive value is
        chosen, the squishing occurs forwards. If a negative value is chosen,
        the squishing occurs backwards.
    """

    def __init__(self, val_range):

        assert val_range[0] <= val_range[1]

        self._val_range_args = val_range if len(val_range) > 2 else \
            (val_range[0], val_range[1], tf.int32)

        super(SquishTime, self).__init__()

    def build_graph(self, spec, f_axis=0, t_axis=1):

        val = tf.random.uniform([], *self._val_range_args)

        return tf.cond(
            tf.equal(val, 0),
            lambda: spec,
            lambda: SpectroTemporal._pad_and_downsample(
                spec,
                t_decr=tf.abs(val), pad_at_later_t=tf.less(val, 0),
                f_axis=f_axis, t_axis=t_axis))


__all__ = ['Temporal', 'SpectroTemporal']
