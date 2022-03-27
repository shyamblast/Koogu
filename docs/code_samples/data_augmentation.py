import tensorflow as tf
from koogu.data.augmentations import Temporal, SpectroTemporal


class MySpectralDataFeeder(koogu.data.feeder.SpectralDataFeeder):

    def pre_transform(self, clip, label, is_training, **kwargs):
        """
        Applying augmentations to waveform.
        """

        output = clip

        # Added noise will have an amplitude that is -30 dB to -18 dB below
        # the peak amplitude of the input.
        gauss_noise = Temporal.AddGaussianNoise((-30, -18))

        # Add Gaussian noise to 25% of inputs.
        output = tf.cond(tf.random.uniform([], 0, 1) <= 1 / 4,
                         lambda: gauss_noise(output),
                         lambda: output)

        # The volume of the input will be linearly lowered/increased over its
        # duration, by a factor ≤ 3 dB.
        vol_ramp = Temporal.RampVolume((-3, 3))

        # Alter volume for 10% of the inputs.
        output = tf.cond(tf.random.uniform([], 0, 1) <= 1 / 10,
                         lambda: vol_ramp(output),
                         lambda: output)

        return output, label

    def post_transform(self, spec, label, is_training, **kwargs):
        """
        Applying augmentations to power spectral density spectrogram.
        """

        output = spec

        # Smear energies along the time-axis while retaining the frequency
        # content intact.
        smear_time = SpectroTemporal.SmearTime((-2, 2))

        # Apply to one in three inputs.
        output = tf.cond(tf.random.uniform([], 0, 1) <= 1 / 3,
                         lambda: smear_time(output),
                         lambda: output)

        return output, label
# [end-first-example]


class MySpectralDataFeeder2(koogu.data.feeder.SpectralDataFeeder):

    # [start-second-example-snippet]
    def pre_transform(self, clip, label, is_training, **kwargs):
        """
        Applying augmentations to waveform.
        """

        # List of time-domain augmentations
        augmentations = [
            Temporal.AddGaussianNoise((-30, -18)),
            Temporal.RampVolume((-3, 3))
        ]

        # At what rates should each be applied (same ordering as above)
        probabilities = [
            0.25,       # apply to 1 in 4 clips
            0.10        # apply to 1 in 10 clips
        ]

        output = Temporal.apply_chain(clip, augmentations, probabilities)

        return output, label

    def post_transform(self, spec, label, is_training, **kwargs):
        """
        Applying augmentations to power spectral density spectrogram.
        """

        # List of spectrogram augmentations
        augmentations = [
            SpectroTemporal.SmearTime((-2, 2)),
            SpectroTemporal.SquishFrequency((-1, 1))
        ]

        # At what rates should each be applied (same ordering as above)
        probabilities = [
            0.33,       # apply to 1 in 3 input spectrograms
            0.20        # apply to 1 in 5 input spectrograms
        ]

        output = SpectrroTemporal.apply_chain(spec, augmentations, probabilities)

        return output, label
