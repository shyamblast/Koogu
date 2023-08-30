import numpy as np
from koogu.data.augmentations import *
from koogu.data.raw import Settings, Convert
from tests.data import save_display_to_disk
import pytest


def aug_output_comparisons(aug_output, orig_shape):

    # Check dtype
    assert aug_output.dtype == np.float32

    # Check if shape remains intact
    assert aug_output.shape == orig_shape


def test_Temporal_AddGaussianNoise(narw_json_clips_and_settings, outputroot):

    data, fs, spec_settings, expected_spec_shape = narw_json_clips_and_settings

    with_noise = apply_aug(data, Temporal.AddGaussianNoise((-6, -5.9999)))

    aug_output_comparisons(with_noise, data.shape)

    spec_settings_c = Settings.Spectral(fs, **spec_settings)
    save_display_to_disk(
        {
            'Original': Convert.audio2spectral(data, fs, spec_settings_c),
            'With noise': Convert.audio2spectral(with_noise,
                                                fs, spec_settings_c)
        },
        outputroot, 'test_augmentations', 'Temporal_AddGaussianNoise')


def test_Temporal_AddEcho(narw_json_clips_and_settings, outputroot):

    data, fs, spec_settings, expected_spec_shape = narw_json_clips_and_settings

    early_echo = apply_aug(data,
                           Temporal.AddEcho((0.005, 0.01), fs, (-3.1, -3)))
    delayed_echo = apply_aug(data,
                             Temporal.AddEcho((0.295, 0.3), fs, (-3.1, -3)))

    aug_output_comparisons(early_echo, data.shape)
    aug_output_comparisons(delayed_echo, data.shape)

    spec_settings_c = Settings.Spectral(fs, **spec_settings)
    save_display_to_disk(
        {
            'Original': Convert.audio2spectral(data, fs, spec_settings_c),
            'Early echo': Convert.audio2spectral(early_echo,
                                                 fs, spec_settings_c),
            'Delayed echo': Convert.audio2spectral(delayed_echo,
                                                   fs, spec_settings_c)
        },
        outputroot, 'test_augmentations', 'Temporal_AddEcho')


def test_Temporal_RampVolume(narw_json_clips_and_settings, outputroot):

    data, fs, spec_settings, expected_spec_shape = narw_json_clips_and_settings

    ramp_down = apply_aug(data, Temporal.RampVolume((-18, -17.9999)))
    ramp_up = apply_aug(data, Temporal.RampVolume((17.9999, 18)))

    aug_output_comparisons(ramp_down, data.shape)
    aug_output_comparisons(ramp_up, data.shape)

    spec_settings_c = Settings.Spectral(fs, **spec_settings)
    save_display_to_disk(
        {
            'Original': Convert.audio2spectral(data, fs, spec_settings_c),
            'Ramp down': Convert.audio2spectral(ramp_down,
                                                fs, spec_settings_c),
            'Ramp up': Convert.audio2spectral(ramp_up,
                                              fs, spec_settings_c)
        },
        outputroot, 'test_augmentations', 'Temporal_RampVolume')


def test_Temporal_ShiftPitch(narw_json_clips_and_settings, outputroot):

    data, fs, spec_settings, expected_spec_shape = narw_json_clips_and_settings

    shift_down = apply_aug(data, Temporal.ShiftPitch((0.75, 0.95)))
    shift_up = apply_aug(data, Temporal.ShiftPitch((1.05, 1.5)))

    aug_output_comparisons(shift_down, data.shape)
    aug_output_comparisons(shift_up, data.shape)

    spec_settings_c = Settings.Spectral(fs, **spec_settings)
    save_display_to_disk(
        {
            'Original': Convert.audio2spectral(data, fs, spec_settings_c),
            'Shifted down': Convert.audio2spectral(shift_down,
                                                   fs, spec_settings_c),
            'Shifted up': Convert.audio2spectral(shift_up,
                                                 fs, spec_settings_c)
        },
        outputroot, 'test_augmentations', 'Temporal_ShiftPitch')


def test_SpectroTemporal_AlterDistance(narw_json_clips_and_settings,
                                       outputroot):

    data, fs, spec_settings, expected_spec_shape = narw_json_clips_and_settings

    orig_clips = Convert.pcm2float(np.stack([data[0, ...], data[1, ...]]))
    spec_settings_c = Settings.Spectral(fs, **spec_settings)
    orig_specs = Convert.audio2spectral(orig_clips, fs, spec_settings_c)

    farther = apply_aug(orig_specs, SpectroTemporal.AlterDistance((-12, -9)))
    closer = apply_aug(orig_specs, SpectroTemporal.AlterDistance((3, 6)))

    aug_output_comparisons(farther, orig_specs.shape)
    aug_output_comparisons(closer, orig_specs.shape)

    save_display_to_disk(
        {
            'Original': orig_specs,
            'Farther': farther,
            'Closer': closer
        },
        outputroot, 'test_augmentations', 'SpectroTemporal_AlterDistance')


def test_SpectroTemporal_SmearFrequency(narw_json_clips_and_settings,
                                        outputroot):

    data, fs, spec_settings, expected_spec_shape = narw_json_clips_and_settings

    orig_clips = Convert.pcm2float(np.stack([data[0, ...], data[1, ...]]))
    spec_settings_c = Settings.Spectral(fs, **spec_settings)
    orig_specs = Convert.audio2spectral(orig_clips, fs, spec_settings_c)

    downward = apply_aug(orig_specs, SpectroTemporal.SmearFrequency((-5, -4)))
    upward = apply_aug(orig_specs, SpectroTemporal.SmearFrequency((9, 10)))

    aug_output_comparisons(downward, orig_specs.shape)
    aug_output_comparisons(upward, orig_specs.shape)

    save_display_to_disk(
        {
            'Original': orig_specs,
            'Downward': downward,
            'Upward': upward
        },
        outputroot, 'test_augmentations', 'SpectroTemporal_SmearFrequency')


def test_SpectroTemporal_SmearTime(narw_json_clips_and_settings,
                                   outputroot):

    data, fs, spec_settings, expected_spec_shape = narw_json_clips_and_settings

    orig_clips = Convert.pcm2float(np.stack([data[0, ...], data[1, ...]]))
    spec_settings_c = Settings.Spectral(fs, **spec_settings)
    orig_specs = Convert.audio2spectral(orig_clips, fs, spec_settings_c)

    backward = apply_aug(orig_specs, SpectroTemporal.SmearTime((-10, -9)))
    forward = apply_aug(orig_specs, SpectroTemporal.SmearTime((9, 10)))

    aug_output_comparisons(backward, orig_specs.shape)
    aug_output_comparisons(forward, orig_specs.shape)

    save_display_to_disk(
        {
            'Original': orig_specs,
            'Backward': backward,
            'Forward': forward
        },
        outputroot, 'test_augmentations', 'SpectroTemporal_SmearTime')


def test_SpectroTemporal_SquishFrequency(narw_json_clips_and_settings,
                                         outputroot):

    data, fs, spec_settings, expected_spec_shape = narw_json_clips_and_settings

    orig_clips = Convert.pcm2float(np.stack([data[0, ...], data[1, ...]]))
    spec_settings_c = Settings.Spectral(fs, **spec_settings)
    orig_specs = Convert.audio2spectral(orig_clips, fs, spec_settings_c)

    downward = apply_aug(orig_specs, SpectroTemporal.SquishFrequency((-5, -2)))
    upward = apply_aug(orig_specs, SpectroTemporal.SquishFrequency((2, 6)))

    aug_output_comparisons(downward, orig_specs.shape)
    aug_output_comparisons(upward, orig_specs.shape)

    save_display_to_disk(
        {
            'Original': orig_specs,
            'Downward': downward,
            'Upward': upward
        },
        outputroot, 'test_augmentations', 'SpectroTemporal_SquishFrequency')


def test_SpectroTemporal_SquishTime(narw_json_clips_and_settings,
                                    outputroot):

    data, fs, spec_settings, expected_spec_shape = narw_json_clips_and_settings

    orig_clips = Convert.pcm2float(np.stack([data[0, ...], data[1, ...]]))
    spec_settings_c = Settings.Spectral(fs, **spec_settings)
    orig_specs = Convert.audio2spectral(orig_clips, fs, spec_settings_c)

    backward = apply_aug(orig_specs, SpectroTemporal.SquishTime((-10, -9)))
    forward = apply_aug(orig_specs, SpectroTemporal.SquishTime((9, 10)))

    aug_output_comparisons(backward, orig_specs.shape)
    aug_output_comparisons(forward, orig_specs.shape)

    save_display_to_disk(
        {
            'Original': orig_specs,
            'Backward': backward,
            'Forward': forward
        },
        outputroot, 'test_augmentations', 'SpectroTemporal_SquishTime')


def apply_aug(clips_or_specs, aug):
    retval = np.stack([aug(c_or_s).numpy() for c_or_s in clips_or_specs])
    return retval

