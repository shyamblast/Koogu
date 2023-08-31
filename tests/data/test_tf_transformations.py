import os
import numpy as np
import tensorflow as tf
from matplotlib import colormaps
import pytest
from koogu.data.raw import Settings, Convert
from koogu.data.tf_transformations import \
    Audio2Spectral, Spec2Img, NormalizeAudio, GaussianBlur, LoG
from tests.data import save_display_to_disk


def test_Audio2Spectral(dataroot, narw_json_clips_and_settings):

    data, fs, spec_settings, expected_shape = narw_json_clips_and_settings

    spec = Audio2Spectral(fs, spec_settings)(data)

    assert spec.dtype == tf.float32
    assert spec.shape == tf.TensorShape((2,) + expected_shape)

    spec = spec.numpy()
    prev_spec = np.load(os.path.join(dataroot,
                                     'tf_transformations_specimen',
                                     'dclde_json_spec.npz')
                        )['spec']

    assert np.all(np.isclose(spec, prev_spec))


@pytest.mark.parametrize('colorname, new_shape', [
    ('gray_r', None),
    ('gray_r', None),
    ('jet', (80, 200)),
    ('jet', (80, 200)),
])
def test_Spec2Img(narw_json_clips_and_settings,
                  colorname, new_shape):

    data, fs, spec_settings, expected_shape = narw_json_clips_and_settings

    cmap = colormaps[colorname](range(256))[:, :3]

    # Generate spectrogram, then convert to image
    output = Audio2Spectral(fs, spec_settings)(data)
    output = Spec2Img(cmap, img_size=new_shape)(output)

    assert output.dtype == tf.float32
    if new_shape is None:
        assert output.shape == tf.TensorShape((2,) + expected_shape + (3,))
    else:
        assert output.shape == tf.TensorShape((2,) + new_shape + (3,))

    output = output.numpy()

    assert output.min() == 0.0 and output.max() == 1.0

    # If grayscale, check if pixels have same value in all channels
    if colorname == 'gray_r':
        assert np.all(np.logical_and(
            output[0, :, :, 0] == output[0, :, :, 1],
            output[0, :, :, 0] == output[0, :, :, 2]
        )), 'Possibly non grayscale values generated (clip 1)'
        assert np.all(np.logical_and(
            output[1, :, :, 0] == output[1, :, :, 1],
            output[1, :, :, 0] == output[1, :, :, 2]
        )), 'Possibly non grayscale values generated (clip 2)'


@pytest.mark.parametrize('offset, scale, expected_res', [
    (0.6, 3, ((-1.0, 0.99085236), (-0.71750516, 1.0))),
    (-0.6, 3, ((-1.0, 0.9908523), (-0.71750516, 1.0))),
    (0.1, -5, ((-0.99085236, 1.0), (-1.0, 0.71750516))),
    (-0.2, -5, ((-0.9908525, 1.0), (-1.0, 0.71750516))),
])
def test_NormalizeAudio(narw_json_clips_and_settings,
                        offset, scale, expected_res):

    data = narw_json_clips_and_settings[0]

    output = NormalizeAudio()((data + offset) * scale)

    output = output.numpy()

    val_range = (output[0, :].min(), output[0, :].max())
    assert np.all(np.isclose(val_range, expected_res[0], rtol=1e-06)), \
        f'Clip1: {val_range}'
    val_range = (output[1, :].min(), output[1, :].max())
    assert np.all(np.isclose(val_range, expected_res[1], rtol=1e-06)), \
        f'Clip2: {val_range}'


@pytest.mark.parametrize('apply_2d, tf_spec', [
    (False, True),
    (True, True),
    (False, False),
    (True, False),
])
def test_GaussianBlur(narw_json_clips_and_settings, outputroot,
                      apply_2d, tf_spec):
    """

    Args:
        narw_json_clips_and_settings:
        outputroot:
        apply_2d:
        tf_spec: If True, test TF Audio2Spectral, else test scipy fft
    """

    data, fs, spec_settings, expected_spec_shape = narw_json_clips_and_settings
    sigmas = [1.55, 2.55, 5]

    orig_clips = Convert.pcm2float(np.stack([data[0, ...], data[1, ...]]))

    outputs = dict()
    if tf_spec:
        orig_specs = Audio2Spectral(fs, spec_settings)(orig_clips)
        outputs['Original'] = orig_specs.numpy()
        orig_shape = orig_specs.shape
        orig_specs = tf.expand_dims(orig_specs, -1)
    else:
        spec_settings_c = Settings.Spectral(fs, **spec_settings)
        orig_specs = Convert.audio2spectral(orig_clips, fs, spec_settings_c)
        outputs['Original'] = orig_specs
        orig_shape = orig_specs.shape
        orig_specs = np.expand_dims(orig_specs, -1)

    for s in sigmas:
        o = GaussianBlur(s, apply_2d)(orig_specs).numpy()[:, :, :, 0]

        assert np.all(orig_shape == o.shape), \
            f's={s}: {orig_shape} != {o.shape}'

        outputs[f's={s}'] = o

    save_display_to_disk(
        outputs, outputroot, 'test_tf_transformations',
        'GaussianBlur_' + ('xy' if apply_2d else 'y') +
        ('_tf' if tf_spec else ''))


@pytest.mark.parametrize('add_offsets, with_conv, retain_LoG, tf_spec', [
    (True, False, False, False),    # 1. Output should be same as 2
    (False, False, False, False),   # 2
    (True, False, False, True),     # 3. Output should be same as 4 & 1
    (False, False, False, True),    # 4
    (False, False, True, True),     # 5. Output should be same as 4
    (False, True, False, False),    # 6. Only conv outputs
    (False, True, False, True),     # 7. Only conv outputs
    (False, True, True, False),     # 8. Log + conv outputs
    (False, True, True, True),      # 9. Log + conv outputs
])
def test_LoG(narw_json_clips_and_settings, outputroot,
             add_offsets, with_conv, retain_LoG, tf_spec):
    """

    Args:
        narw_json_clips_and_settings:
        outputroot:
        add_offsets:
        tf_spec: If True, test TF Audio2Spectral, else test scipy fft
    """

    data, fs, spec_settings, expected_spec_shape = narw_json_clips_and_settings
    scales_sigmas = (4, 8, 16)
    conv_filters = 3

    if with_conv:
        tf.random.set_seed(37)
        num_output_chs = (conv_filters * len(scales_sigmas)) + (
            len(scales_sigmas) if retain_LoG else 0)
    else:
        num_output_chs = len(scales_sigmas)

    orig_clips = Convert.pcm2float(np.stack([data[0, ...], data[1, ...]]))

    outputs = dict()
    if tf_spec:
        orig_specs = Audio2Spectral(fs, spec_settings)(orig_clips)
        outputs['Original'] = orig_specs.numpy()
        orig_shape = orig_specs.shape
        orig_specs = tf.expand_dims(orig_specs, -1)
    else:
        spec_settings_c = Settings.Spectral(fs, **spec_settings)
        orig_specs = Convert.audio2spectral(orig_clips, fs, spec_settings_c)
        outputs['Original'] = orig_specs
        orig_shape = orig_specs.shape
        orig_specs = np.expand_dims(orig_specs, -1)

    l_outputs = LoG(scales_sigmas,
                    add_offsets=add_offsets,
                    conv_filters=conv_filters if with_conv else None,
                    retain_LoG=retain_LoG
                    )(orig_specs).numpy()

    assert num_output_chs == l_outputs.shape[-1], 'Num channels mismatch'
    assert np.all(orig_shape == l_outputs.shape[:-1]), \
        f'{orig_shape} != {l_outputs.shape[:-1]}'

    s_idx_offset = 0
    if (with_conv and retain_LoG) or (not with_conv):
        for s_idx, s in enumerate(scales_sigmas):
            outputs[f's={s}'] = l_outputs[:, :, :, s_idx]
        s_idx_offset = len(scales_sigmas)
    if with_conv:
        for s_idx, s in enumerate(scales_sigmas):
            for c_idx in range(conv_filters):
                outputs[f's={s}, c={c_idx+1}'] = \
                    l_outputs[:, :, :, s_idx_offset + s_idx + c_idx]

    save_display_to_disk(
        outputs, outputroot, 'test_tf_transformations',
        'LoG' + (
            '_offsets' if add_offsets else '') + (
            '_c' if with_conv else '') + (
            '_r' if retain_LoG else '') + (
            '_tf' if tf_spec else ''),
        normalize_levels=False
    )
