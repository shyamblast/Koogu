import os
import numpy as np
import tensorflow as tf
from matplotlib import colormaps
import pytest
from koogu.data.raw import Settings, Convert
from koogu.data.tf_transformations import \
    Audio2Spectral, Spec2Img, NormalizeAudio, GaussianBlur
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
