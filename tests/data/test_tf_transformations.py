import os
import numpy as np
import tensorflow as tf
from matplotlib import colormaps
import pytest
from koogu.data.tf_transformations import \
    Audio2Spectral, Spec2Img, NormalizeAudio


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
