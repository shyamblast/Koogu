import os
import librosa
import numpy as np
from timeit import default_timer as timer
import pytest
from koogu.data.raw import Audio


@pytest.mark.parametrize('resamp_fs, channels, offset, duration', [
    (None, None, None, None),
    (None, 0, None, None),
    (None, [0], None, None),
    (8000, None, None, None),
    (407, [0], None, None),
    (None, 0, None, 0.25),
    (None, None, 2.75, 15.987),
    (None, [0], 105.23, 20.19),
    (None, None, 200, None),
    (None, [0], 300.21, 9999999.19),
    (None, [0], 30, -9999.19),
    (None, [0], 30, 0),
    (6000, None, 34.973, None),
    (513, [0], None, 25)
])
def test_soundfile_load(dataroot, resamp_fs, channels, offset, duration):
    audio_file = os.path.join(
        dataroot, 'narw_dclde', 'train_audio', 'NOPP6_EST_20090328',
        'NOPP6_EST_20090328_023000.flac')

    kg_data, lr_data, _, _ = load_via_underlying_libraries(
        audio_file, resamp_fs, channels, offset, duration)

    compare(kg_data, lr_data)


@pytest.mark.parametrize('resamp_fs, channels, offset, duration', [
    (None, None, None, None),
    (None, 0, None, None),
    (None, 1, None, None),
    (None, [0, 1], None, None),
    (8009, 1, None, None),
    (11033, [0], None, None),
    (None, 1, None, 1024 / 44100),
    (None, None, 2.75, 15.987),
    (None, 0, 5.23, 20.19),
    (None, 1, 7.327, 0.329),
    (None, 1, 60, None),
    (None, 1, 60, 9999999),
    (None, [1], 30, -9999),
    (None, None, 30, 0),
    (5016, 0, 23.217, None),
    (10037, 1, None, 23.217)
])
def test_audioread_load(dataroot, resamp_fs, channels, offset, duration):
    audio_file = os.path.join(dataroot, 'Ayyo - Avial.mp3')

    kg_data, lr_data, _, _ = load_via_underlying_libraries(
        audio_file, resamp_fs, channels, offset, duration)

    compare(kg_data, lr_data)


def load_via_underlying_libraries(audio_file,
                                  resamp_fs, channels, offset, duration):

    # Load using Koogu's functionality
    t_start = timer()
    kg_data, _ = Audio.load(audio_file,
                            desired_fs=resamp_fs,
                            channels=channels,
                            offset=offset,
                            duration=duration)
    kg_time = (timer() - t_start)

    # Load using librosa
    # librosa doesn't support extracting specific channels
    t_start = timer()
    lr_data, _ = librosa.load(audio_file,
                              sr=resamp_fs,
                              mono=False,
                              offset=offset,
                              duration=duration)
    lr_time = (timer() - t_start)
    if lr_data.ndim < 2:
        # librosa returns 1d array when only single channel is present
        lr_data = np.expand_dims(lr_data, axis=0)
    if channels is not None:
        lr_data = lr_data[channels if hasattr(channels, '__len__')
                          else [channels], ...]

    return kg_data, lr_data, kg_time, lr_time


def compare(kg_data, lr_data):

    assert kg_data.dtype == lr_data.dtype, \
        f'Data type mismatch: {kg_data.dtype} != {lr_data.dtype}'

    assert kg_data.shape[-1] == lr_data.shape[-1], \
        f'Num samples mismatch: {kg_data.shape[-1]} != {lr_data.shape[-1]}'

    mismatch_mask = (kg_data != lr_data)
    assert not np.any(mismatch_mask), \
        f'{mismatch_mask.sum()} (of {mismatch_mask.size}) samples mismatched'

