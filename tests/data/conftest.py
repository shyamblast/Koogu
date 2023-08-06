import os
import json
import numpy as np
import pytest


@pytest.fixture(scope="module")
def narw_json_clips_and_settings(dataroot):

    # The json file has 2 clips
    with open(os.path.join(dataroot, 'dclde_audio_samples.json'), 'r') as af:
        data = np.asarray(json.load(af)).astype(np.float32)
    fs = 1000
    spec_settings = {
        'win_len': 0.128,
        'win_overlap_prc': 0.75,
        'eps': 1e-10,
        'bandwidth_clip': [46, 391]
    }
    expected_shape = (45, 67)   # spectrogram height x width

    return data, fs, spec_settings, expected_shape
