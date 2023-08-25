import os
import numpy as np
import shutil
import pytest
from koogu import prepare

from tests.data import narw_dclde_selmap_train


@pytest.mark.parametrize(
    'negative_class_label, ignore_zero_annot_files, attempt_salvage, result', [
        (None, None, None, {'NARW': 12862}),
        (None, None, True, {'NARW': 13074}),
        (None, True, None, {'NARW': 12862}),
        (None, True, True, {'NARW': 13074}),
        ('Other', None, None, {'NARW': 12862, 'Other': 808084}),
        ('Other', None, True, {'NARW': 13074, 'Other': 808084}),
        ('Other', True, None, {'NARW': 12862, 'Other': 729474}),
        ('Other', True, True, {'NARW': 13074, 'Other': 729474}),
])
def test_mono_channel(
        dataroot, outputroot,
        negative_class_label, ignore_zero_annot_files, attempt_salvage,
        result):

    audio_settings = {
        'clip_length': 2.0,
        'clip_advance': 0.4,
        'desired_fs': 1000,
        'filterspec': (9, [42.9, 394.53], 'bandpass'),
        'normalize_clips': False
    }

    outputs_dir = os.path.join(outputroot, 'narw_preprocess')

    class_clip_counts = prepare.from_selection_table_map(
        audio_settings, narw_dclde_selmap_train,
        audio_root=os.path.join(dataroot, 'narw_dclde', 'train_audio'),
        seltab_root=os.path.join(dataroot, 'narw_dclde', 'train_annotations'),
        output_root=outputs_dir,
        negative_class_label=negative_class_label,
        ignore_zero_annot_files=ignore_zero_annot_files,
        attempt_salvage=attempt_salvage
    )

    sample_content_info = query_contents(
        os.path.join(outputs_dir, 'NOPP6_EST_20090328',
                     'NOPP6_EST_20090328_000000.flac.npz'))

    shutil.rmtree(outputs_dir)  # Delete, we don't need to keep them

    assert class_clip_counts == result

    num_clips = 37 + \
                (2065 if negative_class_label is not None else 0) + \
                (0 if ignore_zero_annot_files else 0) + \
                (1 if attempt_salvage else 0)
    num_classes = (2 if negative_class_label is not None else 1)
    assert sample_content_info == {
        'fs': (np.dtype('int64'), ()),
        'channels': (np.dtype('uint8'), (num_clips,)),
        'clips': (np.dtype('int16'), (num_clips, 2000)),
        'clip_offsets': (np.dtype('int64'), (num_clips,)),
        'labels': (np.dtype('float16'), (num_clips, num_classes))
    }


@pytest.mark.parametrize(
    'negative_class_label, ignore_zero_annot_files, attempt_salvage, result', [
        (None, None, None, {'BpBor.NWA.A': 76}),
        (None, None, True, {'BpBor.NWA.A': 96}),
        (None, True, None, {'BpBor.NWA.A': 76}),
        (None, True, True, {'BpBor.NWA.A': 96}),
        ('Other', None, None, {'BpBor.NWA.A': 76, 'Other': 17661}),
        ('Other', None, True, {'BpBor.NWA.A': 96, 'Other': 17661}),
        ('Other', True, None, {'BpBor.NWA.A': 76, 'Other': 17661}),
        ('Other', True, True, {'BpBor.NWA.A': 96, 'Other': 17661}),
])
def test_multi_channel(
        dataroot, outputroot,
        negative_class_label, ignore_zero_annot_files, attempt_salvage,
        result):
    audio_settings = {
        'clip_length': 2.0,
        'clip_advance': 1.0,
        'desired_fs': 1000
    }

    outputs_dir = os.path.join(outputroot, 'sei_preprocess')

    class_clip_counts = prepare.from_selection_table_map(
        audio_settings,
        audio_seltab_list=os.path.join(dataroot, 'multichannel_Sei',
                                       'seltab_map.csv'),
        audio_root=os.path.join(dataroot, 'multichannel_Sei', 'audio'),
        seltab_root=os.path.join(dataroot, 'multichannel_Sei', 'annotations'),
        output_root=outputs_dir,
        negative_class_label=negative_class_label,
        ignore_zero_annot_files=ignore_zero_annot_files,
        attempt_salvage=attempt_salvage
    )

    sample_content_info = query_contents(
        os.path.join(outputs_dir, '20200329',
                     '77766NYSDEC08_005K_M10_MARU_20200329_000000Z.aif.npz'))

    shutil.rmtree(outputs_dir)  # Delete, we don't need to keep them

    assert class_clip_counts == result

    num_clips = 29 + \
                (8819 if negative_class_label is not None else 0) + \
                (0 if ignore_zero_annot_files else 0) + \
                (17 if attempt_salvage else 0)
    num_classes = (2 if negative_class_label is not None else 1)
    assert sample_content_info == {
        'fs': (np.dtype('int64'), ()),
        'channels': (np.dtype('uint8'), (num_clips,)),
        'clips': (np.dtype('int16'), (num_clips, 2000)),
        'clip_offsets': (np.dtype('int64'), (num_clips,)),
        'labels': (np.dtype('float16'), (num_clips, num_classes))
    }


def query_contents(preprocessed_file):
    with np.load(preprocessed_file) as data:
        retval = {
            key: (data[key].dtype, data[key].shape)
            for key in ['fs', 'channels', 'clips', 'clip_offsets', 'labels']
        }

    return retval
