import os
import numpy as np
from koogu import prepare, train, recognize, assessments
from koogu.data.feeder import SpectralDataFeeder
from koogu.model import architectures
from tests.data import narw_dclde_selmap_train, narw_dclde_selmap_test
import pytest

random_seed = 7529

# ----- Project settings ------------------------------
#
# Settings for handling raw audio
_audio_settings = {
    'clip_length': 2.0,
    'clip_advance': 0.5,
    'desired_fs': 1000,
    'filterspec': (9, [42.9, 394.53], 'bandpass')
}

# Settings for converting audio to a time-frequency (tf) representation
_spec_settings = {
    'win_len': 0.128,
    'win_overlap_prc': 0.75,
    'nfft_equals_win_len': True,
    'spec_type': 'spec_db',
    'eps': 1e-10,
    'bandwidth_clip': [46, 391],
}

# Settings to control the training process
_training_settings = {
    'batch_size': 64,
    'epochs': 50,
    'dropout_rate': 0.05,
    'learning_rate': 0.01,
    'lr_change_at_epochs': [10, 35],
    'lr_update_factors': [1, 1/10, 1/100]
}


def test_all_stages(dataroot, outputroot):

    outputs_dir = os.path.join(outputroot, 'narw_dclde_full_test')
    clips_dir = os.path.join(outputs_dir, 'clips')
    model_dir = os.path.join(outputs_dir, 'model')
    dets_dir = os.path.join(outputs_dir, 'detections')

    class_clip_counts = prepare.from_annotations(
        _audio_settings, narw_dclde_selmap_train,
        audio_root=os.path.join(dataroot, 'narw_dclde', 'train_audio'),
        annot_root=os.path.join(dataroot, 'narw_dclde', 'train_annotations'),
        output_root=clips_dir,
        negative_class_label='Other'
    )

    assert class_clip_counts == {'NARW': 10265, 'Other': 583615}

    # Instantiate feeder
    data_feeder = SpectralDataFeeder(
        data_dir=clips_dir,
        fs=_audio_settings['desired_fs'],
        spec_settings=_spec_settings,
        validation_split=0.15,
        max_clips_per_class=10000,
        random_state_seed=random_seed
    )

    assert data_feeder.data_shape == [45, 59]
    assert data_feeder.class_names == ['NARW', 'Other']
    assert np.all(data_feeder.training_samples_per_class == 8500)
    assert np.all(data_feeder.validation_samples_per_class == 1500)

    # Instantiate quasi-DenseNet model
    arch = architectures.DenseNet(
        layers_per_block=[2, 2, 2, 1],
        growth_rate=2,
        pool_strides=[(3, 3), (3, 3), (2, 3)],
        quasi_dense=True,
        preproc=[
            ('Conv2D', {'filters': 16})
        ],
        dense_layers=16
    )

    # Train
    history = train(
        data_feeder,
        model_dir,
        dict(audio_settings=_audio_settings,
             spec_settings=_spec_settings),
        arch,
        _training_settings,
        random_seed=random_seed
    )

    assert history['val_binary_accuracy'][-1] >= 0.975, 'Final accuracy is low'
    assert history['val_loss'][-1] <= 0.065, 'Final loss is high'

    # Perform testing
    os.makedirs(dets_dir, exist_ok=True)
    recognize(
        model_dir,
        os.path.join(dataroot, 'narw_dclde', 'test_audio'),
        raw_detections_dir=os.path.join(dets_dir, 'raw'),
        batch_size=128,
        recursive=True,
        clip_advance=0.25
    )

    thresholds = [0.5, 0.6, 0.75, 0.9, 0.99]

    # Assess raw performance
    per_class_pr_raw, _ = assessments.PrecisionRecall(
        narw_dclde_selmap_test,
        os.path.join(dets_dir, 'raw'),
        os.path.join(dataroot, 'narw_dclde', 'test_annotations'),
        reject_classes='Other',
        thresholds=thresholds
    ).assess()
    f1 = compute_f1_scores(per_class_pr_raw['NARW'])
    assert f1[4] > 0.77, \
        f'Th: {thresholds[4]}, ' + \
        f'Pr: {per_class_pr_raw["NARW"]["precision"][4]}, ' + \
        f'Rc: {per_class_pr_raw["NARW"]["recall"][4]}'
    assert all(np.asarray(f1[3:]) > 0.7), \
        f'Th: {thresholds[3:]}, ' + \
        f'Pr: {per_class_pr_raw["NARW"]["precision"][3:]}, ' + \
        f'Rc: {per_class_pr_raw["NARW"]["recall"][3:]}'
    assert all(np.asarray(f1[2:]) > 0.62), \
        f'Th: {thresholds[2:]}, ' + \
        f'Pr: {per_class_pr_raw["NARW"]["precision"][2:]}, ' + \
        f'Rc: {per_class_pr_raw["NARW"]["recall"][2:]}'
    assert all(np.asarray(f1[1:]) > 0.53), \
        f'Th: {thresholds[1:]}, ' + \
        f'Pr: {per_class_pr_raw["NARW"]["precision"][1:]}, ' + \
        f'Rc: {per_class_pr_raw["NARW"]["recall"][1:]}'
    assert all(np.asarray(f1) > 0.47), \
        f'Th: {thresholds}, ' + \
        f'Pr: {per_class_pr_raw["NARW"]["precision"]}, ' + \
        f'Rc: {per_class_pr_raw["NARW"]["recall"]}'
    print(per_class_pr_raw['NARW'])
    print(f1)

    # Assess post-processing performance
    squeeze_min_dur = 1.5
    min_gt_coverage = 0.8
    min_det_usage_dur = 0.75
    per_class_pr_pp, _ = assessments.PrecisionRecall(
        narw_dclde_selmap_test,
        os.path.join(dets_dir, 'raw'),
        os.path.join(dataroot, 'narw_dclde', 'test_annotations'),
        reject_classes='Other',
        thresholds=thresholds,
        post_process_detections=True,
        min_gt_coverage=min_gt_coverage,
        min_det_usage=min_det_usage_dur / _audio_settings['clip_length'],
        squeeze_min_dur=squeeze_min_dur,
    ).assess()
    f1 = compute_f1_scores(per_class_pr_raw['NARW'])
    assert f1[4] > 0.77, \
        f'Th: {thresholds[4]}, ' + \
        f'Pr: {per_class_pr_raw["NARW"]["precision"][4]}, ' + \
        f'Rc: {per_class_pr_raw["NARW"]["recall"][4]}'
    assert all(np.asarray(f1[3:]) > 0.7), \
        f'Th: {thresholds[3:]}, ' + \
        f'Pr: {per_class_pr_raw["NARW"]["precision"][3:]}, ' + \
        f'Rc: {per_class_pr_raw["NARW"]["recall"][3:]}'
    assert all(np.asarray(f1[2:]) > 0.62), \
        f'Th: {thresholds[2:]}, ' + \
        f'Pr: {per_class_pr_raw["NARW"]["precision"][2:]}, ' + \
        f'Rc: {per_class_pr_raw["NARW"]["recall"][2:]}'
    assert all(np.asarray(f1[1:]) > 0.53), \
        f'Th: {thresholds[1:]}, ' + \
        f'Pr: {per_class_pr_raw["NARW"]["precision"][1:]}, ' + \
        f'Rc: {per_class_pr_raw["NARW"]["recall"][1:]}'
    assert all(np.asarray(f1) > 0.47), \
        f'Th: {thresholds}, ' + \
        f'Pr: {per_class_pr_raw["NARW"]["precision"]}, ' + \
        f'Rc: {per_class_pr_raw["NARW"]["recall"]}'
    print(per_class_pr_raw['NARW'])
    print(f1)


def compute_f1_scores(pr_dict):
    return [((2 * p * r) / (p + r))
            for p, r in zip(pr_dict['precision'], pr_dict['recall'])]
