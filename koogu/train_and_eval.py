
import os
import sys
import tensorflow as tf
import logging
from tensorflow.python.tools import freeze_graph
from tensorflow.python.client import device_lib
import argparse

from koogu.model import get_model, TrainedModel
from koogu.data import feeder
from koogu.data.tf_transformations import Audio2Spectral, LoG, Linear2dB, spatial_split
from koogu.utils import instantiate_logging
from koogu.utils.terminal import ArgparseConverters
from koogu.utils.config import Config, ConfigError, datasection2dict, log_config


_program_name = 'train_and_eval'


def train_and_eval(data_dir, model_dir,
                   data_config,
                   model_config,
                   training_config,
                   **kwargs):

    # Check fields in model_config
    required_fields = ['arch', 'arch_params']
    if any([field not in model_config for field in required_fields]):
        print('Required fields missing in \'model_config\'', file=sys.stderr)
        return 1
    if 'preproc' not in model_config:
        model_config['preproc'] = []    # force to be an empty list
    if 'dense_layers' not in model_config:
        model_config['dense_layers'] = []   # force to be an empty list

    # Check fields in training_config
    required_fields = ['batch_size', 'epochs', 'epochs_between_evals',
                       'learning_rate', 'optimizer']
    if any([field not in training_config for field in required_fields]):
        print('Required fields missing in \'training_config\'',
              file=sys.stderr)
        return 1
    if 'weighted_loss' not in training_config:
        training_config['weighted_loss'] = False
    if 'l2_weight_decay' not in training_config:
        training_config['l2_weight_decay'] = None
    if 'lr_change_at_epochs' not in training_config or \
        'lr_update_factors' not in training_config:
        training_config['lr_change_at_epochs'] = None
    else:
        assert len(training_config['lr_change_at_epochs']) == \
               (len(training_config['lr_update_factors']) - 1)

    # Handle kwargs
    if 'data_format' not in kwargs:
        kwargs['data_format'] = 'channels_last'
    else:
        assert kwargs['data_format'] in ['channels_first', 'channels_last']
    if 'dropout_rate' in training_config:
        # Move this into kwargs. It's needed during model-building
        kwargs['dropout_rate'] = training_config.pop('dropout_rate')

    # Instantiate settings that need instantiation
    try:
        model_config['preproc'] = [
            preproc_op(**preproc_params, data_format=kwargs['data_format'])
            for (preproc_op, preproc_params) in model_config['preproc']]
    except Exception as exc:
        print('Error setting-up model pre-processing: {:s}'.format(repr(exc)),
              file=sys.stderr)
        return 2
    try:
        training_config['optimizer'] = training_config['optimizer'][0](
            learning_rate=training_config['learning_rate'],
            **training_config['optimizer'][1])
    except Exception as exc:
        print('Error setting-up optimizer: {:s}'.format(repr(exc)),
              file=sys.stderr)
        return 2

    # If data_dir parameter was already an instantialized DataFeeder object,
    # pass it on. Otherwise, create one.
    if not isinstance(data_dir, feeder.DataFeeder):
        num_gpu_devices = len(
            [x.name for x in device_lib.list_local_devices()
             if x.device_type == 'GPU'])
        data_dir = \
            SpectralDataFeeder(data_dir, data_config,
                               training_config['batch_size'],
                               cache=True,
                               num_prefetch_batches=max(1, num_gpu_devices))

    print('Dataset: {:d} classes, {:d} training & {:d} eval samples'.format(
        data_dir.num_classes, data_dir.training_samples,
        data_dir.validation_samples))

    # Invoke the underlying main function
    _main(data_dir, model_dir,
          data_config, model_config, training_config,
          **kwargs)


# def _validate_fcn_splitting(raw_data_shape, data_cfg, patch_size, patch_overlap):
#
#     patch_size = np.asarray(patch_size).astype(np.int)
#     patch_overlap = np.asarray(patch_overlap if patch_overlap is not None else [0, 0]).astype(np.int)
#
#     model_input_feat_shp = transform_audio_to_spectral(
#         data_cfg, tf.placeholder(tf.float32, raw_data_shape)).get_shape().as_list()
#     model_input_feat_shp = model_input_feat_shp[0:2]
#
#     # Check along frequency axis
#     remainder = spatial_split(int(model_input_feat_shp[0]), 1, patch_size[0], patch_overlap[0])
#     if remainder > 0:
#         logging.error('FCN patch splitting in frequency axis would leave {} remainders.'.format(remainder))
#         return None, None
#
#     # Check along time axis
#     remainder = spatial_split(int(model_input_feat_shp[1]), 2, patch_size[1], patch_overlap[1])
#     if remainder > 0:
#         logging.error('FCN patch splitting in time axis would leave {} remainders.'.format(remainder))
#         return None, None
#
#     return patch_size, patch_overlap


def _main(data_feeder, model_dir, data_cfg, model_cfg, training_cfg,
          show_progress=False,
          **kwargs):

    os.makedirs(model_dir, exist_ok=True)

    tf.keras.backend.clear_session()
    callbacks = []
    if 'random_seed' in kwargs:
        tf.random.set_seed(kwargs.pop('random_seed'))

    callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=model_dir, write_graph=True, write_images=True))

#    # If patch splitting is enabled, validate it and set estimator params accordingly
#    if model_cfg.fcn_patch_size is not None:
#        patch_size, patch_overlap = \
#            _validate_fcn_splitting(feature_shape, data_cfg, model_cfg.fcn_patch_size, model_cfg.fcn_patch_overlap)
#        if patch_size is not None:
#            # Update with corrected values
#            model_cfg.fcn_patch_size = patch_size
#            model_cfg.fcn_patch_overlap = patch_overlap
#        else:
#            print('MODEL:FCN splitting settings are invalid. See log file for details.', file=sys.stderr)
#            return

#    # Build estimator params
#    estimator_params = dict()
#    if model_cfg.fcn_patch_size is not None:
#        estimator_params['FCN_patch_size'] = model_cfg.fcn_patch_size
#        estimator_params['FCN_patch_overlap'] = model_cfg.fcn_patch_overlap

    # Set the learning rate
    if training_cfg['lr_change_at_epochs'] is not None:   # If variable learning rate is enabled
        # Set the piecewise learning rates
        def get_learning_rate(epoch):
            var_rates = [training_cfg['learning_rate'] * factor for factor in training_cfg['lr_update_factors']]
            for idx, boundary in enumerate(training_cfg['lr_change_at_epochs']):
                if epoch <= boundary:
                    tf.summary.scalar('learning rate', data=var_rates[idx], step=epoch)
                    return var_rates[idx]
            tf.summary.scalar('learning rate', data=var_rates[-1], step=epoch)
            return var_rates[-1]
    else:
        # Static learning rate
        def get_learning_rate(_): return training_cfg['learning_rate']
    callbacks.append(tf.keras.callbacks.LearningRateScheduler(get_learning_rate))

    model_input_shape = data_feeder.data_shape[0]
    data_transformation = None
    if data_cfg['spec_settings'] is not None and \
        data_cfg['spec_settings']['tf_rep_type'] is not None:
        data_transformation = \
            Audio2Spectral(data_cfg['audio_settings']['desired_fs'],
                           data_cfg['spec_settings'])
        model_input_shape = \
            data_transformation.compute_output_shape(
                [1, model_input_shape])[1:]
    #print(data_feeder.data_shape, model_input_shape)

    # Create a Classifier instance
    classifier = get_model(model_cfg,
                           input_shape=model_input_shape,
                           num_classes=data_feeder.num_classes,
                           **kwargs)

    # Add L2 regularization, if enabled
    if training_cfg['l2_weight_decay'] is not None and training_cfg['l2_weight_decay'] > 0.0:
        regularizer = tf.keras.regularizers.l2(training_cfg['l2_weight_decay'])
        for layer in classifier.layers:
            if hasattr(layer, 'kernel_regularizer'):
                setattr(layer, 'kernel_regularizer', regularizer)

    loss_fn = tf.keras.losses.BinaryCrossentropy() if model_cfg.get('multilabel', False) \
        else tf.keras.losses.CategoricalCrossentropy()

    classifier.compile(
        optimizer=training_cfg['optimizer'],
        loss=loss_fn,
        metrics=[tf.keras.metrics.CategoricalAccuracy()])
    classifier.summary()

    # Enable loss weighting (if enabled)
    class_weights = None
    if training_cfg['weighted_loss']:
        class_weights = {idx: weight for idx, weight in enumerate(
            data_feeder.training_samples / (data_feeder.num_classes * data_feeder.training_samples_per_class))}

    classifier.fit(
        x=data_feeder(True),
        validation_data=data_feeder(False),
        initial_epoch=0, epochs=training_cfg['epochs'],
        validation_freq=training_cfg['epochs_between_evals'],
        shuffle=False,
        verbose=1 + (0 if show_progress else 1),
        class_weight=class_weights,
#        steps_per_epoch=int(np.ceil(steps_per_epoch)),
#        validation_steps=val_steps,
        callbacks=callbacks)

    # Reset metrics before saving
    classifier.reset_metrics()
    classifier.save(os.path.join(model_dir, 'classifier.h5'))

    new_subdir = '1'
    TrainedModel.finalize_and_save(classifier,
                                   os.path.join(model_dir, new_subdir),
                                   data_feeder.data_shape,
                                   data_transformation,
                                   data_feeder.class_names,
                                   data_cfg['audio_settings'])


class SpectralDataFeeder(feeder.TFRecordFeeder):
    def __init__(self, data_dir, data_cfg, batch_size, **kwargs):

        super(SpectralDataFeeder, self).__init__(
            data_dir, batch_size, **kwargs)

        self._data_cfg = data_cfg

    def transform(self, clip, label, is_training, **kwargs):

        output = clip

        # Normalize the waveforms
        output = output / \
                 tf.reduce_max(tf.abs(output), axis=-1, keepdims=True)

        # Convert to spectrogram
        output = Audio2Spectral(
            self._data_cfg['audio_settings']['desired_fs'],
            self._data_cfg['spec_settings'])(output)

        return output, tf.one_hot(label, self.num_classes)


def _get_settings_from_config(args):
    """Load config settings from the config file and return values from different sections."""

    cfg = Config(args.cfg, ['DATA', 'MODEL', 'TRAINING'])

    data_settings = datasection2dict(cfg.DATA)

    # Handle overriding of training config values
    if args.batch_size is not None:
        cfg.TRAINING.batch_size = args.batch_size
    if args.num_epochs is not None:
        cfg.TRAINING.epochs = args.num_epochs
    if args.epochs_between_evals is not None:
        cfg.TRAINING.epochs_between_evals = args.epochs_between_evals
    if args.dropout_rate is not None:
        cfg.TRAINING.dropout_rate = args.dropout_rate / 100.    # Convert percent to fraction
    if args.learning_rate is not None:
        cfg.TRAINING.learning_rate = args.learning_rate

    # Some run-time stuff couldn't have been eval'ed in utils.Config while loading config file. They were loaded as
    # plain strings. eval() and process them now.

    cfg.TRAINING.optimizer = eval(cfg.TRAINING.optimizer)       # This param has tensorflow stuff
    assert isinstance(cfg.TRAINING.optimizer, tuple) and len(cfg.TRAINING.optimizer) == 2 and \
        isinstance(cfg.TRAINING.optimizer[1], dict), '\'optimizer\' definition is invalid'

#    # This param has data.feature_transformations and tensorflow stuff
#    if cfg.TRAINING.augmentations_time_domain is not None:
#        cfg.TRAINING.augmentations_time_domain = eval(cfg.TRAINING.augmentations_time_domain)
#        if cfg.TRAINING.augmentations_time_domain is not None:
#            if not isinstance(cfg.TRAINING.augmentations_time_domain, list):    # force to be a list
#                cfg.TRAINING.augmentations_time_domain = [cfg.TRAINING.augmentations_time_domain]
#            if len(cfg.TRAINING.augmentations_time_domain) == 0:
#                cfg.TRAINING.augmentations_time_domain = None
#
#    # This param has data.feature_transformations and tensorflow stuff
#    if cfg.TRAINING.augmentations_timefreq_domain is not None:
#        cfg.TRAINING.augmentations_timefreq_domain = eval(cfg.TRAINING.augmentations_timefreq_domain)
#        if cfg.TRAINING.augmentations_timefreq_domain is not None:
#            if not isinstance(cfg.TRAINING.augmentations_timefreq_domain, list):  # force to be a list
#                cfg.TRAINING.augmentations_timefreq_domain = [cfg.TRAINING.augmentations_timefreq_domain]
#            if len(cfg.TRAINING.augmentations_timefreq_domain) == 0:
#                cfg.TRAINING.augmentations_timefreq_domain = None
#
#    # This param has tensorflow stuff
#    if cfg.TRAINING.background_infusion_params is not None and args.additive_backgrounds is not None:
#        cfg.TRAINING.background_infusion_params = eval(cfg.TRAINING.background_infusion_params)
#        assert isinstance(cfg.TRAINING.background_infusion_params, tuple) and \
#            len(cfg.TRAINING.background_infusion_params) == 2 and \
#            isinstance(cfg.TRAINING.background_infusion_params[1], tuple) and \
#            len(cfg.TRAINING.background_infusion_params[1]) == 3

    return data_settings, _ModelCFG(cfg.MODEL, args.arch), cfg.TRAINING


if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog=_program_name, allow_abbrev=False,
                                     description='Train a TF model.')
    parser.add_argument('datadir', metavar='<DATA DIR>',
                        help='Path to the root directory containing training and validation data.')
    parser.add_argument('modeldir', metavar='<MODEL DIR>',
                        help='Path to destination directory into which model-specific contents will be written out.')
    parser.add_argument('cfg', metavar='<CONFIG FILE>',
                        help='Path to config file.')
    parser.add_argument('arch', metavar='<ARCHITECTURE>',
                        choices=['convnet', 'densenet', 'resnet'],
                        help='Model architecture.')
    arg_group_train = parser.add_argument_group('Training config override',
                                                'Overrides settings obtained from the config file.')
    arg_group_train.add_argument('--batch-size', dest='batch_size', type=ArgparseConverters.positive_integer,
                                 metavar='NUM',
                                 help='Size to batch the inputs into.')
    arg_group_train.add_argument('--epochs', dest='num_epochs', type=ArgparseConverters.positive_integer,
                                 metavar='NUM',
                                 help='Number of epochs to train for.')
    arg_group_train.add_argument('--epochs-between-evals', dest='epochs_between_evals',
                                 type=ArgparseConverters.positive_integer, metavar='NUM',
                                 help='How often evaluation is to be performed.')
    arg_group_train.add_argument('--dropout-rate', metavar='0-100', type=ArgparseConverters.valid_percent,
                                 dest='dropout_rate',
                                 help='Dropout probability (as a percent). Higher value = more regularization.')
    arg_group_train.add_argument('--learning-rate', dest='learning_rate', type=ArgparseConverters.positive_float,
                                 metavar='NUM',
                                 help='Static (or initial value of) learning rate.')
    arg_group_misc = parser.add_argument_group('Miscellaneous')
#    arg_group_misc.add_argument('--preproc', choices=['DD', 'dB', 'dBFS'],
#                                help='Include a pre-processing step for transforming features before they enter the ' +
#                                     'network. Choices\' names are case sensitive. \'DD\': Double-differential; if ' +
#                                     'choosing this preproc, (i) make sure dd_settings is defined in the config file,' +
#                                     ' and (ii) it is advised that any pre-conv filters be disabled. \'dB\': convert ' +
#                                     'linear spectral features to decibel scale. \'dBFS\': convert linear spectral ' +
#                                     'features to normalized (in the range [0.0, 1.0]) decibel scale.')
    arg_group_misc.add_argument('--seed', dest='random_state_seed', type=ArgparseConverters.positive_integer,
                                metavar='NUM',
                                help='Seed value (integer) for deterministic shuffling.')
    arg_group_misc.add_argument('--dim-order', dest='dim_order', choices=['NCHW', 'NHWC'],
                                help='Dimension ordering of data. \'NCHW\' (a.k.a channels first) is [batch, channels' +
                                     ', height, width] which is good when training on GPU using cuDNN. \'NHWC\' (a.k.' +
                                     'a channels last) is [batch, height, width, channels] which is good when ' +
                                     'training on CPU. If unspecified, appropriate choice will be made depending on ' +
                                     'GPU availability.')
#    arg_group_misc.add_argument('--track-mismatches', metavar='0-1', type=ArgparseConverters.float_0_to_1,
#                                dest='track_mismatches',
#                                help='Enable tracking of mismatches during training and evaluation. Mismatches with ' +
#                                     'confidence higher than the specified value, will be recorded for later ' +
#                                     'debugging. If enabled, this setting may have no effect if the TFRecords do not ' +
#                                     'contain any tracing info.')
    arg_group_misc.add_argument('--non-augmented-class', dest='non_augmented_class', metavar='CLASS', nargs='+',
                                help='Name (case sensitive) of the class (like \'Noise\' or \'Other\') that need not ' +
                                     'be subject to data augmentation (if enabled). Can specify multiple (separated ' +
                                     'by whitespaces).')
#    arg_group_misc.add_argument('--additive-background', dest='additive_backgrounds', metavar='TFRECORD',
#                                action='append', default=[],
#                                help='Path to a TFRecord file containing data that will be randomly added to training' +
#                                     ' inputs. Records in the TFRecord file must be of the same shape and format as ' +
#                                     'those in the training data. Multiple TFRecord files can be supplied by ' +
#                                     'specifying this argument multiple times. The probability and strength of the' +
#                                     'additive background can be controlled with the \'background_infusion_params\' ' +
#                                     'in the config file.')
#    arg_group_misc.add_argument('--debug', action='store_true', dest='training_debug',
#                                help='Enable this to allow additional summary and checkpoint outputs.')
    arg_group_logging = parser.add_argument_group('Logging')
    arg_group_logging.add_argument('--log', metavar='FILE',
                                   help='Path to file to which logs will be written out.')
    arg_group_logging.add_argument('--loglevel', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'],
                                   default='INFO',
                                   help='Logging level.')
    args = parser.parse_args()

    # Load section-specific config settings
    try:
        data_cfg, model_cfg, training_cfg = _get_settings_from_config(args)
    except FileNotFoundError as exc:
        print('Error loading config file: {}'.format(exc.strerror), file=sys.stderr)
        exit(exc.errno)
    except ConfigError as exc:
        print('Error processing config file: {}'.format(str(exc)), file=sys.stderr)
        exit(1)
    except Exception as exc:
        print('Error processing config file: {}'.format(repr(exc)), file=sys.stderr)
        exit(1)

    kwargs = {
        'random_seed': args.random_state_seed,
        'dim_order': args.dim_order,
#        'track_mismatches_thld': args.track_mismatches,
#        'training_debug': args.training_debug,
        'non_augmented_class': args.non_augmented_class,
#        'additive_backgrounds': args.additive_backgrounds
    }

    instantiate_logging(args.log if args.log is not None else
                        os.path.join(args.model_dir, _program_name + '.log'),
                        args.loglevel, args, filemode='a')
    log_config(logging.getLogger(__name__), data_cfg.originals, model_cfg, training_cfg, **kwargs)

    _main(args.datadir, args.modeldir, data_cfg, model_cfg, training_cfg, **kwargs)

    logging.shutdown()
