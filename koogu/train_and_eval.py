
import os
import sys
import tensorflow as tf
import logging
from tensorflow.python.client import device_lib
import argparse
import json

from koogu.model import get_model, TrainedModel
from koogu.data import feeder
from koogu.data.tf_transformations import LoG
from koogu.utils import instantiate_logging
from koogu.utils.terminal import ArgparseConverters
from koogu.utils.config import Config, ConfigError, datasection2dict, log_config


_program_name = 'train_and_eval'


def train_and_eval(data_dir, model_dir,
                   data_config,
                   model_config,
                   training_config,
                   verbose=2,
                   **kwargs):

    # Check fields in model_config
    required_fields = ['arch', 'arch_params']
    if any([field not in model_config for field in required_fields]):
        print('Required fields missing in \'model_config\'', file=sys.stderr)
        return -1
    # Copy needed vals
    model_cfg = {key: model_config[key] for key in required_fields}
    # Copy/set default vals for non-mandatory fields
    model_cfg['preproc'] = \
        model_config.get('preproc', [])       # default to an empty list
    model_cfg['dense_layers'] = \
        model_config.get('dense_layers', [])  # default to an empty list

    # Check fields in training_config
    required_fields = ['batch_size', 'epochs']
    if any([field not in training_config for field in required_fields]):
        print('Required fields missing in \'training_config\'',
              file=sys.stderr)
        return -1
    # Copy needed vals
    training_cfg = {key: training_config[key] for key in required_fields}
    # Copy/set default vals for non-mandatory fields
    training_cfg['optimizer'] = training_config.get('optimizer', ['Adam', {}])
    training_cfg['weighted_loss'] = training_config.get('weighted_loss', True)
    training_cfg['l2_weight_decay'] = \
        training_config.get('l2_weight_decay', None)
    training_cfg['epochs_between_evals'] = \
        training_config.get('epochs_between_evals', 5)  # default of 5
    training_cfg['learning_rate'] = \
        training_config.get('learning_rate', 0.001)  # default of 0.001

    # Handle kwargs
    if 'data_format' not in kwargs:
        kwargs['data_format'] = 'channels_last'
    else:
        assert kwargs['data_format'] in ['channels_first', 'channels_last']
    if 'dropout_rate' in training_config:
        # Move this into kwargs. It's needed during model-building
        kwargs['dropout_rate'] = training_config['dropout_rate']

    # Instantiate settings that need instantiation
    try:
        model_cfg['preproc'] = [
            _get_preproc_instance(preproc_op,
                                  data_format=kwargs['data_format'],
                                  **preproc_params)
            for (preproc_op, preproc_params) in model_cfg['preproc']]
    except Exception as exc:
        print('Error setting-up model pre-processing: {:s}'.format(repr(exc)),
              file=sys.stderr)
        return -2
    try:
        training_cfg['learning_rate_fn'] = _get_learning_rate_fn(
            training_cfg['learning_rate'],
            training_config.get('lr_change_at_epochs', None),
            training_config.get('lr_update_factors', None))
    except Exception as exc:
        print('Error setting-up learning rates: {:s}'.format(repr(exc)),
              file=sys.stderr)
        return -2
    try:
        if hasattr(training_cfg['optimizer'], '__len__'):
            training_cfg['optimizer'][1]['learning_rate'] = \
                training_cfg['learning_rate_fn'](0)     # initial value
            training_cfg['optimizer'] = tf.keras.optimizers.get({
                'class_name': training_cfg['optimizer'][0],
                'config': training_cfg['optimizer'][1]})
        else:
            # was possibly already an optimizer instance
            training_cfg['optimizer'] = tf.keras.optimizers.get(
                training_cfg['optimizer'])
    except Exception as exc:
        print('Error setting-up optimizer: {:s}'.format(repr(exc)),
              file=sys.stderr)
        return -2

    # If data_dir parameter was already an instantiated BaseFeeder object,
    # pass it on. Otherwise, create a default one.
    data_feeder = data_dir if isinstance(data_dir, feeder.BaseFeeder) \
        else feeder.TFRecordFeeder(data_dir)

    # Invoke the underlying main function
    return _main(data_feeder, model_dir,
                 data_config, model_cfg, training_cfg,
                 verbose, **kwargs)


def _get_learning_rate_fn(lr, lr_change_at_epochs=None,
                          lr_update_factors=None):

    if lr_change_at_epochs is not None:
        # Variable rate is enabled, from command line
        # Set the piecewise learning rates
        def get_learning_rate(epoch):
            var_rates = [lr * factor for factor in lr_update_factors]
            for idx, boundary in enumerate(lr_change_at_epochs):
                if epoch < boundary:
                    tf.summary.scalar('learning rate', data=var_rates[idx],
                                      step=epoch)
                    return var_rates[idx]
            tf.summary.scalar('learning rate', data=var_rates[-1], step=epoch)
            return var_rates[-1]
    elif hasattr(lr, '__len__'):    # If it's a list
        # Not called from command line and per-epoch values are defined
        def get_learning_rate(epoch):
            retval = lr[-1] if epoch >= len(lr) else lr[epoch]
            tf.summary.scalar('learning rate', data=retval, step=epoch)
            return retval
    elif callable(lr):  # If it's a function
        def get_learning_rate(epoch):
            return lr(epoch + 1)
    else:
        # Static learning rate
        def get_learning_rate(epoch):
            tf.summary.scalar('learning rate', data=lr, step=epoch)
            return lr

    return get_learning_rate


def _get_preproc_instance(name, **kwargs):

    if name == 'LoG':
        return LoG(**kwargs)
    # Add others here in an if-elif ladder

    # Raise exception if unknown option requested
    raise ValueError('Unknown preproc option requested: {:s}'.format(name))


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
          verbose=2,
          **kwargs):

    os.makedirs(model_dir, exist_ok=True)

    tf.keras.backend.clear_session()

    if 'random_seed' in kwargs:
        tf.random.set_seed(kwargs.pop('random_seed'))

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

    #print(data_feeder.data_shape)

    # Create a Classifier instance
    classifier = get_model(model_cfg,
                           input_shape=data_feeder.data_shape,
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

    if verbose > 0:
        print('Data: {:d} classes, {:d} training & {:d} eval samples'.format(
            data_feeder.num_classes, data_feeder.training_samples,
            data_feeder.validation_samples))

        classifier.summary()

    # Disable loss weighting if explicitly requested
    class_weights = None if not training_cfg.get('weighted_loss', True) else {
        idx: weight for idx, weight in enumerate(
            data_feeder.training_samples /
            (data_feeder.num_classes * data_feeder.training_samples_per_class))
        }

    callbacks = []
    # Set the learning rate
    callbacks.append(
        tf.keras.callbacks.LearningRateScheduler(training_cfg['learning_rate_fn'])
    )
#    os.makedirs(os.path.join(model_dir, 'checkpoints'), exist_ok=True)
#    callbacks.append(tf.keras.callbacks.ModelCheckpoint(
#        filepath=os.path.join(model_dir, 'checkpoints', 'ckpt_weights_e-{epoch:03d}.h5'),
#        save_weights_only=True,
#        monitor='val_loss', mode='min', save_best_only=True))

    num_gpu_devices = len(
        [x.name for x in device_lib.list_local_devices()
         if x.device_type == 'GPU'])
    feeder_args = dict(batch_size=training_cfg['batch_size'],
                       num_prefetch_batches=max(1, num_gpu_devices),
                       cache=True)

    history = classifier.fit(
        x=data_feeder(True, **feeder_args),
        validation_data=data_feeder(False, **feeder_args),
        initial_epoch=0, epochs=training_cfg['epochs'],
        validation_freq=training_cfg['epochs_between_evals'],
        shuffle=False,
        verbose=verbose,
        class_weight=class_weights,
#        steps_per_epoch=int(np.ceil(steps_per_epoch)),
#        validation_steps=val_steps,
        callbacks=callbacks)

    # Reset metrics before saving
#    classifier.reset_metrics()
#    classifier.trainable = False
    #classifier.save(os.path.join(model_dir, 'classifier.h5'), include_optimizer=False)
    with open(os.path.join(model_dir, 'classifier.json'), 'w') as of:
        of.write(classifier.to_json())
    classifier.save_weights(os.path.join(model_dir, 'classifier_weights.h5'))

    # Write out training history. Include epoch numbers for train & eval
    with open(os.path.join(model_dir, 'training_history.json'), 'w') as of:
        history_c = history.history.copy()
        history_c['train_epochs'] = \
            [x for x in range(1, training_cfg['epochs'] + 1)]
        history_c['eval_epochs'] = \
            [x for x in range(training_cfg['epochs_between_evals'],
                              training_cfg['epochs'] + 1,
                              training_cfg['epochs_between_evals'])]
        of.write(json.dumps(str(history_c)))

    TrainedModel.finalize_and_save(classifier,
                                   model_dir,
                                   data_feeder.data_shape,
                                   data_feeder.get_shape_transformation_info(),
                                   data_feeder.class_names,
                                   data_cfg['audio_settings'],
                                   spec_settings=data_cfg['spec_settings'])

    return history_c


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

    _main(_get_default_data_feeder(args.data_dir, data_cfg, training_cfg['batch_size']),
          args.modeldir, data_cfg, model_cfg, training_cfg,
          verbose=1, **kwargs)

    logging.shutdown()
