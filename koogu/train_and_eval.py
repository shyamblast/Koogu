import os
import sys
import tensorflow as tf
import logging
from tensorflow.python.client import device_lib
import argparse
import json
from datetime import datetime

from koogu.model import architectures, TrainedModel
from koogu.data import feeder
from koogu.utils import instantiate_logging
from koogu.utils.terminal import ArgparseConverters
from koogu.utils.config import Config, ConfigError


_program_name = 'train_and_eval'


def train_and_eval(data_feeder, model_dir,
                   data_settings,
                   model_architecture,
                   training_config,
                   verbose=2,
                   **kwargs):
    """
    Perform training and evaluation.

    :param data_feeder: An instance of a :class:`~data.feeder.BaseFeeder`
        implementation (e.g., :class:`~data.feeder.SpectralDataFeeder`).
    :param model_dir: Path to the directory into which the trained model and its
        supporting files will be written.
    :param data_settings: A Python dictionary containing -

        * `audio_settings` : a sub-dictionary specifying parameters considered
          in data pre-processing,
        * `spec_settings` : (optional) a sub-dictionary specifying parameters
          considered in data transformation (if any considered).

        These settings are not used during training, but must be specified so
        that they will be saved along with the trained model after training
        completes.

    :param model_architecture: An instance of a
        :class:`~model.architectures.BaseArchitecture` implementation (e.g.,
        :class:`~model.architectures.DenseNet`).
    :param training_settings: Training hyperparameters. A Python dictionary
        containing settings for defining the training process, controlling
        regularization, etc.

        `Required fields`

        * `batch_size`: (integer) Number of input samples from the dataset to
          combine in a single batch.
        * `epochs`: (integer) Number of epochs to perform the training for.

        `Optional fields`

        * `optimizer`: The optimizer to use during training. Must be a 2-element
          tuple specifying the name (string) of the optimizer and its
          parameters (a Python dictionary containing key-value pairs). Defaults
          to ``['Adam', {}]``.
        * `weighted_loss`: (boolean; default: True) When enabled, loss function
          during training will be weighted based on the disparities in
          per-class training samples available.
        * `l2_weight_decay`: To enable, set to a reasonable value (e.g., 1e-4).
          Enabling it will add L2 regularization, with the specified decay.
        * `learning_rate`: Learning rate for training (default: 0.001). Can also
          specify dynamic rates, in one of two ways:

          * set this key to a static value, and specify both
            `lr_change_at_epochs` and `lr_update_factors` (see below), or
          * set this key to be a callable (e.g., function) which takes the
            current epoch number as input and returns the desired learning rate
            for the epoch.
        * `lr_change_at_epochs`: List of integers specifying the epochs at which
          the learning rate must be updated. If specifying this, `learning_rate`
          must be static.
        * `lr_update_factors`: List of integers (one element more than
          lr_change_at_epochs) specifying the decimation factor of the current
          learning rate at the set epochs. If specifying this, `learning_rate`
          must be static.
        * `dropout_rate`: Helps the model generalize better. Set to a small
          positive quantity (e.g., 0.05). Functionality is disabled by default.
        * `epochs_between_evals`: (optional; integer) Number of epochs to wait
          before performing another validation run. (default: 5)
    :param verbose: Level of information to display. Set to -

        | 0 - for no display
        | 1 - to display progress bars for each epoch
        | 2 - to display one-line summary per epoch (default)
    :param random_seed: (optional) A seed (integer) used to initialize the
        psuedo-random number generator that makes setting randomized initial
        values for model parameters repeatable.

    :return: A Python dictionary containing a record of the training history,
        including the "training" and "evaluation" accuracies and losses at each
        epoch.
    """

    assert isinstance(data_feeder, feeder.BaseFeeder), '`data_feeder` must ' + \
        'be an instance of a class that implements koogu.data.feeder.BaseFeeder'

    assert isinstance(model_architecture, architectures.BaseArchitecture), \
        '`model_architecture` must be an instance of a class that ' + \
        'implements koogu.model.architectures.BaseArchitecture'

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
    if 'dropout_rate' in training_config:
        # Move this into kwargs. It's needed during model-building
        kwargs['dropout_rate'] = training_config['dropout_rate']

    # Instantiate settings that need instantiation
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

    # Invoke the underlying main function
    return _main(data_feeder, model_dir,
                 data_settings, model_architecture, training_cfg,
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


def _main(data_feeder, model_dir, data_cfg, model_arch, training_cfg,
          verbose=2,
          **kwargs):

    os.makedirs(model_dir, exist_ok=True)

    tf.keras.backend.clear_session()

    if 'random_seed' in kwargs:
        tf.random.set_seed(kwargs.pop('random_seed'))

    # Create a Classifier instance
    classifier = model_arch(input_shape=data_feeder.data_shape,
                            num_classes=data_feeder.num_classes,
                            is_training=True,
                            **kwargs)

    # Add L2 regularization, if enabled
    if training_cfg['l2_weight_decay'] is not None and training_cfg['l2_weight_decay'] > 0.0:
        regularizer = tf.keras.regularizers.l2(training_cfg['l2_weight_decay'])
        for layer in classifier.layers:
            if hasattr(layer, 'kernel_regularizer'):
                setattr(layer, 'kernel_regularizer', regularizer)

    loss_fn = tf.keras.losses.BinaryCrossentropy() if model_arch.multilabel \
        else tf.keras.losses.CategoricalCrossentropy()
    acc_metric = tf.keras.metrics.BinaryAccuracy() if model_arch.multilabel \
        else tf.keras.metrics.CategoricalAccuracy()

    classifier.compile(
        optimizer=training_cfg['optimizer'],
        loss=loss_fn,
        metrics=[acc_metric])

    if verbose == 'auto' or verbose > 0:
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
                       num_prefetch_batches=max(1, num_gpu_devices))

    history = classifier.fit(
        x=data_feeder(True, **feeder_args),
        validation_data=data_feeder(False, **feeder_args),
        initial_epoch=0, epochs=training_cfg['epochs'],
        validation_freq=training_cfg['epochs_between_evals'],
        shuffle=False,
        verbose=verbose,
        class_weight=class_weights,
        callbacks=callbacks)

    with open(os.path.join(model_dir, 'classifier.json'), 'w') as of:
        of.write(classifier.to_json())
    classifier.save_weights(os.path.join(model_dir, 'classifier_weights.h5'))

    # Write out training history. Include epoch numbers for train & eval
    history_c = {
        key: ([float(v) for v in val] if hasattr(val, '__len__') else val)
        for key, val in history.history.items()
    }
    history_c['train_epochs'] = \
        [x for x in range(1, training_cfg['epochs'] + 1)]
    history_c['eval_epochs'] = \
        [x for x in range(training_cfg['epochs_between_evals'],
                          training_cfg['epochs'] + 1,
                          training_cfg['epochs_between_evals'])]
    with open(os.path.join(model_dir, 'training_history.json'), 'w') as of:
        json.dump(history_c, of, indent=0)

    TrainedModel.finalize_and_save(classifier,
                                   model_dir,
                                   data_feeder.data_shape,
                                   data_feeder.get_shape_transformation_info(),
                                   data_feeder.class_names,
                                   data_cfg['audio_settings'],
                                   spec_settings=data_cfg['spec_settings'])

    return history_c


__all__ = ['train_and_eval']


def cmdline_parser(parser=None):

    if parser is None:
        parser = argparse.ArgumentParser(
            prog='koogu.train', allow_abbrev=True,
            description='Train a model using prepared inputs.')

    parser.add_argument(
        'cfg_file', metavar='<CONFIG FILE>',
        help='Path to config file.')

    parser.add_argument(
        '--seed', dest='random_state_seed',
        type=ArgparseConverters.positive_integer, metavar='NUM',
        help='Seed value (integer) for deterministic shuffling.')
    parser.add_argument(
        '--model_name', dest='model_name',
        type=str, metavar='NUM',
        help='Name of the trained model. Output directory will be given this '
             'name.')

    arg_group_logging = parser.add_argument_group('Logging')
    arg_group_logging.add_argument(
        '--loglevel', dest='log_level',
        choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'],
        default='INFO', help='Logging level.')

    parser.set_defaults(exec_fn=cmdline_train)

    return parser


def cmdline_train(cfg_file, log_level,
                  model_name=None, random_state_seed=None,
                  # Other overriding parameters not available via cmdline
                  data_feeder=None
                  ):
    """Functionality invoked via the command-line interface"""

    # Load config
    try:
        cfg = Config(cfg_file, 'data.audio', 'data.spec', 'data.annotations',
                     'data.augmentations', 'train', 'model')
    except FileNotFoundError as exc:
        print(f'Error loading config file: {exc.strerror}', file=sys.stderr)
        exit(exc.errno)
    except ConfigError as exc:
        print(f'Error processing config file: {str(exc)}', file=sys.stderr)
        exit(1)
    except Exception as exc:
        print(f'Error processing config file: {repr(exc)}', file=sys.stderr)
        exit(1)

    if not os.path.exists(cfg.paths.train_audio):
        print('Error: Invalid path specified in train_audio', file=sys.stderr)
        exit(2)

    if cfg.paths.logs is not None:
        instantiate_logging(os.path.join(cfg.paths.logs, 'train.log'),
                            log_level, 'a')

    exit_code = 0

    logger = logging.getLogger(__name__)

    if model_name is None or model_name == '':
        model_name = datetime.now().strftime('%Y%m%dT%H%M%S')
    logger.info(f'Model name: {model_name}')

    spec_settings = cfg.data.spec.as_dict(skip_invalid=True)
    training_config = cfg.train.as_dict(skip_invalid=True)

    # ---- Instantiate feeder ----
    # Pop out (if exists) hyperparameters that were meant for the feeder
    val_split = training_config.pop('validation_split', 0.15)
    min_clips_per_class = training_config.pop('min_clips_per_class', None)
    max_clips_per_class = training_config.pop('max_clips_per_class', None)
    if data_feeder is None:
        data_feeder = feeder.SpectralDataFeeder(
            data_dir=cfg.paths.training_samples,
            fs=cfg.data.audio.desired_fs,
            spec_settings=spec_settings,
            validation_split=val_split,
            min_clips_per_class=min_clips_per_class,
            max_clips_per_class=max_clips_per_class,
            random_state_seed=random_state_seed,
            background_class=cfg.data.annotations.background_class
        )

    # Set up desired pre- and post-transform augmentations
    if cfg.data.augmentations.temporal:
        for (prob, aug, aug_args) in cfg.data.augmentations.temporal:
            data_feeder.add_pre_transform_augmentation(prob, aug, *aug_args)
    if cfg.data.augmentations.spectrotemporal:
        for (prob, aug, aug_args) in cfg.data.augmentations.spectrotemporal:
            data_feeder.add_post_transform_augmentation(prob, aug, *aug_args)

    logger.info(f'Input shape: {data_feeder.data_shape}')
    logger.info(f'{"Class":<35s} : {"Train":>5s} {"Eval":>5s}')
    logger.info(f'{"-----":<35s} : {"-----":>5s} {"-----":>5s}')
    for lbl, tr_samps, ev_samps in zip(
            data_feeder.class_names,
            data_feeder.training_samples_per_class,
            data_feeder.validation_samples_per_class):
        logger.info(f'{lbl:<35s} : {tr_samps:>5d} {ev_samps:>5d}')

    # ---- Instantiate model ----
    model = getattr(architectures, cfg.model.architecture)(
        **cfg.model.architecture_params)

    # ---- Train ----
    history = train_and_eval(
        data_feeder,
        os.path.join(cfg.paths.model, model_name),
        dict(audio_settings=cfg.data.audio.as_dict(skip_invalid=True),
             spec_settings=spec_settings),
        model,
        training_config=training_config,
        random_seed=random_state_seed
    )

    if cfg.paths.logs is not None:
        logging.shutdown()

    print(f'Trained model "{model_name}" saved at {cfg.paths.model}')

    exit(exit_code)
