import abc
import os
import numpy as np
import tensorflow as tf

from koogu.data import DatasetDigest, DirectoryNames, \
    FilenameExtensions, tfrecord_helper
from koogu.data.tf_transformations import Audio2Spectral


class DataFeeder(metaclass=abc.ABCMeta):
    def __init__(self, data_shape,
                 num_training_samples,
                 num_validation_samples,
                 class_names,
                 **kwargs):
        """

        :param data_shape:
        :param num_training_samples: int or list
        :param num_validation_samples: int or list
        :param class_names:
        """

        self._shape = data_shape
        self._class_names = class_names

        if hasattr(num_training_samples, '__len__'):
            assert len(num_training_samples) == len(class_names)
            self._num_training_samples = np.asarray(num_training_samples)
        else:
            self._num_training_samples = num_training_samples

        if hasattr(num_validation_samples, '__len__'):
            assert len(num_validation_samples) == len(class_names)
            self._num_validation_samples = np.asarray(num_validation_samples)
        else:
            self._num_validation_samples = num_validation_samples

    @abc.abstractmethod
    def transform(self, sample, label, is_training, **kwargs):
        """
        This function must be implemented in the derived class.
        It should contain logic to apply any transformations to a single input
        to the model (during training and validation) and is invoked by the
        make_dataset() method.

        :param sample: The sample that must be 'transformed' before
            consumption by a model.
        :param label: The class info pertaining to 'sample'.
        :param is_training: Boolean, indicating if operating in training mode.
        :param kwargs: Any additional parameters.

        :return: A 2-tuple containing transformed sample and label.
        """

        raise NotImplementedError(
            'transform() method not implemented in derived class')

    @abc.abstractmethod
    def make_dataset(self, is_training, batch_size, **kwargs):
        """
        This function must be implemented in the derived class.
        It should contain logic to load training & validation data (usually
        from stored files) and construct a TensorFlow Dataset.

        :param is_training: Boolean, indicating if operating in training mode.
        :param batch_size:

        :return: A tf.data.Dataset
        """

        raise NotImplementedError(
            'make_dataset() method not implemented in derived class')

    def __call__(self, is_training, batch_size, **kwargs):
        """

        :param is_training: Boolean, indicating if operating in training mode.
        :param batch_size:
        :param kwargs: Passed as is to make_dataset() of inherited class.
        """

        return self.make_dataset(is_training, batch_size, **kwargs)

    @property
    def data_shape(self):
        return self._shape

    @property
    def num_classes(self):
        return len(self._class_names)

    @property
    def class_names(self):
        return self._class_names

    @property
    def training_samples(self):
        return \
            self._num_training_samples.sum() \
            if hasattr(self._num_training_samples, '__len__') else \
            self._num_training_samples

    @property
    def training_samples_per_class(self):
        if hasattr(self._num_training_samples, '__len__'):
            return self._num_training_samples
        else:
            raise ValueError('Per-class sample counts were not initialized')

    @property
    def validation_samples(self):
        return \
            self._num_validation_samples.sum() \
            if hasattr(self._num_validation_samples, '__len__') else \
            self._num_validation_samples

    @property
    def validation_samples_per_class(self):
        if hasattr(self._num_validation_samples, '__len__'):
            return self._num_validation_samples
        else:
            raise ValueError('Per-class sample counts were not initialized')


class TFRecordFeeder(DataFeeder):
    def __init__(self, data_dir, **kwargs):
        """

        :param data_dir:
        :param kwargs: Passed as-is to parent class.
        """

        if any([
            not os.path.exists(data_dir),
            not os.path.exists(os.path.join(data_dir, DirectoryNames.TRAIN)),
            not os.path.exists(os.path.join(data_dir, DirectoryNames.EVAL))
        ]) or DatasetDigest.GetNumClasses(data_dir) is None:
            raise ValueError('Invalid data directory: {:s}'.format(
                repr(data_dir)))

        self._data_dir = data_dir

        # Hidden setting; default to what's best for the data
        if 'tfrecord_handler' in kwargs:
            self._tfrecord_handler = kwargs.pop('tfrecord_handler')
        else:
            ndim = DatasetDigest.GetDataShape(data_dir).shape[0]
            if ndim == 1:  # waveforms
                self._tfrecord_handler = \
                    tfrecord_helper.WaveformTFRecordHandler()
            else:   # time-freq representation; assuming 2D if it wasn't 1D
                self._tfrecord_handler = \
                    tfrecord_helper.SpectrogramTFRecordHandler()

        self._in_shape = DatasetDigest.GetDataShape(data_dir).tolist()

        # Load some dataset info & initialize the base class
        train_eval_samples = \
            DatasetDigest.GetPerClassAndGroupSpecCounts(data_dir)[:, :2]
        super(TFRecordFeeder, self).__init__(
            self._in_shape,
            train_eval_samples[:, 0],
            train_eval_samples[:, 1],
            DatasetDigest.GetOrderedClassList(data_dir),
            **kwargs)

    def transform(self, sample, label, is_training, **kwargs):
        # Pass as-is, nothing to do
        return sample, tf.one_hot(label, self.num_classes)

    def make_dataset(self, is_training, batch_size, **kwargs):
        """
        *** Do not call this directly ***

        :param is_training: (bool)
        :param batch_size: (int)
        :param num_prefetch_batches: (optional) Number of prefetch batches.
            Generally tied to the number of GPUs. (default is 1)
        :param num_threads: (optional, int) Number of parallel read/transform
            threads. Generally tied to number of CPUs (default if unspecified)
        :param queue_capacity: (optional, int)
        :param cache: (optional, bool, default: False) Cache loaded TFRecords
        """

        num_threads = \
            kwargs.pop('num_threads',
                       len(os.sched_getaffinity(0)))  # Default to num. CPUs

        # Read in the list of TFRecord files
        filenames = tf.io.gfile.glob(
            os.path.join(self._data_dir,
                         DirectoryNames.TRAIN if is_training
                         else DirectoryNames.EVAL,
                         '*-*_*{:s}'.format(FilenameExtensions.tfrecord)))
        # print('{:d} {:s} TFRecord files'.format(
        #     len(filenames), 'training' if is_training else 'validation'))

        interleave_cycle_length = min(num_threads, len(filenames))

        # Create a dataset of available TFRecord files
        record_fileset = tf.data.Dataset.from_tensor_slices(filenames)

        # Fetch TFRecords in parallel
        dataset = record_fileset.interleave(
            lambda tfr_file: tf.data.TFRecordDataset(tfr_file).map(
                lambda record: self._parse_and_convert(record)),
            cycle_length=interleave_cycle_length,
            num_parallel_calls=interleave_cycle_length)

        if bool(kwargs.pop('cache', False)):
            dataset = dataset.cache()

        num_prefetch_batches = kwargs.pop('num_prefetch_batches', 1)

        if is_training:  # Shuffle and repeat
            if 'queue_capacity' in kwargs:
                queue_capacity = kwargs.pop('queue_capacity')
            else:
                # the more remaining after each dequeue, better the shuffling
                min_after_dequeue = (8 * batch_size)
                queue_capacity = min_after_dequeue + (
                        batch_size * (
                            num_threads + num_prefetch_batches))
            queue_capacity = min(queue_capacity, self.training_samples)
            dataset = dataset.shuffle(queue_capacity,
                                      reshuffle_each_iteration=True)
            # dataset = dataset.repeat(5)

        # Apply the transformation operation(s)
        dataset = dataset.map(lambda x, y: self.transform(x, y, is_training),
                              num_parallel_calls=num_threads)

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(num_prefetch_batches)

        return dataset

    def _parse_and_convert(self, record):

        data, label = self._tfrecord_handler.parse_record(record)

        data = tf.reshape(data, self._in_shape)
        label = tf.cast(label, tf.int32)

        return data, label


class SpectralTFRecordFeeder(TFRecordFeeder):
    """
    A handy TFRecord feeder, which normalizes and converts raw audio to
    time-frequency format.
    """
    def __init__(self, data_dir, data_cfg, **kwargs):

        super(SpectralTFRecordFeeder, self).__init__(
            data_dir, **kwargs)

        self._transformation = Audio2Spectral(
            data_cfg['audio_settings']['desired_fs'],
            data_cfg['spec_settings'])

        # Update to what the transformed output shape would be
        self._shape = self._transformation.compute_output_shape(
            [1] + self._in_shape)[1:]

    def transform(self, clip, label, is_training, **kwargs):

        output = clip

        # Normalize the waveforms
        output = output - tf.reduce_mean(output, axis=-1, keepdims=True)
        output = output / \
            tf.reduce_max(tf.abs(output), axis=-1, keepdims=True)

        # Convert to spectrogram
        output = self._transformation(output)

        return output, tf.one_hot(label, self.num_classes)
