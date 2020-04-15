import abc
import os
import numpy as np
import tensorflow as tf

from koogu.data import DatasetDigest, DirectoryNames, \
    FilenameExtensions, tfrecord_helper


class DataFeeder(metaclass=abc.ABCMeta):
    def __init__(self, data_shape,
                 num_training_samples,
                 num_validation_samples,
                 class_names,
                 batch_size,
                 **kwargs):
        """

        :param data_shape:
        :param num_training_samples: int or list
        :param num_validation_samples: int or list
        :param class_names:
        :param batch_size:
        :param num_prefetch_batches: (optional)
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

        self._batch_size = batch_size

        # Number of prefetch batches. Generally tied to the number of GPUs
        # Set to 1 if not specified
        self._num_prefetch_batches = kwargs['num_prefetch_batches'] \
            if 'num_prefetch_batches' in kwargs else 1

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

        :return: A 2-tuple containing transormed sample and label.
        """

        raise NotImplementedError(
            'transform() method not implemented in derived class')

    @abc.abstractmethod
    def make_dataset(self, is_training):
        """
        This function must be implemented in the derived class.
        It should contain logic to load training & validation data (usually
        from stored files) and construct a TensorFlow Dataset.

        :param is_training: Boolean, indicating if operating in training mode.

        :return: A tf.data.Dataset
        """

        raise NotImplementedError(
            'make_dataset() method not implemented in derived class')

    def __call__(self, is_training):

        dataset = self.make_dataset(is_training)

        dataset = dataset.batch(self._batch_size)
        dataset = dataset.prefetch(self._num_prefetch_batches)

        return dataset

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
    def __init__(self, data_dir, batch_size, **kwargs):
        """

        :param data_dir:
        :param batch_size:
        :param num_threads: (optional, int) Number of parallel read/transform
            threads. Generally tied to number of CPUs (default if unspecified)
        :param queue_capacity: (optional, int)
        :param cache: (optional, bool, default: False) Cache loaded TFRecords
        :param kwargs: Passed as-is to parent class.
        """

        if any([
            not os.path.exists(data_dir),
            not os.path.exists(os.path.join(data_dir, DirectoryNames.TRAIN)),
            not os.path.exists(os.path.join(data_dir, DirectoryNames.EVAL))
        ]) or DatasetDigest.GetNumClasses(data_dir) is None:
            raise ValueError('Invalid data directory: {:s}'.format(
                repr(data_dir)))

        self._num_threads = \
            kwargs.pop('num_threads',
                       len(os.sched_getaffinity(0)))  # Default to num. CPUs
        self._cache = bool(kwargs.pop('cache', False))
        # Process this later
        queue_capacity = kwargs.pop('queue_capacity', None)
        # Hidden setting; default to tfrecord_helper.WaveformTFRecordHandler
        self._tfrecord_handler = \
            kwargs.pop('tfrecord_handler',
                       tfrecord_helper.WaveformTFRecordHandler())

        # Load some dataset info & initialize the base class
        train_eval_samples = \
            DatasetDigest.GetPerClassAndGroupSpecCounts(data_dir)[:, :2]
        super(TFRecordFeeder, self).__init__(
            DatasetDigest.GetDataShape(data_dir),
            train_eval_samples[:, 0],
            train_eval_samples[:, 1],
            DatasetDigest.GetOrderedClassList(data_dir),
            batch_size, **kwargs)

        self._data_dir = data_dir

        if queue_capacity is not None:
            self._queue_capacity = queue_capacity
        else:
            # the more remaining after each dequeue, better the shuffling
            min_after_dequeue = (8 * self._batch_size)
            self._queue_capacity = min_after_dequeue + (
                    self._batch_size * (
                        self._num_threads + self._num_prefetch_batches))
        self._queue_capacity = min(self._queue_capacity,
                                   self.training_samples)

    def transform(self, sample, label, is_training, **kwargs):
        # Pass as-is, nothing to do
        return sample, label

    def make_dataset(self, is_training):

        # Read in the list of TFRecord files
        filenames = tf.io.gfile.glob(
            os.path.join(self._data_dir,
                         DirectoryNames.TRAIN if is_training
                         else DirectoryNames.EVAL,
                         '*-*_*{:s}'.format(FilenameExtensions.tfrecord)))
        # print('{:d} {:s} TFRecord files'.format(
        #     len(filenames), 'training' if is_training else 'validation'))

        interleave_cycle_length = min(self._num_threads, len(filenames))

        # Create a dataset of available TFRecord files
        record_fileset = tf.data.Dataset.from_tensor_slices(filenames)

        # Fetch TFRecords in parallel
        dataset = record_fileset.interleave(
            lambda tfr_file: tf.data.TFRecordDataset(tfr_file).map(
                lambda record: self._parse_and_convert(record)),
            cycle_length=interleave_cycle_length,
            num_parallel_calls=interleave_cycle_length)

        if self._cache:
            dataset = dataset.cache()

        if is_training:  # Shuffle and repeat
            dataset = dataset.shuffle(self._queue_capacity,
                                      reshuffle_each_iteration=True)
            # dataset = dataset.repeat(5)

        # Apply the transformation operation(s)
        dataset = dataset.map(lambda x, y: self.transform(x, y, is_training),
                              num_parallel_calls=self._num_threads)

        return dataset

    def _parse_and_convert(self, record):

        data, label = self._tfrecord_handler.parse_record(record)

        data = tf.reshape(data, self._shape)
        label = tf.cast(label, tf.int32)

        return data, label
