import abc
import os
import json
import numpy as np
import tensorflow as tf

from koogu.data import DatasetDigest, DirectoryNames, \
    AssetsExtraNames, FilenameExtensions, tfrecord_helper, Convert
from koogu.data.tf_transformations import Audio2Spectral
from koogu.utils.filesystem import recursive_listing


class BaseFeeder(metaclass=abc.ABCMeta):
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

    def _queue_and_batch(self, dataset, is_training, batch_size, **kwargs):
        """Boilerplate caching, queueing and batching functionality."""

        if bool(kwargs.pop('cache', False)):
            dataset = dataset.cache()

        num_prefetch_batches = kwargs.get('num_prefetch_batches', 1)
        num_threads = kwargs.get('num_threads', 1)

        if is_training:  # Shuffle and repeat
            if 'queue_capacity' in kwargs:
                queue_capacity = kwargs['queue_capacity']
            else:
                # the more remaining after each dequeue, better the shuffling
                min_after_dequeue = (8 * batch_size)
                queue_capacity = min_after_dequeue + (
                        batch_size * (
                            num_threads + num_prefetch_batches))
            queue_capacity = min(queue_capacity, self.training_samples)
            dataset = dataset.shuffle(queue_capacity,
                                      reshuffle_each_iteration=True)

        # Apply the transformation operation(s)
        dataset = dataset.map(lambda x, y: self.transform(x, y, is_training),
                              num_parallel_calls=num_threads)

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(num_prefetch_batches)

        return dataset

    @property
    def data_shape(self):
        return self._shape

    def get_shape_transformation_info(self):
        """
        Override in inherited class if its transform() alters the shape of the
        read/input data before a dataset is returned. If not None, must return
        a tuple where:
            first value is the untransformed input shape,
            second is the actual transformation function.
        """
        return None

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


class DataFeeder(BaseFeeder):
    """
    A class for loading prepared data from numpy .npz files and feeding them
    into the training/evaluation pipeline.
    """

    def __init__(self, data_dir,
                 validation_split=None,
                 min_clips_per_class=None,
                 max_clips_per_class=None,
                 random_state_seed=None,
                 **kwargs):
        """
        :param data_dir: Directory under which .npz files are available.
        """

        assert ((validation_split is None) or (0.0 <= validation_split <= 1))

        random_state = np.random.RandomState(random_state_seed)

        with open(os.path.join(data_dir, AssetsExtraNames.classes_list),
                  'r') as f:
            class_names = json.load(f)
        num_classes = len(class_names)

        file_label_counts = [
            np.sum(DataFeeder._get_file_labels_mask(
                os.path.join(data_dir, file)), axis=0)
            for file in recursive_listing(
                data_dir,
                match_extensions=FilenameExtensions.numpy)]
        assert num_classes == file_label_counts[0].shape[0]

        self._data_dir = data_dir
        self._valid_class_mask = np.full((num_classes, ), True, dtype=np.bool)

        class_files_train, class_files_eval = [], []
        class_files_items_train, class_files_items_eval = [], []
        for class_idx in range(num_classes):

            # Cumulative counts of class-specific clips across all files
            cumul_counts = np.cumsum(
                [flc[class_idx] for flc in file_label_counts])

            if cumul_counts[-1] >= (min_clips_per_class or 0):
                (cf_tr, cfi_tr), (cf_ev, cfi_ev) = \
                    DataFeeder._helper1(cumul_counts, max_clips_per_class,
                                        validation_split, random_state)

                class_files_train.append(cf_tr)
                class_files_items_train.append(cfi_tr)
                class_files_eval.append(cf_ev)
                class_files_items_eval.append(cfi_ev)

            else:
                # If insufficient clips, mark class for rejection
                self._valid_class_mask[class_idx] = False

        valid_class_idxs = np.where(self._valid_class_mask)[0]
        out_num_classes = len(valid_class_idxs)

        self._files_clips_idxs_train = []
        self._files_clips_idxs_eval = []
        num_per_class_samples_train = np.zeros((len(valid_class_idxs), ),
                                               dtype=np.int)
        num_per_class_samples_eval = np.zeros((len(valid_class_idxs), ),
                                              dtype=np.int)
        file = None
        for f_idx, file in enumerate(
                recursive_listing(data_dir,
                                  match_extensions=FilenameExtensions.numpy)):

            label_mask = DataFeeder._get_file_labels_mask(
                os.path.join(data_dir, file))[:, self._valid_class_mask]

            file_train_clips_idxs = [np.zeros((0, ), dtype=np.int)]
            file_eval_clips_idxs = [np.zeros((0, ), dtype=np.int)]
            for class_idx in range(out_num_classes):
                file_class_idxs = np.where(label_mask[:, class_idx])[0]

                e_idx = np.where(class_files_train[class_idx] == f_idx)[0]
                if e_idx.shape[0] > 0:
                    file_train_clips_idxs.append(
                        file_class_idxs[
                            class_files_items_train[class_idx][e_idx[0]]])
                    # Clear out after use, no longer needed
                    class_files_items_train[class_idx][e_idx[0]] = None

                e_idx = np.where(class_files_eval[class_idx] == f_idx)[0]
                if e_idx.shape[0] > 0:
                    file_eval_clips_idxs.append(
                        file_class_idxs[
                            class_files_items_eval[class_idx][e_idx[0]]])
                    # Clear out after use, no longer needed
                    class_files_items_eval[class_idx][e_idx[0]] = None

            file_clips_idxs = np.unique(np.concatenate(file_train_clips_idxs))
            self._files_clips_idxs_train.append(file_clips_idxs)
            num_per_class_samples_train += \
                np.sum(label_mask[file_clips_idxs, :], axis=0)

            file_clips_idxs = np.unique(np.concatenate(file_eval_clips_idxs))
            self._files_clips_idxs_eval.append(file_clips_idxs)
            num_per_class_samples_eval += \
                np.sum(label_mask[file_clips_idxs, :], axis=0)

        # Read clip length from the first clip in the last file read above
        self._in_shape = [(
            DataFeeder._get_file_clips_and_labels(
                os.path.join(data_dir, file), [0], self._valid_class_mask)[0]
        ).shape[1]]

        super(DataFeeder, self).__init__(
            self._in_shape,
            num_per_class_samples_train,
            num_per_class_samples_eval,
            [c for c, m in zip(class_names, self._valid_class_mask) if m],
            **kwargs)

    def transform(self, sample, label, is_training, **kwargs):
        # Pass as-is, nothing to do
        return sample, label

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
            kwargs.get('num_threads',
                       len(os.sched_getaffinity(0)))  # Default to num. CPUs

        files_clips_idxs = self._files_clips_idxs_train if is_training \
            else self._files_clips_idxs_eval

        interleave_cycle_length = min(num_threads, len(files_clips_idxs))

        # Create a dataset of available .npz files
        fileset = tf.data.Dataset.from_generator(
            lambda: enumerate(
                recursive_listing(self._data_dir,
                                  match_extensions=FilenameExtensions.numpy)),
            args=None,
            output_signature=(tf.TensorSpec(shape=(), dtype=tf.int64),
                              tf.TensorSpec(shape=(), dtype=tf.string))
            )

        # Fetch clips from multiple files in parallel
        dataset = fileset.interleave(
            lambda file_idx, npz_file: tf.data.Dataset.from_generator(
                lambda a, b: DataFeeder.file_data_generator(
                    os.path.join(self._data_dir, b.decode()),
                    files_clips_idxs[a],
                    self._valid_class_mask),
                args=(file_idx, npz_file),
                output_signature=(
                    tf.TensorSpec(shape=(self._in_shape[0],),
                                  dtype=tf.float32),
                    tf.TensorSpec(shape=(self.num_classes,),
                                  dtype=tf.float32))
            ),
            cycle_length=interleave_cycle_length,
            num_parallel_calls=interleave_cycle_length)

        return self._queue_and_batch(dataset, is_training, batch_size,
                                     **kwargs)

    @staticmethod
    def file_data_generator(npz_filepath, file_clips_idxs, valid_class_mask):

        clips, labels = DataFeeder._get_file_clips_and_labels(
            npz_filepath, file_clips_idxs, valid_class_mask)

        for clip, label in zip(clips, labels):
            yield clip, label

    @staticmethod
    def _helper1(cumul_counts, max_clips, val_split, random_state):

        upper_lim = cumul_counts[-1] if max_clips is None \
            else np.minimum(cumul_counts[-1], max_clips)

        all_idxs = random_state.permutation(cumul_counts[-1])[:upper_lim]

        train_split = upper_lim - \
            np.round((val_split or 0.0) * upper_lim).astype(dtype=np.int64)

        # Find the idx of the file that each item in all_idxs belongs to
        file_refs = np.digitize(all_idxs,
                                np.concatenate([[-1], cumul_counts])) - 1

        # Find the corresponding in-file within-class idxs
        in_file_class_items_refs = \
            all_idxs - np.concatenate([[0], cumul_counts[:-1]])[file_refs]

        return DataFeeder._helper2(file_refs[:train_split],
                                   in_file_class_items_refs[:train_split]), \
               DataFeeder._helper2(file_refs[train_split:],
                                   in_file_class_items_refs[train_split:])

    @staticmethod
    def _helper2(file_refs, in_file_class_items_refs):

        uniq_file_refs, ufr_rev_idxs = np.unique(file_refs,
                                                 return_inverse=True)

        return uniq_file_refs, \
               [in_file_class_items_refs[np.where(ufr_rev_idxs == idx)[0]]
                for idx in range(len(uniq_file_refs))]

    @staticmethod
    def _get_file_labels_mask(filepath):
        with np.load(filepath) as data:
            return data['labels'] == 1

    @staticmethod
    def _get_file_clips_and_labels(filepath, clips_idxs, class_mask):
        # In the npz file, clips are stored as int16 & labels are stored as
        # float16. Convert them appropriately before returning.
        with np.load(filepath) as data:
            return Convert.pcm2float(data['clips'][clips_idxs, :]), \
                   (data['labels'][clips_idxs, :])[:, class_mask].astype(
                       np.float32)


class TFRecordFeeder(BaseFeeder):
    """
    A class for loading records from TFRecords files and feeding them into the
    training/evaluation pipeline.
    """

    def __init__(self, data_dir, **kwargs):
        """
        :param data_dir: Directory in which the TFRecord files are available.
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
            kwargs.get('num_threads',
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

        return self._queue_and_batch(dataset, is_training, batch_size,
                                     **kwargs)

    def _parse_and_convert(self, record):

        data, label = self._tfrecord_handler.parse_record(record)

        data = tf.reshape(data, self._in_shape)
        label = tf.cast(label, tf.int32)

        return data, label


class SpectralDataFeeder(DataFeeder):
    """
    A handy data feeder, which normalizes and converts raw audio to
    time-frequency format.
    """
    def __init__(self, data_dir, fs, spec_settings, **kwargs):

        super(SpectralDataFeeder, self).__init__(
            data_dir, **kwargs)

        self._transformation = Audio2Spectral(fs, spec_settings)

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

        return output, label

    def get_shape_transformation_info(self):
        return self._in_shape, self._transformation


class SpectralTFRecordFeeder(TFRecordFeeder):
    """
    A handy TFRecord feeder, which normalizes and converts raw audio to
    time-frequency format.
    """
    def __init__(self, data_dir, fs, spec_settings, **kwargs):

        super(SpectralTFRecordFeeder, self).__init__(
            data_dir, **kwargs)

        self._transformation = Audio2Spectral(fs, spec_settings)

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

    def get_shape_transformation_info(self):
        return self._in_shape, self._transformation
