import abc
import os
import json
import numpy as np
import tensorflow as tf
import logging
from functools import reduce

from koogu.data import AssetsExtraNames, FilenameExtensions
from koogu.data.raw import Convert
from koogu.data.tf_transformations import NormalizeAudio, Audio2Spectral
from koogu.utils.filesystem import recursive_listing


class BaseFeeder(metaclass=abc.ABCMeta):
    """
    Base class defining the interface for implementing feeder classes for
    building data pipelines in Koogu.

    :param data_shape: Shape of the input samples presented to the model.
    :param num_training_samples: List of per-class counts of training samples
        available.
    :param num_validation_samples: List of per-class counts of validation
        samples available.
    :param class_names: List of names (str) corresponding to the different
        classes in the problem space.
    """

    def __init__(self, data_shape,
                 num_training_samples,
                 num_validation_samples,
                 class_names,
                 **kwargs):

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
        to the model (during training and validation).

        :param sample: The sample that must be 'transformed' before
            consumption by a model.
        :param label: The class info pertaining to ``sample``.
        :param is_training: (boolean) True if operating in training mode.
        :param kwargs: Any additional parameters.

        :return: A 2-tuple containing transformed sample and label.
        """

        raise NotImplementedError(
            'transform() method not implemented in derived class')

    @abc.abstractmethod
    def pre_transform(self, sample, label, is_training, **kwargs):
        """
        Implement this method in the derived class to apply any
        pre-transformation augmentations to a single input to the model (during
        training and validation).

        :param sample: The untransformed sample to which to apply augmentations.
        :param label: The class info pertaining to ``sample``.
        :param is_training: (boolean) True if operating in training mode.
        :param kwargs: Any additional parameters.

        :return: A 2-tuple containing transformed sample and label.
        """

        raise NotImplementedError(
            'transform() method not implemented in derived class')

    @abc.abstractmethod
    def post_transform(self, sample, label, is_training, **kwargs):
        """
        Implement this method in the derived class to apply any
        post-transformation augmentations to a single input to the model (during
        training and validation).

        :param sample: The transformed sample to which to apply augmentations.
        :param label: The class info pertaining to ``sample``.
        :param is_training: (boolean) True if operating in training mode.
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

        :param is_training: (boolean) True if operating in training mode.
        :param batch_size: (integer) Number of input samples from the dataset to
            combine in a single batch.

        :return: A tf.data.Dataset
        """

        raise NotImplementedError(
            'make_dataset() method not implemented in derived class')

    def __call__(self, is_training, batch_size, **kwargs):
        """

        :param is_training: (boolean) True if operating in training mode.
        :param batch_size: Training batch size.
        :param kwargs: Passed as is to make_dataset() of inherited class.
        """

        return self.make_dataset(is_training, batch_size, **kwargs)

    def _apply_augmentations_and_transformations(
            self, sample, label, is_training, **kwargs):
        s_out, l_out = self.pre_transform(sample, label, is_training, **kwargs)
        s_out, l_out = self.transform(s_out, l_out, is_training, **kwargs)
        s_out, l_out = self.post_transform(s_out, l_out, is_training, **kwargs)

        return s_out, l_out

    def _queue_and_batch(self, dataset, is_training, batch_size, **kwargs):
        """Boilerplate caching, queueing and batching functionality."""

        if bool(kwargs.get('cache', False)):
            dataset = dataset.cache()

        num_prefetch_batches = kwargs.get('num_prefetch_batches', 1)
        num_threads = kwargs.get('num_threads',  # default to num CPUs
                                 os.cpu_count() or 1)

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
        dataset = dataset.map(
            lambda x, y: self._apply_augmentations_and_transformations(
                x, y, is_training),
            num_parallel_calls=num_threads)

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(num_prefetch_batches)

        return dataset

    @property
    def data_shape(self):
        """
        The shape of an input sample.
        """
        return self._shape

    def get_shape_transformation_info(self):
        """
        Override in inherited class if its :meth:`transform` alters the shape of
        the read/input data before a dataset is returned. If not None, must
        return a tuple where:

        * first value is the untransformed input shape,
        * second is the actual transformation function.
        """
        return None

    @property
    def num_classes(self):
        """
        The number of classes in the application.
        """
        return len(self._class_names)

    @property
    def class_names(self):
        """
        List of names (str) of the classes in the application.
        """
        return self._class_names

    @property
    def training_samples(self):
        """
        List of per-class training samples available.
        """
        return \
            self._num_training_samples.sum() \
            if hasattr(self._num_training_samples, '__len__') else \
            self._num_training_samples

    @property
    def training_samples_per_class(self):
        """
        List of per-class validation samples available.
        """
        if hasattr(self._num_training_samples, '__len__'):
            return self._num_training_samples
        else:
            raise ValueError('Per-class sample counts were not initialized')

    @property
    def validation_samples(self):
        """
        Total number of training samples available.
        """
        return \
            self._num_validation_samples.sum() \
            if hasattr(self._num_validation_samples, '__len__') else \
            self._num_validation_samples

    @property
    def validation_samples_per_class(self):
        """
        Total number of validation samples available.
        """
        if hasattr(self._num_validation_samples, '__len__'):
            return self._num_validation_samples
        else:
            raise ValueError('Per-class sample counts were not initialized')


class DataFeeder(BaseFeeder):
    """
    A class for loading prepared data from numpy .npz files and feeding them
    untransformed into the training/evaluation pipeline.

    :param data_dir: Directory under which prepared data (.npz files) are
        available.
    :param validation_split: (default: None) Fraction of the available data that
        must be held out for validation. If None, all available data will be
        used as training samples.
    :param min_clips_per_class: (default: None) The minimum number of per-class
        samples that must be available. If fewer samples are available for a
        class, the class will be omitted. If None, no classes will be omitted.
    :param max_clips_per_class: (default: None) The maximum number of per-class
        samples to consider among what is available, for each class. If more
        samples are available for any class, the specified number of samples
        will be randomly selected. If None, no limits will be imposed.
    :param random_state_seed: (default: None) A seed (integer) used to
        initialize the pseudo-random number generator that makes shuffling and
        other randomizing operations repeatable.
    :param cache: (optional; boolean) If True (default), the logic to 'queue &
        batch' training/evaluation samples (loaded from disk) will also cache
        the samples. Helps speed up processing.
    :param suppress_nonmax: (optional; boolean) If True, the class labels will
        be one-hot type arrays, useful for training single-class prediction
        models. Otherwise (default is False), they will be suitable for training
        multi-class prediction models, giving values in the range 0-1 for each
        class.
    """

    def __init__(self, data_dir,
                 validation_split=None,
                 min_clips_per_class=None,
                 max_clips_per_class=None,
                 random_state_seed=None,
                 **kwargs):

        assert ((validation_split is None) or (0.0 <= validation_split <= 1))

        random_state = np.random.RandomState(random_state_seed)

        with open(os.path.join(data_dir, AssetsExtraNames.classes_list),
                  'r') as f:
            class_names = json.load(f)
        num_classes = len(class_names)

        self._suppress_nonmax = kwargs.pop('suppress_nonmax', False)

        # Gather info for building dataset pipeline later
        self._valid_class_mask, self._in_shape, \
            num_per_class_samples_train, self._files_clips_idxs_train, \
            num_per_class_samples_eval, self._files_clips_idxs_eval = \
            DataFeeder.build_dataset_info(
                data_dir, num_classes,
                validation_split, random_state,
                min_clips_per_class, max_clips_per_class,
                self._suppress_nonmax
            )

        self._data_dir = data_dir
        self._cache = kwargs.pop('cache', True)

        if 'background_class' in kwargs:
            background_class = kwargs.pop('background_class')

            if background_class in class_names:
                bg_class_idx = class_names.index(background_class)
                if self._valid_class_mask[bg_class_idx]:

                    # Update 'counts' arrays
                    temp_mask = np.full(
                        (self._valid_class_mask.sum(),), True, dtype=bool)
                    temp_mask[np.sum(
                        self._valid_class_mask[:bg_class_idx + 1]) - 1] = False

                    num_per_class_samples_train = \
                        num_per_class_samples_train[temp_mask]
                    num_per_class_samples_eval = \
                        num_per_class_samples_eval[temp_mask]

                    # Update mask
                    self._valid_class_mask[bg_class_idx] = False
                else:
                    logging.info(
                        f'Class "{background_class}" already suppressed. ' +
                        'Ignoring parameter \'background_class\'.')
            else:
                logging.warning(
                    f'"{background_class}" not in list of classes. ' +
                    'Ignoring parameter \'background_class\'.')

        super(DataFeeder, self).__init__(
            self._in_shape,
            num_per_class_samples_train,
            num_per_class_samples_eval,
            [c for c, m in zip(class_names, self._valid_class_mask) if m],
            **kwargs)

    def transform(self, sample, label, is_training, **kwargs):
        # Pass as-is, nothing to do
        return sample, label

    def pre_transform(self, sample, label, is_training, **kwargs):
        # Pass as-is, nothing to do
        return sample, label

    def post_transform(self, sample, label, is_training, **kwargs):
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
        """

        num_threads = kwargs.get('num_threads',  # default to num CPUs
                                 os.cpu_count() or 1)

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
                    self._valid_class_mask,
                    self._suppress_nonmax),
                args=(file_idx, npz_file),
                output_signature=(
                    tf.TensorSpec(shape=(self._in_shape[0],),
                                  dtype=tf.float32),
                    tf.TensorSpec(shape=(sum(self._valid_class_mask),),
                                  dtype=tf.float32))
            ),
            cycle_length=interleave_cycle_length,
            num_parallel_calls=interleave_cycle_length)

        return self._queue_and_batch(dataset, is_training, batch_size,
                                     cache=self._cache,
                                     **kwargs)

    @staticmethod
    def build_dataset_info(data_dir, num_classes,
                           validation_split, random_state,
                           min_clips_per_class=None, max_clips_per_class=None,
                           suppress_nonmax=False):
        """

        :param suppress_nonmax: If True, the class labels will be one-hot type
            arrays, useful for single-class prediction models. Otherwise, they
            will be suitable for multi-class prediction models, giving values
            in the range 0-1 for each class.

        :meta private:
        """

        get_file_labels_mask = \
            DataFeeder._get_file_labels_mask_one_hot if suppress_nonmax else \
            DataFeeder._get_file_labels_mask

        per_file_class_counts = [
            np.sum(get_file_labels_mask(
                os.path.join(data_dir, file)), axis=0)
            for file in recursive_listing(
                data_dir, match_extensions=FilenameExtensions.numpy)]
        assert num_classes == per_file_class_counts[0].shape[0]

        class_files_tr, class_files_ev = [], []
        class_files_items_tr, class_files_items_ev = [], []
        useable_class_mask = np.full((num_classes, ), True, dtype=np.bool)
        for class_idx in range(num_classes):

            # Cumulative counts of class-specific clips across all files
            cumul_counts = np.cumsum(
                [flc[class_idx] for flc in per_file_class_counts])

            if cumul_counts[-1] >= (min_clips_per_class or 0):
                (cf_tr, cfi_tr), (cf_ev, cfi_ev) = \
                    DataFeeder._helper1(cumul_counts, max_clips_per_class,
                                        validation_split, random_state)

                class_files_tr.append(cf_tr)
                class_files_items_tr.append(cfi_tr)
                class_files_ev.append(cf_ev)
                class_files_items_ev.append(cfi_ev)

            else:
                # If insufficient clips, mark class for non-consideration
                useable_class_mask[class_idx] = False

        del per_file_class_counts

        valid_class_idxs = np.where(useable_class_mask)[0]
        out_num_classes = len(valid_class_idxs)

        files_clips_idxs_tr = []
        files_clips_idxs_ev = []
        per_class_samples_count_tr = np.zeros((len(valid_class_idxs), ),
                                              dtype=np.int)
        per_class_samples_count_ev = np.zeros((len(valid_class_idxs), ),
                                              dtype=np.int)
        file = None
        for f_idx, file in enumerate(
                recursive_listing(data_dir,
                                  match_extensions=FilenameExtensions.numpy)):

            label_mask = get_file_labels_mask(
                os.path.join(data_dir, file))[:, useable_class_mask]

            file_train_clips_idxs = [np.zeros((0, ), dtype=np.int)]
            file_eval_clips_idxs = [np.zeros((0, ), dtype=np.int)]
            for class_idx in range(out_num_classes):
                file_class_idxs = np.where(label_mask[:, class_idx])[0]

                e_idx = np.where(class_files_tr[class_idx] == f_idx)[0]
                if e_idx.shape[0] > 0:
                    file_train_clips_idxs.append(
                        file_class_idxs[
                            class_files_items_tr[class_idx][e_idx[0]]])
                    # Clear out after use, no longer needed
                    class_files_items_tr[class_idx][e_idx[0]] = None

                e_idx = np.where(class_files_ev[class_idx] == f_idx)[0]
                if e_idx.shape[0] > 0:
                    file_eval_clips_idxs.append(
                        file_class_idxs[
                            class_files_items_ev[class_idx][e_idx[0]]])
                    # Clear out after use, no longer needed
                    class_files_items_ev[class_idx][e_idx[0]] = None

            file_clips_idxs = np.unique(np.concatenate(file_train_clips_idxs))
            files_clips_idxs_tr.append(file_clips_idxs)
            per_class_samples_count_tr += \
                np.sum(label_mask[file_clips_idxs, :], axis=0)

            file_clips_idxs = np.unique(np.concatenate(file_eval_clips_idxs))
            files_clips_idxs_ev.append(file_clips_idxs)
            per_class_samples_count_ev += \
                np.sum(label_mask[file_clips_idxs, :], axis=0)

        # Read clip length from the first clip in the last file read above
        clip_shape = [(
            DataFeeder._get_file_clips_and_labels(
                os.path.join(data_dir, file), [0], useable_class_mask)[0]
        ).shape[1]]

        return useable_class_mask, clip_shape, \
            per_class_samples_count_tr, files_clips_idxs_tr, \
            per_class_samples_count_ev, files_clips_idxs_ev

    @staticmethod
    def file_data_generator(npz_filepath, file_clips_idxs, valid_class_mask,
                            suppress_nonmax=False):

        clips, labels = DataFeeder._get_file_clips_and_labels(
            npz_filepath, file_clips_idxs, valid_class_mask)

        if suppress_nonmax:
            # Update labels to be one-hot type, based on the max valued class
            # for each clip. If multiple classes have same max value for a clip,
            # the first of those classes will become the 'hot' item.
            labels = np.where(
                np.arange(labels.shape[1]) ==
                np.expand_dims(labels.argmax(axis=1), axis=1),
                1.0, 0.0
            ).astype(labels.dtype)

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
    def _get_file_labels_mask_one_hot(filepath):
        with np.load(filepath) as data:
            labels = data['labels']

        return (np.arange(labels.shape[1]) ==
                np.expand_dims(labels.argmax(axis=1), axis=1))

    @staticmethod
    def _get_file_clips_and_labels(filepath, clips_idxs, class_mask):
        # In the npz file, clips are stored as int16 & labels are stored as
        # float16. Convert them appropriately before returning.
        with np.load(filepath) as data:
            return Convert.pcm2float(data['clips'][clips_idxs, :]), \
                   (data['labels'][clips_idxs, :])[:, class_mask].astype(
                       np.float32)


class SpectralDataFeeder(DataFeeder):
    """
    A handy data feeder, which converts prepared audio clips into power spectral
    density spectrograms.

    :param data_dir: Directory under which prepared data (.npz files) are
        available.
    :param fs: Sampling frequency of the prepared data.
    :param spec_settings: A Python dictionary. For a list of possible keys and
        values, see parameters to
        :class:`~koogu.data.tf_transformations.Audio2Spectral`.
    :param normalize_clips: (optional; boolean) If True (default), input clips
        will be normalized before applying transform (computing spectrograms).

    Other parameters applicable to the parent :class:`DataFeeder` class may also
    be specified.
    """
    def __init__(self, data_dir, fs, spec_settings, **kwargs):

        normalize = kwargs.pop('normalize_clips', True)

        super(SpectralDataFeeder, self).__init__(data_dir, **kwargs)

        self._transformation = []
        if normalize:
            self._transformation.append(NormalizeAudio())
        self._transformation.append(Audio2Spectral(fs, spec_settings))

        # Update to what the transformed output shape would be
        self._shape = reduce(
            lambda shp, lyr: lyr.compute_output_shape([1] + shp)[1:],
            [self._in_shape] + self._transformation)

    def transform(self, clip, label, is_training, **kwargs):

        output = clip

        # Normalize (if applicable) and convert to spectrogram
        output = reduce(lambda inp, lyr: lyr(inp),
                        [output] + self._transformation)

        return output, label

    def get_shape_transformation_info(self):
        return self._in_shape, self._transformation
