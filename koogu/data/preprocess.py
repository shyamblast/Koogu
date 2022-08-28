
import os
import sys
import logging
import json
import concurrent.futures
from datetime import datetime
import argparse
import csv
import warnings
import numpy as np
import abc

from koogu.data import FilenameExtensions, AssetsExtraNames
from koogu.data.raw import Audio, Settings, Convert
from koogu.utils import instantiate_logging, processed_items_generator_mp
from koogu.utils.detections import SelectionTableReader, LabelHelper, \
    assess_annotations_and_clips_match
from koogu.utils.terminal import ArgparseConverters
from koogu.utils.config import Config, ConfigError, datasection2dict, log_config
from koogu.utils.filesystem import restrict_classes_with_whitelist_file, \
    AudioFileList, get_valid_audio_annot_entries, recursive_listing

_program_name = 'preprocess'


def from_selection_table_map(audio_settings, audio_seltab_list,
                             audio_root, seltab_root, output_root,
                             label_column_name=None,
                             desired_labels=None,
                             remap_labels_dict=None,
                             negative_class_label=None,
                             **kwargs):
    """
    Pre-process training data using info contained in ``audio_seltab_list``.

    :param audio_settings: A dictionary specifying the parameters for processing
        audio from files.
    :param audio_seltab_list: A list containing pairs (tuples or sub-lists) of
        relative paths to audio files and the corresponding annotation
        (selection table) files. Alternatively, you could also specify (path to)
        a 2-column csv file containing these pairs of entries (in the same
        order). Only use the csv option if the paths are simple (i.e., the
        filenames do not contain commas or other special characters).
    :param audio_root: The full paths of audio files listed in
        ``audio_seltab_list`` are resolved using this as the base directory.
    :param seltab_root: The full paths of annotations files listed in
        ``audio_seltab_list`` are resolved using this as the base directory.
    :param output_root: "Prepared" data will be written to this directory.
    :param label_column_name: A string identifying the header of the column in
        the selection table file(s) from which class labels are to be extracted.
        If None (default), will look for a column with the header "Tags".
    :param desired_labels: The target set of class labels. If not None, must be
        a list of class labels. Any selections (read from the selection tables)
        having labels that are not in this list will be discarded. This list
        will be used to populate classes_list.json that will define the classes
        for the project. If None, then the list of classes will be populated
        with the annotation labels read from all selection tables.
    :param remap_labels_dict: If not None, must be a Python dictionary
        describing mapping of class labels. For details, see similarly named
        parameter to the constructor of
        :class:`koogu.utils.detections.LabelHelper`.

        .. note:: If ``desired_labels`` is not None, mappings for which targets
           are not listed in ``desired_labels`` will be ignored.

    :param negative_class_label: A string (e.g. 'Other', 'Noise') which will be
        used as a label to identify the negative class clips (those that did not
        match any annotations). If None (default), saving of negative class
        clips will be disabled.

    Other parameters specific to
    :func:`koogu.utils.detections.assess_annotations_and_clips_match`
    can also be specified, and will be passed as-is to the function.

    :return: A dictionary whose keys are annotation tags (either discovered from
        the set of annotations, or same as ``desired_labels`` if not None) and
        the values are the number of clips produced for the corresponding class.
    """

    logger = logging.getLogger(__name__)

    audio_settings_c = Settings.Audio(**audio_settings)

    # ---------- 1. Input generator --------------------------------------------
    # Discard invalid entries, if any
    v_audio_seltab_list = get_valid_audio_annot_entries(
            audio_seltab_list, audio_root, seltab_root, logger=logger)
    if len(v_audio_seltab_list) == 0:
        print('Nothing to process')
        return {}

    classes_n_counts = annot_classes_and_counts(
        seltab_root,
        [e[-1] for e in v_audio_seltab_list],
        label_column_name or "Tags",
        **({'num_threads': kwargs['num_threads']} if 'num_threads' in kwargs
           else {})
    )

    logger.info('  {:<55s} - {:>5s}'.format('Class', 'Annotations'))
    logger.info('  {:<55s}   {:>5s}'.format('-----', '-----------'))
    for class_name in sorted(classes_n_counts.keys()):
        logger.info('  {:<55s} - {:>5d}'.format(class_name,
                                                classes_n_counts[class_name]))

    ig_kwargs = {}      # Undocumented settings
    if negative_class_label is not None:
        # Deal with these only if there was a request to save non-match clips
        if 'ignore_zero_annot_files' in kwargs:
            ig_kwargs['ignore_zero_annot_files'] = \
                kwargs.pop('ignore_zero_annot_files')
        if 'filetypes' in kwargs:
            ig_kwargs['filetypes'] = kwargs.pop('filetypes')
    input_generator = AudioFileList.from_annotations(
        v_audio_seltab_list,
        audio_root, seltab_root,
        label_column_name or "Tags",
        **ig_kwargs)

    # ---------- 2. LabelHelper ------------------------------------------------
    label_helper = LabelHelper(
        desired_labels if desired_labels else [
            lbl for lbl in classes_n_counts.keys()],
        remap_labels_dict=remap_labels_dict,
        negative_class_label=negative_class_label,
        fixed_labels=(desired_labels is not None),
        assessment_mode=False)

    # ---------- 3. Data aggregator --------------------------------------------
    # Extract args meant for assess_annotations_and_clips_match()
    match_fn_kwargs = dict()
    if 'min_annot_overlap_fraction' in kwargs:
        assert (0.0 < kwargs['min_annot_overlap_fraction'] <= 1.0)
        match_fn_kwargs['min_annot_overlap_fraction'] = \
            kwargs.pop('min_annot_overlap_fraction')
    if 'keep_only_centralized_annots' in kwargs:
        match_fn_kwargs['keep_only_centralized_annots'] = \
            kwargs.pop('keep_only_centralized_annots')
    if label_helper.negative_class_index is not None:
        match_fn_kwargs['negative_class_idx'] = \
            label_helper.negative_class_index

        if 'max_nonmatch_overlap_fraction' in kwargs:
            assert (0.0 <= kwargs['max_nonmatch_overlap_fraction'] <
                    match_fn_kwargs.get('min_annot_overlap_fraction', 1.0))
            match_fn_kwargs['max_nonmatch_overlap_fraction'] = \
                kwargs.pop('max_nonmatch_overlap_fraction')

    aggregator_kwargs = dict(
        match_fn_kwargs=match_fn_kwargs,
        attempt_salvage=kwargs.pop('attempt_salvage', False)
    )

    return _batch_process(
        audio_settings_c, input_generator, label_helper, aggregator_kwargs,
        audio_root, output_root,
        **kwargs)


def from_top_level_dirs(audio_settings, class_dirs,
                        audio_root, output_root,
                        remap_labels_dict=None,
                        **kwargs):
    """
    Pre-process training data available as audio files in ``class_dirs``.

    :param audio_settings: A dictionary specifying the parameters for processing
        audio from files.
    :param class_dirs: A list containing relative paths to class-specific
        directories containing audio files. Each directory's contents will be
        recursively searched for audio files.
    :param audio_root: The full paths of the class-specific directories listed
        in ``class_dirs`` are resolved using this as the base directory.
    :param output_root: "Prepared" data will be written to this directory.
    :param remap_labels_dict: If not None, must be a Python dictionary
        describing mapping of class labels. For details, see similarly named
        parameter to the constructor of
        :class:`koogu.utils.detections.LabelHelper`.
    :param filetypes: (optional) Restrict listing to files matching extensions
        specified in this parameter. Has defaults if unspecified.

    :return: A dictionary whose keys are annotation tags (discovered from the
        set of annotations) and the values are the number of clips produced for
        the corresponding class.
    """

    logger = logging.getLogger(__name__)

    audio_settings_c = Settings.Audio(**audio_settings)

    # ---------- 1. Input generator --------------------------------------------
    file_types = kwargs.pop('filetypes', AudioFileList.default_audio_filetypes)

    logger.info('  {:<55s} - {:>5s}'.format('Class', 'Files'))
    logger.info('  {:<55s}   {:>5s}'.format('-----', '-----'))
    for class_name in class_dirs:
        count = 0
        for _ in recursive_listing(os.path.join(audio_root, class_name),
                                   file_types):
            count += 1
        logger.info('  {:<55s} - {:>5d}'.format(class_name, count))

    input_generator = AudioFileList.from_directories(
        audio_root, class_dirs, file_types)

    # ---------- 2. LabelHelper ------------------------------------------------
    label_helper = LabelHelper(
        class_dirs,
        remap_labels_dict=remap_labels_dict,
        fixed_labels=False,
        assessment_mode=False)

    # ---------- 3. Data aggregator --------------------------------------------
    aggregator_kwargs = {}

    return _batch_process(
        audio_settings_c, input_generator, label_helper, aggregator_kwargs,
        audio_root, output_root,
        **kwargs)


def _batch_process(audio_settings, input_generator, label_helper,
                   aggregator_kwargs,
                   audio_root, dest_root,
                   **kwargs):

    logger = logging.getLogger(__name__)

    if kwargs.pop('show_progress', False):
        warnings.showwarning(
            'The parameter \'show_progress\' is deprecated and will be ' +
            'removed in a future release. Currently, the parameter is ignored' +
            ' and has no effect.',
            DeprecationWarning, __name__, '')

    t_start = datetime.now()
    logger.info('Started at: {}'.format(t_start))

    # Warn about existing output directory
    if os.path.exists(dest_root) and os.path.isdir(dest_root):
        warnings.showwarning(
            f'Output directory {dest_root} already exists. Contents may get ' +
            'overwritten. Also, stale files within the directory could lead ' +
            'corruption of training inputs.', Warning, _program_name, '')
    else:
        os.makedirs(dest_root, exist_ok=True)

    # Invoke the parallel processor
    total_per_class_clip_counts = np.zeros((len(label_helper.classes_list), ),
                                           dtype=np.uint64)
    for _, file_per_class_clip_counts in processed_items_generator_mp(
            kwargs.pop('num_threads', os.cpu_count() or 1),
            _single_threaded_single_file_preprocess,
            input_generator,
            audio_root, dest_root,
            audio_settings, label_helper, aggregator_kwargs,
            **kwargs):
        if len(file_per_class_clip_counts) > 0:
            total_per_class_clip_counts += file_per_class_clip_counts

    t_end = datetime.now()
    logger.info('Finished at: {}'.format(t_end))
    logger.info('Processing time (hh:mm:ss.ms) {}'.format(t_end - t_start))

    retval = {cl: cc
              for cl, cc in zip(label_helper.classes_list,
                                total_per_class_clip_counts)}

    logger.info('Results:')
    logger.info('  {:<55s} - {:>5s}'.format('Class', 'Clips'))
    logger.info('  {:<55s}   {:>5s}'.format('-----', '-----'))
    for class_name, clip_count in retval.items():
        logger.info('  {:<55s} - {:>5d}'.format(class_name, clip_count))

    # Write out the list of class names
    json.dump(
        label_helper.classes_list,
        open(os.path.join(dest_root, AssetsExtraNames.classes_list), 'w'))

    return retval


def _single_threaded_single_file_preprocess(
        input_generator_tuple, audio_root, dest_root,
        audio_settings, label_helper, aggregator_kwargs,
        **kwargs):
    """
    Apply pre-processing (resampling and filtering, as defined) to contents of
    an audio file, break up the resulting time-domain data into fixed-length
    clips, optionally (if annotations are available) match clips to annotations,
    and write the clips to disk along with one-hot style ground-truth labels.
    """

    (audio_file, annots_times, annots_labels, annot_channels) = \
        input_generator_tuple

    audio_file_fullpath = os.path.join(audio_root, audio_file)

    # Derive destination paths. Create directories as necessary
    rel_path, filename = os.path.split(audio_file)
    target_filename = filename + FilenameExtensions.numpy
    target_dir = os.path.join(dest_root, rel_path)
    os.makedirs(target_dir, exist_ok=True)

    # Create the appropriate GroundTruthDataAggregator instance.
    if annots_times is None:  # invoked by from_top_level_dirs()
        # if annots_labels not in label_helper.labels_to_indices:
        #     continue  # Not in desired classes. Skip. (will never happen)

        output_aggregator = GroundTruthDataAggregatorNoAnnots(
            os.path.join(target_dir, target_filename),
            len(label_helper.classes_list), audio_settings.fs,
            label_helper.labels_to_indices[annots_labels],
            **aggregator_kwargs,
            audio_filepath=audio_file)

    else:  # invoked by from_selection_table_map()
        # Based on remap_labels_dict or desired_labels, some labels may
        # become invalid. Keep only annots with valid labels.
        valid_annots_idxs = [
            idx for idx, lbl in enumerate(annots_labels)
            if lbl in label_helper.labels_to_indices]

        output_aggregator = GroundTruthDataAggregatorWithAnnots(
            os.path.join(target_dir, target_filename),
            len(label_helper.classes_list), audio_settings,
            annots_times[valid_annots_idxs, :],
            np.asarray([
                label_helper.labels_to_indices[annots_labels[idx]]
                for idx in valid_annots_idxs]),
            annot_channels[valid_annots_idxs],
            **aggregator_kwargs,
            audio_filepath=audio_file)

    return Audio.get_file_clips(
        audio_file_fullpath, audio_settings,
        **kwargs,
        labels_accumulator=output_aggregator)


def annot_classes_and_counts(seltab_root, annot_files, label_column_name,
                             **kwargs):
    """
    Query the list of annot_files to determine the unique labels present and
    their respective counts.
    Returns a dictionary mapping unique labels to respective counts.
    """

    logger = logging.getLogger(__name__)
    num_workers = kwargs['num_threads'] if 'num_threads' in kwargs else \
        max(1, os.cpu_count() - 1)

    filespec = [
        (label_column_name, str),
        ('Begin Time (s)', float),
        ('End Time (s)', float)]

    # Discard invalid entries, if any
    valid_entries_mask = [
        os.path.isfile(
            af if seltab_root is None else os.path.join(seltab_root, af))
        for af in annot_files
    ]

    if seltab_root is None:
        full_path = lambda x: x
    else:
        full_path = lambda x: os.path.join(seltab_root, x)

    futures_dict = dict()
    retval = dict()
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        for is_valid, annot_file in zip(valid_entries_mask, annot_files):
            if not is_valid:
                logger.error('File {:s} not found. Skipping entry...'.format(annot_file))
            else:
                futures_dict[executor.submit(_get_labels_counts_from_annot_file,
                                             full_path(annot_file), filespec)] = annot_file

        if len(futures_dict) == 0:
            logger.error('Nothing to process')
            return {}

        for future in concurrent.futures.as_completed(futures_dict):
            try:
                uniq_labels, label_counts = future.result()
            except Exception as ho_exc:
                logger.error('Reading file {:s} generated an exception: {:s}'.format(
                    repr(futures_dict[future]), repr(ho_exc)))
            else:
                for ul, lc in zip(uniq_labels, label_counts):
                    if ul in retval:
                        retval[ul] += lc
                    else:
                        retval[ul] = int(lc)

    return retval


def _get_labels_counts_from_annot_file(annot_filepath, filespec):
    """Helper function for annot_classes_and_counts()"""
    labels = [entry[0]
              for entry in SelectionTableReader(annot_filepath, filespec)
              if all([e is not None for e in entry])]
    return np.unique(labels, return_counts=True)


class GroundTruthDataAggregator:

    gt_type = np.float16         # Type of ground-truth label ("coverage") array
    ch_type = np.uint8           # Type of array containing channel indices
    ret_counts_type = np.uint64  # Type of per-class counts array

    def __init__(self, output_filepath, num_classes):

        self._output_filepath = output_filepath
        self._num_classes = num_classes

        # Will populate these in calls to accrue()
        self._clips = []
        self._clip_offsets = []
        self._channels = []

    @abc.abstractmethod
    def accrue(self, channel, clips, clip_offsets, channel_data):
        raise NotImplementedError(
            'accrue() method not implemented in derived class')

    @abc.abstractmethod
    def serialize(self):
        raise NotImplementedError(
            'serialize() method not implemented in derived class')

    @classmethod
    def save(cls, filepath, fs, channels, clip_offsets, clips, labels,
             normalize_clips=False):
        # Save the clips & infos
        np.savez_compressed(
            filepath,
            fs=fs,
            labels=labels,
            channels=np.concatenate([
                    np.broadcast_to(cls.ch_type(ch), (c.shape[0], ))
                    for ch, c in zip(channels, clips)]),
            clip_offsets=np.concatenate(clip_offsets),
            clips=Convert.float2pcm(    # Convert to 16-bit PCM
                np.concatenate(clips) if not normalize_clips else \
                Audio.normalize(np.concatenate(clips)),
                dtype=np.int16)
        )

    @classmethod
    def load(cls):
        """     -xxx-     Currently unimplemented     -xxx-     """
        pass


class GroundTruthDataAggregatorNoAnnots(GroundTruthDataAggregator):
    """
    All accrued clips will be associated with the class corresponding to the
    single class index ( < `num_classes`) specified by `label`.
    """

    def __init__(self, output_filepath, num_classes,
                 fs, label,
                 audio_filepath=None):

        assert (not hasattr(label, '__len__')), 'label must be a scalar index'

        self._fs = fs
        self._file_level_label = label
        self._audio_filepath = audio_filepath       # Only used for logging

        super(GroundTruthDataAggregatorNoAnnots, self).__init__(
            output_filepath, num_classes)

    def accrue(self, channel, clips, clip_offsets, _):
        self._clips.append(clips)
        self._clip_offsets.append(clip_offsets)
        self._channels.append(channel)

    def serialize(self, normalize_clips=False):

        retval = np.zeros((self._num_classes, ), dtype=self.ret_counts_type)

        num_clips = sum([c.shape[0] for c in self._clips])
        action_str = ''
        if num_clips > 0:

            # Create labels container, setting same class on all.
            # No need to create a large container; simply create one row and
            # "broadcast" to mimic replicating rows.
            labels = np.zeros((self._num_classes, ), dtype=self.gt_type)
            labels[self._file_level_label] = 1.0
            labels = np.broadcast_to(labels, (num_clips, labels.size))

            retval[self._file_level_label] = num_clips  # update retval too

            GroundTruthDataAggregator.save(
                self._output_filepath,
                self._fs,
                self._channels,
                self._clip_offsets,
                self._clips,
                labels,
                normalize_clips=normalize_clips
            )

            action_str = 'Wrote '

        logging.getLogger(__name__).info(
            ((self._audio_filepath + ': ') if self._audio_filepath else '') +
            action_str + f'{num_clips} clips')

        return retval


class GroundTruthDataAggregatorWithAnnots(GroundTruthDataAggregator):

    def __init__(self, output_filepath, num_classes,
                 audio_settings, annots_times, annots_labels, annots_channels,
                 match_fn_kwargs,
                 attempt_salvage=False,
                 audio_filepath=None):

        # assert annots_times.shape[0] == len(annots_labels)
        # assert annots_times.shape[0] == len(annots_channels)

        self._annots_times = annots_times
        self._annots_labels = annots_labels
        self._annots_channels = annots_channels
        self._audio_settings = audio_settings
        self._match_fn_kwargs = match_fn_kwargs
        self._match_fn_salvage_kwargs = None
        if attempt_salvage:
            self._match_fn_salvage_kwargs = dict()
            # During salvaging, non-matches and negatives aren't to be
            # considered. Hence, copying only these below two args.
            if 'min_annot_overlap_fraction' in match_fn_kwargs:
                self._match_fn_salvage_kwargs['min_annot_overlap_fraction'] = \
                    match_fn_kwargs['min_annot_overlap_fraction']
            if 'keep_only_centralized_annots' in match_fn_kwargs:
                self._match_fn_salvage_kwargs[
                    'keep_only_centralized_annots'] = \
                    match_fn_kwargs['keep_only_centralized_annots']

        self._audio_filepath = audio_filepath       # Only used for logging

        self._labels = []
        self._unmatched_annots_mask = \
            np.full((len(annots_labels),), False, dtype=np.bool)

        super(GroundTruthDataAggregatorWithAnnots, self).__init__(
            output_filepath, num_classes)

    def accrue(self, channel, clips, clip_offsets, channel_data):

        annots_mask = (self._annots_channels == channel)
        annots_class_idxs = self._annots_labels[annots_mask]

        annots_offsets = np.round(
            self._annots_times[annots_mask, :] * self._audio_settings.fs
        ).astype(clip_offsets.dtype)

        num_clips, num_samps = clips.shape

        keep_clips = [np.zeros((0, num_samps), dtype=clips.dtype)]
        keep_clip_offsets = [np.zeros((0, ), dtype=clip_offsets.dtype)]
        keep_labels = [np.zeros((0, self._num_classes), dtype=self.gt_type)]

        clip_class_coverage, matched_annots_mask = \
            assess_annotations_and_clips_match(
                clip_offsets, num_samps, self._num_classes,
                annots_offsets, annots_class_idxs,
                **self._match_fn_kwargs)

        # Clips having satisfactory coverage with at least one annot
        min_annot_overlap_fraction = self._match_fn_kwargs.get(
            'min_annot_overlap_fraction', 1.0)
        keep_clips_mask = np.any(
            clip_class_coverage >= min_annot_overlap_fraction, axis=1)
        # Add clips and info to collection
        if np.any(keep_clips_mask):
            keep_clips.append(clips[keep_clips_mask, :])
            keep_clip_offsets.append(clip_offsets[keep_clips_mask])
            keep_labels.append(
                clip_class_coverage[keep_clips_mask, :].astype(self.gt_type))

        # If requested, attempt to salvage any unmatched annotations
        for_salvage_annot_idxs = np.where(
            np.logical_not(matched_annots_mask))[0]

        if self._match_fn_salvage_kwargs is not None and \
                len(for_salvage_annot_idxs) > 0:

            salvaged_clips, salvaged_clip_offsets = \
                self._salvage_clips(
                    channel_data, num_samps,
                    annots_offsets[for_salvage_annot_idxs, :])

            if salvaged_clips.shape[0] > 0:
                salvaged_clip_class_coverage, s_matched_annots_mask = \
                    assess_annotations_and_clips_match(
                        salvaged_clip_offsets, num_samps, self._num_classes,
                        annots_offsets,
                        annots_class_idxs,
                        **self._match_fn_salvage_kwargs)

                # Clips having satisfactory coverage with >= 1 annot
                keep_clips_mask = np.any(
                    salvaged_clip_class_coverage >= min_annot_overlap_fraction,
                    axis=1)
                # Add clips and info to collection
                if np.any(keep_clips_mask):
                    keep_clips.append(salvaged_clips[keep_clips_mask, :])
                    keep_clip_offsets.append(
                        salvaged_clip_offsets[keep_clips_mask])
                    keep_labels.append(
                        salvaged_clip_class_coverage[keep_clips_mask, :].astype(
                            self.gt_type))

                # Update curr channel annots mask
                matched_annots_mask[for_salvage_annot_idxs] = \
                    s_matched_annots_mask[for_salvage_annot_idxs]

        # Update overall mask
        self._unmatched_annots_mask[
            np.where(annots_mask)[0][np.logical_not(
                matched_annots_mask)]] = True

        self._clips.append(np.concatenate(keep_clips, axis=0))
        self._clip_offsets.append(np.concatenate(keep_clip_offsets, axis=0))
        self._channels.append(channel)
        self._labels.append(
            self._adjust_clip_annot_coverage(
                np.concatenate(keep_labels, axis=0), min_annot_overlap_fraction)
        )

    def serialize(self, normalize_clips=False):

        logger = logging.getLogger(__name__)

        file_str = (self._audio_filepath + ': ') if self._audio_filepath else ''

        # Offer a warning about unmatched annotations, if any
        if np.any(self._unmatched_annots_mask):
            logger.warning(
                file_str + '{:d} annotations unmatched [{:s}]'.format(
                    sum(self._unmatched_annots_mask),
                    ', '.join([
                        '{:f} - {:f} (ch-{:d})'.format(
                            self._annots_times[annot_idx, 0],
                            self._annots_times[annot_idx, 1],
                            self._annots_channels[annot_idx] + 1)
                        for annot_idx in np.where(
                            self._unmatched_annots_mask)[0]
                    ])))

        num_clips = sum([c.shape[0] for c in self._clips])
        action_str = ''
        if num_clips > 0:

            all_labels = np.concatenate(self._labels, axis=0)
            retval = np.sum(all_labels == 1.0, axis=0,
                            dtype=self.ret_counts_type)

            GroundTruthDataAggregator.save(
                self._output_filepath,
                self._audio_settings.fs,
                self._channels,
                self._clip_offsets,
                self._clips,
                all_labels,
                normalize_clips=normalize_clips
            )

            action_str = 'Wrote '

        else:
            retval = np.zeros((self._num_classes,), dtype=self.ret_counts_type)

        logger.info(file_str + f'{self._annots_times.shape[0]} annotations. ' +
                    action_str + f'{num_clips} clips')

        return retval

    def _salvage_clips(self, data, clip_len, unmatched_annots_offsets):

        salvaged_clips = []
        salvaged_clip_offsets = []
        half_len = clip_len // 2

        # Gather clips corresponding to all yet-unmatched annots
        for annot_idx in range(unmatched_annots_offsets.shape[0]):
            annot_num_samps = (unmatched_annots_offsets[annot_idx, 1] -
                               unmatched_annots_offsets[annot_idx, 0]) + 1

            if annot_num_samps < clip_len:
                # If annotation is shorter than clip size, then we need to
                # center the annotation within a clip
                annot_start_samp = unmatched_annots_offsets[annot_idx, 0] + \
                                   (annot_num_samps // 2) - half_len
                annot_end_samp = annot_start_samp + clip_len - 1
            else:
                # otherwise, take full annotation extents
                annot_start_samp, annot_end_samp = \
                    unmatched_annots_offsets[annot_idx, :]

            short_clips, short_clip_offsets = Audio.buffer_to_clips(
                data[max(0, annot_start_samp):
                     min(annot_end_samp + 1, len(data))],
                self._audio_settings.clip_length,
                self._audio_settings.clip_advance,
                consider_trailing_clip=
                self._audio_settings.consider_trailing_clip
            )

            if short_clips.shape[0] > 0:
                salvaged_clips.append(short_clips)
                salvaged_clip_offsets.append(short_clip_offsets +
                                             annot_start_samp)

        if len(salvaged_clips) > 0:
            return np.concatenate(salvaged_clips, axis=0), \
                np.concatenate(salvaged_clip_offsets, axis=0)

        else:
            # Nothing could be salvaged, return empty containers
            return np.zeros((0, clip_len), dtype=data.dtype), \
                np.zeros((0,), dtype=np.int)

    @classmethod
    def _adjust_clip_annot_coverage(cls, coverage, upper_thld,
                                    lower_thld_frac=1/3):
        # Adjust "coverage":
        #  force values >= upper_thld to 1.0
        #  retain remaining values >= (lower_thld_frac * upper_thld) as is
        #  force all other small values to 0.0
        return np.select(
            [coverage >= upper_thld, coverage >= upper_thld * lower_thld_frac],
            [1.0, coverage],
            default=0.0
        )


def _instantiate_logging(args, audio_settings):
    # Create the logger
    logging.basicConfig(filename=args.log if args.log is not None else os.path.join(args.dst, _program_name + '.log'),
                        filemode='w', level=args.loglevel, format='[%(levelname).1s] %(funcName)s: %(message)s')

    logger = logging.getLogger(__name__)

    logger.info('Command-line arguments: {}'.format({k: v for k, v in vars(args).items() if v is not None}))
    logger.info('Audio settings: {}'.format(audio_settings))


# def _audio_settings_from_config(cfg_file):
#     """Load audio settings parameters from the config file and return a Settings.Audio instance"""
#
#     cfg = Config(cfg_file, 'DATA')
#
#     audio_settings = {
#         'clip_length': cfg.DATA.audio_clip_length,
#         'clip_advance': cfg.DATA.audio_clip_advance,
#         'desired_fs': cfg.DATA.audio_fs,
#         'filterspec': cfg.DATA.audio_filterspec
#     }
#
#     # Validate settings
#     _ = Settings.Audio(**audio_settings)    # Will throw, if failure. Will be caught by caller
#
#     return audio_settings


__all__ = ['from_selection_table_map', 'from_top_level_dirs']


if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog=_program_name, allow_abbrev=False,
                                     description='Prepare audio data before their conversion to TFRecords.')
    parser.add_argument('cfg', metavar='<CONFIG FILE>',
                        help='Path to config file.')
    parser.add_argument('src', metavar='<AUDIO SOURCE>',
                        help='Path to either a single audio file or to a directory. When a directory, if selection ' +
                             'table info is also provided (using \'selmap\'), then this must be the root path from ' +
                             'which relative paths to audio files in the selection tables will be resolved. ' +
                             'Otherwise, this must be the root directory containing per-class top-level ' +
                             'subdirectories which in turn contain audio files.')
    parser.add_argument('dst', metavar='<DST DIRECTORY>',
                        help='Path to destination directory into which prepared data will be written.')
    parser.add_argument('--whitelist', metavar='FILE',
                        help='Path to text file containing names (one per line) of whitelisted classes.')
    arg_group_seltab = parser.add_argument_group('Selection tables',
                                                 'Control which sections of audio files are retained in the output, ' +
                                                 'with the use of Raven selection tables.')
    arg_group_seltab.add_argument('--selmap', metavar='CSVFILE',
                                  help='Path to csv file containing one-to-one mappings from audio files to selection' +
                                  ' table files. Audio filepaths must be relative to <AUDIO_SOURCE>. If selection ' +
                                  'table files are not absolute paths, use \'selroot\' to specify the root directory ' +
                                  'path.')
    arg_group_seltab.add_argument('--selroot', metavar='DIRECTORY',
                                  help='Path to the root directory containing selection table files. Note that, if ' +
                                  'this is specified, all selection table paths in \'selmap\' file be treated as ' +
                                  'relative paths.')
    arg_group_seltab.add_argument('--accept-thld', metavar='0-100', type=ArgparseConverters.valid_percent,
                                  default=90., dest='seltab_accept_thld',
                                  help='Clips from the source audio files are retained in the output only if the ' +
                                  'percentage of their temporal overlap with any annotation in a matched selection ' +
                                  'table is above this threshold value. Default: 90%%.')
    arg_group_seltab.add_argument('--save-reject-class', dest='save_reject_class', action='store_true',
                                  help='Enable saving of clips that do not match annotations as \'other\' class. ' +
                                       'Default: False.')
    arg_group_seltab.add_argument('--reject-thld', metavar='0-100', type=ArgparseConverters.valid_percent,
                                  default=0., dest='seltab_reject_thld',
                                  help='Clips from the source audio files are retained in the output \'other\' class ' +
                                       'only if the percentage of their temporal overlap with any annotation in a ' +
                                       'matched selection table is under this threshold value. Default: 0%%.')
    arg_group_prctrl = parser.add_argument_group('Process control')
    arg_group_prctrl.add_argument('--threads', metavar='NUM', type=ArgparseConverters.positive_integer,
                                  help='Number of threads to spawn for parallel execution (default: as many CPUs).')
    arg_group_logging = parser.add_argument_group('Logging')
    arg_group_logging.add_argument('--log', metavar='FILE',
                                   help='Path to file to which logs will be written out.')
    arg_group_logging.add_argument('--loglevel', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'],
                                   default='INFO',
                                   help='Logging level.')
    arg_group_misc = parser.add_argument_group('Miscellaneous')
    arg_group_misc.add_argument('--filetypes', metavar='EXTN', nargs='+', default=AudioFileList.default_audio_filetypes,
                                help='Audio file types to restrict processing to. Option is ignored if processing ' +
                                     'selection tables or a single file. Can specify multiple types separated by ' +
                                     'whitespaces. By default, will include for processing all discovered files with ' +
                                     'the following extensions: ' + ', '.join(AudioFileList.default_audio_filetypes))
    arg_group_misc.add_argument('--maxdur', metavar='SECONDS', dest='max_file_duration', type=float,
                                help='Maximum duration of an audio file to consider it for processing. Larger files ' +
                                     'will be ignored. Default: no limit.')
    args = parser.parse_args()

    if not os.path.exists(args.src):
        print('Error: Invalid source specified', file=sys.stderr)
        exit(2)

    try:
        data_settings = datasection2dict(Config(args.cfg, 'DATA').DATA)
    except FileNotFoundError as exc:
        print('Error loading config file: {}'.format(exc.strerror), file=sys.stderr)
        exit(exc.errno)
    except ConfigError as exc:
        print('Error processing config file: {}'.format(str(exc)), file=sys.stderr)
        exit(1)
    except Exception as exc:
        print('Error processing config file: {}'.format(repr(exc)), file=sys.stderr)
        exit(1)

    if os.path.isfile(args.src):    # If src is an audio file by itself. Process and exit immediately.

        # Warn about ignoring whitelist and seltab info, if also provided
        if args.selmap is not None:
            warnings.showwarning('Processing a single file, will ignore \'selmap\'.', Warning, _program_name, '')
        if args.whitelist is not None:
            warnings.showwarning('Processing a single file, will ignore \'whitelist\'.', Warning, _program_name, '')

        num_clips, _ = Process.audio2clips(Settings.Audio(**data_settings['audio_settings']), args.src, args.dst)

        print('{:s}: {:d} clips'.format(os.path.split(args.src)[-1], num_clips[0]))

        exit(0)

    other_args = {'show_progress': False}
    if args.max_file_duration:
        other_args['max_file_duration'] = args.max_file_duration
    if args.threads:
        other_args['num_threads'] = args.threads

    instantiate_logging(args.log if args.log is not None else
                            os.path.join(args.dst, _program_name + '.log'),
                        args.loglevel, args)
    log_config(logging.getLogger(__name__), data_cfg={'audio_settings': data_settings['audio_settings']})

    exit_code = 0

    if args.selmap is not None:   # If selmap file is given, build a container with all relevant info

        with open(args.selmap, 'r', newline='') as f:
            seltab_filemap = [(entry[0], entry[1], entry[2])
                              for entry in csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONE)
                              if len(entry) == 3]
        if len(seltab_filemap) == 0:
            print('No (valid) mappings found in {:s}'.format(args.selmap), file=sys.stderr)

        else:

            other_args['positive_overlap_threshold'] = args.seltab_accept_thld / 100
            if args.save_reject_class:
                other_args['negative_overlap_threshold'] = args.seltab_reject_thld / 100

            # Warn about ignoring whitelist, if also provided
            if args.whitelist is not None:
                warnings.showwarning('Will ignore \'whitelist\' because \'selmap\' is also provided.',
                                     Warning, _program_name, '')

            from_selection_table_map(data_settings['audio_settings'],
                                     audio_seltab_list=seltab_filemap,
                                     audio_root=args.src,
                                     seltab_root=args.selroot,
                                     output_root=args.dst,
                                     **other_args)

    else:
        # If seltab info wasn't available, build the list of classes from the combination of the dir listing of src and
        # classes whitelist.

        # List of classes (first level directory names)
        try:
            class_dirs = sorted([c for c in os.listdir(args.src) if os.path.isdir(os.path.join(args.src, c))])
        except FileNotFoundError as exc:
            print('Error reading source directory: {}'.format(exc.strerror), file=sys.stderr)
            exit_code = exc.errno
        else:

            if args.whitelist is not None:  # Apply whitelist
                class_dirs = restrict_classes_with_whitelist_file(class_dirs, args.whitelist)
                print('Application of whitelist from {:s} results in {:d} classes.'.format(
                    args.whitelist, len(class_dirs)))

            if len(class_dirs) == 0:
                print('No classes to process.')

            else:

                if args.filetypes:
                    other_args['filetypes'] = args.filetypes

                from_top_level_dirs(data_settings['audio_settings'], class_dirs, args.src, args.dst, **other_args)

    logging.shutdown()

    exit(exit_code)
