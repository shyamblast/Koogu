
import os
import logging
import json
from datetime import datetime
import warnings
import numpy as np
import abc

from koogu.data import FilenameExtensions, AssetsExtraNames
from koogu.data.raw import Audio, Convert
from koogu.utils import processed_items_generator_mp
from koogu.utils.detections import assess_annotations_and_clips_match


def batch_process(audio_settings, input_generator, label_helper,
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
    logger.debug('Started at: {}'.format(t_start))

    # Warn about existing output directory
    if os.path.exists(dest_root) and os.path.isdir(dest_root):
        warnings.showwarning(
            f'Output directory {dest_root} already exists. Contents may get ' +
            'overwritten. CAUTION: Stale files within the directory could ' +
            'lead to corruption of training inputs.',
            Warning, 'preprocess', '')
    else:
        os.makedirs(dest_root, exist_ok=True)

    # Invoke the parallel processor
    total_per_class_clip_counts = np.zeros(
        (len(label_helper.classes_list), ),
        dtype=GroundTruthDataAggregator.ret_counts_type)
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
    logger.debug('Finished at: {}'.format(t_end))
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

    :param ignore_zero_annot_files: Parameter is only considered when processing
        selmap pairs. Where the audio reference in selmap points to an audio
        file and the corresponding annot file contains zero annotations, a True
        value in this parameter (default) will cause the function to skip
        processing of the audio file. When the audio reference in selmap points
        to a directory instead, a True value in this parameter will cause the
        function to skip processing of discovered audio files for which there
        were no valid annotations. In either scenario, a False value would only
        result in processing of audio files without valid annotations when the
        ``label_helper`` has a defined negative_class_idx.

    :meta private:
    """

    (audio_file, annots_times, annots_labels, annot_channels) = \
        input_generator_tuple

    audio_file_fullpath = os.path.join(audio_root, audio_file)

    # Derive destination paths
    rel_path, filename = os.path.split(audio_file)
    target_filename = filename + FilenameExtensions.numpy
    target_dir = os.path.join(dest_root, rel_path)

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

        if len(valid_annots_idxs) == 0 and (
                label_helper.negative_class_index is None or
                kwargs.pop('ignore_zero_annot_files', True)):
            # Nothing to save, so no need to even load the audio file
            return np.zeros((len(label_helper.classes_list), ),
                            dtype=GroundTruthDataAggregator.ret_counts_type)

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

    # Create destination directories as necessary
    os.makedirs(target_dir, exist_ok=True)

    try:
        retval = Audio.get_file_clips(
            audio_file_fullpath, audio_settings,
            **kwargs,
            labels_accumulator=output_aggregator)
    except Exception as exc:
        logging.getLogger(__name__).error(
            f'Failure loading audio file {audio_file}: {repr(exc)}.' + (
                '' if (annots_times is None or len(annots_times) == 0) else
                f' Discarding {len(annots_times)} corresponding annotations.'))
        retval = np.zeros((len(label_helper.classes_list), ),
                          dtype=GroundTruthDataAggregator.ret_counts_type)

    return retval


def get_unique_labels_from_annotations(
        annot_root, annot_files, annotation_reader,
        num_threads=None):
    """
    Query the list of annot_files to determine the unique labels present.

    Returns:
        - a list of discovered unique labels
        - a mask of invalid entries in `annot_files`
    """

    if annot_root is None:
        fp_to_af = {af: af
                    for af in annot_files}
    else:
        fp_to_af = {os.path.join(annot_root, af): af
                    for af in annot_files}

    classes_n_counts = {}
    invalid_entries_map = {af: True
                           for af in annot_files}

    for pr_fp, (status, (_, _, tags, _, _)) in processed_items_generator_mp(
            num_threads or max(1, (os.cpu_count() or 1) - 1),
            annotation_reader.safe_fetch,
            (fp for fp in fp_to_af.keys()),
            multi_file=False):

        if status:
            uniq_labels, label_counts = np.unique(tags, return_counts=True)
            for ul, lc in zip(uniq_labels, label_counts):
                if ul in classes_n_counts:
                    classes_n_counts[ul] += lc
                else:
                    classes_n_counts[ul] = int(lc)

            invalid_entries_map[fp_to_af[pr_fp]] = False

    return \
        classes_n_counts, \
        [invalid_entries_map[af] for af in annot_files]


class GroundTruthDataAggregator:
    """
    Abstract base class, for implementing functionality to aggregate clips and
    labels whilst loading individual channels from an audio file.
    To accumulate, one must call accrue() for each channel, and subsequently
    call serialize() to write the accumulated data to disk storage.

    :meta private:
    """

    gt_type = np.float16         # Type of ground-truth label ("coverage") array
    ch_type = np.uint8           # Type of array containing channel indices
    ret_counts_type = np.uint32  # Type of per-class counts array

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

    :meta private:
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

        logging.getLogger(__name__).debug(
            ((self._audio_filepath + ': ') if self._audio_filepath else '') +
            action_str + f'{num_clips} clips')

        return retval


class GroundTruthDataAggregatorWithAnnots(GroundTruthDataAggregator):
    """
    Accrued clips will be assigned a per-class "coverage" score based on their
    temporal overlaps with available annotations.

    :meta private:
    """

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
            np.full((len(annots_labels),), False, dtype=bool)

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

        logger.debug(file_str + f'{self._annots_times.shape[0]} annotations. ' +
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
                np.zeros((0,), dtype=int)

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

