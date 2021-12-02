import os
import abc
import numpy as np
import json
import logging

from koogu.data import FilenameExtensions, AssetsExtraNames
from koogu.utils.detections import assess_clips_and_labels_match, \
    assess_annotations_and_detections_match, postprocess_detections, \
    nonmax_suppress_mask
from koogu.utils.filesystem import AudioFileList


class _Metric(metaclass=abc.ABCMeta):
    """
    Base class for implementing performance assessment logic.
    """

    def __init__(self, audio_annot_list,
                 raw_results_root, annots_root,
                 reject_classes=None,
                 **kwargs):
        """
        :param audio_annot_list: A list containing pairs (tuples or sub-lists)
            of relative paths to audio files and the corresponding annotation
            (selection table) files.
        :param raw_results_root: The full paths of the raw result container
            files whose filenames will be derived from the audio files listed in
            'audio_annot_list' will be resolved using this as base directory.
        :param annots_root: The full paths of annotations files listed in
            'audio_annot_list' will be resolved using this as base directory.
        :param reject_class: Name (case sensitive) of the class (like 'Noise' or
            'Other') for which performance assessments are not to be computed.
            Can specify multiple classes for rejection, as a list.
        """

        if os.path.exists(
                os.path.join(raw_results_root, AssetsExtraNames.classes_list)):
            with open(os.path.join(raw_results_root,
                                   AssetsExtraNames.classes_list), 'r') as f:
                self._class_names = json.load(f)
        else:
            raise ValueError(f'{AssetsExtraNames.classes_list} not found in ' +
                             f'{raw_results_root}. Check if path is correct.')

        logger = logging.getLogger(__name__)

        self._raw_results_root = raw_results_root
        self._annots_root = annots_root

        # Discard invalid entries, if any
        valid_entries_mask = [
            (self._validate_seltab_filemap_lhs(lhs) and
             self._validate_seltab_filemap_rhs(rhs))
            for (lhs, rhs) in audio_annot_list]
        for entry in (e for e, e_mask in zip(audio_annot_list,
                                             valid_entries_mask) if not e_mask):
            logger.warning(
                'Entry ({:s},{:s}) is invalid. Skipping...'.format(*entry))

        if sum(valid_entries_mask) == 0:
            logger.warning('Nothing to process')
            return

        self._audio_annot_list = [
            e for e, e_mask in zip(audio_annot_list, valid_entries_mask)
            if e_mask]

        # Undocumented settings
        self._ig_kwargs = {}
        if not kwargs.pop('ignore_zero_annot_files', False):
            self._ig_kwargs['ignore_zero_annot_files'] = False
            if 'filetypes' in kwargs:
                self._ig_kwargs['filetypes'] = kwargs.pop('filetypes')
            # Need to look for files with added extension. Hidden setting.
            self._ig_kwargs['added_ext'] = FilenameExtensions.numpy

        num_classes = len(self._class_names)

        self._class_label_to_idx = {c: ci
                                    for ci, c in enumerate(self._class_names)}

        self._valid_class_mask = np.full((num_classes,), True, dtype=np.bool)
        if reject_classes is not None:
            for rj_class in ([reject_classes] if isinstance(reject_classes, str)
                             else reject_classes):
                if rj_class in self._class_names:
                    self._valid_class_mask[
                        self._class_label_to_idx[rj_class]] = False
                else:
                    logger.warning(
                        f'Reject class {rj_class:s} not found in list of ' +
                        'classes. Setting will be ignored.')

    def assess(self, show_progress=False, **kwargs):

        # kwargs will simply be passed as-is to the overridden internal methods.

        input_generator = AudioFileList.from_annotations(
            self._audio_annot_list,
            self._raw_results_root, self._annots_root,
            show_progress=show_progress,
            **self._ig_kwargs)

        if not np.all(self._valid_class_mask):
            def discard_non_valid_class_annots(times, labels, channels):
                mask = self._valid_class_mask[labels]
                return times[mask, :], labels[mask], channels[mask]
        else:
            def discard_non_valid_class_annots(times, labels, channels):
                return times, labels, channels

        self._init_containers(**kwargs)

        for audio_file, annots_times, annots_labels, annots_channels in \
            input_generator:

            # Convert textual annotation labels to integers
            annots_class_idxs = np.asarray(
                [self._class_label_to_idx[c] for c in annots_labels],
                dtype=np.uint16)

            # Keep only valid classes' entries
            annots_times, annots_class_idxs, annots_channels = \
                discard_non_valid_class_annots(
                    annots_times, annots_class_idxs, annots_channels)

            self._assess_and_accumulate(
                # Annotations info
                annots_times, annots_class_idxs, annots_channels,
                # Raw detections info (derivable from filename
                audio_file,
                **kwargs)

        result = self._produce_result(**kwargs)

        self._clear_containers(**kwargs)

        return result

    @property
    def num_classes(self):
        return len(self._class_names)

    @abc.abstractmethod
    def _init_containers(self, **kwargs):
        raise NotImplementedError(
            '_init_containers() method not implemented in derived class')

    @abc.abstractmethod
    def _clear_containers(self, **kwargs):
        raise NotImplementedError(
            '_clear_containers() method not implemented in derived class')

    @abc.abstractmethod
    def _assess_and_accumulate(self,
                               # Annotations info
                               annots_times, annots_class_idxs, annots_channels,
                               # Raw detections info (derivable from filename)
                               audio_file,
                               **kwargs):
        raise NotImplementedError(
            '_assess_and_accumulate() method not implemented in derived class')

    @abc.abstractmethod
    def _produce_result(self, **kwargs):
        raise NotImplementedError(
            '_produce_result() method not implemented in derived class')

    def load_raw_detection_info(self, audio_file):

        # Derive result file path from audio_file
        res_file_fullpath = os.path.join(
            self._raw_results_root, audio_file + FilenameExtensions.numpy)

        with np.load(res_file_fullpath) as res:
            fs = res['fs']
            clip_length = res['clip_length']
            clip_offsets = res['clip_offsets']
            scores = res['scores']
            channels = res['channels'] if 'channels' in res \
                else np.arange(scores.shape[0])

        return scores, clip_offsets, clip_length, fs, channels

    def _validate_seltab_filemap_lhs(self, entry):
        return len(entry) > 0 and (
            os.path.isdir(os.path.join(self._raw_results_root, entry)) or
            os.path.exists(
                os.path.join(self._raw_results_root,
                             entry + FilenameExtensions.numpy))
        )

    def _validate_seltab_filemap_rhs(self, entry):
        return len(entry) > 0 and os.path.isfile(
            entry if self._annots_root is None else
            os.path.join(self._annots_root, entry))


class PrecisionRecall(_Metric):
    """
    Class for assessing precision-recall values.

    When calling assess(), passing return_counts=True will return the per-class
    counts for the numerators and denominators of precision and recall.
    Otherwise, per-class and overall precision-recall values will be returned.
    """

    def __init__(self, audio_annot_list,
                 raw_results_root, annots_root,
                 reject_classes=None,
                 thresholds=None,
                 post_process_detections=False,
                 **kwargs):
        """

        :param audio_annot_list: A list containing pairs (tuples or sub-lists)
            of relative paths to audio files and the corresponding annotation
            (selection table) files.
        :param raw_results_root: The full paths of the raw result container
            files whose filenames will be derived from the audio files listed in
            'audio_annot_list' will be resolved using this as base directory.
        :param annots_root: The full paths of annotations files listed in
            'audio_annot_list' will be resolved using this as base directory.
        :param reject_class: Name (case sensitive) of the class (like 'Noise' or
            'Other') for which performance assessments are not to be computed.
            Can specify multiple classes for rejection, as a list.
        :param threshold: If not None, must be either a scalar quantity or a
            list of non-decreasing values (float values in the range 0-1) at
            which precision and recall value(s) will be assessed. If None, will
            default to the range 0-1 with an interval of 0.05.
        :param post_process_detections: If True (default is False), a
            post-processing algorithm will be applied to the raw detections
            before assessing performance.

        Other optional parameters:
        :param suppress_nonmax: If True (default is False), only the top-scoring
            class per clip will be considered.
        :param negative_class: A string label (e.g. 'Other', 'Noise'; defaults
            to None) identifying the negative class. If set, clips corresponding
            to the negative class will not be considered in. This parameter
            applies only when 'post_process_detections' is False.
        :param squeeze_min_dur: (default None). If set (duration in seconds), an
            algorithm 'to squeeze together' temporally overlapping regions from
            successive raw clips will be applied. The 'squeezing' will be
            restricted to produce detections that are at least as long as the
            specified value. The value must be smaller than the duration of the
            model inputs.

        Other parameters specific to
            - koogu.data.Process.audio2clips() (when post-processing isn't
              enabled) or
            - koogu.utils.detections.postprocess_detections() (when
              post-processing is enabled)
        can also be specified, and will be passed as-is to the respective
        functions.
        """

        if thresholds is None:  # Apply defaults
            self._thresholds = np.round(np.arange(0, 1.0 + 1e-8, 0.05), 2)
        elif not hasattr(thresholds, '__len__'):    # Scalar given
            self._thresholds = np.asarray([thresholds])  # force to be a list
        else:
            self._thresholds = thresholds   # Assume it is already a list-like

        self._pp = False
        self._negative_class_idx = None
        if post_process_detections:
            self._pp = True

            # Post-processing function kwargs
            pp_fn_kwargs = {}
            if 'suppress_nonmax' in kwargs:
                pp_fn_kwargs['suppress_nonmax'] = kwargs.pop('suppress_nonmax')
            if 'squeeze_min_dur' in kwargs:
                pp_fn_kwargs['squeeze_min_dur'] = kwargs.pop('squeeze_min_dur')

            # Match-making function kwargs
            match_fn_kwargs = {}
            if 'min_gt_coverage' in kwargs:
                match_fn_kwargs['min_gt_coverage'] = kwargs.pop(
                    'min_gt_coverage')
            if 'min_det_usage' in kwargs:
                match_fn_kwargs['min_det_usage'] = kwargs.pop('min_det_usage')

            def assessment_fn(a_times, a_classes, a_chs, audio_file, **akwargs):
                return self.assess_from_processed_scores(
                    a_times, a_classes, a_chs, audio_file,
                    pp_fn_kwargs, match_fn_kwargs, **akwargs)

        else:
            # Match-making function kwargs. Any unspecified parameters will have
            # same defaults as Process.audio2clips().
            match_fn_kwargs = {}
            if 'positive_overlap_threshold' in kwargs:
                match_fn_kwargs['min_annot_overlap_fraction'] = \
                    kwargs.pop('positive_overlap_threshold')
            if 'keep_only_centralized_annots' in kwargs:
                match_fn_kwargs['keep_only_centralized_annots'] = \
                    kwargs.pop('keep_only_centralized_annots')
            if 'negative_overlap_threshold' in kwargs:
                match_fn_kwargs['max_nonmatch_overlap_fraction'] = \
                    kwargs.pop('negative_overlap_threshold')

            if 'negative_class' in kwargs:
                self._negative_class_idx = \
                    self._class_label_to_idx[kwargs.pop('negative_class')]

            # The post-processing counterpart handles nonmax-suppression within
            # lower-level functions. For this option, we do need to handle it
            # explicitly.
            def assessment_fn(a_times, a_classes, a_chs, audio_file, **akwargs):
                return self.assess_from_raw_scores(
                    a_times, a_classes, a_chs, audio_file,
                    match_fn_kwargs,
                    suppress_nonmax=kwargs.pop('suppress_nonmax', False),
                    **akwargs)

        self._assessment_fn = assessment_fn

        # Counts containers (assessment outputs)
        self._prec_numers = None
        self._prec_denoms = None
        if post_process_detections:         # If not post-processing dets,
            self._reca_numers = None        # will be same as prec_numers.
        self._reca_denom = None

        super(PrecisionRecall, self).__init__(
            audio_annot_list, raw_results_root, annots_root, reject_classes,
            **kwargs)

    @property
    def thresholds(self):
        return self._thresholds

    def _init_containers(self, **kwargs):
        num_thlds = len(self._thresholds)
        num_valid_classes = sum(self._valid_class_mask)

        # Initialize counts containers
        self._prec_numers = np.zeros((num_thlds, num_valid_classes),
                                     dtype=np.uint)
        self._prec_denoms = np.zeros((num_thlds, num_valid_classes),
                                     dtype=np.uint)
        if self._pp:
            # same as prec_numers if not post-processing dets
            self._reca_numers = np.zeros((num_thlds, num_valid_classes),
                                         dtype=np.uint)
        self._reca_denom = np.zeros((num_valid_classes, ), dtype=np.uint)

    def _clear_containers(self, **kwargs):
        self._prec_numers = None
        self._prec_denoms = None
        if self._pp:
            # same as prec_numers if didn't post-process dets
            self._reca_numers = None
        self._reca_denom = None

    def _produce_result(self, **kwargs):

        reca_numers = self._reca_numers if self._pp else self._prec_numers
        # same as prec_numers if didn't post-process dets

        if not kwargs.get('return_counts', False):
            th_prec = np.full(self._prec_numers.shape, np.nan, dtype=np.float32)
            temp = self._prec_denoms > 0
            th_prec[temp] = self._prec_numers[temp].astype(np.float32) / \
                self._prec_denoms[temp]
            th_reca = np.full(self._prec_numers.shape, np.nan, dtype=np.float32)
            temp = self._reca_denom > 0
            th_reca[:, temp] = (reca_numers[:, temp].astype(np.float32) /
                                np.expand_dims(self._reca_denom[temp], axis=0))
            per_class_perf = {
                class_name: dict(
                    precision=th_prec[:, self._class_label_to_idx[class_name]],
                    recall=th_reca[:, self._class_label_to_idx[class_name]]
                )
                for class_name, v in zip(self._class_names,
                                         self._valid_class_mask)
                if (v and
                    self._reca_denom[self._class_label_to_idx[class_name]] > 0)
            }

            overall_perf = dict(
                precision=np.full((self._prec_numers.shape[0],), np.nan,
                                  dtype=np.float32),
                recall=np.full((self._prec_numers.shape[0],), np.nan,
                               dtype=np.float32)
            )
            denom = self._prec_denoms.sum(axis=1)
            temp = denom > 0
            overall_perf['precision'][temp] = (
                    self._prec_numers[temp, :].sum(axis=1).astype(np.float32) /
                    denom[temp])
            temp = self._reca_denom.sum()
            if temp > 0:
                overall_perf['recall'] = (
                        reca_numers.sum(axis=1).astype(np.float32) / temp)

            return per_class_perf, overall_perf

        else:
            per_class_counts = {
                class_name: dict(
                    tp=self._prec_numers[
                       :, self._class_label_to_idx[class_name]],
                    tp_plus_fp=self._prec_denoms[
                               :, self._class_label_to_idx[class_name]],
                    recall_numerator=reca_numers[
                                     :, self._class_label_to_idx[class_name]],
                    tp_plus_fn=self._reca_denom[
                        self._class_label_to_idx[class_name]]
                )
                for class_name, v in zip(self._class_names,
                                         self._valid_class_mask)
                if v
            }

            return per_class_counts

    def _assess_and_accumulate(
            self,
            # Annotations info
            annots_times, annots_class_idxs, annots_channels,
            # Raw detections info (derivable from file path)
            audio_file,
            **kwargs):
        return self._assessment_fn(
            annots_times, annots_class_idxs, annots_channels, audio_file,
            **kwargs)

    def assess_from_raw_scores(
            self,
            # Annotations info
            annots_times, annots_class_idxs, annots_channels,
            # Raw detections info (derivable from file path)
            audio_file,
            # Other info
            match_fn_kwargs_dict, suppress_nonmax, **kwargs):
        """
        Function that assesses clip-level matches from raw scores.

        The "post-processing counterpart" of this function handles
        nonmax-suppression internally within lower-level functions. For this
        function, we do need to handle it explicitly.
        """

        # Load raw detections info
        scores, clip_offsets, clip_length, fs, channels = \
            self.load_raw_detection_info(audio_file)

        if suppress_nonmax:     # Set non-max classes' scores to NaN
            scores = np.where(nonmax_suppress_mask(scores), np.nan, scores)

        annots_times_int = np.round(
            annots_times * fs).astype(clip_offsets.dtype)

        unmatched_annots_mask = \
            np.full((len(annots_class_idxs),), False, dtype=np.bool)

        for ch in channels:
            curr_ch_annots_mask = (annots_channels == ch)

            clip_class_mask, matched_annots_mask = \
                assess_clips_and_labels_match(
                    clip_offsets, clip_length,
                    self.num_classes,
                    annots_times_int[curr_ch_annots_mask, :],
                    annots_class_idxs[curr_ch_annots_mask],
                    negative_class_idx=self._negative_class_idx,
                    **match_fn_kwargs_dict)

            # Update "missed" annots mask
            unmatched_annots_mask[
                np.where(curr_ch_annots_mask)[0][
                    np.logical_not(matched_annots_mask)]] = True

            clip_class_mask = clip_class_mask[:, self._valid_class_mask]

            gt_mask = (clip_class_mask == 1)
            above_thld_mask = np.stack([
                (scores[ch, ...][:, self._valid_class_mask] >= th)
                for th in self._thresholds])

            self._prec_numers += np.logical_and(
                above_thld_mask, np.expand_dims(gt_mask, axis=0)
            ).sum(axis=1).astype(np.uint)

            self._prec_denoms += np.logical_and(
                above_thld_mask, np.expand_dims(clip_class_mask > 0, axis=0)
            ).sum(axis=1).astype(np.uint)

            self._reca_denom += gt_mask.sum(axis=0).astype(np.uint)

        # Offer a warning if there were any unmatched annotations
        if np.any(unmatched_annots_mask):
            logging.getLogger(__name__).warning(
                '{:s}: {:d} annotations unmatched [{:s}]'.format(
                    audio_file, sum(unmatched_annots_mask), ', '.join(
                        ['{:f} - {:f} (ch-{:d})'.format(
                            annots_times[annot_idx, 0],
                            annots_times[annot_idx, 1],
                            annots_channels[annot_idx] + 1)
                            for annot_idx in
                            np.where(unmatched_annots_mask)[0]]
                    )))

    def assess_from_processed_scores(
            self,
            # Annotations info
            annots_times, annots_class_idxs, annots_channels,
            # Raw detections info (derivable from file path)
            audio_file,
            # Other info
            pp_fn_kwargs_dict, match_fn_kwargs_dict, **kwargs):
        """
        Function that first post-processes detections and then assesses perf.
        """

        # Load raw detections info
        scores, clip_offsets, clip_length, fs, channels = \
            self.load_raw_detection_info(audio_file)

        pp_fn_kwargs = {k: v for k, v in pp_fn_kwargs_dict.items()}  # copy
        temp = pp_fn_kwargs.pop('squeeze_min_dur', None)
        if temp is not None:   # convert from seconds to samples
            pp_fn_kwargs['squeeze_min_samps'] = int(temp * fs)

        class_idx_remapper = np.where(self._valid_class_mask)[0]

        for ch in channels:
            curr_ch_annots_mask = (annots_channels == ch)

            for th_idx, thld in enumerate(self._thresholds):

                # Apply post-processing algorithm
                pp_times_int, pp_scores, pp_class_idxs = \
                    postprocess_detections(
                        scores[ch, ...][:, self._valid_class_mask],
                        clip_offsets, clip_length,
                        threshold=thld,
                        **pp_fn_kwargs)

                # Remap class IDs to make good for gaps from ignored class(es)
                pp_class_idxs = class_idx_remapper[pp_class_idxs]

                tp, tp_plus_fp, reca_num, _, _ = \
                    assess_annotations_and_detections_match(
                        self.num_classes,
                        annots_times[curr_ch_annots_mask, :],
                        annots_class_idxs[curr_ch_annots_mask],
                        pp_times_int.astype(annots_times.dtype) / fs,
                        pp_class_idxs,
                        **match_fn_kwargs_dict)

                self._prec_numers[th_idx] += tp[self._valid_class_mask]
                self._prec_denoms[th_idx] += tp_plus_fp[self._valid_class_mask]
                self._reca_numers[th_idx] += reca_num[self._valid_class_mask]

        for gt_c_idx in annots_class_idxs:
            self._reca_denom[gt_c_idx] += 1


__all__ = ['PrecisionRecall']
