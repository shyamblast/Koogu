import os
import numpy as np
import json
import logging

from koogu.data import FilenameExtensions, AssetsExtraNames
from koogu.utils.detections import assess_clips_and_labels_match
from koogu.utils.filesystem import AudioFileList


def precision_recall_raw_data(
        audio_annot_list,
        raw_results_root, annots_root,
        negative_class_label=None,
        reject_class=None,
        threshold=None,
        return_counts=None,
        **kwargs):
    """
    Assess precision and recall values.

    :param audio_annot_list: A list containing pairs (tuples or sub-lists) of
        relative paths to audio files and the corresponding annotation
        (selection table) files.
    :param raw_results_root: The full paths of the raw result container files
        whose filenames will be derived from the audio files listed in
        'audio_annot_list' will be resolved using this as the base directory.
    :param annots_root: The full paths of annotations files listed in
        'audio_annot_list' are resolved using this as the base directory.
    :param negative_class_label: A string (e.g. 'Other', 'Noise') identifying
        the negative class (if any; otherwise defaults to None). Clips
        corresponding to the negative class will not be considered in
        assessments if set to None.
    :param reject_class: Name (case sensitive) of the class (like 'Noise' or
        'Other') for which performance assessments are not to be computed. Can
        specify multiple classes for rejection, as a list.
    :param threshold: If not None, must be either a scalar quantity or a list
        of non-decreasing values (float values in the range 0-1) at which
        precision and recall value(s) will be assessed. If None, will default
        to the range 0-1 with a granularity of 0.1.
    :param return_counts: If set to True, instead of returning precision and
        recall values, will return numerator and denominator counts for
        computing precision and recall values.

    :return: Returns a 3-tuple, the first of which is the input/default
        'threshold' and the other two are precision and recall values (in that
        order). The contents of the second and third values will be -
          - scalars if 'threshold' was scalar, or
          - numpy arrays of the same length as 'threshold' if 'threshold' was
            list-like, returns a 2-tuple.
    """

    if os.path.exists(os.path.join(raw_results_root,
                                   AssetsExtraNames.classes_list)):
        with open(os.path.join(raw_results_root,
                               AssetsExtraNames.classes_list), 'r') as f:
            class_names = json.load(f)
    else:
        raise ValueError(f'{AssetsExtraNames.classes_list} not found in ' +
                         f'{raw_results_root}. Check if path is correct.')

    # Discard invalid entries, if any
    valid_entries_mask = [
        (_validate_seltab_filemap_lhs(raw_results_root, lhs) and
         _validate_seltab_filemap_rhs(annots_root, rhs))
        for (lhs, rhs) in audio_annot_list]
    for entry in (e for e, e_mask in zip(audio_annot_list,
                                         valid_entries_mask) if not e_mask):
        print('Entry ({:s},{:s}) is invalid. Skipping...'.format(*entry))

    if sum(valid_entries_mask) == 0:
        print('Nothing to process')
        return [], {}, {}

    # Undocumented settings
    ig_kwargs = {}
    if not kwargs.pop('ignore_zero_annot_files', False):
        ig_kwargs['ignore_zero_annot_files'] = False
        if 'filetypes' in kwargs:
            ig_kwargs['filetypes'] = kwargs.pop('filetypes')
        # Need to look for files with added extension
        ig_kwargs['added_ext'] = FilenameExtensions.numpy   # hidden setting
    input_generator = AudioFileList.from_annotations(
        [e for e, e_mask in zip(audio_annot_list, valid_entries_mask)
         if e_mask],
        raw_results_root, annots_root,
        show_progress=False,
        **ig_kwargs)

    # Defaults for parameters (if unspecified), same as Process.audio2clips()
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

    num_classes = len(class_names)
    class_label_to_idx = {c: ci for ci, c in enumerate(class_names)}

    negative_class_idx = None if negative_class_label is None \
        else class_label_to_idx[negative_class_label]

    valid_class_mask = np.full((num_classes,), True, dtype=np.bool)
    if reject_class is not None:
        for rj_class in ([reject_class] if isinstance(reject_class, str)
                         else reject_class):
            if rj_class in class_names:
                valid_class_mask[class_label_to_idx[rj_class]] = False
            else:
                logging.getLogger(__name__).warning(
                    f'Reject class {rj_class:s} not found in list of ' +
                    'classes. Setting will be ignored.')

    if threshold is None:
        thld = np.round(np.arange(0.0, 1.0 + 1e-8, 0.1), 1)
    elif not hasattr(threshold, '__len__'):
        thld = [threshold]     # force to be a list
    else:
        thld = threshold

    # Initialize counts containers
    th_prec_denoms = [np.zeros((sum(valid_class_mask), ), dtype=np.uint)
                      for _ in thld]
    th_tps = [np.zeros((sum(valid_class_mask), ), dtype=np.uint)
              for _ in thld]
    reca_denom = np.zeros((sum(valid_class_mask), ), dtype=np.uint)

    for audio_file, annots_times, annots_labels, annots_channels in \
            input_generator:

        res_file_fullpath = os.path.join(
            raw_results_root, audio_file + FilenameExtensions.numpy)

        with np.load(res_file_fullpath) as res:
            res_fs = res['fs']
            res_clip_length = res['clip_length']
            res_clip_offsets = res['clip_offsets']
            res_scores = res['scores'][:, :, valid_class_mask]
            res_channels = res['channels'] if 'channels' in res \
                else np.arange(res_scores.shape[0])

        annots_times_int = np.round(
            annots_times * res_fs).astype(res_clip_offsets.dtype)
        annots_class_idxs = np.asarray(
            [class_label_to_idx[c] for c in annots_labels])

        # Keep only valid classes' entries
        if reject_class is not None:
            temp = np.asarray(
                [valid_class_mask[a_c_idx] for a_c_idx in annots_class_idxs],
                dtype=np.bool)
            annots_times = annots_times[temp, :]
            annots_times_int = annots_times_int[temp, :]
            annots_class_idxs = annots_class_idxs[temp]
            annots_channels = annots_channels[temp]

        unmatched_annots_mask = \
            np.full((len(annots_class_idxs),), False, dtype=np.bool)

        for ch_idx, ch in enumerate(res_channels):

            curr_ch_annots_mask = (annots_channels == ch)

            clip_class_mask, matched_annots_mask = \
                assess_clips_and_labels_match(
                    res_clip_offsets, res_clip_length,
                    num_classes, annots_times_int[curr_ch_annots_mask, :],
                    annots_class_idxs[curr_ch_annots_mask],
                    negative_class_idx=negative_class_idx,
                    **match_fn_kwargs)

            # Update "missed" annots mask
            unmatched_annots_mask[
                np.where(curr_ch_annots_mask)[0][
                    np.logical_not(matched_annots_mask)]] = True

            gt_mask = (clip_class_mask[:, valid_class_mask] == 1)
            reca_denom += gt_mask.sum(axis=0).astype(np.uint)

            for th_idx, th in enumerate(thld):
                th_mask = (res_scores[ch_idx, :, :] >= th)

                th_prec_denoms[th_idx] += np.logical_and(
                    th_mask, clip_class_mask[:, valid_class_mask] > 0
                ).sum(axis=0).astype(np.uint)
                th_tps[th_idx] += np.logical_and(
                    th_mask, gt_mask
                ).sum(axis=0).astype(np.uint)

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

    th_prec_denoms = np.stack(th_prec_denoms)
    th_tps = np.stack(th_tps)

    if not return_counts:
        th_prec = np.full(th_tps.shape, np.nan, dtype=np.float32)
        temp = th_prec_denoms > 0
        th_prec[temp] = th_tps[temp].astype(np.float32) / th_prec_denoms[temp]
        th_reca = np.full(th_tps.shape, np.nan, dtype=np.float32)
        temp = reca_denom > 0
        th_reca[:, temp] = (th_tps[:, temp].astype(np.float32) /
                            np.expand_dims(reca_denom[temp], axis=0))
        overall_perf = dict(
            precision=(th_tps.sum(axis=1).astype(np.float32) /
                       th_prec_denoms.sum(axis=1)),
            recall=th_tps.sum(axis=1).astype(np.float32) / reca_denom.sum()
        )
        per_class_perf = {
            class_name: dict(
                precision=th_prec[:, class_label_to_idx[class_name]],
                recall=th_reca[:, class_label_to_idx[class_name]]
            )
            for class_name, v in zip(class_names, valid_class_mask)
            if v and reca_denom[class_label_to_idx[class_name]] > 0
        }

        return thld, overall_perf, per_class_perf

    else:
        per_class_counts = {
            class_name: dict(
                tp=th_tps[:, class_label_to_idx[class_name]],
                tp_plus_fp=th_prec_denoms[:, class_label_to_idx[class_name]],
                tp_plus_fn=reca_denom[class_label_to_idx[class_name]]
            )
            for class_name, v in zip(class_names, valid_class_mask)
            if v
        }

        return thld, per_class_counts


def _validate_seltab_filemap_lhs(raw_dets_root, entry):
    return len(entry) > 0 and (
        os.path.isdir(os.path.join(raw_dets_root, entry)) or
        os.path.exists(
            os.path.join(raw_dets_root, entry + FilenameExtensions.numpy))
    )


def _validate_seltab_filemap_rhs(annot_root, entry):
    return len(entry) > 0 and os.path.isfile(
        os.path.join(annot_root, entry) if annot_root is not None
        else entry)
