
import numpy as np
from collections import Generator
import csv


def _squeeze_streak(starts, scores, num_samples, group_size):
    """Internal helper function for combine_streaks().
    Adjusts the start-end times of successive detections of the same label to contain the maximal overlapping duration
    and aggregates the scores."""

    # Find maximal overlapping regions within a streak of detections and group those detections together.
    # identify the group extents and aggregate scores within each grouping
    grp_start_det_idxs = np.arange(min(group_size, len(starts)) - 1, len(starts))
    grp_end_det_idxs = np.arange(len(starts) - min(group_size, len(starts)) + 1)
    group_extents = np.stack([starts[grp_start_det_idxs], starts[grp_end_det_idxs] + num_samples - 1]).T
    group_extent_idxs = np.stack([grp_start_det_idxs, grp_end_det_idxs]).T
    group_scores = np.asarray([np.max(scores[st_idx:(en_idx + 1)])
                               for st_idx, en_idx in zip(grp_end_det_idxs, grp_start_det_idxs)])

    # Now combine successive maximal overlapping groups if they are contiguous (or also further overlapping)
    contiguous_groups_mask = (group_extents[:-1, 1] + 1) >= group_extents[1:, 0]
    contiguous_groups_onsets = np.where(
        np.concatenate([contiguous_groups_mask[0:1],
                        np.logical_and(np.logical_not(contiguous_groups_mask[:-1]), contiguous_groups_mask[1:])])
        )[0]
    contiguous_groups_ends = np.where(
        np.concatenate([np.logical_and(contiguous_groups_mask[:-1], np.logical_not(contiguous_groups_mask[1:])),
                        contiguous_groups_mask[-1:]]))[0] + 1

    # Find non-contiguous groups, if any
    noncontiguous_groups_mask = np.full((group_extents.shape[0], ), True)
    noncontiguous_groups_mask[[
        d_idx for s_idx, e_idx in zip(contiguous_groups_onsets, contiguous_groups_ends)
        for d_idx in range(s_idx, e_idx + 1)]] = False

    # Combine results of both contiguous groups and non-contiguous ones
    group_extent_idxs = np.concatenate([
        np.stack([group_extent_idxs[contiguous_groups_onsets, 1], group_extent_idxs[contiguous_groups_ends, 0]]).T,
        group_extent_idxs[noncontiguous_groups_mask, ...]],
        axis=0)
    group_extents = np.concatenate([
        np.stack([group_extents[contiguous_groups_onsets, 0], group_extents[contiguous_groups_ends, 1]]).T,
        group_extents[noncontiguous_groups_mask, ...]],
        axis=0)
    group_scores = np.concatenate([
        [np.median(group_scores[s_idx:e_idx+1])
         for s_idx, e_idx in zip(contiguous_groups_onsets, contiguous_groups_ends)],
        group_scores[noncontiguous_groups_mask]],
        axis=0)

    return group_extents, group_scores, np.sort(group_extent_idxs, axis=1)


def combine_streaks(det_scores, clip_start_samples, num_samples, squeeze_min_len=None, return_idxs=False):
    """
    Combine together groupings of successive independent detections.
    :param det_scores: An [N x M] array containing M per-class scores for each of the N clips.
    :param clip_start_samples: An N-length integer array containing indices of the first sample in each clip.
    :param num_samples: Number of samples in each clip.
    :param squeeze_min_len: If not None, will run the algorithm to squish contiguous detections of the same class.
        Squeezing will be limited to produce detections that are at least squeeze_min_len samples long.
    :return:
        A tuple containing sample idxs (array of start and end pairs), aggregated scores, class IDs and, if requested,
        start-end indices making up each combined streak.
    """

    assert squeeze_min_len is None or squeeze_min_len <= num_samples

    good_dets_mask = np.logical_not(np.isnan(det_scores))   # Only take valid score clips

    # Find the extents of every streak
    streak_class_idxs, streak_onset_idxs = np.where(
        np.concatenate([good_dets_mask[0:1, :],
                        np.logical_and(np.logical_not(good_dets_mask[:-1, :]), good_dets_mask[1:, :])]).T)
    _, streak_end_idxs = np.where(
        np.concatenate([np.logical_and(good_dets_mask[:-1, :], np.logical_not(good_dets_mask[1:, :])),
                        good_dets_mask[-1:, :]]).T)

    num_detections = len(streak_class_idxs)
    if num_detections == 0:
        return np.zeros((0, 2), dtype=np.uint64), np.zeros((0,), dtype=np.float), streak_class_idxs

    if squeeze_min_len is not None:
        max_num_overlapping_clips = \
            1 + (clip_start_samples[1:] <= (clip_start_samples[0] + (num_samples - squeeze_min_len))).sum()

        ret_samp_extents = list()
        ret_extents = list()
        ret_scores = list()
        ret_class_idxs = list()
        for idx in range(num_detections):
            str_st_idx = streak_onset_idxs[idx]
            str_en_idx = streak_end_idxs[idx] + 1
            c_samp_exts, c_scores, c_exts = _squeeze_streak(clip_start_samples[str_st_idx:str_en_idx],
                                       det_scores[str_st_idx:str_en_idx, streak_class_idxs[idx]],
                                       num_samples, max_num_overlapping_clips)

            ret_samp_extents.append(c_samp_exts)
            ret_extents.append(c_exts + str_st_idx)
            ret_scores.append(c_scores)
            ret_class_idxs.append(np.full((len(c_scores),), streak_class_idxs[idx]))

        ret_samp_extents = np.concatenate(ret_samp_extents, axis=0)
        ret_extents = np.concatenate(ret_extents, axis=0)
        ret_scores = np.concatenate(ret_scores, axis=0)
        streak_class_idxs = np.concatenate(ret_class_idxs, axis=0)
    else:
        ret_samp_extents = np.asarray(
            [[clip_start_samples[streak_onset_idxs[idx]], clip_start_samples[streak_end_idxs[idx]] + num_samples - 1]
             for idx in range(num_detections)], dtype=np.uint64)
        ret_extents = np.asarray(
            [[streak_onset_idxs[idx], streak_end_idxs[idx]]
             for idx in range(num_detections)], dtype=np.uint64)
        ret_scores = np.asarray(
            [np.max(det_scores[streak_onset_idxs[idx]:(streak_end_idxs[idx] + 1), streak_class_idxs[idx]])
             for idx in range(num_detections)])

    if return_idxs:
        return ret_samp_extents, ret_scores, streak_class_idxs, ret_extents
    else:
        return ret_samp_extents, ret_scores, streak_class_idxs


class SelectionTableReader(Generator):
    """
    A generator for reading Raven selection tables. A simple, fast yet efficient way for processing selection tables.
    Pass in the path to the file and a list containing field specifications, and retrieve table entries iteratively.
    A field specification must be a tuple containing the name of the field (column header), the corresponding data
    type, and optionally, a default value. The field names (including case) should match the actual column headers in
    the selection table file. The generator returns a tuple containing each entry's fields in the same order as that of
    the provided fields specifications list. If no matching column header is found for a field, the respective value in
    the output will be None.
    Example:
        >>> fields_spec = [('Selection', int, 0),
        ...                ('Begin Time (s)', float, 0),
        ...                ('Tags', str),
        ...                ('Score', float)]
        ...
        >>> for entry in SelectionTableReader('something_something.selection.txt', fields_spec):
        ...     print(entry[0], entry[1], entry[2], entry[3])
    """

    def __init__(self, seltab_file, fields_spec, delimiter='\t'):
        """

        :param seltab_file: Path to a selection table file.
        :param fields_spec: A list of field specifiers. Read description above for details.
        :param delimiter: (Optional; default = '\t') The delimiter in the selection table file.
        """

        self._delimiter = delimiter
        self._conversion_spec = [None] * len(fields_spec)

        self._seltab_file_h = open(seltab_file, 'r', newline='')
        self._csv_reader = csv.reader(self._seltab_file_h, delimiter=delimiter)

        col_headers = next(self._csv_reader)

        for f_idx, field_spec in enumerate(fields_spec):
            if field_spec[0] in col_headers:
                self._conversion_spec[f_idx] = (col_headers.index(field_spec[0]),
                                                field_spec[1],
                                                field_spec[2] if len(field_spec) > 2 else None)

    @staticmethod
    def _convert(entry, out_type, default_val):
        return out_type(entry) if entry is not '' else default_val

    # internal function
    def send(self, ignored_arg):
        entry = next(self._csv_reader)
        return tuple([SelectionTableReader._convert(entry[c_spec[0]], c_spec[1], c_spec[2])
                      if c_spec is not None else None
                      for c_spec in self._conversion_spec])

    # internal function
    def throw(self, type=None, value=None, traceback=None):
        raise StopIteration

    def __del__(self):
        self._seltab_file_h.close()


def assess_clips_and_labels_match(
        clip_offsets, clip_len,
        num_classes, annots_times_int, annots_class_idxs,
        negative_class_idx=None,
        min_annot_overlap_fraction=1.0,
        keep_only_centralized_annots=False,
        max_nonmatch_overlap_fraction=0.0):
    """
    Match clips to annotations and return info about 'clips matching
    annotations' and about 'matched annotations'.

    :param clip_offsets: Start samples of clips.
    :param clip_len: Number of samples in each clip.
    :param num_classes: Number of classes in the given application.
    :param annots_times_int: A numpy array (shape Nx2) of start-end pairs
        defining annotations' temporal extents, in terms of sample indices.
    :param annots_class_idxs: An N-length list of zero-based indices to the
        class corresponding to each annotation.
    :param negative_class_idx: If not None, clips that do not have enough
        overlap with any annotation will be marked as clips of the non-target
        class whose index this parameter specifies.
    :param min_annot_overlap_fraction: Lower threshold on how much overlap
        a clip must have with an annotation.
    :param keep_only_centralized_annots: For very short annotations, consider
        only those overlapping clips that have the annotation occurring within
        the central 50% extents of the clip.
    :param max_nonmatch_overlap_fraction: A clip without enough overlap with
        any annotations will be marked as non-target class only if it's
        overlap with any annotation is less than this amount.

    :return: A 2-tuple.
        - PxN matrix mask (but as unsigned int instead of bool) corresponding
          to P clips and N annotations. The values in the matrix will be:
            0   if there was a partial overlap (within the limits)
            1   if there was full (or as much requested) coverage
            2   if there was no (or less than requested) overlap
          Typically, only cells with values 0 or 1 will be used for preparing
          training data. Cells with value 2 are most useful during testing,
          for assessing 'TP+FP' counts.
        - Mask (bool) of annotations that were matched with at least one clip.
    """

    ret_mask = np.full((len(clip_offsets), num_classes), 0, dtype=np.uint8)
    matched_annots_mask = np.full(
        (annots_times_int.shape[0], ), False, dtype=np.bool)

    annots_num_samps = (annots_times_int[:, 1] - annots_times_int[:, 0]) + 1
    pos_overlap_thld = np.maximum(
        np.round(annots_num_samps *
                 min_annot_overlap_fraction).astype(clip_offsets.dtype),
        1)
    neg_overlap_thld = np.ceil(
        annots_num_samps *
        max_nonmatch_overlap_fraction).astype(clip_offsets.dtype)

    half_len = clip_len // 2
    quarter_len = clip_len // 4

    for c_idx in np.arange(num_classes):
        class_annots_mask = (annots_class_idxs == c_idx)  # curr class annots

        # Mx# grid of num common samples between each of the M clips and the #
        # annotations from the current class
        overlaps_samps = np.stack([
            (np.minimum(annots_times_int[class_annots_mask, 1], clip_e) -
             np.maximum(annots_times_int[class_annots_mask, 0], clip_s)) + 1
            for clip_s, clip_e in zip(clip_offsets,
                                      clip_offsets + clip_len - 1)])

        # Initialize a Mx# matrix with:
        #   2  if no (or small, if requested) overlap, or
        #   0  if there is any (or above the requested thld) overlap.
        clip_class_annots_mask = np.where(
            overlaps_samps < np.expand_dims(
                neg_overlap_thld[class_annots_mask], axis=0),
            np.uint8(2), np.uint8(0))

        # Mask of 'sufficiently' matching pairs in the above Mx# matrix
        pos_matches_mask = np.logical_or(
            # occurs when annot is longer than clip
            overlaps_samps == clip_len,
            # if min fraction of annot is covered
            overlaps_samps >= np.expand_dims(
                pos_overlap_thld[class_annots_mask], axis=0)
        )

        # If requested, for very short annotations, turn off 'match' mask if
        # they don't occur within the central 50% of a clip.
        if keep_only_centralized_annots:
            temp = np.logical_and(
                np.any(pos_matches_mask, axis=0),
                annots_num_samps[class_annots_mask] < half_len)

            for l_annot_idx, g_annot_idx in zip(
                    np.where(temp)[0], np.where(class_annots_mask)[0][temp]):
                ov_clip_idxs = np.where(pos_matches_mask[:, l_annot_idx])[0]
                temp2 = np.logical_not(np.logical_and(
                    clip_offsets[ov_clip_idxs] + quarter_len <=
                    annots_times_int[g_annot_idx, 0],
                    clip_offsets[ov_clip_idxs] + clip_len - 1 - quarter_len >=
                    annots_times_int[g_annot_idx, 1]))

                pos_matches_mask[ov_clip_idxs[temp2], l_annot_idx] = False

        # Update "matched" conditions in the Mx# matrix
        clip_class_annots_mask[pos_matches_mask] = np.uint8(1)

        # Update mask for curr annots that have at least one True in the
        # respective column
        matched_annots_mask[class_annots_mask] = \
            np.any(pos_matches_mask, axis=0)

        ret_mask[np.all(clip_class_annots_mask == 2, axis=1), c_idx] = 2
        ret_mask[np.any(clip_class_annots_mask == 1, axis=1), c_idx] = 1

    if negative_class_idx is not None:
        non_neg_mask = np.full((num_classes,), True, dtype=np.bool)
        non_neg_mask[negative_class_idx] = False

        ret_mask[
            np.logical_not(np.any(ret_mask[:, non_neg_mask] == 0, axis=1)),
            negative_class_idx] = 2
        ret_mask[np.all(ret_mask[:, non_neg_mask] == 2, axis=1),
                 negative_class_idx] = 1

    return ret_mask, matched_annots_mask


def assess_annotations_and_detections_match(
        num_classes,
        gt_times, gt_labels,
        det_times, det_labels,
        min_gt_coverage=0.75,
        min_det_usage=0.75):
    """
    Match elements describing time-spans from two collections. Typically, one
    collection corresponds to ground-truth (gt) temporal extents and the other
    collection corresponds to detection (det) temporal extents.

    :param num_classes: Number of classes of the various time-events.
    :param gt_times: Mx2 numpy array representing the start-end times of M
        ground-truth events.
    :param gt_labels: M-length integer array indicating the class of each of
        the M ground-truth events.
    :param det_times: Nx2 numpy array representing the start-end times of N
        detection events.
    :param det_labels: N-length integer array indicating the class of each of
        the N detection events.
    :param min_gt_coverage: A floating point value (in the range 0-1)
        indicating the minimum fraction of a ground-truth event that must be
        covered by one or more detections for it to be considered "recalled".
    :param min_det_usage: A floating point value (in the range 0-1)
        indicating the minimum fraction of a detection event that must have
        covered parts of one or more ground-truth events for it to be
        considered a "true positive".

    :return: A 5-tuple.
        - per-class counts of true positives
        - per-class counts of detections (true + false positives)
        - numerator for computing recall (not that given our definition of
          'true positive' and 'recall', this value may not the same as the
          per-class counts of true positives).
        - mask of ground-truth events that were "recalled"
        - mask of detections that were true positives
    """

    tps = np.zeros((num_classes, ), dtype=np.uint)
    tp_plus_fp = np.zeros((num_classes, ), dtype=np.uint)
    reca_numerator = np.zeros((num_classes, ), dtype=np.uint)
    tp_mask = np.full((len(det_labels), ), False, dtype=np.bool)
    recall_mask = np.full((len(gt_labels), ), False, dtype=np.bool)

    for c_idx in np.arange(num_classes):
        # Current class GTs and dets
        class_gts_mask = (gt_labels == c_idx)
        class_dets_mask = (det_labels == c_idx)

        tp_plus_fp[c_idx] = class_dets_mask.astype(np.uint).sum()

        recall_mask[class_gts_mask] = [
            _coverage(gt_ext, det_times[class_dets_mask, :]) >= (
                    (gt_ext[1] - gt_ext[0]) * min_gt_coverage)
            for gt_ext in gt_times[class_gts_mask, :]]

        reca_numerator[c_idx] = \
            recall_mask[class_gts_mask].astype(np.uint).sum()

        tp_mask[class_dets_mask] = [
            _coverage(det_ext, gt_times[class_gts_mask, :]) >= (
                    (det_ext[1] - det_ext[0]) * min_det_usage)
            for det_ext in det_times[class_dets_mask, :]]

        tps[c_idx] = tp_mask[class_dets_mask].astype(np.uint).sum()

    return tps, tp_plus_fp, reca_numerator, recall_mask, tp_mask


def _coverage(base_ext, items):
    """
    Ascertain how much of base_ext (a 2-element array-like specifying temporal
    extents) is covered by entries in items (a Nx2 numpy array specifying
    start and end times of N items).
    Internal helper function.
    """

    if items.shape[0] == 0:
        return 0.0

    s_mask = items[:, 0] <= base_ext[0]
    e_mask = items[:, 1] >= base_ext[1]
    if not (np.any(s_mask) or np.any(e_mask)):
        # All items are contained within the base extents. So, parts of the
        # base outside the extremities of the outermost items cannot be
        # "covered". Only work with what's inside of those extremities.
        return _coverage([items[:, 0].min(), items[:, 1].max()], items)

    # One or more items either start before or end after the base.

    # Determine new base start & end. Could remain same if no item cuts across
    # the base's extents, or move inward if any item(s) do cut across. If
    # there is some change, that will be counted towards "coverage".
    nb_start = max(items[s_mask, 1].max(initial=base_ext[0]), base_ext[0])
    nb_end = min(items[e_mask, 0].min(initial=base_ext[1]), base_ext[1])

    if nb_start >= nb_end:  # everything was "covered"
        return base_ext[1] - base_ext[0]

    return (nb_start - base_ext[0]) + (base_ext[1] - nb_end) + \
        _coverage(
            [nb_start, nb_end],
            items[np.logical_and(items[:, 1] > nb_start,
                                 items[:, 0] < nb_end), :])


def postprocess_detections(clip_scores, clip_offsets, clip_length,
                           threshold=None,
                           suppress_nonmax=False,
                           squeeze_min_samps=None):
    """
    Post-process detections to group together successive detections from each
    class.
    """

    # Apply non-max suppression, if enabled, and the threshold.
    # Only build a mask for now.
    if threshold is not None and suppress_nonmax:
        nan_mask = np.logical_or(clip_scores < threshold,
                                 _nonmax_suppress_mask(clip_scores))
    elif threshold is not None:
        nan_mask = (clip_scores < threshold)
    elif suppress_nonmax:
        nan_mask = _nonmax_suppress_mask(clip_scores)
    else:
        nan_mask = np.full_like(clip_scores, False, dtype=np.bool)

    return combine_streaks(np.where(nan_mask, np.nan, clip_scores),
                           clip_offsets, clip_length,
                           squeeze_min_len=squeeze_min_samps)


def _nonmax_suppress_mask(scores):

    nonmax_mask = np.full(scores.shape, True, dtype=np.bool)
    nonmax_mask[np.arange(scores.shape[0]), scores.argmax(axis=1)] = False

    return nonmax_mask
