
import numpy as np
from collections import Generator
import csv


def write_csv(rawlabels_filepath, det_times, det_scores, labels=None):
    """
    Write out detections [defined by det_times (an Nx2 array of start and end times) and det_scores (N-length array)] in
    raw CSV format to rawlabels_filepath. The output file will have 3 or 4 columns depending on whether labels is None.
    """

    with open(rawlabels_filepath, 'w') as outfile:
        if labels is not None:
            outfile.write('Begin time (s),End time (s),Score,Label\n')

            for idx in range(det_times.shape[0]):
                outfile.write('{:.3f},{:.3f},{:.2f},{:s}\n'.format(
                    det_times[idx, 0], det_times[idx, 1], det_scores[idx], labels[idx]))
        else:
            outfile.write('Begin time (s),End time (s),Score\n')

            for idx in range(det_times.shape[0]):
                outfile.write('{:.3f},{:.3f},{:.2f}\n'.format(
                    det_times[idx, 0], det_times[idx, 1], det_scores[idx]))


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
    contiguous_groups_onsets = np.asarray(np.where(
        np.concatenate([contiguous_groups_mask[0:1],
                        np.logical_and(np.logical_not(contiguous_groups_mask[:-1]), contiguous_groups_mask[1:])])
        )).ravel()
    contiguous_groups_ends = np.asarray(np.where(
        np.concatenate([np.logical_and(contiguous_groups_mask[:-1], np.logical_not(contiguous_groups_mask[1:])),
                        contiguous_groups_mask[-1:]]))).ravel() + 1

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

    good_dets_mask = det_scores > 0.0   # Only take non-zero score clips

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


def match_detections_to_groundtruth(gt_times, det_times, overlap_thld=0.6,
                                    gt_labels=None, det_labels=None,
                                    det_scores=None, score_thlds=None,
                                    return_gt_matches=False):
    """

    :param gt_times: An Nx2 numpy array containing start (col 1) and end (col 2) times of ground truth annotations.
    :param det_times: An Mx2 numpy array containing start (col 1) and end (col 2) times of detections.
    :param overlap_thld: A detection is considered a True Positive (TP) if the amount of overlap (fraction) with a
        matching annotation's temporal extents is at least this value.
    :param gt_labels: If not None, must be a list of N ground truth labels. If None, label comparisons will be disabled
        and instead all ground truth annotations and detections will be considered to bear the same label.
    :param det_labels: If not None, must be a list of M detection labels that will be compared to ground truth labels.
    :param det_scores: If not None, must be a list of M detection scores.
    :param score_thlds: If not None, can be a single value or a list of values that are applied to det_scores matching.
    :param return_gt_matches: If True, the returned tuple will also include a list of N arrays, each containing indices
        to matched (if any) detections in det_times.
    :return:
        A 2- or 3-tuple containing
            TP mask - an M-length boolean array indicating which of the detections matched ground truth annotations,
            stats - a Px3 table. If det_scores and score_thlds are not None, P will equal the number of elements in
                score_thlds, otherwise P will be 1. Each row includes [num GTs detected, num detections, num TPs]
                determined at the respective thresholds (or overall values if score comparisons were not enabled).
            GT matches - When enabled, a nested list of matching indices. See description for return_gt_matches.
    """

    assert len(gt_times.shape) == 2 and gt_times.shape[1] == 2, '\'gt_times\' must be an Nx2 array'
    assert len(det_times.shape) == 2 and det_times.shape[1] == 2, '\'gt_times\' must be an Mx2 array'
    assert gt_labels is None or len(gt_labels) == gt_times.shape[0]
    assert det_labels is None or len(det_labels) == det_times.shape[0]
    assert gt_labels is None or det_labels is not None, 'If valid gt_labels is given, det_labels must also be valid'
    assert det_scores is None or len(det_scores) == det_times.shape[0]

    gt_matches = [None] * gt_times.shape[0]

    if gt_labels is not None:
        # Convert labels to integers for quicker comparisons (works even if they were already integers)
        temp = {uniq_label: label_idx
                for label_idx, uniq_label in enumerate(list(set(gt_labels).union(set(det_labels))))}
        # the above variable contains a mapping of unique labels (from both sets gt_ & det_ combined) to an index
        gt_int_labels = np.asarray([temp[label] for label in gt_labels])
        det_int_labels = np.asarray([temp[label] for label in det_labels])
        del temp

        label_compare = lambda a, a_idx, b: np.asarray([a[a_idx] == b_val for b_val in b])
    else:
        # Not comparing labels
        gt_int_labels = None
        det_int_labels = None
        label_compare = lambda a, a_idx, b: np.full((det_times.shape[0], ), True, dtype=np.bool)

    gt_durations = gt_times[:, 1] - gt_times[:, 0]
    det_durations = det_times[:, 1] - det_times[:, 0]

    assert all(gt_durations >= 0) and all(det_durations >= 0), 'Durations can\'t be negative'

    for gt_idx in range(gt_times.shape[0]):

        # Temporal overlap with each detection (will be negative for detections without any overlap)
        overlap_durations = np.minimum(det_times[:, 1], gt_times[gt_idx, 1]) - \
                            np.maximum(det_times[:, 0], gt_times[gt_idx, 0])
        overlap_fractions = (
                overlap_durations /
                np.where(np.logical_and(det_times[:, 0] >= gt_times[gt_idx, 0], det_times[:, 1] <= gt_times[gt_idx, 1]),
                         det_durations, gt_durations[gt_idx]))  # if annot fully contains det, divide by det duration
                #np.minimum(det_durations, gt_durations[gt_idx]))

        # Mask of detections that have enough overlap and have matching labels
        matches_mask = np.logical_and(overlap_fractions >= overlap_thld,
                                      label_compare(gt_int_labels, gt_idx, det_int_labels))

        # Store the indices to matched detections
        gt_matches[gt_idx] = np.asarray(np.where(matches_mask)).ravel()

    all_matching_dets = np.unique(np.concatenate(gt_matches)) if gt_times.shape[0] > 0 \
        else np.zeros((0,), dtype=np.int)

    # True Positives (TP) mask. Will be set to True where a detection matches at least one ground truth annotation.
    tp_mask = np.full((det_times.shape[0], ), False, dtype=np.bool)
    tp_mask[all_matching_dets] = True

    if det_scores is None or score_thlds is None:
        # If detection scores weren't given or no thresholds available for comparisons, provide overall stats
        stats = np.zeros((1, 3), dtype=np.uint64)
        stats[0, :] = [
            sum([len(gt_m) > 0 for gt_m in gt_matches]),                # num GTs detected (not always == TP)
            det_times.shape[0],                                         # num detections (TP + FP)
            sum(tp_mask)]                                               # num TPs

    else:
        if not hasattr(score_thlds, '__len__'):
            score_thlds = [score_thlds]     # Force to be a list if not already a list-like container

        det_scores = np.asarray(det_scores)     # Change to be a numpy array if not already

        stats = np.zeros((len(score_thlds), 3), dtype=np.uint64)
        for thld_idx, score_thld in enumerate(score_thlds):
            score_mask = (det_scores >= score_thld)

            stats[thld_idx, :] = [
                sum([any(score_mask[gt_m]) for gt_m in gt_matches]),    # num GTs detected @ threshold (not always = TP)
                score_mask.sum(),                                       # num detections @ threshold (TP + FP)
                (score_mask[tp_mask]).sum()]                            # num TPs @ threshold

    return (tp_mask, stats, gt_matches) if return_gt_matches else (tp_mask, stats)


def match_clip_detections_to_groundtruth(clip_starts, clip_dur,
                                         gt_extents, gt_class_ids,
                                         clip_det_scores,
                                         score_thlds,
                                         fn_lenient_frac_thld=0.50,
                                         centralize_match=None):
    """

    :param clip_starts: An M-length list containing start times of clips.
    :param clip_dur: Duration (in seconds) of each clip.
    :param gt_extents: An Nx2 numpy array containing start (col 1) and end (col 2) times of ground truth annotations.
    :param gt_class_ids: Must be a list of N ground truth class IDs (as integer indices in the range [0, P-1]
        corresponding to P classes).
    :param clip_det_scores: Must be an MxP numpy array containing per-clip per-class detection scores.
    :param score_thlds: A S-length list of monotonically increasing threshold values that are applied to
        clip_det_scores for matching.
    :param fn_lenient_frac_thld: A clip having lesser overlap (than this threshold) with an annotation will not be
        considered as FN when it doesn't have a matching detection. Expressed as a fraction of the clip duration and
        must be in the range [0.0, 1.0).
    :param centralize_match: When an annotation is shorter in duration than a concurrent clip which doesn't have a
        corresponding matching detection, the clip will not be considered as FN if the annotation didn't occur
        "centralized" within the clip. The parameter value, in the range (0.0, 1.0], defines the fractional duration
        around it's center of the clip within which an annotation must be fully contained. An annotation is considered
        "centralized" if it satisfies the above condition or if it spans across the temporal mid-point of the clip.
        Setting the parameter to None disables this feature.

    :return:
        SxP-sized counts of TP, FP and FN
    """

    assert len(gt_extents.shape) == 2 and gt_extents.shape[1] == 2
    assert gt_extents.shape[0] == len(gt_class_ids)
    assert len(clip_det_scores.shape) == 2
    assert len(clip_starts) == clip_det_scores.shape[0]
    assert max(gt_class_ids) <= clip_det_scores.shape[1]
    assert 0.0 <= fn_lenient_frac_thld < 1.0
    assert centralize_match is None or 0.0 < centralize_match <= 1.0

    num_clips, num_classes = clip_det_scores.shape

    clip_starts = np.asarray(clip_starts)
    clip_ends = clip_starts + clip_dur
    gt_class_ids = np.asarray(gt_class_ids)

    # MxN array of temporal overlap amounts expressed as a fraction.
    # The denominators will either be -
    #   annot duration if annotation is fully contained within a clip, or
    #   clip duration otherwise.
    # Resulting values will be <= clip_dur and, where clips have no overlap with a gt, will be non-positive.
    overlaps = np.stack([
        (np.minimum(gt_extents[:, 1], clip_e) - np.maximum(gt_extents[:, 0], clip_s))
        for clip_s, clip_e in zip(clip_starts, clip_ends)])

    # MxP mask array indicating clips having any overlap with one or more gt annotations per class
    clip_class_mask = np.stack([
        (overlaps[:, gt_class_ids == cl_idx] > 0.0).any(axis=1)
        for cl_idx in range(num_classes)
    ], axis=1)

    # MxN mask array indicating whether a gt is fully contained within a clip
    containment_mask = np.stack([
        np.logical_and(clip_s <= gt_extents[:, 0], clip_e >= gt_extents[:, 1])
        for clip_s, clip_e in zip(clip_starts, clip_ends)])

    if centralize_match is not None:
        diff = clip_dur * ((1.0 - centralize_match) / 2.0)
        # Update mask array to further indicate whether a gt is either
        #   fully contained within reduced bounds inside of a clip, or
        #   spans across the midpoint of the clip.
        containment_mask = \
            np.logical_and(containment_mask,
                           np.stack([
                               np.logical_or(
                                   np.logical_and(clip_s <= gt_extents[:, 0], clip_e >= gt_extents[:, 1]),
                                   np.logical_and(clip_m >= gt_extents[:, 0], clip_m <= gt_extents[:, 1])
                                   ) for clip_s, clip_e, clip_m in zip(clip_starts + diff,
                                                                       clip_ends - diff,
                                                                       clip_starts + (clip_dur / 2.0))])
                           )

    clip_gt_lenient_mask = np.logical_or(
        overlaps > (fn_lenient_frac_thld * clip_dur), containment_mask)

    # MxP mask array indicating clips having necessary minimum overlap with one or more gt annotations per class
    clip_class_lenient_mask = np.stack([
        clip_gt_lenient_mask[:, gt_class_ids == cl_idx].any(axis=1)
        for cl_idx in range(num_classes)
    ], axis=1)

    # This will be an SxMxP mask
    score_dets_mask = np.stack([(clip_det_scores >= score_thld) for score_thld in score_thlds])

    # Return SxP-sized counts of TP, FP and FN
    return \
        np.logical_and(score_dets_mask, np.expand_dims(clip_class_mask, axis=0)).sum(axis=1), \
        np.logical_and(score_dets_mask, np.expand_dims(np.logical_not(clip_class_mask), axis=0)).sum(axis=1), \
        np.logical_and(np.logical_not(score_dets_mask), np.expand_dims(clip_class_lenient_mask, axis=0)).sum(axis=1)
