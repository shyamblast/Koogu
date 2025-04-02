import numpy as np
from warnings import showwarning


def _squeeze_streak(starts, scores, num_samples, group_size):
    """
    Internal helper function for combine_streaks().
    Adjusts the start-end times of successive detections of the same label to
    contain the maximal overlapping duration and aggregates the scores.
    """

    # Find maximal overlapping regions within a streak of detections and group
    # those detections together.
    # Identify the group extents and aggregate scores within each grouping
    grp_start_det_idxs = np.arange(min(group_size, len(starts)) - 1,
                                   len(starts))
    grp_end_det_idxs = np.arange(len(starts) - min(group_size, len(starts)) + 1)
    group_extents = np.stack([starts[grp_start_det_idxs],
                              starts[grp_end_det_idxs] + num_samples - 1]).T
    group_extent_idxs = np.stack([grp_start_det_idxs, grp_end_det_idxs]).T
    group_scores = np.asarray([
        np.max(scores[st_idx:(en_idx + 1)])
        for st_idx, en_idx in zip(grp_end_det_idxs, grp_start_det_idxs)
    ])

    # Now combine successive maximal overlapping groups if they are contiguous
    # (or also further overlapping)
    contiguous_groups_mask = (group_extents[:-1, 1] + 1) >= group_extents[1:, 0]
    contiguous_groups_onsets = np.where(np.concatenate([
        contiguous_groups_mask[0:1],
        np.logical_and(np.logical_not(contiguous_groups_mask[:-1]),
                       contiguous_groups_mask[1:])
    ]))[0]
    contiguous_groups_ends = np.where(np.concatenate([
        np.logical_and(contiguous_groups_mask[:-1],
                       np.logical_not(contiguous_groups_mask[1:])),
        contiguous_groups_mask[-1:]
    ]))[0] + 1

    # Find non-contiguous groups, if any
    noncontiguous_groups_mask = np.full((group_extents.shape[0], ), True)
    noncontiguous_groups_mask[[
        d_idx
        for s_idx, e_idx in zip(contiguous_groups_onsets,
                                contiguous_groups_ends)
        for d_idx in range(s_idx, e_idx + 1)
    ]] = False

    # Combine results of both contiguous groups and non-contiguous ones
    group_extent_idxs = np.concatenate([
        np.stack([group_extent_idxs[contiguous_groups_onsets, 1],
                  group_extent_idxs[contiguous_groups_ends, 0]]).T,
        group_extent_idxs[noncontiguous_groups_mask, ...]],
        axis=0)
    group_extents = np.concatenate([
        np.stack([group_extents[contiguous_groups_onsets, 0],
                  group_extents[contiguous_groups_ends, 1]]).T,
        group_extents[noncontiguous_groups_mask, ...]],
        axis=0)
    group_scores = np.concatenate([
        [
            np.median(group_scores[s_idx:e_idx+1])
            for s_idx, e_idx in zip(contiguous_groups_onsets,
                                    contiguous_groups_ends)
        ],
        group_scores[noncontiguous_groups_mask]],
        axis=0)

    return group_extents, group_scores, np.sort(group_extent_idxs, axis=1)


def combine_streaks(det_scores, clip_start_samples, num_samples,
                    squeeze_min_len=None, return_idxs=False):
    """
    Combine together groupings of successive independent detections.

    :param det_scores: An [N x M] array containing M per-class scores for each
        of the N clips.
    :param clip_start_samples: An N-length integer array containing indices of
        the first sample in each clip.
    :param num_samples: Number of samples in each clip.
    :param squeeze_min_len: If not None, will run the algorithm to squish
        contiguous detections of the same class.
        Squeezing will be limited to produce detections that are at least
        squeeze_min_len samples long.
    :return:
        A tuple containing sample idxs (array of start and end pairs),
        aggregated scores, class IDs and, if requested, start-end indices making
        up each combined streak.

    :meta private:
    """

    assert squeeze_min_len is None or squeeze_min_len <= num_samples

    # Only take valid score clips
    good_dets_mask = np.logical_not(np.isnan(det_scores))

    # Find the extents of every streak
    streak_class_idxs, streak_onset_idxs = np.where(np.concatenate([
        good_dets_mask[0:1, :],
        np.logical_and(np.logical_not(good_dets_mask[:-1, :]),
                       good_dets_mask[1:, :])
    ]).T)
    _, streak_end_idxs = np.where(np.concatenate([
        np.logical_and(good_dets_mask[:-1, :],
                       np.logical_not(good_dets_mask[1:, :])),
        good_dets_mask[-1:, :]
    ]).T)

    num_detections = len(streak_class_idxs)
    if num_detections == 0:
        return np.zeros((0, 2), dtype=np.uint64), \
            np.zeros((0,), dtype=det_scores.dtype), \
            streak_class_idxs

    if squeeze_min_len is not None:
        max_num_overlapping_clips = 1 + (
                clip_start_samples[1:] <= (
                    clip_start_samples[0] + (num_samples - squeeze_min_len))
        ).sum()

        ret_samp_extents = list()
        ret_extents = list()
        ret_scores = list()
        ret_class_idxs = list()
        for idx in range(num_detections):
            str_st_idx = streak_onset_idxs[idx]
            str_en_idx = streak_end_idxs[idx] + 1
            c_samp_exts, c_scores, c_exts = _squeeze_streak(
                clip_start_samples[str_st_idx:str_en_idx],
                det_scores[str_st_idx:str_en_idx, streak_class_idxs[idx]],
                num_samples,
                max_num_overlapping_clips)

            ret_samp_extents.append(c_samp_exts)
            ret_extents.append(c_exts + str_st_idx)
            ret_scores.append(c_scores)
            ret_class_idxs.append(
                np.full((len(c_scores),), streak_class_idxs[idx]))

        ret_samp_extents = np.concatenate(ret_samp_extents, axis=0)
        ret_extents = np.concatenate(ret_extents, axis=0)
        ret_scores = np.concatenate(ret_scores, axis=0)
        streak_class_idxs = np.concatenate(ret_class_idxs, axis=0)
    else:
        ret_samp_extents = np.asarray(
            [
                [clip_start_samples[streak_onset_idxs[idx]],
                 clip_start_samples[streak_end_idxs[idx]] + num_samples - 1]
                for idx in range(num_detections)
            ],
            dtype=np.uint64)
        ret_extents = np.asarray(
            [[streak_onset_idxs[idx], streak_end_idxs[idx]]
             for idx in range(num_detections)], dtype=np.uint64)
        ret_scores = np.asarray([
            np.max(
                det_scores[
                    streak_onset_idxs[idx]:(streak_end_idxs[idx] + 1),
                    streak_class_idxs[idx]]
            )
            for idx in range(num_detections)
        ])

    if return_idxs:
        return ret_samp_extents, ret_scores, streak_class_idxs, ret_extents
    else:
        return ret_samp_extents, ret_scores, streak_class_idxs


def assess_annotations_and_clips_match(
        clip_offsets, clip_len,
        num_classes, annots_times, annots_class_idxs,
        min_annot_overlap_fraction=1.0,
        keep_only_centralized_annots=False,
        negative_class_idx=None,
        max_nonmatch_overlap_fraction=0.0):
    """
    Match clips to annotations and return "coverage scores" and a mask of
    'matched annotations'. Coverage score is a value between 0.0 and 1.0 and
    describes how much of a particular class' annotation(s) is/are covered by
    each clip.

    :param clip_offsets: M-length array of start samples (offset from the start
        of the audio file) of M clips.
    :param clip_len: Number of waveform samples in each clip.
    :param num_classes: Number of classes in the given application.
    :param annots_times: A numpy array (shape Nx2) of start-end pairs defining
        annotations' temporal extents, in terms of sample indices.
    :param annots_class_idxs: An N-length list of zero-based indices to the
        class corresponding to each annotation.
    :param min_annot_overlap_fraction: Lower threshold on how much coverage
        a clip must have with an annotation for the annotation to be considered
        "matched".
    :param keep_only_centralized_annots: If enabled (default: False), very short
        annotations (< half of ``clip_len``) will generate full coverage (1.0)
        only if they occur within the central 50% extents of the clip or if the
        annotation cuts across the center of the clip. For short annotations
        that do not satisfy these conditions, their normally-computed coverage
        value will be scaled down based on the annotation's distance from the
        center of the clip.
    :param negative_class_idx: If not None, clips that do have no (or small)
        overlap with any annotation will be marked as clips of the non-target
        class whose index this parameter specifies. See
        ``max_non_match_overlap_fraction`` for further control.
    :param max_nonmatch_overlap_fraction: A clip without enough overlap with
        any annotations will be marked as non-target class only if its
        overlap with any annotation is less than this amount (default 0.0). This
        parameter is only used when ``negative_class_idx`` is set.

    :return: A 2-element tuple containing -

      * MxP "coverage" matrix corresponding to the M clips and P classes. The
        values in the matrix will be:

        | 1.0   - if either the m-th clip fully contained an annotation from the
        |         p-th class or vice versa (possible when annotation is longer
        |         than ``clip_len``);
        | <1.0  - if there was partial coverage (the number of overlapping
        |         samples is divided by the shorter of ``clip_len`` or
        |         annotation length);
        | 0.0   - if the m-th clip had no overlap with any annotations from the
        |         p-th class.

      * N-length boolean mask of annotations that were matched with at least
        one clip under the condition of ``min_annot_overlap_fraction``.
    """

    assert negative_class_idx is None or negative_class_idx < num_classes

    coverage = np.full((len(clip_offsets), num_classes), 0.0, dtype=np.float16)
    matched_annots_mask = np.full((annots_times.shape[0], ), False, dtype=bool)

    neg_clips_mask = None
    if negative_class_idx is not None:
        neg_clips_mask = np.full((len(clip_offsets), ), True, dtype=bool)

    clip_central_extents = None
    clip_centers = None
    if keep_only_centralized_annots:
        clip_central_extents = np.stack(
            [clip_offsets + (clip_len // 4),
             clip_offsets + clip_len - 1 - int(np.ceil(clip_len / 4))], axis=1)
        clip_centers = clip_offsets + ((clip_len - 1) / 2)

    for c_idx in np.arange(num_classes):
        class_annots_mask = (annots_class_idxs == c_idx)  # curr class annots

        if not np.any(class_annots_mask):
            continue

        class_annots_coverage, c_neg_clips_mask = \
            _compute_clips_annots_coverage(clip_offsets, clip_len,
                                           annots_times[class_annots_mask, :],
                                           negative_class_idx,
                                           max_nonmatch_overlap_fraction,
                                           clip_central_extents, clip_centers)

        if negative_class_idx is not None:
            neg_clips_mask = np.logical_and(neg_clips_mask, c_neg_clips_mask)

        # Update "matched" mask for curr annots that have enough "coverage"
        matched_annots_mask[class_annots_mask] = \
            np.any(class_annots_coverage >= min_annot_overlap_fraction, axis=0)

        # Update clips' "class coverage"
        coverage[:, c_idx] = np.max(class_annots_coverage, axis=1)

    if negative_class_idx is not None:
        coverage[:, negative_class_idx] = np.where(neg_clips_mask, 1.0, 0.0)

    return coverage, matched_annots_mask


def assess_annotations_and_detections_match(
        num_classes,
        gt_times, gt_labels,
        det_times, det_labels,
        min_gt_coverage=0.5,
        min_det_usage=0.5):
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

    :return: A 5-element tuple containing -

        - per-class counts of true positives
        - per-class counts of detections (true + false positives)
        - numerator for computing recall (note that given our definition of
          'true positive' and 'recall', this value may not be the same as the
          per-class counts of true positives).
        - mask of ground-truth events that were "recalled"
        - mask of detections that were true positives
    """

    tps = np.zeros((num_classes, ), dtype=np.uint)
    tp_plus_fp = np.zeros((num_classes, ), dtype=np.uint)
    reca_numerator = np.zeros((num_classes, ), dtype=np.uint)
    tp_mask = np.full((len(det_labels), ), False, dtype=bool)
    recall_mask = np.full((len(gt_labels), ), False, dtype=bool)

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


def _compute_clips_annots_coverage(clip_offsets, clip_len,
                                   annots_times,
                                   negative_class_idx=None,
                                   max_nonmatch_overlap_fraction=0.0,
                                   clip_central_extents=None,
                                   clip_centers=None):
    """
    Ascertain coverage of annotations (start & end times specified in
    'annots_times') by clips (specified by 'clip_offsets' & 'clip_len').
    'clip_central_extents' and 'clip_centers' are derived values from clips and
    must be pre-computed.

    Returns:
         NxM matrix (N clips, M annotations) describing the "coverage"
         N-length mask of non-matched clips (if negative_class_idx isn't None)

    :meta private:
    """

    annots_num_samps = (annots_times[:, 1] - annots_times[:, 0]) + 1
    coverage_denom = np.where(clip_len >= annots_num_samps,
                              annots_num_samps, clip_len)

    # Mx# grid of num common samples between each of the M clips and the #
    # annotations from the current class
    overlaps_samps = np.maximum(0, np.stack([
        (np.minimum(annots_times[:, 1], clip_e) -
         np.maximum(annots_times[:, 0], clip_s)) + 1
        for clip_s, clip_e in zip(clip_offsets, clip_offsets + clip_len - 1)]))

    # Compute "coverage" as:
    #   (the number common samples between each of the M clips and the
    #    annotations from the current class), divided by
    #   (the longer one between annotation length or clip length)
    annots_coverage = (
            overlaps_samps / np.expand_dims(coverage_denom, axis=0))

    # If requested, for very short annotations, penalize measured "coverage"
    # if they don't occur within the central 50% of a clip.
    if clip_central_extents is not None and clip_centers is not None:
        short_annot_l_mask = annots_num_samps < (clip_len // 2)

        for l_annot_idx in np.where(short_annot_l_mask)[0]:
            # Short annot that neither lies completely within the central
            # half of clips nor cuts across the clips' centers.
            # This is a mask of the clips that satisfy the conditions.
            penalize_mask = np.logical_not(
                np.logical_or(
                    # Annot with both extents within clips' central halves
                    np.logical_and(
                        annots_times[l_annot_idx, 0] >=
                        clip_central_extents[:, 0],
                        annots_times[l_annot_idx, 1] <=
                        clip_central_extents[:, 1]),
                    # Annot cuts across the center points of clips
                    np.logical_and(
                        annots_times[l_annot_idx, 0] < clip_centers,
                        annots_times[l_annot_idx, 1] > clip_centers)
                ))

            # Apply penalty.
            # The above mask will have True also for clips that didn't have
            # any overlap. It's okay to apply penalty there since they will
            # have zero "coverage" anyway.
            annots_coverage[penalize_mask, l_annot_idx] *= (1 - (
                    np.minimum(  # Distance from clip center to the nearest edge
                        np.abs(clip_centers[penalize_mask] -
                               annots_times[l_annot_idx, 0]),
                        np.abs(clip_centers[penalize_mask] -
                               annots_times[l_annot_idx, 1])
                    ) / (clip_len // 2)))

    # Invalidate neg_clips_mask entries where a clip has more than allowed
    # overlap with one or more annotations. Note that for this purpose, the
    # denominator is always clip_len, regardless of which is longer.
    neg_clips_mask = None
    if negative_class_idx is not None:
        neg_clips_mask = np.all(
            (overlaps_samps / clip_len) <= max_nonmatch_overlap_fraction,
            axis=1)

    return annots_coverage, neg_clips_mask


def _coverage(base_ext, items):
    """
    Ascertain how much of base_ext (a 2-element array-like specifying temporal
    extents) is covered by entries in items (a Nx2 numpy array specifying
    start and end times of N items).

    :meta private:
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

    :param clip_scores: An [N x M] array containing M per-class scores for each
        of the N clips.
    :param clip_offsets: An N-length integer array containing indices of the
        first sample in each clip.
    :param clip_length: Number of waveform samples in each clip.
    :param threshold: (default: None) If not None, scores below this value will
        be ignored.
    :param suppress_nonmax: (bool; default: False) If True, will apply
        non-max suppression to only consider the top-scoring class for each
        clip.
    :param squeeze_min_samps: (default: None) If not None, will run the
        algorithm to squish contiguous detections of the same class. Squeezing
        will be limited to produce detections that are at least this many
        samples long.

    :return: A 3-element or 4-element tuple containing -

        * sample indices (array of start and end pairs),
        * aggregated scores,
        * class IDs, and
        * if requested, start-end indices making up each combined streak.
    """

    # Apply non-max suppression, if enabled, and the threshold.
    # Only build a mask for now.
    if threshold is not None and suppress_nonmax:
        nan_mask = np.logical_or(clip_scores < threshold,
                                 nonmax_suppress_mask(clip_scores))
    elif threshold is not None:
        nan_mask = (clip_scores < threshold)
    elif suppress_nonmax:
        nan_mask = nonmax_suppress_mask(clip_scores)
    else:
        nan_mask = np.full_like(clip_scores, False, dtype=bool)

    return combine_streaks(np.where(nan_mask, np.nan, clip_scores),
                           clip_offsets, clip_length,
                           squeeze_min_len=squeeze_min_samps)


def nonmax_suppress_mask(scores):
    """
    Returned mask will have True where the corresponding class doesn't have the
    top score, and False elsewhere.

    :meta private:
    """

    nonmax_mask = np.full(scores.shape, True, dtype=bool)
    nonmax_mask[np.arange(scores.shape[0]), scores.argmax(axis=1)] = False

    return nonmax_mask


class LabelHelper:
    """
    Provides functionality for manipulating and managing class labels in a
    problem space, without resorting to altering original annotation files.

    :param classes_list: List of class labels. When used during data
        preparation, the list may be generated from available classes or
        be provided as a pre-defined list. When used during performance
        assessments, it is typically populated from the classes_list.json
        file that is saved alongside raw detections.
    :param remap_labels_dict: (default: None) If not None, must be a dictionary
        describing mapping of class labels. Use this to

        - | update existing class' labels
          |   (e.g. ``{'c1': 'new_c1'}``),
        - | merge together existing classes
          |   (e.g. ``{'c4': 'c1'}``), or
        - | combine existing classes into new ones
          |   (e.g. ``{'c4': 'new_c2', 'c23', 'new_c2'}``).

        Avoid chaining of mappings (e.g. ``{'c1': 'c2', 'c2': 'c3'}``).
    :param negative_class_label: (default: None) If not None, must be a string
        (e.g. 'Other', 'Noise') which will be used as a label to identify the
        negative class clips (those that did not match any annotations). If
        specified, will be used in conjunction with ``remap_labels_dict``.
    :param fixed_labels: (bool; default: True) When True, ``classes_list`` will
        remain unchanged - any new mapping targets specified in
        ``remap_labels_dict`` will not be added and any mapped-out class labels
        will not be omitted. Typically, it should be set to True when
        ``classes_list`` is a pre-defined list during data preparation, and
        always during performance assessments.
    :param assessment_mode: (bool; default: False) Set to True when invoked
        during performance assessments.

    .. seealso::
        :func:`koogu.prepare.from_selection_table_map`
        :func:`koogu.prepare.from_top_level_dirs`
        :func:`koogu.utils.assessments.BaseMetric`
    """

    def __init__(self,
                 classes_list,
                 remap_labels_dict=None,
                 negative_class_label=None,
                 fixed_labels=True,
                 assessment_mode=False):

        # fixed_labels cannot be False when in assessment mode
        assert ((not assessment_mode) or (assessment_mode and fixed_labels)), \
            '\'fixed_labels\' cannot be False when \'assessment_mode\' is True'

        adjusted_neg_label = negative_class_label
        if assessment_mode and (negative_class_label is not None) and \
                negative_class_label not in classes_list:
            # In assessment mode, but neg class results were not saved in raw
            # detections. So, invalidate given neg label.
            adjusted_neg_label = None

        valid_mappings = {}
        if remap_labels_dict is not None:
            self._classes_list, valid_mappings = \
                LabelHelper._handle_mappings(
                    classes_list, remap_labels_dict,
                    adjusted_neg_label,
                    fixed_labels)
        else:
            self._classes_list = [c for c in classes_list]  # make a copy

        self._neg_class_idx = None
        if adjusted_neg_label is not None:
            # Add neg label if not already existing
            if adjusted_neg_label not in self._classes_list:
                self._classes_list.append(adjusted_neg_label)

            self._neg_class_idx = self._classes_list.index(adjusted_neg_label)

        # - Generate string-to-int mappings -
        # First add existing ones ...
        self._class_label_to_idx = {c: ci for ci, c in
                                    enumerate(self._classes_list)}
        # then add for identified valid label mappings (if any)
        for lhs, rhs in valid_mappings.items():
            self._class_label_to_idx[lhs] = self._class_label_to_idx[rhs]

    @property
    def classes_list(self):
        """
        The final list of class names in the problem space, after performing
        manipulations based on ``remap_labels_dict`` (if specified).
        """
        return self._classes_list

    @property
    def negative_class_index(self):
        """
        Index (zero-based) of the negative class (if specified) in
        ``classes_list``.
        """
        return self._neg_class_idx

    @property
    def labels_to_indices(self):
        """
        A Python dictionary mapping class names (string) to zero-based indices.
        """
        return self._class_label_to_idx

    @staticmethod
    def _handle_mappings(classes_list,
                         remap_labels_dict,
                         neg_class_label,
                         fixed_labels):
        """
        Internal function to figure out label mappings.
        """

        # First check for stupid mappings
        all_lhs = [lhs for lhs in remap_labels_dict.keys()]
        if any([(rhs in all_lhs)
                for _, rhs in remap_labels_dict.items()]):
            raise ValueError(
                'Self-, chained- and/or circular mappings not allowed. ' +
                'Please fix conflicting entries in \'remap_labels_dict\'.')

        if neg_class_label is not None:
            def is_neg_label(lbl):
                return lbl == neg_class_label
        else:
            def is_neg_label(_):
                return False

        ignore_lhs = []
        ignore_rhs = []
        valid_mappings = {}
        ghost_mappings = {}
        mapped_out_lhs = []     # labels to be deleted from classes_list
        new_labels = []         # labels to be added to classes_list
        for lhs, rhs in remap_labels_dict.items():
            if is_neg_label(lhs):
                # Following this rule so that neg class label doesn't get
                # inadvertently mapped to one of the positive classes.
                showwarning(
                    f'"{lhs}" is \'negative class\' label, which is a special' +
                    ' case and cannot be mapped. Will ignore the mapping ' +
                    f'"{lhs}" -> "{rhs}".',
                    Warning, __file__, '')
                continue

            existing_lhs = (lhs in classes_list)
            existing_rhs = (rhs in classes_list)

            if fixed_labels:    # classes_list is not amenable
                # Only allowed to map a label that doesn't already exist in
                # classes_list to a label that does exist in classes_list
                # (or is a neg class label).

                if existing_lhs:
                    ignore_lhs.append(lhs)  # complain once for all later
                else:
                    if existing_rhs or is_neg_label(rhs):
                        # Mappable target
                        valid_mappings[lhs] = rhs
                    else:
                        # An attempt at creating a new label
                        ignore_rhs.append(rhs)   # Complain later

            else:               # classes_list is amenable
                # Following mappings are possible:
                #   existing -> existing    [action: del lhs]
                #   existing -> new         [action: del lhs, add rhs]
                #   new      -> existing
                # Can't map if both are new.
                if existing_lhs or existing_rhs:    # At least one is valid
                    if existing_lhs:
                        mapped_out_lhs.append(lhs)  # Mark for deletion

                    if not existing_rhs:
                        new_labels.append(rhs)

                    valid_mappings[lhs] = rhs
                else:
                    # Both lhs & rhs are new labels
                    ghost_mappings[lhs] = rhs   # Complain later

        if len(ignore_lhs) > 0:        # only possible if fixed_labels=True
            showwarning(
                'Mappings in \'remap_labels_dict\' with the following ' +
                'left-hand side values cannot be mapped: {}. '.format(
                    ignore_lhs) +
                'Will ignore, since \'fixed_labels\' is set to True.',
                Warning, __file__, '')

        if len(ignore_rhs) > 0:        # only possible if fixed_labels=True
            showwarning(
                'Mappings in \'remap_labels_dict\' with the following ' +
                'right-hand side values cannot be mapped: {}. '.format(
                    ignore_rhs) +
                'Will ignore, since \'fixed_labels\' is set to True.',
                Warning, __file__, '')

        if len(ghost_mappings) > 0:    # only possible if fixed_labels=False
            showwarning(
                'Both sides of the following mapping(s) are not in ' +
                '\'classes_list\': [{}]. '.format(
                    ', '.join([f'"{lhs}" -> "{rhs}"'
                               for lhs, rhs in ghost_mappings])) +
                'Will ignore, since there isn\'t anything to map.',
                Warning, __file__, '')

        if fixed_labels:
            # Return a copy
            return \
                [c for c in classes_list], \
                valid_mappings
        else:
            # Del "mapped out", add "new", then sort
            return \
                sorted(
                    [c for c in classes_list if c not in mapped_out_lhs] +
                    new_labels), \
                valid_mappings

