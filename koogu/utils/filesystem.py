import os
import numpy as np
import logging
import csv


def recursive_listing(root_dir, match_extensions=None):
    """
    A generator for producing a recursive listing.
    'match_extensions', if not None, must either be a string (e.g. '.txt') or
    a list of strings (e.g. ['.txt', '.TXT', '.csv']), and will result in only
    matched files to be returned.

    The returned paths will all be relative to 'root_dir'.
    """

    if match_extensions is None:
        filtered = __rl_no_filter
    elif isinstance(match_extensions, str):
        filtered = __rl_single_filter
    else:  # assuming now that it's a list/tuple of strings
        filtered = __rl_multi_filter

    for parent, dirs, filenames in os.walk(root_dir):

        relpath = os.path.relpath(parent, start=root_dir)
        adjust = __rl_no_relpath if relpath == '.' else __rl_with_relpath

        for file in sorted(filtered(filenames, match_extensions)):
            yield adjust(file, relpath)

        # Sort this in-place to obtain children subdirs in sorted order
        dirs.sort()


def __rl_no_filter(fn_list, _):
    return fn_list


def __rl_single_filter(fn_list, extn):
    return [f for f in fn_list if f.endswith(extn)]


def __rl_multi_filter(fn_list, extn):
    return [f for f in fn_list if any((f.endswith(e) for e in extn))]


def __rl_no_relpath(fl, _):
    return fl


def __rl_with_relpath(fl, relp):
    return os.path.join(relp, fl)


def restrict_classes_with_whitelist_file(classes, wl_file, ):
    """Among the names in the list 'classes', only return the subset
    that was found in the file 'wl_file'.
    """

    with open(wl_file, 'r') as f:
        # Strip out leading & trailing whitespaces and retain a list of
        # non-empty lines in the input file
        wl_classes = [entry
                      for entry in (line.strip() for line in f)
                      if entry != '']

    return [d for d in classes if d in wl_classes]


class AudioFileList:

    default_audio_filetypes = ['.wav', '.WAV', '.flac', '.aif', '.mp3']

    @staticmethod
    def from_directories(audio_root, class_dirs, filetypes=None):

        match_extensions = filetypes if filetypes else \
            AudioFileList.default_audio_filetypes

        for class_dir in class_dirs:
            for file in recursive_listing(os.path.join(audio_root, class_dir),
                                          match_extensions):
                yield os.path.join(class_dir, file), None, class_dir, None

    @staticmethod
    def from_annotations(selmap, audio_root, seltab_root, annotation_handler,
                         filetypes=None, added_ext=None):
        """

        :param selmap: Mapping from audio file/dir to corresponding annotation
            file. Only specify paths that are relative to 'audio_root' and
            'seltab_root'.
        :param audio_root: Root directory for all audio files.
        :param seltab_root: Root directory for all annotation files.
        :param annotation_handler: An instance of one of the annotation handlers
            from data.annotations.
        :param filetypes: This parameter applies only when an audio reference
            in selmap points to a directory.
        :param added_ext: This parameter applies only when an audio reference
            in selmap points to a directory. Useful when looking for
            'raw result' files, during performance assessments.
        """

        logger = logging.getLogger(__name__)

        if seltab_root is None:
            def full_path(x): return x
        else:
            def full_path(x): return os.path.join(seltab_root, x)

        match_extensions = filetypes if filetypes else \
            AudioFileList.default_audio_filetypes
        if added_ext is not None:
            # Append the "added" extension, remove later when yielding
            if isinstance(match_extensions, str):
                match_extensions = match_extensions + added_ext
            else:  # assuming now that it's a list/tuple of strings
                match_extensions = [(m + added_ext) for m in match_extensions]

        for (audio_path, seltab_path) in selmap:

            is_multi_file = os.path.isdir(os.path.join(audio_root, audio_path))

            # Fetch annotations
            status, (
                annots_times, _, annots_tags, annots_channels, annots_files
            ) = annotation_handler.safe_fetch(full_path(seltab_path),
                                              multi_file=is_multi_file)

            if not status:
                continue        # error message already logged. just skip

            num_annots = len(annots_times)
            if num_annots > 0:
                # Convert to numpy arrays
                annots_times = np.asarray(annots_times)
                annots_channels = \
                    np.asarray(annots_channels).astype(np.uint8)
            else:
                logger.warning(
                    f'No valid annotations found in {seltab_path:s}')
                annots_times = np.zeros((0, 2))
                annots_tags = []
                annots_channels = np.zeros((0, ), dtype=np.uint8)

            if not is_multi_file:
                # Individual audio file; return everything at once

                if len(annots_times) > 0:
                    yield audio_path, annots_times, annots_tags, annots_channels

            else:
                # The selection table file applied to the directory's contents.
                # Yield each listed audio file individually.

                all_entries_idxs = np.arange(num_annots)
                remaining_entries_mask = np.full((num_annots, ), True)

                # First process annotations (if any) corresponding to
                # directory-listed files
                for dir_file in recursive_listing(
                        os.path.join(audio_root, audio_path),
                        match_extensions):
                    dir_file_e = dir_file if added_ext is None else \
                        dir_file[:-len(added_ext)]

                    # Annotations matching dir_file
                    dir_file_matches_mask = np.full((num_annots,), False)
                    dir_file_matches_mask[[
                        idx
                        for idx in all_entries_idxs[remaining_entries_mask]
                        if annots_files[idx] == dir_file_e]] = True

                    # Update for next iteration
                    remaining_entries_mask[dir_file_matches_mask] = False

                    yield os.path.join(audio_path, dir_file_e), \
                        annots_times[dir_file_matches_mask, :], \
                        [annots_tags[idx]
                         for idx in np.where(dir_file_matches_mask)[0]], \
                        annots_channels[dir_file_matches_mask]

                # Process any remaining entries
                for uniq_file in list(set([     # get unique files
                        af
                        for af, v in zip(annots_files, remaining_entries_mask)
                        if v])):

                    # Annotations matching uniq_file
                    uniq_file_matches_mask = np.full((num_annots,), False)
                    uniq_file_matches_mask[[
                        idx
                        for idx in all_entries_idxs[remaining_entries_mask]
                        if annots_files[idx] == uniq_file]] = True

                    # Update for next iteration
                    remaining_entries_mask[uniq_file_matches_mask] = False

                    # Check if file even exists on disk
                    ret_path = os.path.join(audio_path, uniq_file)
                    if os.path.exists(ret_path + (
                            '' if added_ext is None else added_ext)):

                        yield ret_path, \
                            annots_times[uniq_file_matches_mask, :], \
                            [annots_tags[idx]
                             for idx in np.where(uniq_file_matches_mask)[0]], \
                            annots_channels[uniq_file_matches_mask]

                    else:
                        logger.error(
                            'File {} corresponding to {} '.format(
                                ret_path + (
                                    '' if added_ext is None else added_ext),
                                np.sum(uniq_file_matches_mask)
                            ) + 'annotations could not be found.')


def get_valid_audio_annot_entries(audio_annot_list_or_csv,
                                  audio_root, annot_root,
                                  plus_extn=None, logger=None):
    """
    Validate presence of files in `audio_annot_list_or_csv` and return a list of
    only valid entries. Each entry is a pair of audio file/dir and annot file.
    Alternatively, `audio_annot_list_or_csv` could also be specified as (a path
    to) a 2-column csv file containing audio-annot pairs. Only use the csv
    option if the paths are simple (i.e., the filenames do not contain commas or
    other special characters).

    `plus_extn` if not None (e.g., '.npz') will be appended to each audio file.

    :meta private:
    """

    if logger is None:
        logger = logging.getLogger(__name__)

    if plus_extn is None:
        f_type = 'Audio'
        def lhs_fullname(x): return x
    else:
        f_type = 'Raw scores'
        def lhs_fullname(x): return x + plus_extn

    if annot_root is None:
        def rhs_fullpath(x): return x
    else:
        def rhs_fullpath(x): return os.path.join(annot_root, x)

    def validate_lhs(entry):
        retval = len(entry) > 0 and (
            os.path.isdir(os.path.join(audio_root, entry)) or
            os.path.exists(os.path.join(audio_root, lhs_fullname(entry)))
        )
        if not retval:
            logger.error(
                '{} file path "{}" is either invalid or unreachable'.
                format(f_type, lhs_fullname(entry)))
        return retval

    def validate_rhs(entry):
        retval = len(entry) > 0 and (os.path.isfile(rhs_fullpath(entry)))
        if not retval:
            logger.error(
                'Annotation file path "{}" is either invalid or unreachable'.
                format(entry))
        return retval

    if isinstance(audio_annot_list_or_csv, (list, tuple)):
        audio_annot_list = audio_annot_list_or_csv
    elif isinstance(audio_annot_list_or_csv, str):
        if os.path.exists(audio_annot_list_or_csv):
            # Attempt reading it as a csv file
            with open(audio_annot_list_or_csv, 'r', newline='') as fh:
                audio_annot_list = [
                    entry[:2] for entry in csv.reader(fh) if len(entry) >= 2]
        else:
            raise ValueError('Path specified in audio_annot_list ' +
                             f'({audio_annot_list_or_csv}) does not exist.')
    else:
        raise ValueError(
            'Audio file & annotation pairs must either be specified as a list' +
            ' of pairs, or as a path to a csv file')

    valid_entries_mask = [
        (validate_lhs(lhs) and validate_rhs(rhs))
        for (lhs, rhs) in audio_annot_list
    ]

    # Discard invalid entries, if any
    return [entry
            for entry, v in zip(audio_annot_list, valid_entries_mask)
            if v]

