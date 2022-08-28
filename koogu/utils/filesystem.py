import os
import numpy as np
import logging
import csv

from koogu.utils.detections import SelectionTableReader


def recursive_listing(root_dir, match_extensions=None):
    """
    A generator for producing a recursive listing.
    'match_extensions', if not None, must either be a string (e.g. '.txt') or
    a list of strings (e.g. ['.txt', '.TXT', '.csv']), and will result in only
    matched files to be returned.

    The returned paths will all be relative to 'root_dir'.
    """

    if match_extensions is None:
        matcher_fn = lambda x: True
    elif isinstance(match_extensions, str):
        matcher_fn = lambda x: x.endswith(match_extensions)
    else:  # assuming now that it's a list/tuple of strings
        matcher_fn = lambda x: any((x.endswith(e) for e in match_extensions))

    for parent, dirs, filenames in os.walk(root_dir):

        for file in [f for f in sorted(filenames) if matcher_fn(f)]:
            relpath = os.path.relpath(parent, start=root_dir)
            yield os.path.join(relpath, file) if relpath != '.' else file

        # Sort this in-place to obtain children subdirs in sorted order
        dirs.sort()


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
    def from_annotations(selmap, audio_root, seltab_root, label_column_name,
                         filetypes=None, added_ext=None):
        """

        :param selmap: Mapping from audio file/dir to corresponding annotation
            file. Only specify paths that are relative to 'audio_root' and
            'seltab_root'.
        :param audio_root: Root directory for all audio files.
        :param seltab_root: Root directory for all annotation files.
        :param label_column_name: A string (e.g., "Tags") identifying the header
            of the column in the selection table file(s) from which class labels
            are to be extracted.
        :param filetypes: This parameter applies only when an audio reference
            in selmap points to a directory and 'ignore_zero_annot_files' is
            set to False.
        :param added_ext: This parameter applies only when an audio reference
            in selmap points to a directory. Useful when looking for
            'raw result' files, during performance assessments.
        """

        single_file_filespec = [('Begin Time (s)', float),
                                ('End Time (s)', float),
                                (label_column_name, str),
                                ('Channel', int, 1)]
        multi_file_filespec = [('Begin Time (s)', float),
                               ('End Time (s)', float),
                               ('Begin File', str),
                               ('File Offset (s)', float),
                               ('Relative Path', str),
                               (label_column_name, str),
                               ('Channel', int, 1)]

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

            if os.path.isdir(os.path.join(audio_root, audio_path)):
                # The selection table file applies to the directory's contents.

                # Derive annot start & end times from in-file offsets and
                # durations and yield each listed audio file individually.
                files_times_tags = [
                    [entry[2] if entry[4] is None else os.path.join(entry[4], entry[2]),
                     (entry[3], entry[3] + (entry[1] - entry[0])),
                     entry[5],
                     entry[6] or 1]
                    for entry in SelectionTableReader(full_path(seltab_path),
                                                      multi_file_filespec)
                    if ((entry[0] is not None) and
                        (entry[1] is not None) and
                        (entry[2] is not None) and
                        (entry[3] is not None) and
                        (entry[5] is not None))
                ]

                if len(files_times_tags) == 0:
                    logger.warning(
                        f'No valid annotations found in {seltab_path:s}')

                    for file in recursive_listing(
                            os.path.join(audio_root, audio_path),
                            match_extensions):
                        ret_path = os.path.join(audio_path, file)
                        if added_ext is not None:
                            ret_path = ret_path[:-len(added_ext)]
                        yield ret_path, np.zeros((0, 2)), [], \
                              np.zeros((0, ), dtype=np.uint8)

                else:

                    all_entries_idxs = np.arange(len(files_times_tags))
                    remaining_entries_mask = np.full((len(files_times_tags),),
                                                     True)

                    # First process entries corresponding to directory-listed
                    # files
                    for uniq_file in recursive_listing(
                            os.path.join(audio_root, audio_path),
                            match_extensions):
                        if added_ext is not None:
                            uniq_file = uniq_file[:-len(added_ext)]

                        curr_file_items_idxs = np.asarray(
                            [idx
                             for idx in all_entries_idxs[remaining_entries_mask]
                             if files_times_tags[idx][0] == uniq_file])

                        if len(curr_file_items_idxs) > 0:
                            # Update for next iteration
                            remaining_entries_mask[curr_file_items_idxs] = False

                            yield os.path.join(audio_path, uniq_file), \
                                np.asarray([files_times_tags[idx][1]
                                            for idx in curr_file_items_idxs]), \
                                [files_times_tags[idx][2]
                                 for idx in curr_file_items_idxs], \
                                np.asarray([(files_times_tags[idx][3] - 1)
                                            for idx in curr_file_items_idxs],
                                           dtype=np.uint8)
                        else:
                            yield os.path.join(audio_path, uniq_file), \
                                  np.zeros((0, 2)), [], \
                                  np.zeros((0, ), dtype=np.uint8)

                    # Process all (remaining) entries
                    for uniq_file in list(set([
                            ftt[0]
                            for ftt, v in zip(files_times_tags,
                                              remaining_entries_mask)
                            if v])):

                        curr_file_items_idxs = np.asarray(
                            [idx
                             for idx in all_entries_idxs[remaining_entries_mask]
                             if files_times_tags[idx][0] == uniq_file])

                        # Update for next iteration
                        remaining_entries_mask[curr_file_items_idxs] = False

                        # Check if file even exists on disk
                        if os.path.exists(
                                os.path.join(audio_path, uniq_file) +
                                ('' if added_ext is None else added_ext)):

                            yield os.path.join(audio_path, uniq_file), \
                                np.asarray([files_times_tags[idx][1]
                                            for idx in curr_file_items_idxs]), \
                                [files_times_tags[idx][2]
                                 for idx in curr_file_items_idxs], \
                                np.asarray([(files_times_tags[idx][3] - 1)
                                            for idx in curr_file_items_idxs],
                                           dtype=np.uint8)

                        else:
                            logger.error(
                                'File {} corresponding to {} '.format(
                                    uniq_file + ('' if added_ext is None
                                                 else added_ext),
                                    len(curr_file_items_idxs)
                                ) + 'annotations could not be found.')

            else:
                # Individual audio file

                times_tags = [
                    (entry[0], entry[1], entry[2], entry[3] or 1)
                    for entry in SelectionTableReader(full_path(seltab_path),
                                                      single_file_filespec)
                    if ((entry[0] is not None) and
                        (entry[1] is not None) and
                        (entry[2] is not None))
                ]

                if len(times_tags) > 0:
                    yield audio_path, \
                          np.asarray([[tt[0], tt[1]] for tt in times_tags]), \
                          [tt[2] for tt in times_tags], \
                          np.asarray([(tt[3] - 1) for tt in times_tags],
                                     dtype=np.uint8)
                else:
                    logger.warning(
                        f'No valid annotations found in {seltab_path:s}')
                    yield audio_path, np.zeros((0, 2)), [], \
                        np.zeros((0, ), dtype=np.uint8)


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

    def validate_lhs(entry):
        return len(entry) > 0 and (
            os.path.isdir(os.path.join(audio_root, entry)) or
            os.path.exists(os.path.join(
                audio_root, entry if plus_extn is None else (entry + plus_extn)
            ))
        )

    def validate_rhs(entry):
        return len(entry) > 0 and (
            os.path.isfile(
                entry if annot_root is None else os.path.join(annot_root, entry)
            )
        )

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

    valid_entries_mask = [False] * len(audio_annot_list)
    for e_idx, (lhs, rhs) in enumerate(audio_annot_list):
        l_v = validate_lhs(lhs)
        r_v = validate_rhs(rhs)
        if l_v and r_v:
            valid_entries_mask[e_idx] = True
        else:
            logger.error(
                f'Validity of elements in entry ({lhs}, {rhs}) are ' +
                f'({l_v}, {r_v}). Will discard entry.')

    # Discard invalid entries, if any
    return [entry
            for entry, v in zip(audio_annot_list, valid_entries_mask)
            if v]

