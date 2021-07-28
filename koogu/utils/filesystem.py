import os
import numpy as np
import logging

from koogu.utils.detections import SelectionTableReader
from koogu.utils.terminal import ProgressBar


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
    def from_directories(audio_root, class_dirs, filetypes=None, show_progress=False):

        match_extensions = filetypes if filetypes else AudioFileList.default_audio_filetypes

        pbar = ProgressBar(len(class_dirs), prefix='Processing', length=60, show_start=True) if show_progress else None

        for class_dir in class_dirs:
            for file in recursive_listing(os.path.join(audio_root, class_dir), match_extensions):
                yield os.path.join(class_dir, file), None, class_dir

            if pbar is not None:
                pbar.increment()

    @staticmethod
    def from_annotations(selmap, audio_root, seltab_root, show_progress=False,
                         ignore_zero_annot_files=True,
                         filetypes=None, added_ext=None):
        """

        :param selmap: Mapping from audio file/dir to corresponding annotation
            file. Only specify paths that are relative to 'audio_root' and
            'seltab_root'.
        :param audio_root: Root directory for all audio files.
        :param seltab_root: Root directory for all annotation files.
        :param show_progress: Show progress bar during processing.
        :param ignore_zero_annot_files: Where the audio reference in selmap
            points to an audio file and the corresponding annot file contains
            zero annotations, a True value in this parameter will cause the
            function to skip the audio file, while a False value will cause
            the function to yield zero-length arrays for times & tags.
            When the audio reference in selmap points to a directory instead
            a True value in this parameter will cause the function to yield
            only those files for which there are annotations available, while
            a False value will cause the function to return times & tags for
            all files discovered in the directory (times & tags arrays will be
            zero-length arrays for files for which there were no annotations).
            Also see 'match_extensions'.
        :param filetypes: This parameter applies only when an audio reference
            in selmap points to a directory and 'ignore_zero_annot_files' is
            set to False.
        :param added_ext: This parameter applies only when an audio reference
            in selmap points to a directory and 'ignore_zero_annot_files' is
            set to False. Useful when looking for 'raw result' files, mostly
            only during performance assessments.
        """

        single_file_filespec = [('Begin Time (s)', float),
                                ('End Time (s)', float),
                                ('Tags', str)]
        multi_file_filespec = [('Begin Time (s)', float),
                               ('End Time (s)', float),
                               ('Begin File', str),
                               ('File Offset (s)', float),
                               ('Relative Path', str),
                               ('Tags', str)]

        logger = logging.getLogger(__name__)

        if seltab_root is None:
            full_path = lambda x: x
        else:
            full_path = lambda x: os.path.join(seltab_root, x)

        match_extensions = filetypes if filetypes else AudioFileList.default_audio_filetypes
        if added_ext is not None:
            # Append the "added" extension, remove later when yielding
            if isinstance(match_extensions, str):
                match_extensions = match_extensions + added_ext
            else:  # assuming now that it's a list/tuple of strings
                match_extensions = [m + added_ext for m in match_extensions]

        pbar = ProgressBar(len(selmap), prefix='Processing', length=60, show_start=True) if show_progress else None

        for (audio_path, seltab_path) in selmap:

            if os.path.isdir(os.path.join(audio_root, audio_path)):
                # The selection table file applies to the directory's contents.

                # Derive annot start & end times from in-file offsets and durations and yield each listed audio file
                # individually.
                files_times_tags = [[entry[2] if entry[4] is None else os.path.join(entry[4], entry[2]),
                                     (entry[3], entry[3] + (entry[1] - entry[0])),
                                     entry[5]]
                                    for entry in SelectionTableReader(full_path(seltab_path), multi_file_filespec)
                                    if any([e is not None for e in entry])]

                if len(files_times_tags) == 0:
                    logger.warning(
                        f'No valid annotations found in {seltab_path:s}')

                    if not ignore_zero_annot_files:
                        for file in recursive_listing(os.path.join(audio_root, audio_path), match_extensions):
                            ret_path = os.path.join(audio_path, file)
                            if added_ext is not None:
                                ret_path = ret_path[:-len(added_ext)]
                            yield ret_path, np.zeros((0, 2)), []

                else:

                    all_entries_idxs = np.arange(len(files_times_tags))
                    remaining_entries_mask = np.full((len(files_times_tags),), True)

                    # First process entries corresponding to directory-listed files (if specified)
                    if not ignore_zero_annot_files:
                        for uniq_file in recursive_listing(os.path.join(audio_root, audio_path), match_extensions):
                            if added_ext is not None:
                                uniq_file = uniq_file[:-len(added_ext)]

                            curr_file_items_idxs = np.asarray(
                                [idx for idx in all_entries_idxs[remaining_entries_mask]
                                 if files_times_tags[idx][0] == uniq_file])

                            if len(curr_file_items_idxs) > 0:
                                remaining_entries_mask[curr_file_items_idxs] = False    # Update for next iteration

                                yield os.path.join(audio_path, uniq_file), \
                                      np.asarray([files_times_tags[idx][1] for idx in curr_file_items_idxs]), \
                                      [files_times_tags[idx][2] for idx in curr_file_items_idxs]
                            else:
                                yield os.path.join(audio_path, uniq_file), np.zeros((0, 2)), []

                    # Process all (remaining) entries
                    for uniq_file in list(set([ftt[0] for ftt, v in zip(files_times_tags, remaining_entries_mask)
                                               if v])):

                        curr_file_items_idxs = np.asarray(
                            [idx for idx in all_entries_idxs[remaining_entries_mask]
                             if files_times_tags[idx][0] == uniq_file])

                        remaining_entries_mask[curr_file_items_idxs] = False    # Update for next iteration

                        yield os.path.join(audio_path, uniq_file), \
                              np.asarray([files_times_tags[idx][1] for idx in curr_file_items_idxs]), \
                              [files_times_tags[idx][2] for idx in curr_file_items_idxs]

            else:
                # Individual audio file

                times_tags = [entry
                              for entry in SelectionTableReader(full_path(seltab_path), single_file_filespec)
                              if any([e is not None for e in entry])]

                if len(times_tags) > 0:
                    yield audio_path, np.asarray([[tt[0], tt[1]] for tt in times_tags]), [tt[2] for tt in times_tags]
                elif not ignore_zero_annot_files:
                    yield audio_path, np.zeros((0, 2)), []
                else:
                    logger.warning(
                        f'No valid annotations found in {seltab_path:s}')

            if pbar is not None:
                pbar.increment()

