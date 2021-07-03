import os
import numpy as np
import warnings

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

    @staticmethod
    def from_directories(audio_root, class_dirs, match_extensions, show_progress=False):

        pbar = ProgressBar(len(class_dirs), prefix='Processing', length=60, show_start=True) if show_progress else None

        for class_dir in class_dirs:
            for file in recursive_listing(os.path.join(audio_root, class_dir), match_extensions):
                yield os.path.join(class_dir, file), None, class_dir

            if pbar is not None:
                pbar.increment()

    @staticmethod
    def from_annotations(selmap, audio_root, seltab_root, show_progress=False):

        single_file_filespec = [('Begin Time (s)', float),
                                ('End Time (s)', float),
                                ('Tags', str)]
        multi_file_filespec = [('Begin Time (s)', float),
                               ('End Time (s)', float),
                               ('Begin File', str),
                               ('File Offset (s)', float),
                               ('Relative Path', str),
                               ('Tags', str)]

        if seltab_root is None:
            full_path = lambda x: x
        else:
            full_path = lambda x: os.path.join(seltab_root, x)

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
                    warnings.showwarning(
                        'No valid annotations found in {:s}'.format(seltab_path),
                        Warning, AudioFileList.from_annotations, '')

                else:

                    all_entries_idxs = np.arange(len(files_times_tags))
                    remaining_entries_mask = np.full((len(files_times_tags),), True)

                    for uniq_file in list(set([files_times_tags[idx][0] for idx in range(len(files_times_tags))])):

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

                if len(times_tags) == 0:
                    warnings.showwarning(
                        'No valid annotations found in {:s}'.format(seltab_path),
                        Warning, AudioFileList.from_annotations, '')
                else:
                    yield audio_path, np.asarray([[tt[0], tt[1]] for tt in times_tags]), [tt[2] for tt in times_tags]

            if pbar is not None:
                pbar.increment()

