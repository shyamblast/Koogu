import os
from koogu.utils.detections import SelectionTableReader


class _AnnotationHandler:
    """
    Base class for managing different annotation formats.
    """
    def __init__(self):
        pass

    def load(self, filepath, **kwargs):
        """
        Implementations must return a 4-element tuple -
          - Annotations' times, as a Nx2 numpy array
          - N-length list of tags/class labels
          - N-length numpy array of channel indices (0-based)
          - (optional; set to None if not returning) N-length list of audio
            files corresponding to the returned annotations
        """
        raise NotImplementedError(
            'load() method not implemented in derived class')


class RavenAnnotationHandler(_AnnotationHandler):
    """
    Class for handling Raven selection table format files.

    :param label_column_name: A string (e.g., "Tags") identifying the header
        of the column in the selection table file(s) from which class labels are
        to be extracted. If None (default), will look for a column with the
        header "Tags".
    """

    def __init__(self, label_column_name=None):

        self._single_file_filespec = [('Begin Time (s)', float),
                                      ('End Time (s)', float),
                                      (label_column_name or 'Tags', str),
                                      ('Channel', int, 1)]
        self._multi_file_filespec = [('Begin Time (s)', float),
                                     ('End Time (s)', float),
                                     ('Begin File', str),
                                     ('File Offset (s)', float),
                                     ('Relative Path', str),
                                     (label_column_name or 'Tags', str),
                                     ('Channel', int, 1)]

        super(RavenAnnotationHandler, self).__init__()

    def load(self, filepath, multi_file=False):
        """
        Load annotations from ``filepath``.

        :param filepath: Path to a Raven selection table file.
        :param multi_file: Set to True if the selection table file corresponds
            to multiple audio files (usually under a parent directory) instead
            of a single file.

        :return: A 4-element tuple

          - Annotations' times, as a Nx2 numpy array
          - N-length list of tags/class labels
          - N-length numpy array of channel indices (0-based)
          - None if ``multi_file`` was False. Otherwise, an N-length list of
            audio files corresponding to the returned annotations

        """

        if multi_file:
            # Derive annot start & end times from in-file offsets and durations
            files_times_tags_channels = [
                (entry[2] if entry[4] is None else os.path.join(entry[4],
                                                                entry[2]),
                 (entry[3], entry[3] + (entry[1] - entry[0])),
                 entry[5],
                 entry[6] or 1)
                for entry in SelectionTableReader(filepath,
                                                  self._multi_file_filespec)
                if ((entry[0] is not None) and
                    (entry[1] is not None) and
                    (entry[2] is not None) and
                    (entry[3] is not None) and
                    (entry[5] is not None))
            ]

            files = [entry[0] for entry in files_times_tags_channels]
            times = [entry[1] for entry in files_times_tags_channels]
            tags = [entry[2] for entry in files_times_tags_channels]
            channels = [entry[3]-1 for entry in files_times_tags_channels]

        else:
            times_tags_channels = [
                ((entry[0], entry[1]), entry[2], entry[3] or 1)
                for entry in SelectionTableReader(filepath,
                                                  self._single_file_filespec)
                if ((entry[0] is not None) and
                    (entry[1] is not None) and
                    (entry[2] is not None))
            ]

            files = None
            times = [entry[0] for entry in times_tags_channels]
            tags = [entry[1] for entry in times_tags_channels]
            channels = [entry[2]-1 for entry in times_tags_channels]

        return times, tags, channels, files


__all__ = ['RavenAnnotationHandler']
