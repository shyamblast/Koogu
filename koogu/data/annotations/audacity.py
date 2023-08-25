import csv
from koogu.data.annotations import BaseAnnotationReader, BaseAnnotationWriter


class Reader(BaseAnnotationReader):
    """
    Reader class for reading Audacity format annotation files.

    :param fetch_frequencies: (boolean; default: False) If True, will also
        attempt to read annotations' frequency bounds. NaNs will be returned
        for any missing values. If False, the respective item in the tuple
        returned will be set to None.

    Example::

        >>> # Create a reader instance
        >>> reader = Audacity.Reader()
        >>>
        >>> # Read annotations from file and display
        >>> (times, _, tags, _, _) = reader('my_audacity_annots.txt')
        >>> print('Start time, End time, Label')
        >>> print('----------, --------, -----')
        >>> for idx in range(len(times)):
        ...     print(f'{times[idx][0]}, {times[idx][1]}, {tags[idx]}')
    """

    def __init__(self, fetch_frequencies=False):
        super(Reader, self).__init__(fetch_frequencies)

        if fetch_frequencies:
            self._accum_fn = self._add_with_freq
            self._updat_fn = self._filter_freq_update
        else:
            self._accum_fn = self._add_no_freq
            self._updat_fn = self._filter_no_update

    def _fetch(self, filepath, **kwargs):
        """
        Load annotations from ``filepath``.

        :param filepath: Path to a Sonic Visualiser annotation layer file.

        :return: A 5-element tuple

          - N-length list of 2-element tuples denoting annotations' start and
            end times
          - Either None or an N-length list of 2-element tuples denoting
            annotations' frequency bounds
          - N-length list of tags/class labels
          - N-length list of zeros as 0-based channel indices
          - None

        :meta private:
        """

        # An ephemeral container. It's contents (the 0-th item) will be
        # created, updated and re-created within the below map-filter logic.
        item = ([None])

        default_float = self.default_float()
        default_freq = (default_float, default_float)

        with open(filepath, 'r', newline='') as file_h:

            annots = list(
                # map() creates new content in `item` (its 0-th element), and
                # filter() updates, as appropriate, the frequency content within
                # the `item`.
                # In combo, map-filter create a conditionally self-updating
                # iterator, whose values will be collected in the surrounding
                # list().
                map(
                    lambda l: self._accum_fn(item, l, default_freq),
                    filter(
                        lambda l: self._updat_fn(item, l, default_float),
                        csv.reader(file_h, delimiter='\t')
                    )
                )
            )

        # Channel indices (4th return item) must be zero since channel-specific
        # annotations aren't (yet) supported in Audacity.
        return \
            list(map(lambda tt: tt[0], annots)), \
            list(map(lambda tt: tt[2], annots)) if self._fetch_frequencies \
            else None, \
            list(map(lambda tt: tt[1], annots)), \
            [0] * len(annots), \
            None

    @classmethod
    def _add_no_freq(cls, last_annot, fields, _):
        # Create new list and return its reference
        last_annot[0] = [
            (float(fields[0]), float(fields[1])),
            fields[2]
        ]
        return last_annot[0]

    @classmethod
    def _add_with_freq(cls, last_annot, fields, default_freq):
        # Create new list and return its reference
        last_annot[0] = [
            (float(fields[0]), float(fields[1])),
            fields[2],
            default_freq
        ]
        return last_annot[0]

    @classmethod
    def _filter_no_update(cls, last_annot, fields, _):
        return fields[0] != '\\'

    @classmethod
    def _filter_freq_update(cls, last_annot, fields, default_float):
        if fields[0] == '\\':
            # Update the freq elements in the referenced list
            last_annot[0][2] = (
                default_float if fields[1].startswith('-')
                else float(fields[1]),
                default_float if fields[2].startswith('-')
                else float(fields[2])
            )

            return False

        return True


class Writer(BaseAnnotationWriter):
    """
    Writer class for writing annotations/detections to Audacity format files.

    :param write_frequencies: Boolean (default: False) directing whether to
        include bounding (lower and higher) frequency info in the outputs.
    """

    def __init__(self, write_frequencies=False, **kwargs):

        super(Writer, self).__init__(write_frequencies)

    def _write(self, out_file, times, labels, frequencies=None, **kwargs):
        """
        Write out annotations/detections to Audacity format file.

        :param out_file: Output filepath.
        :param times: An N-length list of 2-element list/tuple of start and end
            times.
        :param labels: An N-length list of annotation/detection labels.
        :param frequencies: An N-length list of 2-element list/tuple of low and
            high frequencies.

        :return: Number of annotations/detections written.
        """

        num_rows = len(times)

        basic_fmt = '{0[0]:.6f}\t{0[1]:.6f}\t{1:s}\n'
        freq_fmt = '\\\t{0[0]:.6f}\t{0[1]:.6f}\n'
        with open(out_file, 'w') as out_fh:
            if self._write_frequencies and frequencies is not None:
                for (time, label, freq) in zip(times, labels, frequencies):
                    out_fh.write(basic_fmt.format(time, label))
                    out_fh.write(freq_fmt.format(freq))
            else:
                for (time, label) in zip(times, labels):
                    out_fh.write(basic_fmt.format(time, label))

        return num_rows
