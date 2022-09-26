import csv
from koogu.data.annotations import BaseAnnotationReader


class Reader(BaseAnnotationReader):
    """
    Reader class for reading Audacity format annotation files.

    :param fetch_frequencies: (boolean; default: False) If True, will also
        attempt to read annotations' frequency bounds. NaNs will be returned
        for any missing values. If False, the respective item in the tuple
        returned will be set to None.
    """

    def __init__(self, fetch_frequencies=False):
        super(Reader, self).__init__(fetch_frequencies)

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

        lal = _LastAnnotLink(self.default_float())

        if self._fetch_frequencies:
            accum_fn = lal.add_with_freq
            updat_fn = lal.filter_freq_update
        else:
            accum_fn = lal.add_no_freq
            updat_fn = lal.filter_no_update

        with open(filepath, 'r', newline='') as file_h:
            lines_iterator = csv.reader(file_h, delimiter='\t')

            annots = list(map(accum_fn, filter(updat_fn, lines_iterator)))

        # Channel indices (4th return item) must be zero since channel-specific
        # annotations aren't (yet) supported in Audacity.
        return \
            list(map(lambda tt: tt[0], annots)), \
            list(map(lambda tt: tt[2], annots)) if self._fetch_frequencies \
            else None, \
            list(map(lambda tt: tt[1], annots)), \
            [0] * len(annots), \
            None


class _LastAnnotLink:

    def __init__(self, default_float):
        self._last_annot = None
        self._default_float = default_float
        self._default_freq = (default_float, default_float)

    def add_no_freq(self, fields):
        self._last_annot = [
            (float(fields[0]), float(fields[1])),
            fields[2]
        ]
        return self._last_annot

    def add_with_freq(self, fields):
        self._last_annot = [
            (float(fields[0]), float(fields[1])),
            fields[2],
            self._default_freq
        ]
        return self._last_annot

    def filter_no_update(self, fields):
        return fields[0] != '\\'

    def filter_freq_update(self, fields):
        if fields[0] == '\\':
            self._last_annot[2] = (
                self._default_float if fields[1].startswith('-')
                else float(fields[1]),
                self._default_float if fields[2].startswith('-')
                else float(fields[2])
            )

            return False

        return True
