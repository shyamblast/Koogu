import xml.etree.ElementTree as XElemTree
from koogu.data.annotations import BaseAnnotationReader


class Reader(BaseAnnotationReader):
    """
    Reader class for reading Sonic Visualiser format annotation files.

    :param fetch_frequencies: (boolean; default: False) If True, will also
        attempt to read annotations' frequency bounds. NaNs will be returned
        for any missing values. If False, the respective item in the tuple
        returned will be set to None.

    Example::

        >>> # Create a reader instance
        >>> reader = SonicVisualiser.Reader()
        >>>
        >>> # Read annotations from file and display
        >>> (times, _, tags, _, _) = reader('my_annots.svl')
        >>> print('Start time, End time, Label')
        >>> print('----------, --------, -----')
        >>> for idx in range(len(times)):
        ...     print(f'{times[idx][0]}, {times[idx][1]}, {tags[idx]}')
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

        # Load the "data" element from file
        data = XElemTree.parse(filepath).getroot().find('data')

        # Get the sampling rate, as a float value
        fs = float(data.find('model').get('sampleRate'))

        if self._fetch_frequencies:
            read_point_elem = Reader._read_2d_point_elem
        else:
            read_point_elem = Reader._read_1d_point_elem

        missing_freq_val = self.default_float()

        # Read all "point" elements
        times_tags = list(map(
            lambda elem: read_point_elem(elem, fs, missing_freq_val),
            data.find('dataset').findall('point')
        ))

        # Channel indices (4th return item) must be zero since channel-specific
        # annotations aren't (yet) supported in SV.
        return \
            list(map(lambda tt: tt[0], times_tags)), \
            list(map(lambda tt: tt[2], times_tags)) if self._fetch_frequencies \
            else None, \
            list(map(lambda tt: tt[1], times_tags)), \
            [0] * len(times_tags), \
            None

    @staticmethod
    def _read_1d_point_elem(point_elem, fs, _):
        """
        Convert frame number & duration to start-end time pair and return, along
        with the label.
        """
        start_s = float(point_elem.get('frame')) / fs
        end_s = start_s + (float(point_elem.get('duration')) / fs)
        label = point_elem.get('label')

        return (start_s, end_s), label

    @staticmethod
    def _read_2d_point_elem(point_elem, fs, default_float):
        """
        Convert frame number & duration to start-end time pair, convert value &
        extent to low-high frequency pair, and return, along with the label.
        Use defaults (NaN) if frequency info is unavailable.
        """
        start_s = float(point_elem.get('frame')) / fs
        end_s = start_s + (float(point_elem.get('duration')) / fs)
        low_f = float(point_elem.get('value', default=default_float))
        high_f = low_f + (
            float(point_elem.get('extent', default=default_float)))
        label = point_elem.get('label')

        return (start_s, end_s), label, (low_f, high_f)
