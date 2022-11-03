import abc
import math


class BaseAnnotationReader(metaclass=abc.ABCMeta):
    """
    Base class for reading annotations from storage.

    Within Koogu, the method :meth:`__call__` (and others in chain) will be
    invoked from parallel threads of execution. Exercise caution if an
    implementation of this class needs to use and alter any member variables.

    :meta private:
    """

    def __init__(self, fetch_frequencies=False):
        """
        :param fetch_frequencies: (boolean; default: False) If True, will also
            attempt to read annotations' frequency bounds. NaNs will be returned
            for any missing values. If False, the respective item in the tuple
            returned from :meth:`__call__` will be set to None.
        """

        self._fetch_frequencies = fetch_frequencies

    def __call__(self, source, **kwargs):
        """
        Read annotations from file/database/etc., process as appropriate and
        return a 5-element tuple (see below).

        :param source: Identifier of an annotation source (e.g., path to an
            annotation file).

        :return: A 5-element tuple

          - N-length list of 2-element tuples denoting annotations' start and
            end times
          - Either None or an N-length list of 2-element tuples denoting
            annotations' frequency bounds
          - N-length list of tags/class labels
          - N-length list of channel indices (0-based)
          - (optional; set to None if not returning) N-length list of audio
            sources corresponding to the returned annotations

        :meta private:
        """

        return self._fetch(source, **kwargs)

    @abc.abstractmethod
    def _fetch(self, source, **kwargs):
        """
        Read annotations from file/database/etc., process as appropriate and
        return a 5-element tuple (see below).

        :param source: Identifier of an annotation source (e.g., path to an
            annotation file).

        Implementations must return a 5-element tuple -
          - N-length list of 2-element tuples denoting annotations' start and
            end times
          - None if load_frequencies=False, otherwise an N-length list of
            2-element tuples denoting annotations' frequency bounds
          - N-length list of tags/class labels
          - N-length list of channel indices (0-based)
          - (optional; set to None if not returning) N-length list of audio
            sources corresponding to the returned annotations
        """
        raise NotImplementedError(
            '_fetch() method not implemented in derived class')

    @classmethod
    def default_float(cls):
        return math.nan


class BaseAnnotationWriter(metaclass=abc.ABCMeta):
    """
    Base class for writing annotations/detections to storage.

    Within Koogu, the method :meth:`__call__` (and others in chain) will be
    invoked from parallel threads of execution. Exercise caution if an
    implementation of this class needs to use and alter any member variables.

    :meta private:
    """

    def __init__(self, write_frequencies=False):
        """
        :param write_frequencies: (boolean; default: False) If True, will also
            write out annotations' frequency bounds. Based on the implementation
            appropriate defaults (blank spaces, NaNs, negative values, etc.)
            will be written when missing frequency values. If False, frequency
            values, even if provided, will not be written out, and relevant
            structural constructs will not be created in the output file.
        """

        self._write_frequencies = write_frequencies

    def __call__(self, destination, times, labels, *args, **kwargs):
        """
        Write out annotations/detections to destination.

        :param destination: Identifier of the target where
            annotations/detections will be written to (e.g., path to an
            annotation file).
        :param times: An N-length list of 2-element list/tuple of start and end
            times.
        :param labels: An N-length list of annotation/detection labels.
        :param frequencies: An N-length list of 2-element list/tuple of low and
            high frequencies.

        :return: Number of annotations/detections written.
        """
        self._write(destination, times, labels, *args, **kwargs)

    @abc.abstractmethod
    def _write(self, destination, times, labels, frequencies=None, **kwargs):
        raise NotImplementedError(
            '_write() method not implemented in derived class')


# TODO: revert to using non-CamelCase naming
from . import raven as Raven
from . import sonicvisualiser as SonicVisualiser
from . import audacity as Audacity

__all__ = [
    'BaseAnnotationReader',
    'BaseAnnotationWriter',
    'Raven',
    'SonicVisualiser',
    'Audacity'
]
