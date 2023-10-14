import csv
import abc
import io
import warnings
from functools import lru_cache
from koogu.data.annotations import BaseAnnotationReader, BaseAnnotationWriter


class Reader(BaseAnnotationReader):
    """
    Reader class for reading Raven selection table format files.

    :param fetch_frequencies: (boolean; default: False) If True, will also
        return annotations' frequency bounds. NaNs will be returned for any
        missing values in file. If False, the respective return value will be
        set to None.
    :param label_column_name: A string (e.g., "Tags") identifying the header
        of the column in the selection table file(s) from which class labels are
        to be extracted. If None (default), will look for a column with the
        header "Tags".

    Example::

        >>> # Create a reader instance
        >>> reader = Raven.Reader(fetch_frequencies=True)
        >>>
        >>> # Read annotations from file and display
        >>> (times, freqs, tags, chs, _) = \\
        ...     reader('20180212_060000.selections.txt')
        >>> print('Start time, Upper freq, Channel, Tag')
        >>> print('----------, ----------, -------, ---')
        >>> for idx in range(len(times)):
        ...     print(
        ...         f'{times[idx][0]}, {freqs[idx][1]}, {chs[idx]}, {tags[idx]}'
        ...     )
    """

    def __init__(self, fetch_frequencies=False, label_column_name=None,
                 **kwargs):

        super(Reader, self).__init__(fetch_frequencies)

        # Undocumented parameter: to specify default label value
        default_label = kwargs.get('default_label', None)

        # Undocumented parameter: to facilitate incorporating checks on
        # additional fields. Specify
        #    - a list of 2- or 3-element fieldspec tuples corresponding to the
        #      additional fields to be read
        #    - a "filter" function that accepts a list containing all the read
        #      fields (basic + the additional fieldspec) and returns a boolean
        #      indicating the validity of the annotation entry
        # For example, to only keep annotations that have a "Quality" value > 5,
        # pass
        #     additional_fieldspec=[('Quality', float)],
        #     filter_fn=check_fn
        # with `check_fn` defined like `def check_fn(sel): return sel[-1] > 5`.
        # If `filter_fn` is not specified, `additional_filterspec` will be
        # ignored.
        self._filter_fn = kwargs.get('filter_fn', None)
        additional_fields_spec = []
        if self._filter_fn is not None and 'additional_fieldspec' in kwargs:
            additional_fields_spec = kwargs['additional_fieldspec']

        # Build the 'read orchestrator's for both simple and multi-file cases
        self._read_orchestrators = {}
        for mf in [True, False]:
            fields_spec, extractors = Reader._get_fields_spec_and_extractors(
                fetch_frequencies, label_column_name,
                default_label=default_label, multi_file=mf)

            self._read_orchestrators[mf] = _ReadOrchestrator(
                fields_spec + additional_fields_spec, *extractors)

    def _fetch(self, filepath, multi_file=False,
               **kwargs):
        """
        Load annotations from ``filepath``.

        :param filepath: Path to a Raven selection table file.
        :param multi_file: Set to True if the selection table file corresponds
            to multiple audio files (usually under a parent directory) instead
            of a single file.

        :return: A 5-element tuple

          - N-length list of 2-element tuples denoting annotations' start and
            end times
          - Either None or an N-length list of 2-element tuples denoting
            annotations' frequency bounds
          - N-length list of tags/class labels
          - N-length list of channel indices (0-based)
          - None if `multi_file` was False. Otherwise, an N-length list of
            audio files corresponding to the returned annotations

        :meta private:
        """

        read_orchestrator = self._read_orchestrators[multi_file]

        with open(filepath, 'r', newline='') as file_h:
            # Get/create an iterator
            annots_itr = _SelectionTableReader.get_annotation_iterator_for_file(
                file_h,
                lambda headers: _SelectionTableReader.get_reader(
                    read_orchestrator, *headers))

            if self._filter_fn is not None:
                annots_itr = filter(self._filter_fn, annots_itr)

            annots = list(annots_itr)

        return read_orchestrator.package(annots)

    @classmethod
    def get_annotations_from_file(cls, seltab_file, fields_spec,
                                  delimiter='\t'):
        """
        A generator for reading Raven selection tables. A simple, fast yet
        efficient way for processing selection tables. Pass in the path to the
        file and a list containing field specifications, and retrieve table
        entries iteratively, without having to load the entire selection table
        file into memory.

        :param seltab_file: Path to a Raven selection table file.
        :param fields_spec: A list of field specifiers. A field specifier must
            be a tuple containing -

            * the name of the field (column header),
            * the corresponding data type, and
            * optionally, a default value.

            The field names (case-sensitive) should match the actual column
            headers in the selection table file. If no matching column header is
            found for aspecified field, then the respective value will be None
            in every returned output. If an annotation entry is blank for a
            specified field, then the respective value returned will be set to
            either the default (if specified) or to None.
        :param delimiter: (optional; default is the tab character) The delimiter
            in the selection table file.

        :return: The generator iteratively yields tuples containing
            type-converted values corresponding to the chosen fields from each
            annotation read. The fields in the tuple will be in the same order
            as that of ``fields_spec``.

        Example::

            >>> file_fields_spec = [('Selection', int, 0),
            ...                     ('Begin Time (s)', float, 0),
            ...                     ('Tags', str),
            ...                     ('Score', float)]
            ...
            >>> for entry in Raven.Reader.get_annotations_from_file(
            ...         'my_annots.selection.txt', file_fields_spec):
            ...     print(entry[0], entry[1], entry[2], entry[3])
        """

        with open(seltab_file, 'r', newline='') as file_h:
            # Get/create an iterator
            annots_itr = _SelectionTableReader.get_annotation_iterator_for_file(
                file_h,
                lambda headers: _SelectionTableReader(fields_spec, headers),
                delimiter=delimiter)

            # Process all remaining lines in file and "yield" each selection
            yield from annots_itr

    @classmethod
    def _get_fields_spec_and_extractors(
            cls, fetch_frequencies, label_column_name,
            default_label=None, multi_file=False):
        """
        Builds the necessary "fields_spec" based on the available function
        parameter choices and returns it along with 5 callables that can be
        applied to selections, read from a Raven SelectionTable file, for
        extracting the appropriate values to return (see `_fetch()`).

        :meta private:
        """

        in_fields_spec = [
            _ReadWriteSpecs.sttime[:3],
            _ReadWriteSpecs.entime[:3],
            (label_column_name or _ReadWriteSpecs.clabel[0],
             None,
             default_label or _ReadWriteSpecs.clabel[2]),
            _ReadWriteSpecs.chlnum[:3]
        ]
        extract_times_fn = cls._extract_times

        if fetch_frequencies:
            in_fields_spec += [  # Include default bandwidth [0, inf]
                _ReadWriteSpecs.lofreq[:3],
                _ReadWriteSpecs.hifreq[:3]
            ]
            extract_freqs_fn = cls._extract_freqs
        else:
            extract_freqs_fn = cls._nothing

        if multi_file:
            in_fields_spec += [
                _ReadWriteSpecs.bgfile[:3],
                _ReadWriteSpecs.foffst[:3]
            ]
            if fetch_frequencies:
                extract_times_fn = cls._extract_times_with_offset_past_freq
                extract_files_fn = cls._extract_files_past_freq
            else:
                extract_times_fn = cls._extract_times_with_offset_no_freq
                extract_files_fn = cls._extract_files_no_freq
        else:
            extract_files_fn = cls._nothing

        return in_fields_spec, (
            extract_times_fn,
            extract_freqs_fn,
            cls._extract_tags,
            cls._extract_channels,
            extract_files_fn)

    @staticmethod
    def _nothing(_):
        return None

    @staticmethod
    def _extract_times(selections):
        return list(map(lambda sel: [sel[0], sel[1]], selections))

    @staticmethod
    def _extract_times_with_offset_no_freq(selections):
        return list(map(lambda sel: [sel[5], sel[5] + (sel[1] - sel[0])],
                    selections))

    @staticmethod
    def _extract_times_with_offset_past_freq(selections):
        return list(map(lambda sel: [sel[7], sel[7] + (sel[1] - sel[0])],
                    selections))

    @staticmethod
    def _extract_freqs(selections):
        return list(map(lambda sel: [sel[4], sel[5]], selections))

    @staticmethod
    def _extract_tags(selections):
        return list(map(lambda sel: sel[2], selections))

    @staticmethod
    def _extract_channels(selections):
        # turn channel numbers into zero based indices
        return list(map(lambda sel: sel[3] - 1, selections))

    @staticmethod
    def _extract_files_no_freq(selections):
        return list(map(lambda sel: sel[4], selections))

    @staticmethod
    def _extract_files_past_freq(selections):
        return list(map(lambda sel: sel[6], selections))


class _ReadWriteSpecs(metaclass=abc.ABCMeta):
    # RHS tuples' contents:
    #       Column name, data type, default or None, output formatting specifier
    selnum = ('Selection', int, None, 'd')
    chlnum = ('Channel', int, 1, 'd')        # To default to 1
    sttime = ('Begin Time (s)', float, None, '.6f')
    entime = ('End Time (s)', float, None, '.6f')
    lofreq = (('Low Freq (Hz)', 'Low Frequency (Hz)'),
              float, BaseAnnotationReader.default_float(), '.2f')
    hifreq = (('High Freq (Hz)', 'High Frequency (Hz)'),
              float, BaseAnnotationReader.default_float(), '.2f')
    clabel = ('Tags', None, None, 's')
    bgfile = ('Begin File', None, None, 's')
    foffst = ('File Offset (s)', float, None, '.6f')
    dscore = ('Score', float, None, '.2f')


class _ReadOrchestrator:
    """
    Also serves as a convenient wrapper around the `fields_spec` list, so that
    functions having `fields_spec` list as an input parameter can be cached
    (since the parameter can now be hashed).
    """
    def __init__(self, fields_spec,
                 extract_times_fn,
                 extract_freqs_fn,
                 extract_tags_fn,
                 extract_channels_fn,
                 extract_files_fn):

        self.fields_spec_list = fields_spec

        self._extract_times_fn = extract_times_fn
        self._extract_freqs_fn = extract_freqs_fn
        self._extract_tags_fn = extract_tags_fn
        self._extract_channels_fn = extract_channels_fn
        self._extract_files_fn = extract_files_fn

    def package(self, selections):
        return \
            self._extract_times_fn(selections), \
            self._extract_freqs_fn(selections), \
            self._extract_tags_fn(selections), \
            self._extract_channels_fn(selections), \
            self._extract_files_fn(selections)


class _SelectionTableReader:
    """
    Primarily used via the :meth:`get_reader` interface, which is efficient due
    to reuse of instantiated objects.
    """

    def __init__(self, fields_spec, field_headers):
        """
        Builds a list of 2-element tuples:
          - a function for conversion
          - a tuple of items making 2nd & beyond arguments to the function
        The first argument to each conversion function will be a list of items
        read from a row of the selection table file.
        """

        self._converters_n_args = []
        for field_spec in fields_spec:
            pos = None
            if isinstance(field_spec[0], (list, tuple)):  # See if list-like
                for fieldname_option in field_spec[0]:    # Use first matched
                    if fieldname_option in field_headers:
                        pos = field_headers.index(fieldname_option)
                        break
            elif field_spec[0] in field_headers:          # Must be a string
                pos = field_headers.index(field_spec[0])

            converter = _SelectionTableReader._get_field_converter(
                pos is not None, *(field_spec[1:]))

            if converter is None:  # Puke!
                raise LookupError(
                    f'Required field "{field_spec[0]}" does not exist')

            self._converters_n_args.append(
                (converter, (pos, *(field_spec[1:]))))

    @staticmethod
    @lru_cache(maxsize=16)
    def _get_field_converter(field_present, cast_type=None, default_val=None):

        if field_present:                   # Field exists in file.
            if cast_type is not None:           # Conversion requested
                if default_val is not None:         # Default value available
                    # Set to attempt conversion only if entry not empty
                    return _SelectionTableReader.check_n_convert
                else:                               # No defaults
                    # Set to attempt conversion. Could blow up!
                    return _SelectionTableReader.convert_nocheck
            else:                               # No conversion
                if default_val is not None:         # Default value available
                    if type(default_val) is not str:
                        warnings.warn(
                            'Non-string default '
                            f'{default_val} ({type(default_val)})'
                            ' specified for a field without conversion')
                    return _SelectionTableReader.asis_or_default
                else:                               # Pass as-is
                    return _SelectionTableReader.asis
        else:                               # Missing column in file.
            if default_val is not None:         # Default value specified
                return _SelectionTableReader.default

        return None     # Nothing found. Must Puke!

    @staticmethod
    def asis(values, pos, cast_type=None, default=None):
        return values[pos]

    @staticmethod
    def asis_or_default(values, pos, unused, default):
        return default if values[pos] == '' else values[pos]

    @staticmethod
    def convert_nocheck(values, pos, cast_type, default=None):
        return cast_type(values[pos])

    @staticmethod
    def check_n_convert(values, pos, cast_type, default):
        return default if values[pos] == '' else cast_type(values[pos])

    @staticmethod
    def default(rain, hail, sunshine, default):
        # Rain, hail or sunshine, simply return the default.
        return default

    @staticmethod
    @lru_cache(maxsize=16)
    def get_reader(read_orchestrator, *field_headers):
        return _SelectionTableReader(read_orchestrator.fields_spec_list,
                                     field_headers)

    def convert_selection(self, selection_fields):
        """
        Apply field-specific converters at appropriate positions and return a
        tuple representing the selection.
        `selection_fields` is a list of fields read (in order, as per file's
        contents) from a single line of a selection table file.
        """
        return tuple(map(
            lambda conv_arg: conv_arg[0](selection_fields, *conv_arg[1]),
            self._converters_n_args))

    @classmethod
    def get_annotation_iterator_for_file(
            cls, seltab_file_h, reader_creator_fn, delimiter='\t'):
        """
        Return an iterator that will generate processed selections.
        `seltab_file_h` must be an open file handle.
        `reader_creator_fn` must be a callable that returns an instance of
        _SelectionTableReader.
        """

        lines_iterator = csv.reader(seltab_file_h, delimiter=delimiter)

        # Read header
        field_headers = next(lines_iterator)

        # Get a reader (an instance of _SelectionTableReader), based on the
        # header, for appropriate processing of fields from the remaining lines
        # of the file.
        reader = reader_creator_fn(field_headers)

        # Return an iterator that will eventually process all remaining lines in
        # file.
        return map(reader.convert_selection, lines_iterator)


class Writer(BaseAnnotationWriter):
    """
    Writer class for writing annotations/detections to Raven selection table
    format files.

    :param write_frequencies: Boolean (default: False) directing whether to
        include the "Low Freq (Hz)" and "High Freq (Hz)" fields in the outputs.
    :param extra_fields_spec: Optional list of 2-element tuples identifying any
        additional fields to add to the output and their respective formats.
        E.g., [('Model used', 's')] will add an extra field named "Model used"
        and set the values in the fields to be formatted as strings.
    :param add_selection_number: Boolean (default: True) directing whether to
        include the "Selection" field.
    :param add_channel: Boolean (default: True) directing whether to include the
        "Channel" field.
    :param add_score: Boolean (default: False) directing whether to include the
        "Scores" field. Use when saving detections.
    """

    def __init__(self, write_frequencies=False, extra_fields_spec=None,
                 **kwargs):

        super(Writer, self).__init__(write_frequencies)

        self._add_selnum = kwargs.get('add_selection_number', True)
        self._add_channel = kwargs.get('add_channel', True)
        self._add_score = kwargs.get('add_score', False)

        self._extra_fields_spec = []
        if extra_fields_spec is not None:
            for field in extra_fields_spec:
                if isinstance(field, (list, tuple)):
                    # Assume field format exists
                    self._extra_fields_spec.append((field[0], field[1]))
                else:
                    # Only field name given, use default formatting
                    self._extra_fields_spec.append((field, ''))

    def _write(self, out_file, times, labels,
               frequencies=None, channels=None, scores=None,
               file_offset=None, begin_file=None, selection_num_offset=0,
               new_file=True,  # 'False' condition used only in multi-file case
               extra_fields_values_dict=None, **kwargs):
        """
        Write out annotations/detections to Raven selection table file.

        :param out_file: Can be a path string or an open file handle (with write
            access). The latter case is useful when combining outputs from
            multiple audio files into a single selection table file.
        :param times: An N-length list of 2-element list/tuple of start and end
            times.
        :param labels: An N-length list of annotation/detection labels.
        :param frequencies: An N-length list of 2-element list/tuple of low and
            high frequencies.
        :param channels: An N-length list of channel numbers.
        :param scores: An N-length list of detection scores.
        :param file_offset: If specified (must be a scalar value), values in
            `times` will be shifted accordingly, and the "File Offset (s)"
            field will be added. (useful when combining outputs)
        :param begin_file: If specified (must be a single filename string), the
            "Begin File" field will be added. (useful when combining outputs)
        :param selection_num_offset: If specified (must be a positive integer),
            the selection numbers of to-be-written annotations will be advanced
            by this amount. (useful when combining outputs)
        :param new_file: If True, will add the header row to the output file
            (useful when combining outputs). Also dictates the file mode to open
            the output file with when `out_file` is a path string.
        :param extra_fields_values_dict: A dictionary containing N-length lists
            of corresponding values for each item in the `extra_fields_spec`
            that was passed to the constructor. The keys in the dict must match
            the field names from `extra_fields_spec`.

        :return: Number of annotations/detections written.
        """

        num_rows = len(times)

        # Validate extra fields
        extra_fields_values = []    # Copy in the order of _extra_fields_spec
        if extra_fields_values_dict is None:
            # Set everything (if any) to false
            extra_fields_validity = [False for _ in self._extra_fields_spec]
        else:
            extra_fields_validity = []
            for ef_name, _ in self._extra_fields_spec:
                if ef_name in extra_fields_values_dict:
                    extra_fields_values.append(
                        extra_fields_values_dict[ef_name])
                    extra_fields_validity.append(True)
                else:
                    extra_fields_values.append((None for _ in times))
                    extra_fields_validity.append(False)

        header, fmt_str = self._get_header_and_fmt_str(
            file_offset is not None, begin_file is not None,
            channels is not None,
            frequencies is not None,
            scores is not None,
            *extra_fields_validity)

        with _FileOrPath(out_file, 'w' if new_file else 'a') as out_fh:

            if new_file:        # Add header if it was a new file
                out_fh.write(header)

            for line_items in zip(
                range(selection_num_offset + 1,
                      selection_num_offset + num_rows + 1),
                (None for _ in times) if channels is None else channels,
                times if file_offset is None else map(
                    lambda t: (t[0] + file_offset, t[1] + file_offset), times),
                (None for _ in times) if frequencies is None else frequencies,
                labels,
                (None for _ in times) if scores is None else scores,
                ((t[0] for t in times) if file_offset is not None
                 else (None for _ in times)),
                (begin_file for _ in times) or (None for _ in times),
                *extra_fields_values
            ):
                out_fh.write(fmt_str.format(*line_items))

        return num_rows

    @lru_cache(maxsize=16)
    def _get_header_and_fmt_str(self,
                                file_offset_available, begin_file_available,
                                channel_available,
                                freqs_available,
                                scores_available,
                                *extra_fields_validity):

        header = []
        formatter = []

        next_idx = 0        # Selection
        if self._add_selnum:
            header.append(_ReadWriteSpecs.selnum[0])
            formatter.append(f'{{{next_idx}:{_ReadWriteSpecs.selnum[-1]}}}')

        next_idx = 1        # Channel
        if self._add_channel:
            header.append(_ReadWriteSpecs.chlnum[0])
            formatter.append(f'{{{next_idx}:{_ReadWriteSpecs.chlnum[-1]}}}'
                             if channel_available else
                             f'{_ReadWriteSpecs.chlnum[2]}')

        next_idx = 2        # Begin Time (s)
        header.append(_ReadWriteSpecs.sttime[0])
        formatter.append(f'{{{next_idx}[0]:{_ReadWriteSpecs.sttime[-1]}}}')
        # next_idx = 2      # End Time (s)
        header.append(_ReadWriteSpecs.entime[0])
        formatter.append(f'{{{next_idx}[1]:{_ReadWriteSpecs.entime[-1]}}}')

        if self._write_frequencies:
            next_idx = 3    # Low Freq (Hz)
            header.append(_ReadWriteSpecs.lofreq[0][0])
            formatter.append(f'{{{next_idx}[0]:{_ReadWriteSpecs.lofreq[-1]}}}'
                             if freqs_available else '')
            # next_idx = 3  # High Freq (Hz)
            header.append(_ReadWriteSpecs.hifreq[0][0])
            formatter.append(f'{{{next_idx}[1]:{_ReadWriteSpecs.hifreq[-1]}}}'
                             if freqs_available else '')

        next_idx = 4        # Tags
        header.append(_ReadWriteSpecs.clabel[0])
        formatter.append(f'{{{next_idx}:{_ReadWriteSpecs.clabel[-1]}}}')

        next_idx = 5        # Score
        if self._add_score:
            header.append(_ReadWriteSpecs.dscore[0])
            formatter.append(f'{{{next_idx}:{_ReadWriteSpecs.dscore[-1]}}}'
                             if scores_available else '')

        if file_offset_available:
            next_idx = 6    # File Offset (s)
            header.append(_ReadWriteSpecs.foffst[0])
            formatter.append(f'{{{next_idx}:{_ReadWriteSpecs.foffst[-1]}}}')
        if begin_file_available:
            next_idx = 7    # Begin File
            header.append(_ReadWriteSpecs.bgfile[0])
            formatter.append(f'{{{next_idx}:{_ReadWriteSpecs.bgfile[-1]}}}')

        next_idx = 8
        for (field_name, field_fmt), field_validity in zip(
                self._extra_fields_spec, extra_fields_validity):
            header.append(field_name)
            formatter.append(f'{{{next_idx}:{field_fmt}}}'
                             if field_validity else '')

            next_idx += 1

        return '\t'.join(header) + '\n', '\t'.join(formatter) + '\n'


class _FileOrPath:
    """
    Convenient 'context' interface for writing to new file or continue writing
    to an existing one.
      - If `file` was already a file object, nothing to do.
      - If `file` was a path string, open the file with the chosen `open_mode`
        (must be one of 'w' or 'a').
    """
    def __init__(self, file, open_mode):
        self._path_or_file = file

        if isinstance(file, io.TextIOBase) and hasattr(file, 'write'):
            # Was a file handle already
            if file.closed:
                raise ValueError('File already closed. Cannot write further.')

            self._must_open = False
            self._open_mode = None
        else:
            self._must_open = True
            self._open_mode = open_mode

    def __enter__(self):
        if self._must_open:
            self._path_or_file = open(self._path_or_file, self._open_mode)

        return self._path_or_file

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._must_open:
            temp = self._path_or_file.name
            self._path_or_file.close()
            self._path_or_file = temp   # Restore value?
