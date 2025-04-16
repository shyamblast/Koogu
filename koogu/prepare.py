import os
import logging
import sys
import argparse
import warnings
from .data import annotations
from .data.preprocess import batch_process, get_unique_labels_from_annotations
from .data.raw import Settings
from .utils.detections import LabelHelper
from .utils.terminal import ArgparseConverters
from .utils.config import Config, ConfigError
from .utils.filesystem import AudioFileList, \
    get_valid_audio_annot_entries, recursive_listing
from .utils import instantiate_logging


def from_selection_table_map(audio_settings, audio_seltab_list,
                             audio_root, seltab_root, output_root,
                             annotation_reader=None,
                             desired_labels=None,
                             remap_labels_dict=None,
                             negative_class_label=None,
                             **kwargs):
    """
    Deprecated interface, retained for backwards compatibility.

    :meta private:
    """
    warnings.showwarning(
        'The interface koogu.prepare.from_selection_table_map is deprecated '
        'and will be removed in a future release. Instead of this, please use '
        'koogu.prepare.from_annotations.',
        DeprecationWarning, __name__, '')

    return from_annotations(audio_settings, audio_seltab_list,
                            audio_root, seltab_root, output_root,
                            annotation_reader=annotation_reader,
                            desired_labels=desired_labels,
                            remap_labels_dict=remap_labels_dict,
                            negative_class_label=negative_class_label,
                            **kwargs)


def from_annotations(audio_settings, audio_annot_list,
                     audio_root, annot_root, output_root,
                     annotation_reader=None,
                     desired_labels=None,
                     remap_labels_dict=None,
                     negative_class_label=None,
                     **kwargs):
    """
    Pre-process training data using info contained in ``audio_annot_list``.

    :param audio_settings: A dictionary specifying the parameters for processing
        audio from files.
    :param audio_annot_list: A list containing pairs (list-like) of relative
        paths to audio files and the corresponding annotation file(s). The
        latter can be a single path string or a nested list of path strings.
        Alternatively, you could also specify (path to) a csv file containing
        these pairs of entries (in the same order; include 3rd, 4th, ...
        columns if you need to specify additional annotation files corresponding
        to an audio path). Only use the csv option if the paths are simple
        (i.e., the filenames do not contain commas or other special characters).
    :param audio_root: The full paths of audio files listed in
        ``audio_annot_list`` are resolved using this as the base directory.
    :param annot_root: The full paths of annotations files listed in
        ``audio_annot_list`` are resolved using this as the base directory.
    :param output_root: "Prepared" data will be written to this directory.
    :param annotation_reader: If not None, must be an annotation reader instance
        from the :mod:`~koogu.data.annotations` module. Defaults to Raven
        :class:`~koogu.data.annotations.Raven.Reader`.
    :param desired_labels: The target set of class labels. If not None, must be
        a list of class labels. Any annotations (read from the annotation files)
        having labels that are not in this list will be discarded. This list
        will be used to populate classes_list.json that will define the classes
        for the project. If None, then the list of classes will be populated
        with the annotation labels read from all annotation files.
    :param remap_labels_dict: If not None, must be a Python dictionary
        describing mapping of class labels. For details, see similarly named
        parameter to the constructor of
        :class:`koogu.utils.detections.LabelHelper`.

        .. note:: If ``desired_labels`` is not None, mappings for which targets
           are not listed in ``desired_labels`` will be ignored.

    :param negative_class_label: A string (e.g. 'Other', 'Noise') which will be
        used as a label to identify the negative class clips (those that did not
        match any annotations). If None (default), saving of negative class
        clips will be disabled.

    Other parameters specific to
    :func:`koogu.utils.detections.assess_annotations_and_clips_match`
    can also be specified, and will be passed as-is to the function.

    :return: A dictionary whose keys are annotation tags (either discovered from
        the set of annotations, or same as ``desired_labels`` if not None) and
        the values are the number of clips produced for the corresponding class.

    .. seealso::
        :mod:`koogu.data.annotations`
    """

    logger = logging.getLogger(__name__)

    audio_settings_c = Settings.Audio(**audio_settings)

    # Discard invalid entries, if any
    v_audio_annot_list = get_valid_audio_annot_entries(
            audio_annot_list, audio_root, annot_root, logger=logger)

    if len(v_audio_annot_list) == 0:
        print('Nothing to process')
        return {}

    if annotation_reader is None:
        annotation_reader = annotations.Raven.Reader()

    if desired_labels is not None:
        is_fixed_classes = True
        classes_list = desired_labels
    else:       # need to discover list of classes
        is_fixed_classes = False
        classes_list = get_unique_labels_from_annotations(
            annot_root, [af for _, rhs in v_audio_annot_list for af in rhs],
            annotation_reader, num_threads=kwargs.get('num_threads', None))

    # ---------- 1. Input generator --------------------------------------------
    ig_kwargs = {}      # Undocumented settings
    if negative_class_label is not None:
        # Deal with this only if there was a request to save non-match clips
        auf_types = kwargs.pop('filetypes', None)
        if auf_types is not None:
            ig_kwargs['filetypes'] = auf_types
    input_generator = AudioFileList.from_annotations(
        v_audio_annot_list,
        audio_root, annot_root,
        annotation_reader,
        **ig_kwargs)

    # ---------- 2. LabelHelper ------------------------------------------------
    label_helper = LabelHelper(
        classes_list,
        remap_labels_dict=remap_labels_dict,
        negative_class_label=negative_class_label,
        fixed_labels=is_fixed_classes,
        assessment_mode=False)

    # ---------- 3. Data aggregator --------------------------------------------
    # Extract args meant for assess_annotations_and_clips_match()
    match_fn_kwargs = dict()
    if 'min_annot_overlap_fraction' in kwargs:
        assert (0.0 < kwargs['min_annot_overlap_fraction'] <= 1.0)
        match_fn_kwargs['min_annot_overlap_fraction'] = \
            kwargs.pop('min_annot_overlap_fraction')
    if 'keep_only_centralized_annots' in kwargs:
        match_fn_kwargs['keep_only_centralized_annots'] = \
            kwargs.pop('keep_only_centralized_annots')
    if label_helper.negative_class_index is not None:
        match_fn_kwargs['negative_class_idx'] = \
            label_helper.negative_class_index

        if 'max_nonmatch_overlap_fraction' in kwargs:
            assert (0.0 <= kwargs['max_nonmatch_overlap_fraction'] <
                    match_fn_kwargs.get('min_annot_overlap_fraction', 1.0))
            match_fn_kwargs['max_nonmatch_overlap_fraction'] = \
                kwargs.pop('max_nonmatch_overlap_fraction')

    aggregator_kwargs = dict(
        match_fn_kwargs=match_fn_kwargs,
        attempt_salvage=kwargs.pop('attempt_salvage', False)
    )

    return batch_process(
        audio_settings_c, input_generator, label_helper, aggregator_kwargs,
        audio_root, output_root,
        **kwargs)


def from_top_level_dirs(audio_settings, class_dirs,
                        audio_root, output_root,
                        remap_labels_dict=None,
                        **kwargs):
    """
    Pre-process training data available as audio files in ``class_dirs``.

    :param audio_settings: A dictionary specifying the parameters for processing
        audio from files.
    :param class_dirs: A list containing relative paths to class-specific
        directories containing audio files. Each directory's contents will be
        recursively searched for audio files.
    :param audio_root: The full paths of the class-specific directories listed
        in ``class_dirs`` are resolved using this as the base directory.
    :param output_root: "Prepared" data will be written to this directory.
    :param remap_labels_dict: If not None, must be a Python dictionary
        describing mapping of class labels. For details, see similarly named
        parameter to the constructor of
        :class:`koogu.utils.detections.LabelHelper`.
    :param filetypes: (optional) Restrict listing to files matching extensions
        specified in this parameter. Has defaults if unspecified.

    :return: A dictionary whose keys are annotation tags (discovered from the
        set of annotations) and the values are the number of clips produced for
        the corresponding class.
    """

    logger = logging.getLogger(__name__)

    audio_settings_c = Settings.Audio(**audio_settings)

    # ---------- 1. Input generator --------------------------------------------
    file_types = kwargs.pop('filetypes', AudioFileList.default_audio_filetypes)

    logger.info('  {:<55s} - {:>5s}'.format('Class', 'Files'))
    logger.info('  {:<55s}   {:>5s}'.format('-----', '-----'))
    for class_name in class_dirs:
        count = 0
        for _ in recursive_listing(os.path.join(audio_root, class_name),
                                   file_types):
            count += 1
        logger.info('  {:<55s} - {:>5d}'.format(class_name, count))

    input_generator = AudioFileList.from_directories(
        audio_root, class_dirs, file_types)

    # ---------- 2. LabelHelper ------------------------------------------------
    label_helper = LabelHelper(
        class_dirs,
        remap_labels_dict=remap_labels_dict,
        fixed_labels=False,
        assessment_mode=False)

    # ---------- 3. Data aggregator --------------------------------------------
    aggregator_kwargs = {}

    return batch_process(
        audio_settings_c, input_generator, label_helper, aggregator_kwargs,
        audio_root, output_root,
        **kwargs)


__all__ = ['from_annotations', 'from_top_level_dirs']


def cmdline_parser(parser=None):

    if parser is None:
        parser = argparse.ArgumentParser(
            prog='koogu.prepare', allow_abbrev=True,
            description='Prepare audio data for training.')

    parser.add_argument(
        'cfg_file', metavar='<CONFIG FILE>',
        help='Path to config file.')

    arg_group_prctrl = parser.add_argument_group('Process control')
    arg_group_prctrl.add_argument(
        '--threads', dest='num_threads', metavar='NUM',
        type=ArgparseConverters.positive_integer,
        help='Number of threads to spawn for parallel execution (default: as ' +
             'many CPUs).')

    arg_group_logging = parser.add_argument_group('Logging')
    arg_group_logging.add_argument(
        '--loglevel', dest='log_level',
        choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'],
        default='INFO', help='Logging level.')

    parser.set_defaults(exec_fn=_prepare)

    return parser


def _prepare(cfg_file, log_level, num_threads=None):
    """Functionality invoked via the command-line interface"""

    # Load config
    try:
        cfg = Config(cfg_file, 'data.audio', 'data.annotations', 'prepare')
    except FileNotFoundError as exc:
        print(f'Error loading config file: {exc.strerror}', file=sys.stderr)
        exit(exc.errno)
    except ConfigError as exc:
        print(f'Error processing config file: {str(exc)}', file=sys.stderr)
        exit(1)
    except Exception as exc:
        print(f'Error processing config file: {repr(exc)}', file=sys.stderr)
        exit(1)

    if not os.path.exists(cfg.paths.train_audio):
        print('Error: Invalid path specified in train_audio', file=sys.stderr)
        exit(2)

    other_args = {'show_progress': False}
    if num_threads is not None:
        other_args['num_threads'] = num_threads

    if cfg.paths.logs is not None:
        instantiate_logging(os.path.join(cfg.paths.logs, 'prepare.log'),
                            log_level)

    exit_code = 0

    if cfg.paths.train_audio_annotations_map is not None:
        # train_audio_annotations_map is given

        if cfg.paths.train_annotations is None or \
                (not os.path.exists(cfg.paths.train_annotations)):
            print('Error: No or invalid path specified in train_annotations',
                  file=sys.stderr)
            exit(2)

        if cfg.data.annotations.annotation_reader is not None:
            ar_type = getattr(annotations,
                              cfg.data.annotations.annotation_reader).Reader
        else:
            ar_type = annotations.Raven.Reader  # Default to Raven.Reader
        ar_kwargs = dict()
        if ar_type == annotations.Raven.Reader:
            if cfg.data.annotations.raven_label_column_name is not None:
                ar_kwargs['label_column_name'] = \
                    cfg.data.annotations.raven_label_column_name
            if cfg.data.annotations.raven_default_label is not None:
                ar_kwargs['default_label'] = \
                    cfg.data.annotations.raven_default_label

        if cfg.prepare.min_annotation_overlap_fraction is not None:
            other_args['min_annot_overlap_fraction'] = \
                cfg.prepare.min_annotation_overlap_fraction
        if cfg.prepare.max_nonmatch_overlap_fraction is not None:
            other_args['max_nonmatch_overlap_fraction'] = \
                cfg.prepare.max_nonmatch_overlap_fraction
        if cfg.prepare.attempt_salvage is not None:
            other_args['attempt_salvage'] = cfg.prepare.attempt_salvage

        from_annotations(
            cfg.data.audio.as_dict(),
            audio_annot_list=cfg.paths.train_audio_annotations_map,
            audio_root=cfg.paths.train_audio,
            annot_root=cfg.paths.train_annotations,
            output_root=cfg.paths.training_samples,
            annotation_reader=ar_type(**ar_kwargs),
            desired_labels=cfg.data.annotations.desired_labels,
            remap_labels_dict=cfg.data.annotations.remap_labels_dict,
            negative_class_label=cfg.prepare.negative_class,
            **other_args)

    else:
        # Annotation map wasn't given. Build the list of classes from the
        # directory listing of cfg.paths.train_audio

        # List of classes (first level directory names)
        try:
            class_dirs = sorted([
                c for c in os.listdir(cfg.paths.train_audio)
                if os.path.isdir(os.path.join(cfg.paths.train_audio, c))])
        except FileNotFoundError as exc:
            print(f'Error reading source directory: {exc.strerror}',
                  file=sys.stderr)
            exit_code = exc.errno
        else:

            if len(class_dirs) == 0:
                print('No classes to process.')

            else:
                from_top_level_dirs(
                    cfg.data.audio.as_dict(),
                    class_dirs,
                    cfg.paths.train_audio,
                    cfg.paths.prepared_training_samples,
                    **other_args)

    if cfg.paths.logs is not None:
        logging.shutdown()

    exit(exit_code)
