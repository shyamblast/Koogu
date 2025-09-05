import os
import sys
import argparse
import logging
import numpy as np

from . import recognize
from .data import annotations
from .model.trained_model import TrainedModel
from .utils import instantiate_logging
from .utils.detections import LabelHelper
from .utils.terminal import ArgparseConverters
from .utils.filesystem import AudioFileList, get_valid_audio_annot_entries
from .utils.config import Config, ConfigError


def from_annotations(
        model_dir, audio_annot_list,
        audio_root, annots_root, output_root,
        annotation_reader=None,
        remap_labels_dict=None,
        negative_class_label=None,
        **kwargs
):
    """
    Apply a trained model on a test dataset with audio-annotation mappings.

    :param model_dir: Path to directory where the trained model is available.
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
    :param annots_root: The full paths of annotations files listed in
        ``audio_annot_list`` are resolved using this as the base directory.
    :param output_root: Raw detection scores will be written to this directory.

    **Optional parameters**

    :param annotation_reader: If not None, must be an annotation reader instance
        from the :mod:`~koogu.data.annotations` module. Defaults to Raven
        :class:`~koogu.data.annotations.Raven.Reader`.
    :param remap_labels_dict: If not None, must be a Python dictionary
        describing mapping of class labels. For details, see similarly named
        parameter to the constructor of
        :class:`koogu.utils.detections.LabelHelper`.

        .. note:: Mappings for which targets are not among the model's classes
           will be ignored.

    :param negative_class_label: Specify the same string (e.g. 'Other', 'Noise')
        that was used as a label to identify negative class clips (those that
        did not match any annotations) when pre-processing the training dataset
        using :func:`koogu.prepare.from_annotations()`. Set to None (default),
        if training did not consider auto-extracted negative class samples.
    :param clip_advance: The value defines the amount of clip advance when
        pre-processing audio. If specified, overrides the value that was read
        from the model's metadata.
    :param filetypes: Audio file types to restrict processing to. Option is
        ignored if processing a single file. Can specify multiple types, as a
        list. Defaults to ['.wav', '.WAV', '.flac', '.aif', '.mp3'].
    :param channels: (int or list of ints) When audio files have multiple
        channels, set which channels to restrict processing to. If
        unspecified, all available channels will be processed. E.g., setting
        to 0 saves the first channel, setting to [0, 2] saves the first and
        third channels.
    :param batch_size: (int; default: 1) Size to batch audio file's clips into.
        Increasing this may improve speed on computers with high RAM.
    :param num_fetch_threads: (int; default: 1) Number of background threads
        that will fetch audio from files in parallel.
    :param show_progress: (bool) If enabled, messages indicating progress of
        processing will be shown on console output.

    .. seealso::
            :mod:`koogu.data.annotations`
    """

    logger = logging.getLogger(__name__)

    model_classes = TrainedModel.get_model_class_names(model_dir)

    if annotation_reader is None:
        annotation_reader = annotations.Raven.Reader()

    label_helper = LabelHelper(
        model_classes,
        remap_labels_dict=remap_labels_dict,
        negative_class_label=negative_class_label,
        fixed_labels=True,
        assessment_mode=False   # Force this to be False
    )

    # Discard invalid entries, if any
    v_audio_annot_list = get_valid_audio_annot_entries(
        audio_annot_list,
        audio_root,
        annots_root,
        logger=logger)

    ig_kwargs = {}  # Undocumented settings
    auf_types = kwargs.pop('filetypes', None)
    if auf_types is not None:
        ig_kwargs['filetypes'] = auf_types
    src_generator = AudioFileList.from_annotations(
        v_audio_annot_list,
        audio_root, annots_root,
        annotation_reader,
        **ig_kwargs)

    if kwargs.pop('ignore_zero_annot_files', False):
        # Undocumented setting. When explicitly set, must process only files
        # having valid annots

        req_chs = kwargs.get('channels', None)  # get() coz it's needed later

        input_generator = (
            audio_file
            for audio_file, _, annots_labels, annots_channels in src_generator
            if _file_has_target_annots(annots_labels, annots_channels, req_chs,
                                       label_helper)
        )

    else:
        # Must process everything
        input_generator = (
            audio_file
            for audio_file, _, _, _ in src_generator
        )

    recognize(model_dir, audio_root,
              output_dir=None,
              raw_detections_dir=output_root,
              src_generator=input_generator,  # Undocumented, special arg.
              **kwargs)


def from_top_level_dirs(model_dir, audio_root, output_root, **kwargs):
    """
    Apply a trained model on a test dataset with classes defined by top-level
    folders' names.

    :param model_dir: Path to directory where the trained model is available.
    :param audio_root: Full path to the base directory containing the audio
        files of the test dataset.
    :param output_root: Raw detection scores will be written to this directory.

    **Optional parameters**

    :param clip_advance: The value defines the amount of clip advance when
        pre-processing audio. If specified, overrides the value that was read
        from the model's metadata.
    :param filetypes: Audio file types to restrict processing to. Option is
        ignored if processing a single file. Can specify multiple types, as a
        list. Defaults to ['.wav', '.WAV', '.flac', '.aif', '.mp3'].
    :param channels: (int or list of ints) When audio files have multiple
        channels, set which channels to restrict processing to. If
        unspecified, all available channels will be processed. E.g., setting
        to 0 saves the first channel, setting to [0, 2] saves the first and
        third channels.
    :param batch_size: (int; default: 1) Size to batch audio file's clips into.
        Increasing this may improve speed on computers with high RAM.
    :param num_fetch_threads: (int; default: 1) Number of background threads
        that will fetch audio from files in parallel.
    :param show_progress: (bool) If enabled, messages indicating progress of
        processing will be shown on console output.

    """

    recognize(model_dir, audio_root,
              output_dir=None,
              raw_detections_dir=output_root,
              recursive=True,   # Force this, since everything must be processed
              **kwargs)


def _file_has_target_annots(annots_labels, annots_channels, req_chs,
                            label_helper):
    if req_chs is not None:
        # Keep only annots corresponding to req_chs
        in_req_chs_annots_idxs = [idx for idx, annot_ch in
                                  enumerate(annots_channels) if
                                  annot_ch in req_chs]
        annots_labels = [annots_labels[idx] for idx in
                         in_req_chs_annots_idxs]

    # Get mapped idxs for unique labels
    annot_class_idxs = [
        label_helper.labels_to_indices[a_lbl]
        for a_lbl in list(set(annots_labels))
        if a_lbl in label_helper.labels_to_indices
    ]

    # Remove negative_class' idx
    valid_classes_in_file = list(
        set(annot_class_idxs) - set([label_helper.negative_class_index])
    )

    return len(valid_classes_in_file) > 0


__all__ = ['from_annotations', 'from_top_level_dirs']


def cmdline_parser(parser=None):

    if parser is None:
        parser = argparse.ArgumentParser(
            prog='koogu-test', allow_abbrev=False,
            description='Test a trained model using the test dataset.')

    parser.add_argument(
        'cfg_file', metavar='<CONFIG FILE>',
        help='Path to config file.')
    parser.add_argument(
        'model_name', metavar='<MODEL NAME>',
        help='Name of the trained model.')

    arg_group_in_ctrl = parser.add_argument_group('Input control')
    arg_group_in_ctrl.add_argument(
        '--filetypes', metavar='EXTN', nargs='+',
        help='Audio file types to restrict processing to. Can specify multiple '
             'types separated by whitespaces. By default, will process all '
             'discovered files with the following extensions: [' + ', '.join(
                 AudioFileList.default_audio_filetypes) +
             ']. When test_audio_annotations_map is available, this option is '
             'only considered where audio sources are directories.')
    arg_group_in_ctrl.add_argument(
        '--channels', metavar='#', nargs='+',
        type=ArgparseConverters.all_or_posint,
        help='Channels to restrict processing to. List out the desired channel '
             'indices, separated by whitespaces. If unspecified, all available '
             'channels will be processed. Channel indices must be 0-based.')
    arg_group_in_ctrl.add_argument(
        '--clip-advance', metavar='SECONDS', dest='clip_advance',
        type=ArgparseConverters.positive_float,
        help='Override \"clip advance\". When audio files\' contents are broken'
             ' up into clips, by default the amount of overlap between '
             'successive clips is determined by the settings that were in place'
             ' during model training. Use this flag to override that quantity, '
             'by setting a different amount of advance (in seconds) between '
             'successive clips.')

    arg_group_prctrl = parser.add_argument_group('Process control')
    arg_group_prctrl.add_argument(
        '--threads', dest='num_fetch_threads', metavar='NUM', default=1,
        type=ArgparseConverters.positive_integer,
        help='Number of threads that will fetch audio from files in parallel.')
    arg_group_prctrl.add_argument(
        '--batch-size', dest='batch_size', metavar='NUM', default=1,
        type=ArgparseConverters.positive_integer,
        help='Size to batch audio file\'s clips into (default: %(default)d). '
             'Increasing this may improve speed on computers with higher RAM.')
    arg_group_prctrl.set_defaults(show_progress=True)   # Force this

    arg_group_logging = parser.add_argument_group('Logging')
    arg_group_logging.add_argument(
        '--log', dest='log_file', metavar='LOGFILE',
        help='If specified, logging will be written out to this file instead of'
             ' the default. (Default: <PROJECT-LOGS-DIR>/test.log)')
    arg_group_logging.add_argument(
        '--loglevel', dest='log_level',
        choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'],
        default='INFO', help='Logging level.')

    parser.set_defaults(exec_fn=cmdline_test)

    return parser


def cmdline_test(cfg_file, model_name, log_level,
                 **kwargs):
    """
    Functionality invoked via the command-line interface.
    """

    # Load config
    try:
        cfg = Config(cfg_file, 'data.annotations', 'prepare')
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

    model_dir = os.path.join(cfg.paths.model, model_name)
    if not os.path.exists(model_dir):
        print(f'Error: Model "{model_name}" not found at "{cfg.paths.model}"',
              file=sys.stderr)
        exit(1)

    logfile = kwargs.pop('log_file', None) or os.path.join(cfg.paths.logs,
                                                           'test.log')
    instantiate_logging(logfile, log_level, 'a')

    logger = logging.getLogger(__name__)

    logger.info(f'Model name: {model_name}')

    recog_kwargs = {}   # kwargs tunneled thru to inference.recognize()

    # 'Process control' args
    num_fetch_threads = kwargs.pop('num_fetch_threads', None)
    if num_fetch_threads is not None:
        recog_kwargs['num_fetch_threads'] = num_fetch_threads
    val = kwargs.pop('batch_size', None)
    if val is not None:
        recog_kwargs['batch_size'] = val
    val = kwargs.pop('show_progress', None)
    if val is not None:
        recog_kwargs['show_progress'] = val

    # 'Input control' args
    auf_types = kwargs.pop('filetypes', None)
    if auf_types is not None:
        recog_kwargs['filetypes'] = auf_types
    val = kwargs.pop('channels', None)
    if val is not None:
        recog_kwargs['channels'] = np.sort(np.unique(val).astype(np.uint32))
    val = kwargs.pop('clip_advance', None)
    if val is not None:
        recog_kwargs['clip_advance'] = val

    if cfg.paths.test_audio_annotations_map is not None:
        # test_audio_annotations_map is given

        if cfg.paths.test_annotations is None or \
                (not os.path.exists(cfg.paths.test_annotations)):
            print('Error: No or invalid path specified in test_annotations',
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
        annotation_reader = ar_type(**ar_kwargs)

        from_annotations(
            model_dir, cfg.paths.test_audio_annotations_map,
            cfg.paths.test_audio,
            cfg.paths.test_annotations,
            os.path.join(cfg.paths.test_detections, model_name),
            annotation_reader=annotation_reader,
            remap_labels_dict=cfg.data.annotations.remap_labels_dict,
            negative_class_label=cfg.prepare.negative_class,
            **recog_kwargs
        )

    else:
        # Annotation map wasn't given. Process everything under test_audio
        from_top_level_dirs(
            model_dir, cfg.paths.test_audio,
            os.path.join(cfg.paths.test_detections, model_name),
            **recog_kwargs
        )

    logging.shutdown()


def main():
    # for setting up script entry point

    args = cmdline_parser().parse_args()

    args.exec_fn(**{k: getattr(args, k)
                    for k in vars(args) if k != 'exec_fn'})
