
import os
import sys
import logging
from datetime import datetime
import numpy as np
import argparse
import warnings
import h5py
import concurrent.futures
import tensorflow as tf

from koogu.data import tfrecord_helper, FilenameExtensions, DirectoryNames, DatasetDigest, Convert, Settings
from koogu.utils import instantiate_logging
from koogu.utils.terminal import ProgressBar, ArgparseConverters
from koogu.utils.config import Config, ConfigError, datasection2dict, log_config
from koogu.utils.filesystem import recursive_listing, restrict_classes_with_whitelist_file


_program_name = 'clips_to_tfrecords'


def write_time_domain_records(src_root, output_root,
                              validation_split=0.0, test_split=0.0,
                              **kwargs):

    tfrecord_handler = tfrecord_helper.WaveformTFRecordHandler(tracing_on=False)
    transformation_fn = _TransformationFunction(_TransformationFunction.pcm2float, np.float32)

    _batch_process(tfrecord_handler, transformation_fn,
                   src_root, output_root,
                   validation_split, test_split,
                   show_progress=True,
                   **kwargs)


def write_spectral_records(src_root, output_root,
                           fs, spec_settings,
                           validation_split=0.0, test_split=0.0,
                           **kwargs):

    spec_settings = Settings.Spectral(fs, **spec_settings)

    tfrecord_handler = tfrecord_helper.SpectrogramTFRecordHandler(tracing_on=False)
    transformation_fn = [
        _TransformationFunction(_TransformationFunction.pcm2float, np.float32),
        _TransformationFunction(_TransformationFunction.audio2spectral, fs, spec_settings)
        ]

    _batch_process(tfrecord_handler, transformation_fn,
                   src_root, output_root,
                   validation_split, test_split,
                   show_progress=True,
                   **kwargs)


_npz_data_container_fieldname = 'clips'


def _batch_process(tfrecord_handler, transformation_fn,
                   src_root, dest_root,
                   val_split=0.0, test_split=0.0,
                   **kwargs):

    # List of classes (first level directory names)
    try:
        class_dirs = sorted([c for c in os.listdir(src_root) if os.path.isdir(os.path.join(src_root, c))])
    except FileNotFoundError as exc:
        print('Error reading source directory: {}'.format(exc.strerror), file=sys.stderr)
        return

    if 'whitelist' in kwargs:  # Apply whitelist
        whitelist_file = kwargs.pop('whitelist')
        class_dirs = restrict_classes_with_whitelist_file(class_dirs, whitelist_file)
        print('Application of whitelist from {:s} results in {:d} classes.'.format(whitelist_file, len(class_dirs)))

    if len(class_dirs) == 0:
        print('No classes to process.')
        return

    min_clips_per_class = kwargs.pop('min_clips_per_class') if 'min_clips_per_class' in kwargs else 0
    max_clips_per_class = kwargs.pop('max_clips_per_class') if 'max_clips_per_class' in kwargs else None
    max_per_tfrecord_file = kwargs.pop('max_records_per_file') if 'max_records_per_file' in kwargs else np.inf
    random_state = np.random.RandomState(kwargs.pop('random_state_seed') if 'random_state_seed' in kwargs else None)
    masksdir = kwargs.pop('masksdir') if 'masksdir' in kwargs else None
    show_progress = kwargs.pop('show_progress') if 'show_progress' in kwargs else False

    # Warn about existing output directory
    if os.path.exists(dest_root) and os.path.isdir(dest_root):
        warnings.showwarning('Output directory {:s} already exists. Contents may get overwritten'.format(dest_root),
                             Warning, _program_name, '')
    os.makedirs(dest_root, exist_ok=True)

    logger = logging.getLogger()

    t_start = datetime.now()
    logger.info('Started at: {}'.format(t_start))

    DatasetDigest.Wipeout(dest_root)  # Start the digest afresh

    group_subdirs = [DirectoryNames.TRAIN, DirectoryNames.EVAL, DirectoryNames.TEST]
    # Create subdirectories as needed
    if val_split > 0.:
        os.makedirs(os.path.join(dest_root, DirectoryNames.EVAL), exist_ok=True)
    if test_split > 0.:
        os.makedirs(os.path.join(dest_root, DirectoryNames.TEST), exist_ok=True)
    if val_split + test_split < 1.:
        os.makedirs(os.path.join(dest_root, DirectoryNames.TRAIN), exist_ok=True)

    output_path_fmt = os.path.join(
        dest_root, '{:s}', '{{:0{:d}d}}-{{:s}}_{{:d}}{:s}'.format(np.floor(np.log10(len(class_dirs))).astype(np.int) + 1,
                                                                 FilenameExtensions.tfrecord))

    curr_class_idx = -1
    valid_classes = [False] * len(class_dirs)
    class_stats = np.zeros((len(class_dirs), 5), np.uint32)  # [files count, clips count, num train, num eval, num test]
    data_shape = None
    num_workers = kwargs.pop('num_threads') if 'num_threads' in kwargs else os.cpu_count()
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:

        # Start the operations for assessing class details and map each future to class idx
        future_to_idx = {
            executor.submit(_get_class_details,
                            src_root, class_dir,
                            masks_dir=masksdir): class_idx
            for class_idx, class_dir in enumerate(class_dirs)}
        step2_futures = []

        # As the "details" become available, they will be staged for later processing in the same order as 'class_dirs'
        class_details_avail = [False] * len(class_dirs)
        class_details = [None] * len(class_dirs)
        unprocessed_class_idx = 0

        if show_progress:
            pbar = ProgressBar(len(class_dirs), prefix='Gathering details', length=53, show_start=True)
        else:
            print('Gathering details', end='')
        for future in concurrent.futures.as_completed(future_to_idx):
            c_idx = future_to_idx[future]
            class_details_avail[c_idx] = True
            try:
                spec_files, spec_files_mask_idxs, clip_shape = future.result()
            except Exception as exc:
                logger.warning('Loading class info for {:s} generated an exception: {:s}'.format(
                    repr(class_dirs[c_idx]), exc))
            else:
                num_clips = np.sum([len(idxs) for idxs in spec_files_mask_idxs]) if len(spec_files_mask_idxs) > 0 else 0
                class_stats[c_idx, :2] = [len(spec_files), num_clips]
                if num_clips < min_clips_per_class:
                    logger.warning('Ignoring class {:s}, has only {:d} clips'.format(repr(class_dirs[c_idx]), num_clips))
                else:
                    # Stage the "details" for now, so that they can be processed in order in the below while loop
                    class_details[c_idx] = (spec_files, spec_files_mask_idxs, clip_shape)

                    #print('{:<55s}: {:5d} files, {:6d} clips'.format(class_dirs[c_idx], len(spec_files), num_clips))
                del(spec_files, spec_files_mask_idxs, clip_shape)

            if show_progress:
                pbar.increment()
            else:
                print('.', end='')

            while unprocessed_class_idx < len(class_dirs) and class_details_avail[unprocessed_class_idx]:
                if class_details[unprocessed_class_idx] is None:
                    # Appropriate message was already logged. Simply skip over now.
                    unprocessed_class_idx += 1
                    continue

                class_dir = class_dirs[unprocessed_class_idx]
                num_clips = class_stats[unprocessed_class_idx, 1]
                (spec_files, spec_files_mask_idxs, data_shape) = class_details[unprocessed_class_idx]
                class_details[unprocessed_class_idx] = None     # No need to retain "details" in the list anymore

                valid_classes[unprocessed_class_idx] = True
                curr_class_idx += 1

                # Determine splits
                num_output_clips = min(num_clips, max_clips_per_class) \
                    if max_clips_per_class is not None else num_clips
                curr_validation_split = np.round(val_split * num_output_clips).astype(dtype=np.int64)
                curr_test_split = np.round(test_split * num_output_clips).astype(dtype=np.int64)
                train_split = num_output_clips - curr_validation_split - curr_test_split

                # Update stats fields for class
                class_stats[unprocessed_class_idx, 2:] = [train_split, curr_validation_split, curr_test_split]
                #print('{:3d}. {:<55s}: {:6d} train, {:6d} eval, {:6d} test'.format(curr_class_idx, class_dir,
                #                                                            *(class_stats[unprocessed_class_idx, 2:])))

                # Per group (train/val/test) extents within 'clips_full_idxs'
                group_offsets = [0, train_split, train_split + curr_validation_split]
                group_extents = [train_split, train_split + curr_validation_split,
                                 train_split + curr_validation_split + curr_test_split]

                # Add file idxs with clip idxs to build (file idx, clip idx) pairs
                clips_full_idxs = np.concatenate([
                    np.stack([np.full((len(s), ), s_idx), s]).T for s_idx, s in enumerate(spec_files_mask_idxs)])

                # Shuffle; for randomized trimming (if enabled) and splitting (if necessary)
                if (max_clips_per_class is not None and num_clips > max_clips_per_class) or \
                             (curr_validation_split > 0) or (curr_test_split > 0):
                    clips_full_idxs = random_state.permutation(clips_full_idxs)

                # Limit the number of clips, if applicable
                clips_full_idxs = clips_full_idxs[:num_output_clips, ...]

                curr_class_root = os.path.join(src_root, class_dir)

                # Divide and distribute writing to TFRecord files
                for offset, limit, subdir in zip(group_offsets, group_extents, group_subdirs):
                    if offset == limit:     # Nothing to do for this group
                        continue

                    # Serialize, in chunks up to the max size chosen (if not unlimited) per output file
                    fc = 1  # file counter within class and group
                    while offset + max_per_tfrecord_file < limit:
                        futr = executor.submit(_write_single_tfrecord_file,
                                               curr_class_root, curr_class_idx,
                                               spec_files, clips_full_idxs[offset:(offset + max_per_tfrecord_file)],
                                               output_path_fmt.format(subdir, curr_class_idx, class_dir, fc),
                                               tfrecord_handler,
                                               transformation_fn=transformation_fn)
                        offset += max_per_tfrecord_file
                        fc += 1
                        step2_futures.append(futr)

                    # Serialize the remaining/unchunked part
                    futr = executor.submit(_write_single_tfrecord_file,
                                           curr_class_root, curr_class_idx,
                                           spec_files, clips_full_idxs[offset:limit],
                                           output_path_fmt.format(subdir, curr_class_idx, class_dir, fc),
                                           tfrecord_handler,
                                           transformation_fn=transformation_fn)
                    step2_futures.append(futr)

                DatasetDigest.AddClassOrderedSpecFileList(dest_root, class_dir, spec_files)

                # Clear out some of the (potentially) big containers
                del(spec_files, spec_files_mask_idxs, clips_full_idxs)

                unprocessed_class_idx += 1

        if show_progress:
            pbar = ProgressBar(len(step2_futures), prefix='Writing TFRecords', length=53, show_start=True)
        else:
            print()
            print('Writing TFRecords', end='')
        for _ in concurrent.futures.as_completed(step2_futures):
            if show_progress:
                pbar.increment()
            else:
                print('.', end='')

    if not show_progress:
        print()

    class_dirs = [class_dirs[c_idx] for c_idx in range(len(class_dirs)) if valid_classes[c_idx]]
    class_stats = class_stats[np.asarray(valid_classes), ...]

    # Write out the rest of the digest while the child threads continue writing
    DatasetDigest.AddOrderedClassList(dest_root, class_dirs)
    DatasetDigest.AddPerClassAndGroupSpecCounts(dest_root, class_stats[:, 2:])
    if transformation_fn is not None:
        if hasattr(transformation_fn, '__len__'):
            for tf_fn in transformation_fn:
                data_shape = tf_fn.outshape(data_shape)
        else:
            data_shape = transformation_fn.outshape(data_shape)
    DatasetDigest.AddDataShape(dest_root, data_shape)

    t_end = datetime.now()
    logger.info('Started at: {}'.format(t_end))

    stats_fmt = '{:>3d}. {:<55s} - {:>5d}, {:>6d} [{:s}]'
    counts_fmts = np.asarray(['{:>6d}', '{:>6d}', '{:>6d}'])
    counts_headers = np.asarray(['{:>6s}'.format('Train'), '{:>6s}'.format('Eval'), '{:>6s}'.format('Test')])
    counts_fields_mask = np.asarray([val_split + test_split < 1., val_split > 0., test_split > 0.])
    _print_and_log('{:>3s}. {:<55s} - {:>5s}, {:>6s} [{:s}]'.format(
        'Id', 'Class', 'Files', 'Clips', ', '.join(counts_headers[counts_fields_mask])), logger)

    for c_idx in range(len(class_dirs)):
        _print_and_log(stats_fmt.format(c_idx, class_dirs[c_idx], class_stats[c_idx, 0], class_stats[c_idx, 1],
                                        (', '.join(counts_fmts[counts_fields_mask]).format(
                                            *class_stats[c_idx, 2:][counts_fields_mask]))), logger)

    _print_and_log('{:<60s} : {:>5d}, {:>6d} [{:s}]'.format('Total {:d} classes'.format(len(class_dirs)),
                                                            np.sum(class_stats[:, 0]), np.sum(class_stats[:, 1]),
                                                            (', '.join(counts_fmts[counts_fields_mask]).format(
                                                                *(np.sum(class_stats[:, 2:], axis=0))[counts_fields_mask]))
                                                            ), logger)

    logging.info('Processing time (hh:mm:ss.ms) {}'.format(t_end - t_start))

    print('Processing time (hh:mm:ss.ms) {}'.format(t_end - t_start))


def _print_and_log(msg, logger):
    print(msg)
    logger.info(msg)


def _get_class_details(src_dir, class_name, masks_dir=None, mask_val=1):
    """Load info about data available for the class"""

    assert isinstance(mask_val, int)

    logger = logging.getLogger(__name__)

    clip_shape = None

    # If usable masks container exists for the class, use it. Else, load the individual clip files and collect.
    if masks_dir is not None and \
            os.path.exists(os.path.join(masks_dir, class_name + FilenameExtensions.hdf5)):
        with h5py.File(os.path.join(masks_dir, class_name + FilenameExtensions.hdf5), 'r') as hf:
            # Apply mask and gather indices
            spec_files, spec_files_mask_idxs = ([], []) if len(hf.items()) == 0 else \
                map(list, zip(*[(a, np.asarray(np.equal(b[()], mask_val).nonzero()).ravel()) for a, b in hf.items()]))

        # Discard spec_files entries (+ '.npz') if the actual files don't exist
        valid_files = np.asarray([os.path.exists(os.path.join(src_dir, class_name, f + FilenameExtensions.numpy))
                                  for f in spec_files])
        if len(spec_files) > np.sum(valid_files):
            logger.info('Clip files missing for {:d} (of {:d}) listed entries for class {:s}'.format(
                len(spec_files) - np.sum(valid_files), len(spec_files), class_name))
        spec_files = [spec_files[s_idx] for s_idx in range(len(spec_files)) if valid_files[s_idx]]
        spec_files_mask_idxs = [spec_files_mask_idxs[s_idx] for s_idx in range(len(spec_files)) if valid_files[s_idx]]

        for spec_file in (os.path.join(src_dir, class_name, s + FilenameExtensions.numpy) for s in spec_files):
            try:
                with np.load(spec_file) as filedata:
                    clip_shape = filedata[_npz_data_container_fieldname].shape[1:]
            except Exception as exc:
                logger.error('Exception encountered while loading {:s}: {:s}'.format(repr(spec_file), exc))
            else:
                break

    else:
        spec_files = sorted([f for f in recursive_listing(os.path.join(src_dir, class_name),
                                                          match_extensions=FilenameExtensions.numpy)])
        spec_files_mask_idxs = [None] * len(spec_files)
        valid_files = np.full((len(spec_files), ), True)

        for s_idx, spec_file in enumerate(spec_files):
            spec_filepath = os.path.join(src_dir, class_name, spec_file)
            try:
                filedata = np.load(spec_filepath)
            except Exception as exc:
                logger.error('Exception encountered while loading {:s}: {:s}'.format(repr(spec_filepath), exc))
                valid_files[s_idx] = False
            else:
                spec_files_mask_idxs[s_idx] = np.arange(filedata[_npz_data_container_fieldname].shape[0])
                if clip_shape is None:
                    clip_shape = filedata[_npz_data_container_fieldname].shape[1:]
                filedata.close()

        valid_files = np.asarray(np.where(valid_files)).ravel()
        spec_files = [os.path.splitext(spec_files[s_idx])[0] for s_idx in valid_files]  # Strip the '.npz' extensions
        spec_files_mask_idxs = [spec_files_mask_idxs[s_idx] for s_idx in valid_files]

    # Only keep those files that offers at least one clip
    valid_files = [s_idx for s_idx, s in enumerate(spec_files_mask_idxs) if len(s) > 0]
    if len(spec_files) > len(valid_files):
        logger.info('Discarding {:d} (of {:d}) files for class {:s}'.format(
            len(spec_files) - len(valid_files), len(spec_files), class_name))
    spec_files = [spec_files[s_idx] for s_idx in valid_files]
    spec_files_mask_idxs = [spec_files_mask_idxs[s_idx] for s_idx in valid_files]

    return spec_files, spec_files_mask_idxs, clip_shape


def _write_single_tfrecord_file(class_root, class_idx, files_list, file_spec_idxs, output_path,
                                tfrecord_handler,
                                transformation_fn=None):
    """
    Write clips/specs to a TFRecord file.
    :param class_root: Root directory of the class (whose clips are being written) that contains the .npz files.
    :param class_idx: Number index of the class.
    :param files_list: Ordered list of .npz files (without the filename extension).
    :param file_spec_idxs: An Nx2 numpy array. The first column contains indices to files in the files_list, and the
        second column contains indices to clips in the corresponding .npz file. This array defines which clips in each
        file gets written to the TFRecord file.
    :param output_path: Full path to the TFRecord file to be written.
    :param tfrecord_handler: An instance of one of data.tfrecord_helper.TFRecordHandler's inheritors.
    :param transformation_fn: A callable (that takes one argument - a clip) that transforms the clip as needed. Multiple
        transformations can be chained by specifying them in a list.
    """

    logger = logging.getLogger(__name__)

    logger.debug('Will write {:5d} records to {:s}'.format(file_spec_idxs.shape[0], repr(output_path)))

    writer = tf.io.TFRecordWriter(output_path)

    written_records_count = int(0)

    # For each unique file idx ...
    for file_idx in np.unique(file_spec_idxs[:, 0]):

        infile_clip_idxs = file_spec_idxs[np.asarray((file_spec_idxs[:, 0] == file_idx).nonzero()).ravel(), 1]

        # ... load the corresponding .npz file, and ...
        spec_filepath = os.path.join(class_root, files_list[file_idx] + FilenameExtensions.numpy)
        try:
            with np.load(spec_filepath) as d:
                data = d[_npz_data_container_fieldname][infile_clip_idxs, ...]

        except Exception as exc:
            logger.error('Exception loading file {:s} while writing tfrecords to {:s}: {:s}'.format(
                repr(spec_filepath), repr(output_path), exc))
            continue

        # ... apply transformation(s), if requested, and...
        if transformation_fn is not None:
            try:
                if hasattr(transformation_fn, '__len__'):   # if chained, apply in sequence
                    for tf_fn in transformation_fn:
                        data = tf_fn(data)
                else:
                    data = transformation_fn(data)
            except Exception as exc:
                logger.error('Exception applying requested transformation(s). File {:s}: {:s}'.format(
                    repr(spec_filepath), exc))
                continue

        # ... write out as TFRecords.
        for idx, clip_idx in enumerate(infile_clip_idxs):
            tfrecord_handler.write_record(writer, data[idx], class_idx, file_idx, clip_idx)

            written_records_count += 1

    writer.close()

    if written_records_count < file_spec_idxs.shape[0]:
        logger.warning('Failed to write {:d} specs to {:s}'.format(
            file_spec_idxs.shape[0] - written_records_count, repr(output_path)))

    logger.info('Wrote {:5d} (of {:5d}) records to {:s}'.format(
        written_records_count, file_spec_idxs.shape[0], repr(output_path)))


class _TransformationFunction:
    """A wrapper providing a callable interface to some common data transformation functions."""

    # Supported transformations
    pcm2float = 0
    float2pcm = 1
    audio2spectral = 2

    _fn_mapping = {
        pcm2float: Convert.pcm2float,
        float2pcm: Convert.float2pcm,
        audio2spectral: Convert.audio2spectral
    }

    def __init__(self, fn_id, *args, **kwargs):

        assert fn_id in _TransformationFunction._fn_mapping

        self._fn = _TransformationFunction._fn_mapping[fn_id]
        self._fn_args = args
        self._fn_kwargs = kwargs

    def __call__(self, data):
        return self._fn(data, *self._fn_args, **self._fn_kwargs)

    def outshape(self, in_shp):
        # Create dummy data of the required type with an added batch dimension
        if self._fn == _TransformationFunction._fn_mapping[_TransformationFunction.pcm2float]:
            dummy_data = (100 * np.random.random([1] + list(in_shp))).astype(np.int16)  # Needs integer data
        elif self._fn == _TransformationFunction._fn_mapping[_TransformationFunction.float2pcm]:
            dummy_data = np.random.random([1] + list(in_shp))
        else:
            dummy_data = np.random.random([1] + list(in_shp))

        # Return everything but the batch dimension after applying the transformation
        return self(dummy_data).shape[1:]


# def _audio_n_spec_settings_from_config(cfg_file):
#     """Load audio & spectrogram settings parameters from the config file"""
#
#     cfg = Config(cfg_file, 'DATA')
#
#     fs = cfg.DATA.audio_fs
#     spec_settings = {
#         'win_len': cfg.DATA.spec_win_len,
#         'win_overlap_prc': cfg.DATA.spec_win_overlap_prc,
#         'nfft_equals_win_len': cfg.DATA.spec_nfft_equals_win_len,
#         'eps': cfg.DATA.spec_eps,
#         'bandwidth_clip': cfg.DATA.spec_bandwidth_clip,
#         'tf_rep_type': cfg.DATA.spec_type,
#         'num_mels': cfg.DATA.spec_num_mels
#     }
#     return fs, spec_settings


__all__ = [write_time_domain_records, write_spectral_records]


if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog=_program_name, allow_abbrev=False,
                                     description='Write TFRecords files from prepared data (clips).')
    parser.add_argument('src', metavar='<CLIPS SOURCE DIR>',
                        help='Path to the root directory containing per-class top-level subdirectories which in turn' +
                             ' contain audio clips (.npz) files that were produced by prepare_data.py.')
    parser.add_argument('dst', metavar='<OUTPUT DIR>',
                        help='Path to destination directory into which TFRecord files will be written.')
    arg_group_inctrl = parser.add_argument_group('Input control')
    arg_group_inctrl.add_argument('--whitelist', metavar='FILE',
                                  help='Path to text file containing names (one per line) of whitelisted classes.')
    arg_group_inctrl.add_argument('--min-clips', dest='min_clips_per_class', metavar='NUM',
                                  type=ArgparseConverters.positive_integer, default=50,
                                  help='Minimum number of clips per class. If number of available clips for a class' +
                                       'is lower than this value, the class will be ignored. (default: %(default)s).')
    arg_group_inctrl.add_argument('--max-clips', dest='max_clips_per_class',  metavar='NUM',
                                  type=ArgparseConverters.positive_integer,
                                  help='Maximum number of clips per class. If a class has more clips available, then ' +
                                       'clips will be randomly chosen to attain this target. (default: no limit).')
    arg_group_outctrl = parser.add_argument_group('Output control')
    arg_group_outctrl.add_argument('--type', choices=['1d', '2d'], default='1d',
                                   help='1d: Store time-domain data in TFRecords. 2d: Store time-frequency domain ' +
                                        'data in TFRecords. If \'2d\' is chosen, then \'cfg\' must also be specified ' +
                                        'so the program can obtain spectral settings.')
    arg_group_outctrl.add_argument('--val-split', metavar='0-100', type=ArgparseConverters.valid_percent,
                                   default=15., dest='val_split',
                                   help='Validation split, as a percent. (default: %(default)s).')
    arg_group_outctrl.add_argument('--test-split', metavar='0-100', type=ArgparseConverters.valid_percent,
                                   default=0., dest='test_split',
                                   help=argparse.SUPPRESS)
    arg_group_outctrl.add_argument('--max-records', dest='max_records_per_file',
                                   type=ArgparseConverters.positive_integer, metavar='NUM',
                                   help='Maximum num of TFRecord entries to be written into a record file. When the ' +
                                        'number of clips exceeds this value, any excess will be written to a new file.')
#    arg_group_outctrl.add_argument('--save-tracing-info', dest='save_tracing_info', action='store_true',
#                                   help='Enable saving of clips\' tracing info in the output TFRecord files. ' +
#                                        '(default: disabled).')
#    arg_group_outctrl.add_argument('--masksdir', metavar='DIRECTORY',
#                                   help=argparse.SUPPRESS)
    arg_group_prctrl = parser.add_argument_group('Process control')
    arg_group_prctrl.add_argument('--threads', metavar='NUM', type=ArgparseConverters.positive_integer,
                                  help='Number of threads to spawn for parallel execution (default: as many CPUs).')
    arg_group_logging = parser.add_argument_group('Logging')
    arg_group_logging.add_argument('--log', metavar='FILE',
                                   help='Path to file to which logs will be written out.')
    arg_group_logging.add_argument('--loglevel', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'],
                                   default='INFO',
                                   help='Logging level.')
    arg_group_misc = parser.add_argument_group('Miscellaneous')
    arg_group_misc.add_argument('--cfg', metavar='CONFIG FILE',
                                help='Path to config file. Necessary when output type is chosen to be \'2d\'.')
    arg_group_misc.add_argument('--seed', dest='random_state_seed', type=ArgparseConverters.positive_integer,
                                metavar='NUM',
                                help='Seed value (integer) for deterministic shuffling.')
    args = parser.parse_args()

    other_args = {}
    if args.min_clips_per_class:
        other_args['min_clips_per_class'] = args.min_clips_per_class
    if args.max_clips_per_class:
        other_args['max_clips_per_class'] = args.max_clips_per_class
    if args.max_records_per_file:
        other_args['max_records_per_file'] = args.max_records_per_file
    if args.whitelist:
        other_args['whitelist'] = args.whitelist
    if args.threads:
        other_args['num_threads'] = args.threads
    if args.masksdir:
        other_args['masksdir'] = args.masksdir
    other_args['show_progress'] = True

    instantiate_logging(args.log if args.log is not None else
                        os.path.join(args.dst, _program_name + '.log'),
                        args.loglevel, args)

    exit_code = 0

    if args.type == '2d':   # Time-frequency format output requested
        if args.cfg is not None:
            # Load the config file
            try:
                data_settings = datasection2dict(Config(args.cfg, 'DATA').DATA)
            except FileNotFoundError as exc:
                print('Error loading config file: {}'.format(exc.strerror), file=sys.stderr)
                exit_code = exc.errno
            except ConfigError as exc:
                print('Error processing config file: {}'.format(str(exc)), file=sys.stderr)
                exit_code = 1
            except Exception as exc:
                print('Error processing config file: {}'.format(repr(exc)), file=sys.stderr)
                exit_code = 1
            else:
                log_config(logging.getLogger(__name__), data_cfg=data_settings)

                write_spectral_records(args.src, args.dst,
                                       data_settings['audio_settings']['desired_fs'],
                                       data_settings['spec_settings'],
                                       args.val_split / 100, args.test_split / 100,
                                       **other_args)

    else:       # Time domain format output requested
        write_time_domain_records(args.src, args.dst,
                                  args.val_split / 100, args.test_split / 100,
                                  **other_args)

    logging.shutdown()

    exit(exit_code)
