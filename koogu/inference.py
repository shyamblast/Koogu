import os
import sys
import numpy as np
from timeit import default_timer as timer
import json
import argparse
import concurrent.futures
import logging

from koogu.data import FilenameExtensions, AssetsExtraNames
from koogu.data.raw import Audio, Settings
from koogu.data.annotations import Raven
from koogu.model import TrainedModel
from koogu.utils import processed_items_generator_mp, \
    processed_items_generator_mp_ordered, instantiate_logging
from koogu.utils.detections import postprocess_detections
from koogu.utils.terminal import ProgressBar, ArgparseConverters
from koogu.utils.filesystem import recursive_listing, AudioFileList

_program_name = 'predict'
_selection_table_file_suffix = '.selections.txt'


def analyze_clips(trained_model, clips, batch_size=1, audio_filepath=None):
    """
    Apply a trained model to one or more audio clips and obtain scores.

    :param trained_model: A :class:`koogu.model.TrainedModel` instance.
    :param clips: An [N x ?] numpy array of N input waveforms.
    :param batch_size: (default: 1) Control how many clips are processed in a
        single batch. Increasing this helps improve throughput, but requires
        more RAM.
    :param audio_filepath: (default: None) If not None, will display a progress
        bar.

    :returns: A 2-element tuple consisting of -

        * detection/classification scores ([N x M] numpy array corresponding to
          the N clips and M target classes), and
        * the total time taken to process all the clips.
    """

    pbar = None if audio_filepath is None else \
        ProgressBar(clips.shape[0],
                    prefix='{:>59s}'.format(audio_filepath[-59:]),
                    length=10, show_start=True)

    batch_start_idxs = np.arange(0, clips.shape[0], batch_size)
    batch_end_idxs = np.minimum(batch_start_idxs + batch_size, clips.shape[0])
    det_scores = [None] * len(batch_end_idxs)
    predict_time = 0.
    for idx, (s_idx, e_idx) in enumerate(zip(batch_start_idxs, batch_end_idxs)):

        t_start = timer()
        det_scores[idx] = trained_model.infer(inputs=clips[s_idx:e_idx, ...])
        predict_time += (timer() - t_start)

        if pbar is not None:
            pbar.increment(e_idx - s_idx)

    return np.concatenate(det_scores, axis=0), predict_time


def _combine_and_write(raven_writer, out_file_info,
                       det_scores, clip_start_samples, num_samples, fs,
                       class_names, class_frequencies,
                       threshold=None,
                       channel_IDs=None,
                       offset_info=None,
                       ignore_class=None,
                       suppress_nonmax=False,
                       squeeze_min_dur=None):

    num_channels, num_clips, num_classes = det_scores.shape

    if num_clips == 0:
        # Write the header only (if creating a new file) and return immediately
        if out_file_info[1]:
            raven_writer(
                out_file_info[0], [], [],
                file_offset=None if offset_info is None else offset_info[1],
                begin_file=None if offset_info is None else offset_info[2]
                )
        return 0

    # Mask out the ignore_class(es), so that we don't waste time post-processing
    # those results
    write_class_mask = np.full((num_classes, ), True)
    if ignore_class is not None:
        if hasattr(ignore_class, '__len__'):
            write_class_mask[np.asarray([c for c in ignore_class])] = False
        else:
            write_class_mask[ignore_class] = False
    class_idx_remapper = np.where(write_class_mask)[0]

    # First, combine detections within each channel and gather per-channel
    # combined results
    channel_combined_det_times = [None] * num_channels
    channel_combined_det_scores = [None] * num_channels
    channel_combined_det_labels = [None] * num_channels
    num_combined_dets_per_channel = np.zeros((num_channels,), np.uint32)
    min_det_len = None if squeeze_min_dur is None else int(squeeze_min_dur * fs)
    for ch in range(num_channels):
        channel_combined_det_times[ch], \
            channel_combined_det_scores[ch], \
            channel_combined_det_labels[ch] = \
            postprocess_detections(det_scores[ch, ...][:, write_class_mask],
                                   clip_start_samples, num_samples,
                                   threshold=threshold,
                                   suppress_nonmax=suppress_nonmax,
                                   squeeze_min_samps=min_det_len)

        num_combined_dets_per_channel[ch] = \
            channel_combined_det_scores[ch].shape[0]

    if int(num_combined_dets_per_channel.sum()) == 0:  # No detections available
        # Write the header only (if creating a new file) and return immediately
        if out_file_info[1]:
            raven_writer(out_file_info[0], [], [])
        return 0

    # Flatten
    combined_det_times = np.concatenate(channel_combined_det_times, axis=0)
    combined_det_scores = np.concatenate(channel_combined_det_scores)
    combined_det_labels = np.concatenate(channel_combined_det_labels)
    if channel_IDs is not None:
        combined_det_channels = np.concatenate([
            np.full((num_combined_dets_per_channel[ch],), channel_IDs[ch])
            for ch in range(num_channels)])

    # Remap class IDs to make good for the gaps from ignore_class
    combined_det_labels = class_idx_remapper[combined_det_labels]

    # Sort the detections across channels (based on detection start time)
    sort_idx = np.argsort(combined_det_times[:, 0])
    combined_det_times = combined_det_times[sort_idx, ...]
    combined_det_scores = combined_det_scores[sort_idx]
    combined_det_labels = combined_det_labels[sort_idx]
    if channel_IDs is not None:
        combined_det_channels = combined_det_channels[sort_idx]

    # Convert detection extents from samples to seconds
    combined_det_times = combined_det_times.astype(float) / float(fs)

    if isinstance(class_frequencies[0], list):  # is a 2D list
        def freq_output(l_idx): return class_frequencies[l_idx]
    else:
        def freq_output(_): return class_frequencies    # same for all

    # Dump!
    raven_writer(
        out_file_info[0],
        combined_det_times,
        [class_names[cd_idx] for cd_idx in combined_det_labels],
        frequencies=[freq_output(cd_idx) for cd_idx in combined_det_labels],
        channels=None if channel_IDs is None else combined_det_channels,
        scores=combined_det_scores,
        selection_num_offset=0 if offset_info is None else offset_info[0],
        file_offset=None if offset_info is None else offset_info[1],
        begin_file=None if offset_info is None else offset_info[2],
        new_file=out_file_info[1]
    )

    return combined_det_scores.shape[0]


def write_raw_detections(file_path, fs, det_scores,
                         clip_start_samples, num_samples, channel_IDs):

    os.makedirs(os.path.split(file_path)[0], exist_ok=True)

    res = dict(
        fs=fs,
        clip_length=num_samples,
        clip_offsets=clip_start_samples,
        scores=det_scores
    )
    if channel_IDs is not None:
        res['channels'] = channel_IDs.astype(np.uint8)

    np.savez_compressed(file_path, **res)


def recognize(model_dir, audio_root,
              output_dir=None, raw_detections_dir=None,
              **kwargs):
    """
    Batch-process audio files using a trained model.

    :param model_dir: Path to directory where the trained model for use in
        making inferences is available.
    :param audio_root: Path to directory from which to load audio files for
        inferences. Can also set this to a single audio file instead of a
        directory. See optional parameters ``recursive`` and ``combine_outputs``
        that may be used when ``audio_root`` points to a directory.
    :param output_dir: If not None, processed recognition results (Raven
        selection tables) will be written out into this directory. At least
        one of ``output_dir`` or ``raw_detections_dir`` must be specified.
    :param raw_detections_dir: If not None, raw outputs from the model will be
        written out into this directory. At least one of ``output_dir`` or
        ``raw_detections_dir`` must be specified.

    **Optional parameters**

    :param clip_advance: If specified, override the value that was read from
        the model's files. The value defines the amount of clip advance when
        preparing audio.
    :param threshold: (float, 0-1) Suppress writing of detections with scores
        below this value. Defaults to 0.
    :param recursive: (bool) If set, the contents of ``audio_root`` will be
        processed recursively.
    :param filetypes: Audio file types to restrict processing to. Option is
        ignored if processing a single file. Can specify multiple types, as a
        list. Defaults to ['.wav', '.WAV', '.flac', '.aif', '.mp3'].
    :param combine_outputs: (bool) When processing audio files from entire
        directories, enabling this option combines recognition results of
        processing every file within a directory and writes them to a single
        output file. When enabled, outputs will contain 2 additional fields
        describing offsets of detections in the corresponding audio files.
    :param channels: (int or list of ints) When audio files have multiple
        channels, set which channels to restrict processing to. If
        unspecified, all available channels will be processed. E.g., setting
        to 0 saves the first channel, setting to [0, 2] saves the first and
        third channels.
    :param scale_scores: (bool) Enabling this will scale the raw scores before
        they are written out. Use of this setting is recommended only when the
        output of a model is based on softmax (not multi-label) and the model
        was trained with training data where each input corresponded to a
        single class.
    :param frequency_extents: A dictionary of per-class frequency bounds of
        each label class. Will be used when producing the output selection
        table files. If unspecified, the "Low Frequency (Hz)" and
        "High Frequency (Hz)" fields in the output table will be the same for
        all classes and will be set equal to the bandwidth used in preparing
        model inputs.
    :param reject_class: Name (case sensitive) of the class (like 'Noise' or
        'Other') that must be ignored from the recognition results. The
        corresponding detections will not be written to the output selection
        tables. Can specify multiple classes for rejection, as a list.
    :param batch_size: (int; default: 1) Size to batch audio file's clips into.
        Increasing this may improve speed on computers with high RAM.
    :param num_fetch_threads: (int; default: 1) Number of background threads
        that will fetch audio from files in parallel.
    :param show_progress: (bool) If enabled, messages indicating progress of
        processing will be shown on console output.
    """

    assert os.path.exists(model_dir) and os.path.exists(audio_root)
    assert output_dir is not None or raw_detections_dir is not None
    assert 'threshold' not in kwargs or 0.0 <= kwargs['threshold'] <= 1.0

    # Load the classifier
    classifier = TrainedModel(model_dir)

    # Query some dataset info
    class_names = classifier.class_names
    audio_settings = classifier.audio_settings
    spec_settings = None if classifier.spec_settings is None \
        else Settings.Spectral(audio_settings['desired_fs'],
                               **classifier.spec_settings)

    # Override clip_advance, if requested
    if 'clip_advance' in kwargs:
        audio_settings['clip_advance'] = kwargs['clip_advance']

    raw_output_executor = None
    if raw_detections_dir:
        os.makedirs(raw_detections_dir, exist_ok=True)
        # Write out the list of class names
        json.dump(class_names,
                  open(os.path.join(raw_detections_dir,
                                    AssetsExtraNames.classes_list), 'w'))
        raw_output_executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=1)

    # Set up function to scale scores, if enabled
    if kwargs.get('scale_scores', False):
        frac = 1.0 / float(len(class_names))
        def scale_scores(scores): return \
            np.maximum(0.0, (scores - frac) / (1.0 - frac))
    else:
        def scale_scores(scores): return scores

    output_executor = None
    combine_outputs = False  # Also controls multi-threaded processor type
    if output_dir:
        reject_class_idx = None
        if kwargs.get('reject_class', None) is not None:
            reject_classes = kwargs['reject_class']
            reject_classes = [reject_classes] \
                if isinstance(reject_classes, str) else reject_classes
            reject_class_idx = []
            for rj_class in reject_classes:
                if rj_class in class_names:
                    reject_class_idx.append(class_names.index(rj_class))
                else:
                    print('Reject class {:s} not found in list of classes. Will ignore setting.'.format(
                        repr(rj_class)))

        # Handle frequency extents in detection outputs
        if spec_settings is not None:
            default_freq_extents = spec_settings.bandwidth_clip
        else:
            default_freq_extents = [0, audio_settings['desired_fs'] / 2]
        if 'frequency_extents' not in kwargs:  # unspecified, set same for all
            class_freq_extents = default_freq_extents
        else:
            # Assign defaults for missing classes
            class_freq_extents = [
                kwargs['frequency_extents'].get(cn, default_freq_extents)
                for cn in class_names
                ]

        # Check post-processing settings
        squeeze_min_dur = kwargs.get('squeeze_detections', None)
        suppress_nonmax = kwargs.get('suppress_nonmax', False)

        combine_outputs = kwargs.get('combine_outputs', False)

        os.makedirs(output_dir, exist_ok=True)
        output_executor = concurrent.futures.ProcessPoolExecutor(max_workers=1)
        output_executor_future = None
        raven_writer = Raven.Writer(write_frequencies=True, add_score=True)

    if not os.path.isdir(audio_root):     # Single file input
        src_generator = [audio_root]  # turn it into a list

        # No need of fields describing offsets of detections
        combine_outputs = False    # Disable this (if it was enabled)

    else:
        # Prepare the input file generator
        filetypes = kwargs.get('filetypes',
                               AudioFileList.default_audio_filetypes)
        if kwargs.get('recursive', False):
            src_generator = (
                os.path.join(audio_root, f)
                for f in recursive_listing(audio_root,
                                           match_extensions=filetypes))
        else:
            # Only get the top-level files
            src_generator = (
                os.path.join(audio_root, f)
                for f in os.listdir(audio_root)
                if (any((f.endswith(e) for e in filetypes)) and
                    os.path.isfile(os.path.join(audio_root, f))))

    logger = logging.getLogger(__name__)

    if output_dir and squeeze_min_dur is not None and squeeze_min_dur > audio_settings['clip_length']:
        logger.warning('Squeeze min duration ({:f} s) is larger than model input length ({:f} s)'.format(
            squeeze_min_dur, audio_settings['clip_length']))

    # Convert to a container that is needed by Audio.get_file_clips()
    audio_settings = Settings.Audio(**audio_settings)

    # Prepare parameters for audio_loader
    channels = None    # fetch all channels' clips if nothing specified
    if kwargs.get('channels', None) is not None:    # something specified?
        # fetch selected channel's clips
        channels = np.sort(np.unique(kwargs['channels']).astype(np.uint32))

    # Choose whether to use ordered or unordered processor
    processed_items_generator = \
        processed_items_generator_mp_ordered if combine_outputs else \
        processed_items_generator_mp

    selmap = None    # To store src rel path, seltab file relpath, analysis time
    sel_running_info = None  # To store last sel num, time offset for next file
    last_file_dur = 0.
    total_audio_dur = 0.
    total_time_taken = 0.
    last_file_relpath = 'WTF? Blooper!'
    num_fetch_threads = kwargs.get('num_fetch_threads', 1)
    for audio_filepath, processed_res in processed_items_generator(
            num_fetch_threads,
            Audio.get_file_clips, src_generator,
            settings=audio_settings,
            channels=channels):

        # Unpack processed item container
        (clips, clip_start_samples, curr_file_dur, curr_file_ch_idxs) = \
            processed_res

        if clip_start_samples is None:
            logger.warning('{:s} yielded 0 clips'.format(repr(audio_filepath)))
            continue

        # 'clips container' will be a num_channels length list containing arrays
        # of shape [num clips, num samples per clip]
        num_samples = clips[0].shape[1]

        # Run the model on the audio file's contents, for all channels.
        # At first, concatenate every channels' clips. Analyze together. And,
        # then split the detections back.
        det_scores, time_taken = analyze_clips(
            classifier,
            np.concatenate(clips, axis=0),
            kwargs.get('batch_size', 1),
            None if not kwargs.get('show_progress', False) else audio_filepath)
        det_scores = np.stack(
            np.split(det_scores,
                     np.cumsum([c.shape[0] for c in clips])[:-1],
                     axis=0)
            , axis=0)

        total_audio_dur += curr_file_dur
        total_time_taken += time_taken

        if audio_filepath == audio_root:  # Single file was specified
            audio_relpath = os.path.basename(audio_filepath)
            seltab_relpath = os.path.splitext(audio_relpath)[0]
        else:
            audio_relpath = os.path.relpath(audio_filepath, start=audio_root)
            subdirs = os.path.split(audio_relpath)[0]
            # Seltab filename based on dir (if combining results) or filename
            seltab_relpath = (
                ('results' if subdirs == '' else subdirs) if combine_outputs
                else os.path.splitext(audio_relpath)[0])
        seltab_relpath += _selection_table_file_suffix

        # Scale the scores, if enabled
        det_scores = scale_scores(det_scores)

        if raw_output_executor is not None:
            # Offload writing of raw results (if enabled)
            # Fire and forget. No need to wait for or fetch results.
            raw_output_executor.submit(
                write_raw_detections,
                os.path.join(raw_detections_dir,
                             audio_relpath + FilenameExtensions.numpy),
                audio_settings.fs,
                det_scores.copy(), clip_start_samples.copy(),
                num_samples,
                None if channels is None else curr_file_ch_idxs)

        # Offload writing of processed results (if enabled)
        if output_executor is not None:

            # First, wait for the previous writing to finish (if any)
            if output_executor_future is not None:
                try:
                    num_dets_written = output_executor_future.result()
                except Exception as exc:
                    logger.error(('Writing out recognition results from file {:s} to file {:s} generated exception: ' +
                                  '{:s}').format(repr(last_file_relpath), repr(selmap[1]), repr(exc)))
                    num_dets_written = 0

                sel_running_info[0] += num_dets_written
                sel_running_info[1] += last_file_dur

                if selmap[1] != seltab_relpath:
                    # About to start a new seltab file. Write out logs about
                    # previous seltab file
                    logger.info('{:s} -> {:s}: {:d} detections, {:.3f}s processing time'.format(
                        selmap[0], selmap[1], sel_running_info[0], selmap[2]))

            if selmap is None or selmap[1] != seltab_relpath:
                # First time here, or output seltab file is to be changed.
                # Open new seltab file and (re-)init counters.

                os.makedirs(
                    os.path.join(output_dir, os.path.split(seltab_relpath)[0]),
                    exist_ok=True)
                out_file_info = (os.path.join(output_dir, seltab_relpath), True)
                selmap = [
                    os.path.split(audio_relpath)[0] if combine_outputs
                    else audio_relpath,
                    seltab_relpath, time_taken]
                sel_running_info = [0, 0.]  # sel num offset, file time offset

            else:
                # False here causes 'appending' to existing file
                out_file_info = (os.path.join(output_dir, selmap[1]), False)
                selmap[2] += time_taken

            # Offload writing of recognition results to a separate thread.
            # Send in data for only those valid classes in the mask.
            output_executor_future = output_executor.submit(
                _combine_and_write,
                raven_writer, out_file_info + tuple(), det_scores.copy(),
                clip_start_samples.copy(), num_samples, audio_settings.fs,
                class_names, class_freq_extents,
                threshold=kwargs.get('threshold', None),
                channel_IDs=None if (
                        channels is None and len(curr_file_ch_idxs) == 1
                    ) else curr_file_ch_idxs + 1,
                offset_info=None if not combine_outputs else (
                    sel_running_info[0], sel_running_info[1],
                    os.path.basename(audio_relpath)),
                ignore_class=reject_class_idx,
                suppress_nonmax=suppress_nonmax,
                squeeze_min_dur=squeeze_min_dur)

            last_file_relpath = audio_relpath
            last_file_dur = curr_file_dur

        else:
            logger.info('{:s} -> {:s}: {:.3f}s processing time'.format(
                audio_relpath, audio_relpath + '.npz', time_taken))

    # Done looping. Wait for the last 'write' thread to finish, if any
    if output_executor is not None and output_executor_future is not None:
        try:
            num_dets_written = output_executor_future.result()
        except Exception as exc:
            logger.error(('Writing out recognition results from file {:s} to file {:s} generated exception: ' +
                          '{:s}').format(repr(last_file_relpath), repr(selmap[1]), repr(exc)))
            num_dets_written = 0

        sel_running_info[0] += num_dets_written

        # write out last log
        logger.info('{:s} -> {:s}: {:d} detections, {:.3f}s processing time'.format(
            selmap[0], selmap[1], sel_running_info[0], selmap[2]))

    if raw_output_executor is not None:
        raw_output_executor.shutdown()

    if output_executor is not None:
        output_executor.shutdown()

    if total_audio_dur == 0:
        print('No files processed')
    else:
        msg = '{:.3f} s of audio processed in {:.3f} s. Realtime factor: {:.2f}x.'.format(
            total_audio_dur, total_time_taken, total_audio_dur / total_time_taken)
        logger.info(msg)
        print(msg)


__all__ = ['recognize']


def _fetch_freq_info(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)


def cmdline_parser(parser=None):

    if parser is None:
        parser = argparse.ArgumentParser(
            prog='koogu.recognize', allow_abbrev=False,
            description='Perform recognition using a trained model.')

    parser.add_argument(
        'modeldir', metavar='<MODEL DIR>',
        help='Path to the directory containing a model trained with Koogu.')
    parser.add_argument(
        'src', metavar='<AUDIO SOURCE>',
        help='Path to an audio file or to a directory. When a directory is '
             'provided, all files (of the supported types) within the directory'
             ' will be processed (use the --recursive flag to also process '
             'subdirectories).')
    parser.add_argument(
        'dst', metavar='<OUTPUT ROOT>',
        help='Path to directory into which detection outputs will be written. '
             'If necessary, subdirectories will be automatically created.')

    arg_group_in_ctrl = parser.add_argument_group('Input control')
    arg_group_in_ctrl.add_argument(
        '--filetypes', metavar='EXTN', nargs='+',
        # default=AudioFileList.default_audio_filetypes,
        help='Audio file types to restrict processing to. Can specify multiple '
             'types separated by whitespaces. By default, will process all '
             'discovered files with the following extensions: [' + ', '.join(
                 AudioFileList.default_audio_filetypes) +
             ']. Option is ignored when <AUDIO SOURCE> is a single file.')
    arg_group_in_ctrl.add_argument(
        '--recursive', action='store_true',
        help='Process files also in subdirectories of <AUDIO SOURCE>.')
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

    arg_group_out_ctrl = parser.add_argument_group(
        'Output control',
        description='Store recognition outputs either as raw per-clip '
                    'per-class scores (used in assessing model performance), or'
                    ' apply a threshold followed by a post-processing algorithm'
                    ' and store results in a format suitable for downstream '
                    'analyses. For a group of contiguous clips having scores '
                    '(corresponding to a class) above the specified threshold, '
                    'the default post-processing algorithm reports a single '
                    'detection, with the outermost temporal extents of the '
                    'group and the maximum score among the group as the '
                    'detection\'s temporal extents and score, respectively.')
    out_ctrl_mutex_grp = arg_group_out_ctrl.add_mutually_exclusive_group(
        required=True)
    out_ctrl_mutex_grp.add_argument(
        '--raw-scores', dest='raw_scores', action='store_true',
        help='Option to output raw per-clip per-class recognition scores (only)'
             ' as is. If set, no post-processing algorithm is applied and all '
             'other settings within the \'Output control\' group will be '
             'ignored.')
    out_ctrl_mutex_grp.add_argument(
        '--threshold', metavar='#', type=ArgparseConverters.float_0_to_1,
        help='Suppress detections having scores below this value (valid range: '
             '0.0-1.0). If set, a post-processing algorithm will be applied.')
    arg_group_out_ctrl.add_argument(
        '--reject-class', dest='reject_class', metavar='CLASS', nargs='+',
        help='Name (case sensitive) of the class (like \'Noise\' or \'Other\') '
             'that must be excluded from the recognition results. Can specify '
             'multiple (separated by whitespaces).')
    arg_group_out_ctrl.add_argument(
        '--squeeze', metavar='MIN-DUR', type=ArgparseConverters.positive_float,
        help='If specified, will apply (the non-default) algorithm which \''
             'squeezes together\' successive detections from temporally '
             'overlapping clips. The \'squeezing\' will be restricted to '
             'produce detections which are at least \'MIN-DUR\' seconds long. '
             'MIN-DUR amount specified must be smaller than the duration of '
             'the model\'s input.')
    arg_group_out_ctrl.add_argument(
        '--frequency-info', dest='freq_info', metavar='FILE',
        help='Path to a json file containing a dictionary of per-class '
             'frequency bounds. If unspecified, corresponding fields in the '
             'outputs (if applicable) will be the same for all classes.')
    arg_group_out_ctrl.add_argument(
        '--raven-combine-outputs', dest='raven_combine_outputs',
        action='store_true',
        help='Enable this to combine recognition results of processing every '
             'file within a directory and write them to a single Raven '
             'selection table file. When enabled, the outputs will contain 2 '
             'additional fields describing offsets of detections in the '
             'corresponding audio files.')

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
    # arg_group_prctrl.add_argument(
    #     '--show-progress', dest='show_progress', action='store_true',
    #     help='Show progress of processing on screen.')
    arg_group_prctrl.set_defaults(show_progress=True)   # Force this

    arg_group_logging = parser.add_argument_group('Logging')
    arg_group_logging.add_argument(
        '--log', dest='log_file', metavar='LOGFILE',
        help='If set, logging will be enabled and written out to the specified '
             'file.')
    arg_group_logging.add_argument(
        '--loglevel', dest='log_level',
        choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'],
        default='INFO', help='Logging level.')

    parser.set_defaults(exec_fn=cmdline_recognize)

    return parser


def cmdline_recognize(modeldir, src, dst,
                      raw_scores, threshold,
                      log_level, log_file=None,
                      **kwargs):
    """
    Functionality invoked via the command-line interface.
    NOTE: `raw_scores` and `threshold` are mutually exclusive. If `raw_scores`
          is specified, `threshold` will be ignored.
    """

    if not os.path.exists(modeldir):
        print('Error: Invalid model directory path specified', file=sys.stderr)
        exit(1)

    if not os.path.exists(src):
        print('Error: Invalid audio path specified', file=sys.stderr)
        exit(2)

    os.makedirs(dst, exist_ok=True)

    pp_kwargs = dict()
    if raw_scores:
        output_dir = None
        raw_detections_dir = dst
    else:
        output_dir = dst
        raw_detections_dir = None

        pp_kwargs['threshold'] = threshold

        # Other 'Output control'/post-processing args
        val = kwargs.pop('reject_class', None)
        if val is not None:
            pp_kwargs['reject_class'] = val
        val = kwargs.pop('squeeze', None)
        if val is not None:
            pp_kwargs['squeeze_detections'] = val
        val = kwargs.pop('freq_info', None)
        if val is not None:
            pp_kwargs['frequency_extents'] = _fetch_freq_info(val)      # TODO: Add try-catch around this
        val = kwargs.pop('raven_combine_outputs', None)
        if val is not None:
            pp_kwargs['combine_outputs'] = val

    # 'Input control' args
    ic_args = dict()
    val = kwargs.pop('filetypes', None)
    if val is not None:
        ic_args['filetypes'] = val
    val = kwargs.pop('recursive', None)
    if val is not None:
        ic_args['recursive'] = val
    val = kwargs.pop('channels', None)
    if val is not None:
        ic_args['channels'] = np.sort(np.unique(val).astype(np.uint32))
    val = kwargs.pop('clip_advance', None)
    if val is not None:
        ic_args['clip_advance'] = val

    # 'Process control' args
    prc_args = dict()
    val = kwargs.pop('num_fetch_threads', None)
    if val is not None:
        prc_args['num_fetch_threads'] = val
    val = kwargs.pop('batch_size', None)
    if val is not None:
        prc_args['batch_size'] = val
    val = kwargs.pop('show_progress', None)
    if val is not None:
        prc_args['show_progress'] = val

    if log_file is not None:
        instantiate_logging(log_file, log_level, 'a')
        logging.info(f'Model: {modeldir}')
        logging.info(
            f"{'raw_detections_dir' if raw_scores else 'output_dir'}: {dst}")
        logging.info(f'Output control: {pp_kwargs}')
        logging.info(f'Input control: {ic_args}')
        logging.info(f'Process control: {prc_args}')

    recognize(modeldir, src,
              output_dir=output_dir, raw_detections_dir=raw_detections_dir,
              **ic_args,
              **pp_kwargs,
              **prc_args)

    if log_file is not None:
        logging.shutdown()
