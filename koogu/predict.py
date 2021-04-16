
import os
import sys
import numpy as np
from timeit import default_timer as timer
import json
import argparse
import concurrent.futures
import logging
import librosa

from koogu.data import Audio, Settings, Convert
from koogu.model import TrainedModel
from koogu.utils import processed_items_generator_mp, detections
from koogu.utils.terminal import ProgressBar, ArgparseConverters
from koogu.utils.filesystem import recursive_listing

_program_name = 'predict'
_default_audio_filetypes = ['.wav', '.WAV', '.flac', '.aif', '.mp3']
_selection_table_file_suffix = '.selections.txt'

output_spec = [
    ['Selection',       '{:d}'],
    ['Channel',         '{:d}'],
    ['Begin Time (s)',  '{:.6f}'],
    ['End Time (s)',    '{:.6f}'],
    ['Low Frequency (Hz)',  '{:.2f}'],
    ['High Frequency (Hz)', '{:.2f}'],
    ['Begin File',      '{:s}'],
    ['File Offset (s)', '{:.6f}'],
    ['Score',           '{:.2f}'],
    ['Tags',            '{}'],
    ['Begin Path',      ' ']]


def _fetch_clips(audio_filepath, audio_settings, channels, spec_settings=None):
    """

    If spec_settings is not None, the clips will be converted to time-frequency representation.
    """

    clips, clip_start_samples = Audio.get_file_clips(audio_settings, audio_filepath,
                                                     downmix_channels=False,
                                                     chosen_channels=channels,
                                                     return_clip_indices=True)

    # add the channel axis if it doesn't already exist
    if len(clips.shape) < 3:
        clips = np.expand_dims(clips, 0)

    num_samples = clips.shape[-1]

    if spec_settings is not None:
        clips = np.stack([
            Convert.audio2spectral(clips[ch, ...], audio_settings.fs, spec_settings)
            for ch in range(clips.shape[0])])

    # return file duration, loaded clips, their starting samples, & num samples per clip
    return librosa.get_duration(filename=audio_filepath), clips, clip_start_samples, num_samples


def analyze_clips(classifier, clips, batch_size=1, audio_filepath=None):
    """
    Run predictions on clips and obtain classification results.
    :param classifier: A model.TrainedModel instance.
    :param clips: An [N x ?] numpy array of N input data.
    :param batch_size: Control how many clips are processed in a single batch.
    :param audio_filepath: If not None, will display a progress bar.
    :return: A tuple consisting of -
        confidence values ([N x M] numpy array),
        and the total time taken to process the clips.
    """

    pbar = None if audio_filepath is None else \
        ProgressBar(clips.shape[0], prefix='{:>59s}'.format(audio_filepath[-59:]), length=10, show_start=True)

    batch_start_idxs = np.arange(0, clips.shape[0], batch_size)
    batch_end_idxs = np.minimum(batch_start_idxs + batch_size, clips.shape[0])
    det_scores = [None] * len(batch_end_idxs)
    predict_time = 0.
    for idx, (s_idx, e_idx) in enumerate(zip(batch_start_idxs, batch_end_idxs)):

        t_start = timer()
        det_scores[idx] = classifier.infer(inputs=clips[s_idx:e_idx, ...])
        predict_time += (timer() - t_start)

        if pbar is not None:
            pbar.increment(e_idx - s_idx)

    return np.concatenate(det_scores, axis=0), predict_time


def _combine_and_write(outfile_h, det_scores, clip_start_samples, num_samples, fs,
                       class_names, class_frequencies,
                       channel_IDs=None,
                       offset_info=None,
                       ignore_class=None,
                       threshold=0.0,
                       suppress_nonmax=False,
                       squeeze_min_dur=None):

    valid_cols_mask = np.concatenate([
        [True],                                                     # Sel num
        [False] if channel_IDs is None else [True],                 # Channel
        [True, True],                                               # Start and end times
        [True, True],                                               # Start and end freq
        [False, False] if offset_info is None else [True, True],    # Filename & offset
        [True, True],                                               # Score and label
        [True]                                                      # Bogus
    ]).astype(np.bool)

    # Fields in output selection table. No need of fields describing offsets of detections
    output_header = '\t'.join([h[0] for idx, h in enumerate(output_spec) if valid_cols_mask[idx]]) + '\n'
    output_fmt_str = '\t'.join([h[1] for idx, h in enumerate(output_spec) if valid_cols_mask[idx]]) + '\n'

    num_channels, num_clips, num_classes = det_scores.shape

    if num_clips == 0:
        # Write the header only and return immediately
        if outfile_h[1] == 'w':
            with open(outfile_h[0], outfile_h[1]) as seltab_file:
                seltab_file.write(output_header)
        return 0

    # Suppress non-max classes, if enabled
    if suppress_nonmax:
        nonmax_mask = np.full(det_scores.shape, True, dtype=np.bool)
        for ch in range(num_channels):
            nonmax_mask[ch, np.arange(num_clips), det_scores[ch].argmax(axis=1)] = False
        det_scores[nonmax_mask] = 0.0

    # Mask out the ignore_class(es), so that we don't waste time post-processing those results
    write_class_mask = np.full((num_classes, ), True)
    if ignore_class is not None:
        if hasattr(ignore_class, '__len__'):
            write_class_mask[np.asarray([c for c in ignore_class])] = False
        else:
            write_class_mask[ignore_class] = False
    class_idx_remapper = np.asarray(np.where(write_class_mask)).ravel()

    # First, combine detections within each channel and gather per-channel combined results
    channel_combined_det_times = [None] * num_channels
    channel_combined_det_scores = [None] * num_channels
    channel_combined_det_labels = [None] * num_channels
    num_combined_dets_per_channel = np.zeros((num_channels,), np.uint32)
    min_det_dur = None if squeeze_min_dur is None else int(squeeze_min_dur * fs)
    for ch in range(num_channels):
        channel_combined_det_times[ch], channel_combined_det_scores[ch], channel_combined_det_labels[ch] = \
            detections.combine_streaks(det_scores[ch:ch+1, :, write_class_mask][0],
                                       clip_start_samples, num_samples, min_det_dur)

        num_combined_dets_per_channel[ch] = channel_combined_det_scores[ch].shape[0]

    if int(num_combined_dets_per_channel.sum()) == 0:   # No detections available
        # Write the header only and return immediately
        if outfile_h[1] == 'w':
            with open(outfile_h[0], outfile_h[1]) as seltab_file:
                seltab_file.write(output_header)
        return 0

    # Flatten
    combined_det_times = np.concatenate(channel_combined_det_times, axis=0)
    combined_det_scores = np.concatenate(channel_combined_det_scores)
    combined_det_labels = np.concatenate(channel_combined_det_labels)
    if channel_IDs is not None:
        combined_det_channels = np.concatenate([np.full((num_combined_dets_per_channel[ch],), channel_IDs[ch])
                                                for ch in range(num_channels)])

    # Remap class IDs to make good for the gaps from ignore_class
    combined_det_labels = class_idx_remapper[combined_det_labels]

    # Apply threshold
    valid_dets_mask = combined_det_scores >= threshold
    combined_det_times = combined_det_times[valid_dets_mask, ...]
    combined_det_scores = combined_det_scores[valid_dets_mask]
    combined_det_labels = combined_det_labels[valid_dets_mask]
    if channel_IDs is not None:
        combined_det_channels = combined_det_channels[valid_dets_mask]

    # Sort the detections across channels (based on detection start time)
    sort_idx = np.argsort(combined_det_times[:, 0])
    combined_det_times = combined_det_times[sort_idx, ...]
    combined_det_scores = combined_det_scores[sort_idx]
    combined_det_labels = combined_det_labels[sort_idx]
    if channel_IDs is not None:
        combined_det_channels = combined_det_channels[sort_idx]

    # Convert detection extents from samples to seconds
    combined_det_times = combined_det_times.astype(np.float) / float(fs)

    if isinstance(class_frequencies[0], list):  # is a 2D list
        def freq_output(l_idx): return class_frequencies[l_idx]
    else:
        def freq_output(_): return class_frequencies    # same for all

    if offset_info is not None:     # Apply the offsets
        o_sel, o_time, o_file = offset_info

        if channel_IDs is None:
            def writer(file_h, d_idx):
                l_freq = freq_output(combined_det_labels[d_idx])
                file_h.write(output_fmt_str.format(
                    o_sel + d_idx + 1,
                    o_time + combined_det_times[d_idx, 0], o_time + combined_det_times[d_idx, 1],
                    l_freq[0], l_freq[1],
                    o_file, combined_det_times[d_idx, 0],
                    combined_det_scores[d_idx], class_names[combined_det_labels[d_idx]]))
        else:
            def writer(file_h, d_idx):
                l_freq = freq_output(combined_det_labels[d_idx])
                file_h.write(output_fmt_str.format(
                    o_sel + d_idx + 1,
                    combined_det_channels[d_idx],
                    o_time + combined_det_times[d_idx, 0], o_time + combined_det_times[d_idx, 1],
                    l_freq[0], l_freq[1],
                    o_file, combined_det_times[d_idx, 0],
                    combined_det_scores[d_idx], class_names[combined_det_labels[d_idx]]))

    else:       # Fields for indicating offset info are either meaningless or not needed in the output
        if channel_IDs is None:
            def writer(file_h, d_idx):
                l_freq = freq_output(combined_det_labels[d_idx])
                file_h.write(output_fmt_str.format(
                    d_idx + 1,
                    combined_det_times[d_idx, 0], combined_det_times[d_idx, 1],
                    l_freq[0], l_freq[1],
                    combined_det_scores[d_idx], class_names[combined_det_labels[d_idx]]))
        else:
            def writer(file_h, d_idx):
                l_freq = freq_output(combined_det_labels[d_idx])
                file_h.write(output_fmt_str.format(
                    d_idx + 1,
                    combined_det_channels[d_idx],
                    combined_det_times[d_idx, 0], combined_det_times[d_idx, 1],
                    l_freq[0], l_freq[1],
                    combined_det_scores[d_idx], class_names[combined_det_labels[d_idx]]))

    # Finally, write out the outputs
    with open(outfile_h[0], outfile_h[1]) as seltab_file:
        if outfile_h[1] == 'w':
            seltab_file.write(output_header)
        for d_idx in range(combined_det_scores.shape[0]):
            writer(seltab_file, d_idx)

    return combined_det_scores.shape[0]


def write_raw_detections(file_path, det_scores, clip_start_samples, num_samples, channel_IDs):

    os.makedirs(os.path.split(file_path)[0], exist_ok=True)

    res = dict(
        clip_length=num_samples,
        clip_start_samples=clip_start_samples,
        scores=det_scores
    )
    if channel_IDs is not None:
        res['channels'] = channel_IDs.astype(np.uint8)

    np.savez_compressed(file_path, **res)


def main(args):

    # Load the classifier
    classifier = TrainedModel(args.modeldir)

    # Query some dataset info
    class_names = classifier.class_names
    audio_settings = classifier.audio_settings
    spec_settings = None if classifier.spec_settings is None \
        else Settings.Spectral(audio_settings['desired_fs'], **classifier.spec_settings)

    # Override clip_advance, if specified
    if args.clip_advance is not None:
        audio_settings['clip_advance'] = args.clip_advance

    raw_output_executor = None
    if args.raw_outputs_dir:
        os.makedirs(args.raw_outputs_dir, exist_ok=True)
        raw_output_executor = concurrent.futures.ProcessPoolExecutor(max_workers=1)

    output_executor = None
    if args.proc_outputs_dir:
        reject_class_idx = None
        if args.reject_class is not None:
            reject_class_idx = []
            for rj_class in args.reject_class:
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
        if not args.freq_info:  # none specified, set the same for all classes
            class_freq_extents = default_freq_extents
        else:
            with open(args.freq_info, 'r') as f:
                freq_extents_dict = json.load(f)

            # Assign defaults for missing classes
            class_freq_extents = [
                freq_extents_dict.get(cn, default_freq_extents)
                for cn in class_names
                ]

        # Set up function to scale scores, if enabled
        if args.scale_scores:
            frac = 1.0 / float(len(class_names))
            def scale_scores(scores): return np.maximum(0.0, (scores - frac) / (1.0 - frac))
        else:
            def scale_scores(scores): return scores

        # Check post-processing settings
        squeeze_min_dur = None
        suppress_nonmax = False
        if args.top:
            suppress_nonmax = True
        elif args.squeeze is not None:
            squeeze_min_dur = args.squeeze
        elif args.top_squeeze is not None:
            suppress_nonmax = True
            squeeze_min_dur = args.top_squeeze

        os.makedirs(args.proc_outputs_dir, exist_ok=True)
        output_executor = concurrent.futures.ProcessPoolExecutor(max_workers=1)
        output_executor_future = None

    if not os.path.isdir(args.src):     # Single file input
        src_generator = [args.src]  # turn it into a list

        # Forward all logging to stdout
        logging.basicConfig(stream=sys.stdout, filemode='a', level=args.loglevel,
                            format='%(asctime)s[%(levelname).1s] %(funcName)s: %(message)s', datefmt="%Y%m%dT%H%M%S")

        # No need of fields describing offsets of detections
        args.combine_outputs = False    # Disable this (if it was enabled)

    else:
        # Prepare the input file generator
        if args.recursive:
            src_generator = (os.path.join(args.src, f)
                             for f in recursive_listing(args.src, match_extensions=args.filetypes))
        else:
            # Only get the top-level files
            src_generator = (os.path.join(args.src, f) for f in os.listdir(args.src)
                             if (any((f.endswith(e) for e in args.filetypes)) and
                                 os.path.isfile(os.path.join(args.src, f))))

        # Create logger
        logfile = args.log if args.log is not None else os.path.join((args.proc_outputs_dir or args.raw_outputs_dir),
                                                                     _program_name + '.log')
        logging.basicConfig(filename=logfile, filemode='w', level=args.loglevel,
                            format='%(asctime)s[%(levelname).1s] %(funcName)s: %(message)s', datefmt="%Y%m%dT%H%M%S")

        logging.info('Model : {:s}'.format(repr(args.modeldir)))
        logging.info('Source: {:s}'.format(repr(args.src)))
        logging.info('Config: {}'.format(audio_settings))
        if args.raw_outputs_dir:
            logging.info('Raw Output: {:s}'.format(repr(args.raw_outputs_dir)))
        if args.proc_outputs_dir:
            logging.info('Processed Output: {:s}'.format(repr(args.proc_outputs_dir)))
            if args.reject_class is not None:
                logging.info('Reject class: {:s}'.format(
                    repr([class_names[rid] for rid in reject_class_idx]) if len(reject_class_idx) > 0 else 'None'))
            if args.threshold is not None:
                logging.info('Threshold: {:f}'.format(args.threshold))
            if args.scale_scores is not None and args.scale_scores:
                logging.info('Scale scores: True')
            if args.top is not None and args.top:
                logging.info('Postprocessing algorithm: Top class')
            elif args.squeeze is not None:
                logging.info('Postprocessing algorithm: Squeeze (MIN-DUR = {:f} s)'.format(args.squeeze))
            elif args.top_squeeze is not None:
                logging.info('Postprocessing algorithm: Top class, Squeeze (MIN-DUR = {:f} s)'.format(args.top_squeeze))
            else:
                logging.info('Postprocessing algorithm: Default')

#    tf.logging.set_verbosity(args.loglevel)
    logger = logging.getLogger(__name__)

    if args.proc_outputs_dir and squeeze_min_dur is not None and squeeze_min_dur > audio_settings['clip_length']:
        logger.warning('Squeeze min duration ({:f} s) is larger than model input length ({:f} s)'.format(
            squeeze_min_dur, audio_settings['clip_length']))

    # Convert to a container that is needed by Audio.get_file_clips()
    audio_settings = Settings.Audio(**audio_settings)

    # Prepare parameters for audio_loader
    if args.channels is None:                       # fetch all channels' clips independently
        channels = None
    else:                                           # fetch selected channel's clips
        channels = (args.channels - 1)  # convert indices to be 0-based

    print('Starting to predict...')

    selmap = None       # Will contain src rel path, seltab file relpath, analysis time
    sel_running_info = None  # Will contain running info -> last sel num, time offset for next file
    last_file_dur = 0.
    total_audio_dur = 0.
    total_time_taken = 0.
    last_file_relpath = 'WTF? Blooper!'
    num_fetch_threads = 1 if args.num_fetch_threads is None else args.num_fetch_threads
    for audio_filepath, (curr_file_dur, clips, clip_start_samples, num_samples) in \
            processed_items_generator_mp(num_fetch_threads, _fetch_clips, src_generator,
                                         audio_settings=audio_settings,
                                         spec_settings=spec_settings,
                                         channels=channels):

        # 'clips' container will be of shape [num channels, num clips, ...]
        num_channels, num_clips = clips.shape[:2]

        if num_clips == 0:
            logger.warning('{:s} yielded 0 clips'.format(repr(audio_filepath)))
            continue

        # Run the model on the audio file's contents, separately for each channel
        # At first, concatenate every channels' clips. Analyze together. And, then split the detections back.
        det_scores, time_taken = analyze_clips(classifier,
                                               np.concatenate(np.split(clips, num_channels, axis=0), axis=1)[0],
                                               args.batch_size,
                                               None if not args.show_progress else audio_filepath)
        det_scores = np.stack(np.split(det_scores, num_channels, axis=0), axis=0)

        total_audio_dur += curr_file_dur
        total_time_taken += time_taken

        # Determine the channel number(s) to write out in the outputs
        channels_to_write = np.arange(1, num_channels + 1) \
            if args.channels is not None and len(args.channels) == 0 \
            else args.channels

        if audio_filepath == args.src:  # Single file
            audio_relpath = os.path.basename(audio_filepath)
            seltab_relpath = os.path.splitext(audio_relpath)[0]
        else:
            audio_relpath = os.path.relpath(audio_filepath, start=args.src)
            subdirs = os.path.split(audio_relpath)[0]
            # Seltab filename based on dir (if combining results) or filename
            if subdirs == '':
                seltab_relpath = ('results' if args.combine_outputs else os.path.splitext(audio_relpath)[0])
            else:
                seltab_relpath = (subdirs if args.combine_outputs else os.path.splitext(audio_relpath)[0])
        seltab_relpath += _selection_table_file_suffix

        if raw_output_executor is not None:  # Offload writing of raw results (if enabled)
            # Fire and forget. No need to wait for or fetch results.
            raw_output_executor.submit(write_raw_detections,
                                       os.path.join(args.raw_outputs_dir, audio_relpath + '.npz'),
                                       det_scores.copy(), clip_start_samples.copy(),
                                       num_samples,
                                       channels_to_write)

        if output_executor is not None:  # Offload writing of processed results (if enabled)
            # Scale the scores, if enabled
            det_scores = scale_scores(det_scores)

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
                    # About to start a new seltab file. Write out logs about previous seltab file
                    logger.info('{:s} -> {:s}: {:d} detections, {:.3f}s processing time'.format(
                        selmap[0], selmap[1], sel_running_info[0], selmap[2]))

            if selmap is None or selmap[1] != seltab_relpath:
                # First time here, or output seltab file is to be changed.
                # Open new seltab file and (re-)init counters.

                os.makedirs(os.path.join(args.proc_outputs_dir, os.path.split(seltab_relpath)[0]), exist_ok=True)
                seltab_file_h = (os.path.join(args.proc_outputs_dir, seltab_relpath), 'w')
                selmap = [os.path.split(audio_relpath)[0] if args.combine_outputs else audio_relpath,
                          seltab_relpath, time_taken]
                sel_running_info = [0, 0.]  # sel num offset, file time offset

            else:
                seltab_file_h = (os.path.join(args.proc_outputs_dir, selmap[1]), 'a')
                selmap[2] += time_taken

            # Offload writing of recognition results to a separate thread.
            # Send in data for only those valid classes in the mask.
            output_executor_future = output_executor.submit(
                _combine_and_write,
                seltab_file_h + tuple(), det_scores.copy(), clip_start_samples.copy(),
                num_samples, audio_settings.fs,
                class_names, class_freq_extents,
                channel_IDs=channels_to_write,
                offset_info=None if not args.combine_outputs
                            else (sel_running_info[0], sel_running_info[1], os.path.basename(audio_relpath)),
                ignore_class=reject_class_idx,
                threshold=0.0 if args.threshold is None else args.threshold,
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

    logging.shutdown()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog=_program_name, allow_abbrev=False,
                                     description='Make inferences using a trained model.')
    parser.add_argument('modeldir', metavar='<MODEL DIR>',
                        help='Path to the directory containing a TensorFlow exported model.')
    parser.add_argument('src', metavar='<AUDIO SOURCE>',
                        help='Path to either a single audio file or to a directory. When a directory, all files of ' +
                             'the supported filetypes within the specified directory will be processed (use the ' +
                             '--recursive flag to process subdirectories as well).')
    arg_group_in_ctrl = parser.add_argument_group('Input control')
    arg_group_in_ctrl.add_argument('--filetypes', metavar='EXTN', nargs='+', default=_default_audio_filetypes,
                                   help='Audio file types to restrict processing to. Option is ignored if processing ' +
                                        'a single file. Can specify multiple types separated by whitespaces. By ' +
                                        'default, will include for processing all discovered files with ' +
                                        'the following extensions: ' + ', '.join(_default_audio_filetypes))
    arg_group_in_ctrl.add_argument('--recursive', action='store_true',
                                   help='Process files also in subdirectories of <AUDIO_SOURCE>.')
    arg_group_in_ctrl.add_argument('--channels', metavar='#', nargs='+', type=ArgparseConverters.all_or_posint,
                                   help='Channels to restrict processing to. List out the desired channel numbers, ' +
                                        'separated with whitespaces. If unspecified, all available channels will be' +
                                        'processed.')
    arg_group_in_ctrl.add_argument('--clip-advance', metavar='SEC', dest='clip_advance',
                                   type=ArgparseConverters.positive_float,
                                   help='When audio file\'s contents are broken up into clips, by default the amount ' +
                                        'of overlap between successive clips is determined by the settings that were ' +
                                        'in place during model training. Use this flag to alter that, by setting a ' +
                                        'different amount (in seconds) of gap (or advance) between successive clips.')
    arg_group_type_ctrl = parser.add_argument_group('Output type(s)',
                                                    description='At least one of these must be specified. If multiple' +
                                                                ' audio files are to be processed, as many ' +
                                                                'corresponding output files will be generated, and ' +
                                                                'necessary subdirectories will be created.')
    arg_group_type_ctrl.add_argument('--raw-outputs', dest='raw_outputs_dir', metavar='DIR',
                                     help='If set, raw outputs from the model will be written out into the specified ' +
                                          'directory.')
    arg_group_type_ctrl.add_argument('--processed-outputs', dest='proc_outputs_dir', metavar='DIR',
                                     help='If set, processed recognition results (Raven selection tables) will be ' +
                                          'written out into the specified directory. Use options under \'Output ' +
                                          'control\' and \'Post-process control\' for further control.')
    arg_group_out_ctrl = parser.add_argument_group('Output control',
                                                   description='These options will have no effect if ' +
                                                               '--processed-outputs is not specified.')
    arg_group_out_ctrl.add_argument('--reject-class', dest='reject_class', metavar='CLASS', nargs='+',
                                    help='Name (case sensitive) of the class (like \'Noise\' or \'Other\') that must ' +
                                         'be ignored from the recognition results. The corresponding detections will ' +
                                         'not be written to the output selection tables. Can specify multiple (' +
                                         'separated by whitespaces).')
    arg_group_out_ctrl.add_argument('--frequency-info', dest='freq_info', metavar='FILE',
                                    help='Path to a json file containing a dictionary of per-class frequency bounds. ' +
                                         'If unspecified, the "Low Frequency (Hz)" and "High Frequency (Hz)" fields ' +
                                         'in the output table will be the same for all classes.')
    arg_group_out_ctrl.add_argument('--combine-outputs', dest='combine_outputs', action='store_true',
                                    help='Enable this to combine recognition results of processing every file within ' +
                                         'a directory and write them to a single output file. When enabled, the ' +
                                         'outputs will contain 2 additional fields describing offsets of detections ' +
                                         'in the corresponding audio files.')
    arg_group_out_ctrl.add_argument('--threshold', metavar='[0-1]', type=ArgparseConverters.float_0_to_1,
                                    help='Suppress writing of detections with confidence below this value.')
    arg_group_postproc = parser.add_argument_group('Post-process control',
                                                   description='By default, per-class scores from successive clips ' +
                                                               'are averaged to produce the results. You may choose ' +
                                                               'from one of the below alternative algorithms instead.' +
                                                               ' These options will have no effect if ' +
                                                               '--processed-outputs is not specified.')
    postproc_mutex_grp = arg_group_postproc.add_mutually_exclusive_group(required=False)
    postproc_mutex_grp.add_argument('--top', action='store_true',
                                    help='Same algorithm as default, but only considers the top-scoring class for ' +
                                         'each clip.')
    postproc_mutex_grp.add_argument('--squeeze', metavar='MIN-DUR', type=ArgparseConverters.positive_float,
                                    help='An algorithm \'to squeeze together\' temporally overlapping regions from ' +
                                         'successive raw detections will be applied. The \'squeezing\' will be ' +
                                         'restricted to produce detections that are at least \'MIN-DUR\' seconds long' +
                                         '. MIN-DUR must be smaller than the duration of the model input.')
    postproc_mutex_grp.add_argument('--top-squeeze', metavar='MIN-DUR', type=ArgparseConverters.positive_float,
                                    dest='top_squeeze',
                                    help='Same algorithm as --squeeze, but only considers the top-scoring class from ' +
                                         'each clip.')
    arg_group_postproc.add_argument('--scale-scores', dest='scale_scores', action='store_true',
                                    help='Enable this to scale the raw scores. Use of this setting is most ' +
                                         'recommended when the output of a model is based on softmax and the model ' +
                                         'was trained with training data where each input corresponded to a single ' +
                                         'class.')
    arg_group_misc = parser.add_argument_group('Miscellaneous')
    arg_group_misc.add_argument('--fetch-threads', dest='num_fetch_threads', type=ArgparseConverters.positive_integer,
                                metavar='NUM', default=1,
                                help='Number of threads that will fetch audio from files in parallel.')
    arg_group_misc.add_argument('--batch-size', dest='batch_size', type=ArgparseConverters.positive_integer,
                                metavar='NUM', default=1,
                                help='Size to batch audio file\'s clips into (default: %(default)d). Increasing this ' +
                                     'may improve speed on computers with high RAM.')
    arg_group_misc.add_argument('--show-progress', dest='show_progress', action='store_true',
                                help='Show progress of processing on screen.')
    arg_group_logging = parser.add_argument_group('Logging')
    arg_group_logging.add_argument('--log', metavar='FILE',
                                   help='Path to file to which logs will be written out.')
    arg_group_logging.add_argument('--loglevel', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'],
                                   default='INFO',
                                   help='Logging level.')
    args = parser.parse_args()

    if not os.path.exists(args.src) or not os.path.exists(args.modeldir):
        print('Error: Invalid model and/or audio path specified', file=sys.stderr)
        exit(2)

    if not (args.raw_outputs_dir or args.proc_outputs_dir):
        print('Error: At least one of --raw-outputs and --processed-outputs must be specified.')
        exit(2)

    if args.channels is not None:
        args.channels = np.sort(np.unique(args.channels).astype(np.uint32))

    main(args)
