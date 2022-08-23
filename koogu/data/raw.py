
import numpy as np
from scipy.signal import butter, sosfiltfilt, spectrogram, lfilter
from scipy.signal.windows import hann
import soundfile as sf
import audioread
import resampy
import logging
from koogu.utils.detections import assess_annotations_and_clips_match


class Settings:

    class Audio:
        def __init__(self, desired_fs, clip_length, clip_advance,
                     filterspec=None, normalize_clips=True, consider_trailing_clip=False):
            """
            Validate parameters (usually as loaded from a config file) and set appropriate fields.
            """

            assert 0.0 < clip_advance <= clip_length, 'clip_advance must be in the range [0.0 - clip_length)'
            assert filterspec is None or (isinstance(filterspec, (list, tuple)) and len(filterspec) == 3)

            # Rename 'desired_fs' to 'fs'
            self.fs = desired_fs

            # Convert to num samples
            self.clip_length = int(round(clip_length * self.fs))
            self.clip_advance = int(round(clip_advance * self.fs))

            # Setup filtering if requested (filterspec is needed for this)
            if filterspec is not None:
                self.filter_sos = Filters.butterworth_filter(filterspec, self.fs)
            else:
                self.filter_sos = None

            # Copy other parameters
            self.normalize_clips = normalize_clips
            self.consider_trailing_clip = consider_trailing_clip

    class Spectral:
        def __init__(self, fs, win_len, win_overlap_prc,
                     nfft_equals_win_len=True,
                     bandwidth_clip=None,
                     tf_rep_type='spec_db',
                     eps=1e-10,
                     num_mels=None):
            """
            Validate parameters (usually as loaded from a config file) and set appropriate fields in terms of number of
            samples/points as would be used by some of the code FFT libraries.
            """

            self.win_len_samples = int(round(win_len * fs))     # Convert seconds to samples
            assert self.win_len_samples > 0

            self.win_overlap_samples = int(round(win_len * win_overlap_prc * fs))
            assert 0 <= self.win_overlap_samples < self.win_len_samples

            if nfft_equals_win_len in [None, False]:
                self.nfft = int(2 ** np.ceil(np.log2(self.win_len_samples)).astype(np.int))  # nextpow2
            else:
                self.nfft = self.win_len_samples

            # Set to default ([0, nyquist rate]) if not provided
            if bandwidth_clip is None:
                self.bandwidth_clip = [0., fs/2.]
            else:
                assert len(bandwidth_clip) == 2
                self.bandwidth_clip = bandwidth_clip

            # Set to default ('spec') if not provided
            rep_type = tf_rep_type.lower() if tf_rep_type is not None else 'spec'
            # Currently supported formats
            assert rep_type in ['spec', 'spec_db', 'spec_dbfs', 'spec_pcen',
                                'melspec', 'melspec_db', 'melspec_dbfs', 'melspec_pcen']
            self.tf_rep_type = rep_type

            self.eps = eps
            self.num_mels = num_mels


class Audio:

    @staticmethod
    def load(filepath, desired_fs=None,
             offset=0.0, duration=None, channels=None, dtype=np.float32,
             resample_type=None):
        """
        Load an audio from a file on disk using librosa's logic, which includes
            - reading the file using `SoundFile` first, and falling back to
              using `audioread` if the former failed;
            - resampling (if requested) using the `resampy` and adjusting the
              array-length of the resampled data.
        Intended to be somewhat of a drop-in replacement for librosa.load().
        My addition includes handling of channels - can choose which channels to
        load and process. Also, the returned data is always 2d.
        """

        try:
            # First, attempt with SoundFile
            data, file_fs = Audio.__soundfile_load(
                filepath, offset, duration, channels, dtype)

        except RuntimeError:
            # SoundFile failed. Attempt loading with audioread now
            logging.getLogger(__name__).warning(
                f'SoundFile failed on {filepath}. Trying audioread instead...')
            data, file_fs = Audio.__audioread_load(
                filepath, offset, duration, channels, dtype)

        # Resample if and as requested
        if desired_fs is not None and desired_fs != file_fs:
            target_num_samples = int(
                np.ceil(data.shape[-1] * (float(desired_fs) / file_fs)))

            data = resampy.resample(data, file_fs, desired_fs,
                                    filter=resample_type or 'kaiser_best',
                                    axis=-1)

            # fix length
            if target_num_samples < data.shape[-1]:     # is longer?
                data = data[:, :target_num_samples]     # crop!
            elif target_num_samples > data.shape[-1]:   # is shorter?
                data = np.pad(                          # pad zeros!
                    data,
                    [[0, 0], [0, target_num_samples - data.shape[-1]]],
                    'constant')

        return data, (desired_fs or file_fs)

    @staticmethod
    def get_info(filepath):
        """
        Query an audio file's sampling rate, duration, and num channels.
        """

        # Try SoundFile first. Fall back to audioread only if there was a
        # runtime error. Will puke if any other type of error occurs.
        try:
            ainfo = sf.info(filepath)
            return ainfo.samplerate, ainfo.duration, ainfo.channels
        except RuntimeError:
            with audioread.audio_open(filepath) as fd:
                return fd.samplerate, fd.duration, fd.channels

    @staticmethod
    def get_file_clips(settings, filepath, chosen_channels=None,
                       offset=0.0, duration=None,
                       outtype=np.float32):
        """
        Loads an audio file from disk, applies filtering (if set up) and then
        chunks the data stream into clips.

        :param settings: An instance of Settings.Audio
        :param filepath: Path to the audio file to read.
        :param chosen_channels: A list of 0-based channel indices (or a single
            index). Only the specified channels will be processed and
            corresponding clips returned. If None, all available channels will
            be processed.
        :param offset: start reading after this time (in seconds)
        :param duration: only load up to this much audio (in seconds)
        :param outtype:
        :return:
            A 2-element tuple:
              A clips container of shape [num channels, num clips, clip length],
              A 1d array containing the starting sample indices of each clip.
        """

        # Fetch data from disk
        data, _ = Audio.load(filepath, settings.fs,
                             offset=offset, duration=duration)

        def to_clips(x):    # local helper function: apply filter and convert to clips
            return Audio.buffer_to_clips(
                (sosfiltfilt(settings.filter_sos, x) if settings.filter_sos is not None else x).astype(outtype),
                settings.clip_length, settings.clip_advance,
                normalize_clips=settings.normalize_clips,
                consider_trailing_clip=settings.consider_trailing_clip,
                return_clip_indices=True)

        if len(data.shape) == 1:        # Single channel (will never happen)
            if chosen_channels is not None:
                logging.getLogger(__name__).warning('parameter \'chosen_channels\' will be ignored')

            clips, clip_start_samples = to_clips(data)
            return clips, (None if clip_start_samples is None
                           else (clip_start_samples + int(np.floor(offset * settings.fs))))

        else:   # Multiple channels
            process_channels = np.arange(data.shape[0]) if chosen_channels is None \
                else np.asarray(chosen_channels if hasattr(chosen_channels, '__len__') else [chosen_channels])

            if any([ch not in range(data.shape[0]) for ch in process_channels]):
                raise ValueError('One or more of chosen channels ({}) not available in audio file {:s}'.format(
                    chosen_channels, repr(filepath)))

            channels_clips = [None] * len(process_channels)
            for ch_idx, ch in enumerate(process_channels):
                channels_clips[ch_idx], clip_start_samples = to_clips(data[ch])

            return np.stack(channels_clips), \
                   (clip_start_samples + int(np.floor(offset * settings.fs)))

    @staticmethod
    def buffer_to_clips(data, clip_len, clip_advance,
                        normalize_clips=True,
                        consider_trailing_clip=False,
                        return_clip_indices=False):

        # If there aren't enough samples, nothing to do
        if data.shape[-1] < clip_len:
            retval = np.zeros((0, clip_len), dtype=data.dtype)
            return (retval, None) if return_clip_indices else retval

        clip_overlap = clip_len - clip_advance  # derived value

        # Split up into chunks by creating a strided array (http://stackoverflow.com/a/5568169).
        # Technique copied from numpy's spectrogram() implementation.
        shape = data.shape[:-1] + ((data.shape[-1] - clip_overlap) // clip_advance, clip_len)
        strides = data.strides[:-1] + (clip_advance * data.strides[-1], data.strides[-1])

        remaining_samples = (data.shape[-1]                                   # total samples
                             - (((shape[0] - 1) * clip_advance) + clip_len))  # - consumed samples

        # If requested, deal with trailing unused samples if there are sufficient enough ...
        if consider_trailing_clip and \
                remaining_samples >= min(clip_len // 2, clip_overlap):  # ... at least half of clip length, or amount of
                                                                        # clip overlap, whichever may be smaller.
            # Take in the last chunk
            sliced_data = np.concatenate([
                np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides, writeable=False),
                np.expand_dims(data[(-clip_len):], 0)], axis=0)
            clip_start_samples = np.concatenate([
                np.arange(0, len(data) - clip_len + 1, clip_advance, dtype=np.int),
                [(len(data) - clip_len + 1)]], axis=0)
        else:
            sliced_data = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides, writeable=False)
            clip_start_samples = np.arange(0, len(data) - clip_len + 1, clip_advance, dtype=np.int)

        # Normalize each clip, if not disabled
        if normalize_clips:
            # Remove DC
            sliced_data = sliced_data - sliced_data.mean(axis=1, keepdims=True)
            # Bring to range [-1.0, 1.0]
            sliced_data = sliced_data / np.maximum(np.abs(sliced_data).max(axis=1, keepdims=True), 1e-24)

        return (sliced_data, clip_start_samples) if return_clip_indices else sliced_data

    @staticmethod
    def __soundfile_load(filepath,
                         offset=None, duration=None, channels=None,
                         dtype=np.float32):

        with sf.SoundFile(filepath) as fd:

            fs = fd.samplerate
            n_channels = fd.channels

            num_samps = -1 if duration is None else int(duration * fs)

            if offset:  # Seek to the requested start sample
                fd.seek(int(offset * fs))

            # Load required number of samples
            data = fd.read(frames=num_samps, dtype=dtype, always_2d=True)

        # keep only the requested channels
        if channels is not None:
            req_chs = Audio.__validate_channels(n_channels, channels, filepath)
            data = data[:, req_chs]

        # transpose to make 'channels' the first dimension
        return data.T, fs

    @staticmethod
    def __audioread_load(filepath,
                         offset=None, duration=None, channels=None,
                         dtype=np.float32):

        with audioread.audio_open(filepath) as fd:
            fs = fd.samplerate

            s_start = int(np.round((offset or 0) * fs))
            s_end = int(np.round(fs * fd.duration)) if duration is None else \
                (s_start + (int(np.round(fs * duration))))

            # Gather samples
            data = np.concatenate([
                    pcm_buf
                    for pcm_buf in Audio.__audioread_samp_gen(
                        fd, s_start, s_end, channels, filepath)
                ], axis=-1)

        return Convert.pcm2float(data, dtype), fs

    @staticmethod
    def __audioread_samp_gen(fh, s_start, s_end, channels, filepath):
        # audioread produces int16 samples
        n_bytes = 2
        fmt_str = '<i2'

        n_channels = fh.channels
        samp_bytes = n_bytes * n_channels

        if channels is not None:
            req_channels = Audio.__validate_channels(
                n_channels, channels, filepath)
            out_n_channels = len(req_channels)

            def pull_chs(arr_2d):
                return arr_2d[:, req_channels]
        else:
            out_n_channels = n_channels

            def pull_chs(arr_2d):
                return arr_2d

        buf_end = 0   # a running pointer to 'end' of buffer
        empty_retval = True
        for buf in fh:
            buf_start = buf_end
            buf_end += int(len(buf) / samp_bytes)

            if buf_end < s_start:
                continue
            if buf_start > s_end:
                break

            ret_samps = np.frombuffer(
                buf[max(0, (s_start - buf_start) * samp_bytes):
                    ((s_end - buf_start) * samp_bytes)],
                fmt_str)

            # transpose to make 'channels' the first dimension
            yield pull_chs(ret_samps.reshape((-1, n_channels))).T

            empty_retval = False    # at least one piece was returned

        if empty_retval:
            yield np.zeros((out_n_channels, 0), dtype=np.int16)

    @staticmethod
    def __validate_channels(num_available, requested_idxs, filepath=None):

        if all([(ch in np.arange(num_available))
                for ch in (
                    requested_idxs if hasattr(requested_idxs, '__len__')
                    else [requested_idxs])
                ]):                # All requested channel(s) do exist in file

            # force retval to be a 1d array
            return np.array(requested_idxs, ndmin=1, copy=False)

        else:
            raise ValueError(
                f'One or more requested channels ({requested_idxs}) unavailable'
                + ('.' if filepath is None else f' in audio file "{filepath}".')
            )


class Convert:

    @staticmethod
    def float2pcm(data, dtype=np.int16):
        """Convert waveform from float to integer. Values outside of [-1.0, 1.0) will get clipped."""

        assert dtype in [np.int16, np.int32]

        if data.dtype == dtype:
            return data

        info = np.iinfo(dtype)
        abs_max = 2 ** (info.bits - 1)
        offset = info.min + abs_max

        return (data * abs_max + offset).clip(info.min, info.max).astype(dtype)

    @staticmethod
    def pcm2float(data, dtype=np.float32):
        """
        Convert PCM data from integer to float values in the range [-1.0, 1.0).
        """

        assert dtype in [np.float32, np.float64]

        if data.dtype == dtype:
            return data

        info = np.iinfo(data.dtype)
        abs_max = 2 ** (info.bits - 1)

        return data.astype(dtype) / abs_max

    @staticmethod
    def audio2spectral(data, fs, spec_settings, eps=None, return_f_axis=False, return_t_axis=False):
        """
        Convert time domain data to time-frequency domain.
        :param data: Either a 1-d or 2-d array. If 2-d, the first dimension is the batch dimension.
        :param fs: Sampling rate of the time-domain data.
        :param spec_settings: The container returned by Settings.Spectral().
        :param eps: If not None, will override the eps value contained in spec_settings.
        :param return_f_axis: Include frequency axis values in the returned value.
        :param return_t_axis: Include frequency axis values in the returned value.
        :return:
            If either return_f_axis or return_t_axis (or both) are enabled, the return will be a tuple, where elements
            following the first one are the respective axis values and the first element is the time-frequency
            representation. If neither of these flags are set, only the time-frequency representation is returned.
        """

        eps_val = eps if eps is not None else spec_settings.eps

        if return_f_axis and return_t_axis:
            package_retval = lambda tfrep, fvals, tvals: (tfrep, fvals, tvals)
        elif return_f_axis and not return_t_axis:
            package_retval = lambda tfrep, fvals, tvals: (tfrep, fvals)
        elif not return_f_axis and return_t_axis:
            package_retval = lambda tfrep, fvals, tvals: (tfrep, tvals)
        else:
            package_retval = lambda tfrep, fvals, tvals: tfrep

        # If inputs are not given in batches, then simply emulate input as a batch with just one item
        if np.ndim(data) == 1:
            data = np.expand_dims(data, 0)

        f, t, tf_rep = spectrogram(data, fs=fs, window=hann(spec_settings.win_len_samples),
                                   nperseg=spec_settings.win_len_samples, noverlap=spec_settings.win_overlap_samples,
                                   nfft=spec_settings.nfft, detrend=False, mode='psd')

        # Find out the indices of where to clip the TF representation
        valid_f_idx_start = f.searchsorted(spec_settings.bandwidth_clip[0], side='left')
        valid_f_idx_end = f.searchsorted(spec_settings.bandwidth_clip[1], side='right') - 1

        if spec_settings.tf_rep_type == 'spec':
            # Clip and return the remainder
            return package_retval(tf_rep[:, valid_f_idx_start:(valid_f_idx_end + 1), :],
                                  f[valid_f_idx_start:(valid_f_idx_end + 1)], t)
        elif spec_settings.tf_rep_type.startswith('spec_db'):
            # Clip and return the remainder
            logspec = 10 * np.log10(eps_val + tf_rep[:, valid_f_idx_start:(valid_f_idx_end + 1), :])
            if spec_settings.tf_rep_type == 'spec_dbfs':
                # Normalize to the range 0-1
                logspec = (logspec - (10 * np.log10(eps_val))) / ((10 * np.log10(1.)) - (10 * np.log10(eps_val)))
            return package_retval(logspec, f[valid_f_idx_start:(valid_f_idx_end + 1)], t)
        elif spec_settings.tf_rep_type == 'spec_pcen':
            # Clip and return the remainder
            return package_retval(Convert.pcen(tf_rep[:, valid_f_idx_start:(valid_f_idx_end + 1), :], eps=eps_val),
                                  f[valid_f_idx_start:(valid_f_idx_end + 1)], t)

        # TODO: add functionality to cache the filterbank after first computation
        mel_filterbank, f2 = Filters.mel_filterbanks2(spec_settings.num_mels,
                                                      np.asarray(spec_settings.bandwidth_clip, dtype=data.dtype),
                                                      f, dtype=tf_rep.dtype)

        # Clip to non-zero range so that unnecessary multiplications can be avoided
        mel_filterbank = mel_filterbank[valid_f_idx_start:(valid_f_idx_end + 1), :]

        # Clip the TF representation and apply the mel_filterbank.
        # Due to the nature of np.dot(), tf_rep needs to be transposed prior, and reverted after
        tf_rep = np.transpose(tf_rep[:, valid_f_idx_start:(valid_f_idx_end + 1), :], [0, 2, 1])
        tf_rep = np.dot(tf_rep, mel_filterbank)
        tf_rep = np.transpose(tf_rep, [0, 2, 1])

        if spec_settings.tf_rep_type == 'melspec':
            return package_retval(tf_rep, f2, t)
        elif spec_settings.tf_rep_type.startswith('melspec_db'):
            logval = 10 * np.log10(eps_val + tf_rep)
            if spec_settings.tf_rep_type == 'melspec_dbfs':
                # Normalize to the range 0-1
                logval = (logval - (10 * np.log10(eps_val))) / ((10 * np.log10(1.)) - (10 * np.log10(eps_val)))
            return package_retval(logval, f2, t)
        elif spec_settings.tf_rep_type == 'melspec_pcen':
            return package_retval(Convert.pcen(tf_rep, eps=eps_val), f2, t)

        # Coding for MFCC in progress...
        # tf_rep = dct(np.log(tf_rep), type=2, axis=1, norm='ortho')
        raise ValueError('Not yet implemented')

    @staticmethod
    def pcen(E, alpha=0.98, delta=2, r=0.5, s=0.025, eps=1e-6):
        M = lfilter([s], [1, s - 1], E)
        smooth = (eps + M) ** (-alpha)
        return (E * smooth + delta) ** r - delta ** r


class Filters:
    """
    Abstract class providing filter arrays for various filtering operations.
    """

    @staticmethod
    def gauss_kernel_1d(sigma):
        """
        A 1-dimensional Gaussian kernel for convolutions.

        :param sigma: Std. dev. of the Gaussian.

        :return: Gaussian curve (normalized) of length ((8 x sigma) + 1)
        """

        n = np.ceil(sigma * 4)  # 3x sigma contains >99% of values in a Gaussian
        kernel = np.exp(-(np.arange(-n, n + 1) ** 2) / (2 * (sigma ** 2)))
        kernel = kernel / np.sum(kernel)  # Normalize

        return kernel

    @staticmethod
    def LoG_kernel_1d(sigma):
        """A 1-dimensional Laplacian of Gaussian kernel for convolutions."""

        kernel = Filters.gauss_kernel_1d(sigma)
        kernel_len = np.int32(len(kernel))
        kernel_half_len = kernel_len // 2

        # Compute the Laplacian of the above Gaussian.
        # The first (sigma ** 2) below is the normalising factor to render the outputs scale-invariant. The rest is the
        # second derivative of the Gaussian.
        kernel = (sigma ** 2) * (-1 / (sigma ** 4)) * kernel * \
                 ((np.arange(-kernel_half_len, kernel_half_len + 1) ** 2) - (sigma ** 2))

        # Sum of the points within one-sigma of mean
        scale_threshold_factor = np.sum(kernel) - (2 * np.sum(kernel[0:np.ceil(2 * sigma).astype(np.int)]))
        # Note: Doing this before removal of DC (below) because it undesirably lowers the threshold for larger sigma.

        # Normalize, in order to set the convolution outputs to be closer to putative blobs' original SNRs
        kernel /= scale_threshold_factor

        # Remove DC
        # kernel -= np.mean(kernel)  # Not really necessary

        return kernel, scale_threshold_factor

    @staticmethod
    def mel_filterbanks(num_banks, freq_extents, nfft, fs, dtype=np.float32):
        """
        This method is based on the examples I found on the internet. The hard snapping of bank points (start, middle
        and end) to frequency bins results in weird outputs. These include banks with no matching bins and also, weird
        alignment of banks at the extremities of 'freq_extents'. These issues can be avoided if the "hard snapping to
        frequency bins"is avoided. I've written another equivalent function _get_mel_filterbanks2() for this purpose.
        """

        assert np.asarray(freq_extents).shape[0] == 2

        # convert Hz to mel
        freq_extents_mel = 2595 * np.log10(1 + np.asarray(freq_extents) / 700.)

        # compute points evenly spaced in mels
        melpoints = np.linspace(freq_extents_mel[0], freq_extents_mel[1], num_banks + 2)

        # convert mels to Hz and then on to fft bin idx
        bin_idxs = np.floor((nfft + 1) * (700 * (10 ** (melpoints / 2595.0) - 1)) / fs).astype(np.int)

        filterbank = np.zeros([nfft // 2 + 1, num_banks], dtype=dtype)
        for bank_idx in range(0, num_banks):
            for i in range(int(bin_idxs[bank_idx]), int(bin_idxs[bank_idx + 1])):
                filterbank[i, bank_idx] = (i - bin_idxs[bank_idx]) / (bin_idxs[bank_idx + 1] - bin_idxs[bank_idx])
            for i in range(int(bin_idxs[bank_idx + 1]), int(bin_idxs[bank_idx + 2])):
                filterbank[i, bank_idx] = (bin_idxs[bank_idx + 2] - i) / (
                        bin_idxs[bank_idx + 2] - bin_idxs[bank_idx + 1])

        return filterbank, (fs/(nfft+1)) * bin_idxs[1:-1]

    @staticmethod
    def mel_filterbanks2(num_banks, freq_extents, f_vec, dtype=np.float32):
        """
        An arguably better version of _get_mel_filterbanks() (see above) that avoids issues with "hard snapping". Works
        with an existing vector of frequency bins, as returned from signal.spectrogram(), instead of recalculating them
        and flooring down the bin indices.
        """

        assert np.asarray(freq_extents).shape[0] == 2

        # convert Hz to mel
        freq_extents_mel = 2595 * np.log10(1 + np.asarray(freq_extents, dtype=dtype) / 700.)

        # compute points evenly spaced in mels
        melpoints = np.linspace(freq_extents_mel[0], freq_extents_mel[1], num_banks + 2, dtype=dtype)

        # convert mels to Hz
        banks_ends = (700 * (10 ** (melpoints / 2595.0) - 1))

        filterbank = np.zeros([len(f_vec), num_banks], dtype=dtype)
        for bank_idx in range(1, num_banks+1):
            # Points in the first half of the triangle
            mask = np.logical_and(f_vec >= banks_ends[bank_idx - 1], f_vec <= banks_ends[bank_idx])
            filterbank[mask, bank_idx-1] = (f_vec[mask] - banks_ends[bank_idx - 1]) / \
                (banks_ends[bank_idx] - banks_ends[bank_idx - 1])

            # Points in the second half of the triangle
            mask = np.logical_and(f_vec >= banks_ends[bank_idx], f_vec <= banks_ends[bank_idx+1])
            filterbank[mask, bank_idx-1] = (banks_ends[bank_idx + 1] - f_vec[mask]) / \
                (banks_ends[bank_idx + 1] - banks_ends[bank_idx])

        # Scale and normalize
        temp = filterbank.sum(axis=0)
        non_zero_mask = temp > 0
        filterbank[:, non_zero_mask] /= np.expand_dims(temp[non_zero_mask], 0)

        # print([banks_ends[0], banks_ends[-1]])
        return filterbank, banks_ends[1:-1]

    @staticmethod
    def butterworth_filter(filterspec, fs):

        filter_order, filter_critical_freq, filter_type_str = filterspec

        # Build a filter of the desired type
        wn = np.array(filter_critical_freq) / (fs / 2)  # convert to angular frequency

        filter_sos = butter(filter_order, wn, btype=filter_type_str, output='sos')

        return filter_sos


class Process:
    """Abstract class providing algorithmic operations, combining some of the lower-level functionalities provided by
    the sibling classes Audio and Convert."""

    @staticmethod
    def audio2clips(audio_settings, src_file, dst_file,
                    num_classes, annots_times, annots_class_idxs, annots_channels,
                    min_annot_overlap_fraction=1.0,
                    negative_class_idx=None, max_nonmatch_overlap_fraction=0.0,
                    keep_only_centralized_selections=False,
                    attempt_salvage=False):
        """
        Apply pre-processing (resampling and filtering, as defined) to contents of an audio file, break up the resulting
        time-domain data into fixed-length clips and write the clips to disk.
        :param audio_settings: An instance of Settings.Audio.
        :param src_file: Path to the source audio file.
        :param dst_file: Path to the target '.npz' file.
        :param num_classes: Number of classes in the target dataset. Used to create "one-hot" style ground-truth arrays
            for each clip produced.
        :param annots_times: If not None, must be a numpy array (shape Nx2) of start-end pairs defining annotations'
            temporal extents within the source file. If None, then 'annot_class_idxs' must be a single index, and all
            clips generated from the 'src_file' will be associated with the class corresponding to the single index.
        :param annots_class_idxs: When 'annots_times' is not None, this must be an N-length list of zero-based indices
            to the class corresponding to each annotation. When 'annot_times' is None, this must be a single idx value.
        :param annots_channels: When 'annots_times' is not None, this must be an N-length list of zero-based indices
            to the channels corresponding to each annotation. When 'annot_times' is None, this parameter is ignored.
        :param min_annot_overlap_fraction: Lower threshold on how much overlap a
            clip must have with an annotation.
        :param negative_class_idx: If not None, clips that do not have enough overlap with any annotation will be saved
            as clips of the non-target class whose index this parameter specifies.
        :param max_nonmatch_overlap_fraction: A clip without enough overlap with
            any annotations will be saved (as a negative class clip) only if its
            overlap with all annotations is less than this amount. This
            parameter is only used if negative_class_idx is not None.
        :param keep_only_centralized_selections: (Optional) For very short annotations, consider only those overlapping
            clips that have the annotation occurring within the central 50% extents of the clip.
        :param attempt_salvage: (Optional) When enabled, if an annotation didn't have any matching clip due to the
            automatic way of producing clips, attempt will be made to salvage a match by "forming" clips by expanding
            outwards from the mid-epoch of the annotation.

        :return: A 2-tuple. First value is the number of clips written. Second value is a list of the number of clips
            written per class.
        """

        if annots_times is not None:
            assert annots_times.shape[0] == len(annots_class_idxs)
            assert annots_times.shape[0] == len(annots_channels)
            assert (0.0 < min_annot_overlap_fraction <= 1.0)
            assert negative_class_idx is None or \
                   (0.0 <= max_nonmatch_overlap_fraction <
                    min_annot_overlap_fraction)
        else:
            assert (not hasattr(annots_class_idxs, '__len__'))

        # Fetch data from disk
        data, _ = Audio.load(src_file, audio_settings.fs)

        all_clips = []
        all_clips_offsets = []
        all_clips_channels = []
        all_ground_truth = []
        unmatched_annots_mask = None if annots_times is None \
            else np.full((len(annots_class_idxs), ), False, dtype=np.bool)
        for ch_idx, ch_data in enumerate(data):

            # Apply filter
            if audio_settings.filter_sos is not None:
                ch_data = sosfiltfilt(audio_settings.filter_sos, ch_data).astype(np.float32)

            # Break up into clips
            clips, clip_offsets = Audio.buffer_to_clips(
                ch_data, audio_settings.clip_length, audio_settings.clip_advance,
                normalize_clips=audio_settings.normalize_clips,
                consider_trailing_clip=audio_settings.consider_trailing_clip,
                return_clip_indices=True)

            num_clips, num_samps = clips.shape

            if annots_times is not None:
                # Annotation extents are available, only need to save relevant clips.

                curr_ch_annots_mask = (annots_channels == ch_idx)
                curr_ch_annots_class_idxs = annots_class_idxs[curr_ch_annots_mask]

                curr_ch_annots_offsets = np.round(
                    annots_times[curr_ch_annots_mask, :] * audio_settings.fs
                ).astype(clip_offsets.dtype)

                ch_clip_class_coverage, ch_matched_annots_mask = \
                    assess_annotations_and_clips_match(
                        clip_offsets, num_samps, num_classes,
                        curr_ch_annots_offsets, curr_ch_annots_class_idxs,
                        min_annot_overlap_fraction,
                        keep_only_centralized_selections,
                        negative_class_idx, max_nonmatch_overlap_fraction)

                # Clips having satisfactory coverage with at least one annot
                keep_clips_mask = np.any(
                    ch_clip_class_coverage >= min_annot_overlap_fraction,
                    axis=1)
                # Add clips and info to collection
                if np.any(keep_clips_mask):
                    all_clips.append(clips[keep_clips_mask, :])
                    all_clips_offsets.append(clip_offsets[keep_clips_mask])
                    all_ground_truth.append(
                        ch_clip_class_coverage[keep_clips_mask, :].astype(
                            np.float16))
                    all_clips_channels.append(
                        np.full((keep_clips_mask.sum(), ), ch_idx,
                                dtype=np.uint8))

                # If requested, attempt to salvage any unmatched annotations
                for_salvage_annot_idxs = np.where(
                    np.logical_not(ch_matched_annots_mask))[0]

                if attempt_salvage and len(for_salvage_annot_idxs) > 0:

                    salvaged_clips, salvaged_clip_offsets = \
                        Process._salvage_clips(
                            ch_data, audio_settings, num_samps,
                            curr_ch_annots_offsets[for_salvage_annot_idxs, :])

                    if salvaged_clips.shape[0] > 0:
                        salvaged_clip_class_coverage, s_matched_annots_mask = \
                            assess_annotations_and_clips_match(
                                salvaged_clip_offsets, num_samps, num_classes,
                                curr_ch_annots_offsets,
                                curr_ch_annots_class_idxs,
                                min_annot_overlap_fraction,
                                keep_only_centralized_selections, None)

                        # Clips having satisfactory coverage with >= 1 annot
                        keep_clips_mask = np.any(
                            salvaged_clip_class_coverage >=
                            min_annot_overlap_fraction, axis=1)
                        # Add clips and info to collection
                        if np.any(keep_clips_mask):
                            all_clips.append(salvaged_clips[keep_clips_mask, :])
                            all_clips_offsets.append(
                                salvaged_clip_offsets[keep_clips_mask])
                            all_ground_truth.append(
                                salvaged_clip_class_coverage[
                                    keep_clips_mask, :].astype(np.float16))
                            all_clips_channels.append(
                                np.full((keep_clips_mask.sum(), ), ch_idx,
                                        dtype=np.uint8))

                        # Update curr channel annots mask
                        ch_matched_annots_mask[for_salvage_annot_idxs] = \
                            s_matched_annots_mask[for_salvage_annot_idxs]

                # Update overall mask
                unmatched_annots_mask[
                    np.where(curr_ch_annots_mask)[0][
                        np.logical_not(ch_matched_annots_mask)]] = True

            else:
                curr_ground_truth = np.zeros((num_clips, num_classes), dtype=np.float16)
                # Mark every clip as belonging to the single specified class
                curr_ground_truth[:, annots_class_idxs] = 1.0

                all_clips.append(clips)
                all_clips_offsets.append(clip_offsets)
                all_ground_truth.append(curr_ground_truth)
                all_clips_channels.append(np.full((clips.shape[0], ), ch_idx, dtype=np.uint8))

        # Offer a warning about unmatched annotations, if any
        if unmatched_annots_mask is not None and np.any(unmatched_annots_mask):
            logging.getLogger(__name__).warning('{:s}: {:d} annotations unmatched [{:s}]'.format(
                src_file, sum(unmatched_annots_mask),
                ', '.join([
                    '{:f} - {:f} (ch-{:d})'.format(
                        annots_times[annot_idx, 0], annots_times[annot_idx, 1], annots_channels[annot_idx] + 1)
                    for annot_idx in np.where(unmatched_annots_mask)[0]
                ])))

        if len(all_clips) > 0:
            all_ground_truth = Process._adjust_clip_annot_coverage(
                np.concatenate(all_ground_truth),
                min_annot_overlap_fraction)

            # Save the clips & infos
            np.savez_compressed(
                dst_file,
                fs=audio_settings.fs,
                labels=all_ground_truth,
                channels=np.concatenate(all_clips_channels),
                clip_offsets=np.concatenate(all_clips_offsets),
                clips=Convert.float2pcm(    # Convert to 16-bit PCM
                    np.concatenate(all_clips), dtype=np.int16))

            return \
                all_ground_truth.shape[0], \
                np.sum(all_ground_truth == 1.0, axis=0)
        else:
            return 0, np.zeros((num_classes, ), dtype=np.int)

    @staticmethod
    def _salvage_clips(data, audio_settings, clip_len,
                       unmatched_annots_offsets):
        """Internal function used by Process.audio2clips()"""

        salvaged_clips = []
        salvaged_clip_offsets = []
        half_len = clip_len // 2

        # Gather clips corresponding to all yet-unmatched annots
        for annot_idx in range(unmatched_annots_offsets.shape[0]):
            annot_num_samps = (unmatched_annots_offsets[annot_idx, 1] -
                               unmatched_annots_offsets[annot_idx, 0]) + 1

            if annot_num_samps < clip_len:
                # If annotation is shorter than clip size, then we need to
                # center the annotation within a clip
                annot_start_samp = unmatched_annots_offsets[annot_idx, 0] + \
                                   (annot_num_samps // 2) - half_len
                annot_end_samp = annot_start_samp + clip_len - 1
            else:
                # otherwise, take full annotation extents
                annot_start_samp, annot_end_samp = \
                    unmatched_annots_offsets[annot_idx, :]

            short_clips, short_clip_offsets = Audio.buffer_to_clips(
                data[max(0, annot_start_samp):min(annot_end_samp + 1,
                                                  len(data))],
                audio_settings.clip_length, audio_settings.clip_advance,
                normalize_clips=audio_settings.normalize_clips,
                consider_trailing_clip=audio_settings.consider_trailing_clip,
                return_clip_indices=True)

            if short_clips.shape[0] > 0:
                salvaged_clips.append(short_clips)
                salvaged_clip_offsets.append(short_clip_offsets +
                                             annot_start_samp)

        if len(salvaged_clips) > 0:
            return np.concatenate(salvaged_clips, axis=0), \
                np.concatenate(salvaged_clip_offsets, axis=0)

        else:
            # Nothing could be salvaged, return empty containers
            return np.zeros((0, clip_len), dtype=data.dtype), \
                np.zeros((0,), dtype=np.int)

    @staticmethod
    def _adjust_clip_annot_coverage(coverage, upper_thld, lower_thld_frac=1/3):
        # Adjust "coverage":
        #  force values >= upper_thld to 1.0
        #  retain remaining values >= (lower_thld_frac * upper_thld) as is
        #  force all other small values to 0.0
        return np.select(
            [coverage >= upper_thld,
             coverage >= upper_thld * lower_thld_frac],
            [1.0, coverage],
            default=0.0
        )
