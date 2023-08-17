
import numpy as np
from scipy.signal import butter, sosfiltfilt, spectrogram, lfilter
from scipy.signal.windows import hann
import soundfile as sf
import audioread
import resampy
import logging


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
                # nextpow2
                self.nfft = \
                    int(2 ** np.ceil(np.log2(self.win_len_samples)).astype(int))
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
    def get_file_clips(filepath, settings, channels=None,
                       offset=0.0, duration=None,
                       dtype=np.float32,
                       labels_accumulator=None,
                       **kwargs):
        """
        Loads an audio file from disk, applies filtering (if set up) and then
        chunks the data stream into clips.

        :param filepath: Path to the audio file to read.
        :param settings: An instance of Settings.Audio.
        :param channels: If None (default), all available channels will be
          processed. Otherwise, must be a list of 0-based channel indices (or
          a single index) and only the specified channels (if available) will be
          processed. Any specified channel(s) missing in the audio file will be
          ignored.
        :param offset: If not None, will start reading the file's contents from
          this point in time (in seconds) and forward.
        :param duration: If not None, only loads audio corresponding to the
          specified value (in seconds).
        :param dtype: Output type (default: float32).
        :param labels_accumulator: If not None, must be an instance of inherited
          class of :class:`koogu.data.preprocess.GroundTruthDataAggregator`
          which accumulates labels (or ground truth info), and writes it out to
          disk storage along with clips and other info.
        :return:
          If ``labels_accumulator`` is specified, the return value of
          :func:`serialize()` will be forwarded as is. Otherwise, the returned
          value will be a tuple containing -
            - A list (length = num channels) of numpy arrays each having shape
              [num clips, clip length] of extracted clips in the respective
              channels;
            - A 1d array containing the starting sample indices of each clip;
            - Loaded file's duration;
            - A 1d array of indices to the channels that were successfully
              loaded (matches the order of channels in the clips container).
            When any error occurs, the second item in the tuple will be None.
        """

        ret_clips = None

        _, file_dur, n_channels = Audio.get_info(filepath)

        loggr = logging.getLogger(__name__)

        # Are any of requested channel(s) available?
        if channels is not None:    # there was an explicit request
            return_channels, invalid_ch_mask = \
                Audio.__validate_channels(n_channels, channels)

            req_channels = return_channels   # For passing to Audio.load()
            n_channels = len(return_channels)

            # Log out a warning about missing channels, if any
            if np.any(invalid_ch_mask):
                if n_channels == 0:
                    msg = f'None of requested channels found in {filepath}'
                    ret_clips = []  # so we return immediately (see below)
                else:
                    msg = Audio.__missing_channels_msg(
                        channels, invalid_ch_mask, filepath)
                loggr.warning(msg)

        else:
            req_channels = None   # For passing to Audio.load()
            return_channels = np.arange(n_channels)

        # Is file too long or too short
        within_extents = (
            (settings.clip_length / settings.fs) <= file_dur <=
            kwargs.get('max_file_duration', np.inf))
        if not within_extents:
            loggr.warning(f'{filepath}: duration = {file_dur} s. Ignoring.')

            ret_clips = n_channels * [
                np.zeros((0, settings.clip_length), dtype=dtype)]

        if ret_clips is not None:
            # At least one failure. Return immediately
            if labels_accumulator is not None:
                return labels_accumulator.serialize()
            else:
                return ret_clips, None, file_dur, return_channels

        # Fetch data from disk
        data, _ = Audio.load(filepath, settings.fs,
                             offset=offset, duration=duration, dtype=dtype,
                             channels=req_channels)

        ret_clips = [None] * n_channels
        clip_start_samples = None   # placeholder
        for ch_idx in range(n_channels):

            if settings.filter_sos is not None:
                data[ch_idx, :] = sosfiltfilt(settings.filter_sos, data[ch_idx])

            clips, clip_start_samples = \
                Audio.buffer_to_clips(
                    data[ch_idx].astype(dtype),
                    settings.clip_length, settings.clip_advance,
                    consider_trailing_clip=settings.consider_trailing_clip)

            # clips returned above are not normalized. If invoked from
            # data.preprocess normalization will be applied later when
            # labels_accumulator serializes accrued data. Otherwise (invoked
            # from inference.recognize), normalize right away.

            # Accumulate the per-channel elements
            if labels_accumulator is not None:
                labels_accumulator.accrue(
                    ch_idx,
                    clips,
                    clip_start_samples + int(np.round(offset * settings.fs)),
                    data[ch_idx])
            else:
                ret_clips[ch_idx] = clips if not settings.normalize_clips else \
                    Audio.normalize(clips)

        if labels_accumulator is not None:
            return labels_accumulator.serialize(
                normalize_clips=settings.normalize_clips)
        else:
            return (
                ret_clips,
                clip_start_samples + int(np.round(offset * settings.fs)),
                file_dur,
                return_channels)

    @staticmethod
    def buffer_to_clips(data, clip_len, clip_advance,
                        consider_trailing_clip=False):

        # If there aren't enough samples, nothing to do
        if data.shape[-1] < clip_len:
            retval = np.zeros((0, clip_len), dtype=data.dtype)
            return retval, None

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
                np.arange(0, len(data) - clip_len + 1, clip_advance, dtype=int),
                [(len(data) - clip_len + 1)]], axis=0)
        else:
            sliced_data = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides, writeable=False)
            clip_start_samples = \
                np.arange(0, len(data) - clip_len + 1, clip_advance, dtype=int)

        return sliced_data, clip_start_samples

    @staticmethod
    def normalize(clips):
        # Remove DC
        clips = clips - clips.mean(axis=1, keepdims=True)

        # Bring to range [-1.0, 1.0]
        clips = clips / np.maximum(
            np.abs(clips).max(axis=1, keepdims=True), 1e-24)
        # considering a small quantity to avoid divide-by-zero

        return clips

    @staticmethod
    def __soundfile_load(filepath,
                         offset=None, duration=None, channels=None,
                         dtype=np.float32):

        req_chs = None

        with sf.SoundFile(filepath) as fd:

            fs = fd.samplerate
            n_channels = fd.channels

            if channels is not None:    # Check validity of requested channels
                req_chs, invalid_ch_mask = \
                    Audio.__validate_channels(n_channels, channels)

                if np.any(invalid_ch_mask):   # Puke!
                    raise ValueError(Audio.__missing_channels_msg(
                        channels, invalid_ch_mask, filepath))

            num_samps = -1 if duration is None else int(np.round(duration * fs))

            if offset:  # Seek to the requested start sample
                fd.seek(int(np.round(offset * fs)))

            # Load required number of samples
            data = fd.read(frames=num_samps, dtype=dtype, always_2d=True)

        # keep only the requested channels
        if req_chs is not None:
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
            req_channels, invalid_ch_mask = \
                Audio.__validate_channels(n_channels, channels)

            if np.any(invalid_ch_mask):   # Puke!
                raise ValueError(Audio.__missing_channels_msg(
                    channels, invalid_ch_mask, filepath))

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
    def __validate_channels(num_available, requested_idxs):
        """
        Will return a numpy array of available channels and a mask of
        unavailable channels.

        See companion function __missing_channels_msg() for generating
        appropriate error/warning message.
        """

        # force (retval) to be a 1d array
        req_chs = np.array(requested_idxs, ndmin=1, copy=False)

        valid_mask = req_chs < num_available

        return req_chs[valid_mask], np.logical_not(valid_mask)

    @staticmethod
    def __missing_channels_msg(requested_idxs, invalid_mask, filepath):

        return 'Channel(s) ({}) unavailable in audio file "{}".'.format(
            [ch for ch, m in zip(requested_idxs, invalid_mask) if m], filepath)


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
        scale_threshold_factor = np.sum(kernel) - (
                2 * np.sum(kernel[0:np.ceil(2 * sigma).astype(int)]))
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
        bin_idxs = np.floor((nfft + 1) *
                            (700 * (10 ** (melpoints / 2595.0) - 1)) / fs
                            ).astype(int)

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
