
import numpy as np
from scipy.signal import butter, sosfiltfilt, spectrogram, lfilter
from scipy.signal.windows import hann
import librosa
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
                     tf_rep_type='spec',
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
    def load(in_filepath, desired_fs,
             downmix_channels=True, offset=0.0, duration=None, dtype=np.float32):

        # Load audio from file
        data, fs = librosa.load(in_filepath, sr=desired_fs, mono=downmix_channels, offset=offset, duration=duration)

        assert fs == desired_fs, "load_audio_from_file() got {:f} Hz, expected {:f} Hz".format(fs, desired_fs)

        return data.astype(dtype), fs

    @staticmethod
    def get_file_clips(settings, filepath, downmix_channels=True, chosen_channels=None,
                       offset=0.0, duration=None,
                       return_clip_indices=False, outtype=np.float32):
        """
        Loads an audio file from disk, applies filtering (if set up) and then chunks the data stream into clips.

        :param settings: An instance of Settings.Audio
        :param filepath:
        :param downmix_channels: If True, multi-channel audio files will be downmixed to result in a single channel. If
            set to True, 'chosen_channels' will have no effect.
        :param chosen_channels: A list of 0-based channel indices (or a single index. Only the specified channels will
            be processed and corresponding clips returned. If None, all available channels will be processed. Has no
            effect when 'downmix_channels' is True.
        :param offset: start reading after this time (in seconds)
        :param duration: only load up to this much audio (in seconds)
        :param return_clip_indices:
        :param outtype:
        :return:
            If downmix_channels is True or if the audio file itself has only one channel, the returned clips container
            (a numpy array) will be of shape [num clips, clip length]. Otherwise, it will be of shape
            [num channels, num clips, clip length].
            If return_clip_indices is True, the starting sample indices of each clip will also be returned. This will
            be a 1-dimensional array regardless of how many channels are processed.
        """

        # Fetch data from disk
        data, _ = Audio.load(filepath, settings.fs,
                             downmix_channels=downmix_channels, offset=offset, duration=duration)

        # Presently, librosa doesn't provide an interface to query the number of channels without loading the file.
        # So the checks for validity of chosen_channels is deferred until after the file is loaded.

        def to_clips(x):    # local helper function: apply filter and convert to clips
            return Audio.buffer_to_clips(
                (sosfiltfilt(settings.filter_sos, x) if settings.filter_sos is not None else x).astype(outtype),
                settings.clip_length, settings.clip_advance,
                normalize_clips=settings.normalize_clips,
                consider_trailing_clip=settings.consider_trailing_clip,
                return_clip_indices=return_clip_indices)

        if downmix_channels or len(data.shape) == 1:        # Single channel
            if chosen_channels is not None:
                logging.getLogger(__name__).warning('parameter \'chosen_channels\' will be ignored')

            if return_clip_indices:
                clips, clip_start_samples = to_clips(data)
                return clips, (None if clip_start_samples is None
                               else (clip_start_samples + int(np.floor(offset * settings.fs))))
            else:
                return to_clips(data)

        else:   # Multiple channels
            process_channels = np.arange(data.shape[0]) if chosen_channels is None \
                else np.asarray(chosen_channels if hasattr(chosen_channels, '__len__') else [chosen_channels])

            if any([ch not in range(data.shape[0]) for ch in process_channels]):
                raise ValueError('One or more of chosen channels ({}) not available in audio file {:s}'.format(
                    chosen_channels, repr(filepath)))

            if return_clip_indices:     # Need to handle collecting two return values from buffer_to_clips()
                channels_clips = [None] * len(process_channels)
                for ch_idx, ch in enumerate(process_channels):
                    channels_clips[ch_idx], clip_start_samples = to_clips(data[ch])

                return np.stack(channels_clips), (clip_start_samples + int(np.floor(offset * settings.fs)))
            else:
                return np.stack([to_clips(data[ch]) for ch in process_channels])

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
            sliced_data = sliced_data - sliced_data.mean(axis=1, keepdims=True)     # Remove DC
            sliced_data = sliced_data / np.abs(sliced_data).max(axis=1, keepdims=True)  # Bring to range [-1.0, 1.0]

        return (sliced_data, clip_start_samples) if return_clip_indices else sliced_data


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
        """Convert PCM data from integer to float values in the rance [-1.0, 1.0)."""

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
        :param spec_settings: The container returned by data.process_spec_settings().
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
    """Abstract class providing filter arrays for various filtering operations."""

    @staticmethod
    def gauss_kernel_1d(sigma):
        """A 1-dimensional Gaussian kernel for convolutions."""

        n = np.ceil(sigma * 3)  # 3 x sigma contains >99% of values in a Gaussian
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
                    selections=None, min_selection_overlap_fraction=1.0,
                    keep_only_centralized_selections=False, attempt_salvage=False,
                    non_matching_clips_file=None, max_nonmatch_overlap_fraction=0.0):
        """
        Apply pre-processing (resampling and filtering, as defined) to contents of an audio file, break up the resulting
        time-domain data into fixed-length clips and write the clips to disk.
        :param audio_settings: An instance of Settings.Audio.
        :param src_file: Path to the source audio file.
        :param dst_file: Path to the target '.npz' file.
        :param selections: If not None, must be a numpy array (shape Nx2) of start-end pairs defining annotations'
            temporal extents within the source file. If None, the below parameters will have no effect.
        :param min_selection_overlap_fraction: Lower threshold on how much overlap a clip must have with an annotation.
        :param keep_only_centralized_selections: (Optional) For very short annotations, consider only those overlapping
            clips that have the annotation occurring within the central 50% extents of the clip.
        :param attempt_salvage: (Optional) When enabled, if an annotation didn't have any matching clip due to the
            automatic way of producing clips, attempt will be made to salvage a match by "forming" clips by expanding
            outwards from the mid-epoch of the annotation.
        :param non_matching_clips_file: If not None, clips that did not have enough overlap with any annotation will be
            written to this file. Also see 'max_nonmatch_overlap_fraction'.
        :param max_nonmatch_overlap_fraction: A clip without enough overlap will be saved only if it's overlap with any
            annotation is less than this amount.
        :return: A 2-tuple. First value is the number of clips written (depends on whether 'selections' was specified).
            Second value is the number of non-matching clips written (if enabled, otherwise None).
        """

        if selections is not None:
            assert (0.0 < min_selection_overlap_fraction <= 1.0)
            assert non_matching_clips_file is None or (0.0 <= max_nonmatch_overlap_fraction < 1.0)

        clip_dur = float(audio_settings.clip_length) / float(audio_settings.fs)

        # Fetch data from disk and apply filter
        data, _ = Audio.load(src_file, audio_settings.fs)
        if audio_settings.filter_sos is not None:
            data = sosfiltfilt(audio_settings.filter_sos, data).astype(np.float32)

        # Break up into clips
        clips, clip_offsets = Audio.buffer_to_clips(data, audio_settings.clip_length, audio_settings.clip_advance,
                                                    normalize_clips=audio_settings.normalize_clips,
                                                    consider_trailing_clip=audio_settings.consider_trailing_clip,
                                                    return_clip_indices=True)

        num_non_matching_clips = None
        if selections is not None:
            # Annotation extents are available, only need to save relevant clips.

            temp = clip_offsets.astype(np.float32) / audio_settings.fs
            clip_times = np.concatenate([[temp], [temp + clip_dur]], axis=0).T  # start & end pairs

            annot_dur = selections[:, 1] - selections[:, 0]

            # The minimum of annotation durations or clip length, for each clip
            whole_durations = np.minimum(annot_dur, clip_dur)

            positive_overlap_dur_thlds = min_selection_overlap_fraction * whole_durations

            # Mask of overlapping clips having enough overlap amount. A clip must have the minimum required overlap
            # with at least one annotation.
            overlaps_mask = np.stack([(np.minimum(selections[:, 1], clip_times[idx, 1]) -
                                       np.maximum(selections[:, 0], clip_times[idx, 0])) >= positive_overlap_dur_thlds
                                      for idx in range(len(clip_offsets))])
            # overlaps_mask is a matrix with each row corresponding to a clip and each column corresponding to an
            # annotation.

            if keep_only_centralized_selections:
                # For very short annotations, turn off 'match' mask if they don't occur within the central 50% of a clip
                for annot_idx in [a_idx for a_idx, a_dur in enumerate(annot_dur)
                                  if a_dur < (clip_dur * 0.5) and np.any(overlaps_mask[:, a_idx])]:
                    ov_clip_idxs = np.asarray(np.where(overlaps_mask[:, annot_idx])).ravel()
                    overlaps_mask[ov_clip_idxs, annot_idx] = [
                        (clip_times[ov_clip_idx, 0] + (clip_dur * 0.25) <= selections[annot_idx, 0] and
                         clip_times[ov_clip_idx, 1] - (clip_dur * 0.25) >= selections[annot_idx, 1])
                        for ov_clip_idx in ov_clip_idxs]

            # Clips that were matched with one or more annotations
            positive_clips_mask = np.any(overlaps_mask, axis=1)  # at least one True in each row

            unmatched_annots_mask = np.logical_not(np.any(overlaps_mask, axis=0))  # not even one True in each column

            if attempt_salvage and np.any(unmatched_annots_mask):
                # Handle any unmatched annotation(s)
                salvaged_clips, salvaged_clip_offsets, unsalvaged_annots_mask = \
                    Process._salvage_clips(data, audio_settings, clip_dur,
                                           selections[unmatched_annots_mask, :], annot_dur[unmatched_annots_mask])

                unmatched_annots_mask[unmatched_annots_mask] = unsalvaged_annots_mask   # Update mask

            else:
                # Set to appropriately-shaped empty containers
                salvaged_clips = np.zeros((0, clips.shape[1]), dtype=clips.dtype)
                salvaged_clip_offsets = np.zeros((0,), dtype=clip_offsets.dtype)

            if np.any(unmatched_annots_mask):   # Offer a warning about unmatched annotations
                annot_idxs = np.asarray(np.where(unmatched_annots_mask)).ravel()
                logging.getLogger(__name__).warning('{:s}: {:d} annotations unmatched [{:s}]'.format(
                    src_file, len(annot_idxs),
                    ', '.join(['{:f} - {:f}'.format(selections[annot_idx, 0], selections[annot_idx, 1])
                               for annot_idx in annot_idxs])))

            if non_matching_clips_file:
                # Saving of non-matching clips was requested

                negative_overlap_dur_thlds = max_nonmatch_overlap_fraction * whole_durations

                # Mask of non-overlapping clips and overlapping clips with small overlap amount. A clip must have no
                # (or only up to a very small amount of) overlap with any annotation.
                negative_mask = np.asarray([
                    np.all((np.minimum(selections[:, 1], clip_times[idx, 1]) -
                            np.maximum(selections[:, 0], clip_times[idx, 0])) <= negative_overlap_dur_thlds)
                    for idx in range(len(clip_offsets))])

                num_non_matching_clips = negative_mask.sum()

                # Save the negative case clips
                if num_non_matching_clips > 0:
                    storage_vals = {
                        'fs': audio_settings.fs,
                        'clip_offsets': clip_offsets[negative_mask],
                        'clips': Convert.float2pcm(clips[negative_mask, ...], dtype=np.int16)}  # Convert to 16-bit PCM
                    np.savez_compressed(non_matching_clips_file, **storage_vals)

            # Now apply the positive matches mask. The positive mask'ed clips will be saved outside of this block.
            # Also merge in the salvaged clips, if any.
            clips = np.concatenate([clips[positive_clips_mask, ...], salvaged_clips], axis=0)
            clip_offsets = np.concatenate([clip_offsets[positive_clips_mask], salvaged_clip_offsets], axis=0)

        # Save the positive case clips
        if len(clip_offsets) > 0:
            storage_vals = {
                'fs': audio_settings.fs,
                'clip_offsets': clip_offsets,
                'clips': Convert.float2pcm(clips, dtype=np.int16)}  # Convert to 16-bit PCM
            np.savez_compressed(dst_file, **storage_vals)

        return len(clip_offsets), num_non_matching_clips

    @staticmethod
    def clips2tfrecords():
        pass    # TODO: yet to implement

    @staticmethod
    def _salvage_clips(data, audio_settings, clip_dur, annot_times, annot_dur):
        """Internal function used by Process.audio2clips()"""

        salvaged_clips = []
        salvaged_clip_offsets = []
        unsalvaged_mask = np.full((annot_times.shape[0],), True)

        for annot_idx in range(annot_times.shape[0]):

            if annot_dur[annot_idx] <= clip_dur:
                # If annotation is shorter than clip size, then we need to center the annotation within a clip
                annot_start_samp = int(np.floor(
                    (annot_times[annot_idx, 0] + ((annot_dur[annot_idx] - clip_dur) / 2)) * audio_settings.fs))
                annot_end_samp = int(np.ceil(
                    (annot_times[annot_idx, 1] - ((annot_dur[annot_idx] - clip_dur) / 2)) * audio_settings.fs))
            else:
                # otherwise, take full annotation extents
                annot_start_samp = int(np.floor(annot_times[annot_idx, 0] * audio_settings.fs))
                annot_end_samp = int(np.ceil(annot_times[annot_idx, 1] * audio_settings.fs))

            short_clips, short_clip_offsets = Audio.buffer_to_clips(
                data[max(0, annot_start_samp):min(annot_end_samp, len(data)) + 1],
                audio_settings.clip_length, audio_settings.clip_advance,
                normalize_clips=audio_settings.normalize_clips,
                consider_trailing_clip=audio_settings.consider_trailing_clip,
                return_clip_indices=True)

            if short_clips.shape[0] > 0:
                salvaged_clips.append(short_clips)
                salvaged_clip_offsets.append(short_clip_offsets + annot_start_samp)
                unsalvaged_mask[annot_idx] = False  # mark as "salvaged"

        if len(salvaged_clips) > 0:
            return np.concatenate(salvaged_clips, axis=0), \
                   np.concatenate(salvaged_clip_offsets, axis=0), \
                   unsalvaged_mask
        else:
            # Nothing could be salvaged, return empty containers
            return np.zeros((0, audio_settings.clip_length), dtype=data.dtype), \
                   np.zeros((0,), dtype=np.int), \
                   unsalvaged_mask
