Prepare training inputs
=======================

Prepared data (clips + associated label/class information) are written to the
filesystem for later consumption during model training. In addition to extracting
clips from raw audio, the below interfaces also support the following audio
pre-processing operations -

* standardizing the sampling frequencies of all recordings,
* application of low-pass, high-pass or band-pass filters, and
* waveform normalization.

.. _audio_settings:

The parameters for data preparation are specified using a Python dictionary object that is passed as a parameter (named ``audio_settings``) to the below functions. The following keys are supported:

  * **desired_fs** *(required)* The target sampling frequency (in Hz). Audio files having other sampling frequencies will be resampled to this value. Note that upsampling from a lower sampling rate introduces frequency banding in the resulting audio.
  * **clip_length** *(required)* The duration of each audio segment (in seconds).
  * **clip_advance** *(required)* The amount (in seconds) of overlap between successive segments is controlled by this. If `clip_advance` equals `clip_length`, then the overlap between successive segments will be zero.
  * **filterspec** *(optional)* If specified, must be a 3-element ordered list/tuple specifying -

    * filter order *(integer)*
    * cutoff frequency(ies) *(a 1-element or 2-element list/tuple)*
    * filter type *(string; one of 'lowpass', 'highpass' or 'bandpass')*

    If filter type is *'bandpass'*, the the cutoff frequency must be a 2-element list/tuple.
  * **normalize_clips** *(optional; default: True)* If True, will scale the waveform within each resulting clip to be in the range [-1.0, 1.0].


.. automodule:: koogu.prepare
   :members: