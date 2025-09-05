Recognition and Deployment
==========================

Apply your trained model to new audio recordings in batch-processing mode.

**For post-processed detections** (typically used during deployment):

    .. code-block:: console

       $> # Process field recordings with detection threshold
       $> koogu-recognize /path/to/models/my_first_model \
                /path/to/data/field_recordings \
                /path/to/outputs/detections

    Output format: **Raven selection tables** (`.selections.txt`)


**For raw segment-level scores** (useful for testing and performance assessments):

    .. code-block:: console

       $> # Generate raw per-clip scores
       $> koogu-recognize /path/to/models/my_first_model \
                /path/to/data/test_recordings \
                /path/to/raw_scores/my_first_model --raw-scores

    Output format: **Raw NumPy arrays** (`.npz`)


The recognition process provides real-time feedback:

.. code-block:: console

   est_audio/NOPP6_EST_20090401/NOPP6_EST_20090401_024500.flac |██████████| 100.0%
   est_audio/NOPP6_EST_20090401/NOPP6_EST_20090401_024500.flac |██████    |  60.0%


**Performance considerations**
    - Batch size: Increase ``--batch-size`` for faster processing on high-RAM systems
    - Threading: Increase ``--threads`` for greater parallelization of I/O operations


----

Parameters
^^^^^^^^^^

Positional arguments
--------------------------------------------------------------------------------
    - ``<MODEL DIR>``

      Path to the directory containing a model trained with Koogu.

    - ``<AUDIO SOURCE>``

      Path to an audio file or to a directory. When a directory is provided, all files (of the supported types) within the directory will be processed (use the --recursive flag to also process subdirectories).

    - ``<OUTPUT ROOT>``

      Path to directory into which detection outputs will be written. If necessary, subdirectories will be automatically created.

Input control
--------------------------------------------------------------------------------
    - ``--filetypes EXTN``

      Audio file types to restrict processing to. Can specify multiple types separated by whitespaces. By default, will process all discovered files with the following extensions: [.wav, .WAV, .flac, .aif, .mp3]. Option is ignored when is a single file.

    - ``--recursive``

      Process files also in subdirectories of ``<AUDIO SOURCE>``.

    - ``--channels #``

      Channels to restrict processing to. List out the desired channel indices, separated by whitespaces. If unspecified, all available channels will be processed. Channel indices must be 0-based.

    - ``--clip-advance SECONDS``

      Override "clip advance". When audio files' contents are broken up into clips, by default the amount of overlap between successive clips is determined by the settings that were in place during model training. Use this flag to override that quantity, by setting a different amount of advance (in seconds) between successive clips.

Output control
--------------------------------------------------------------------------------
    - ``--raw-scores``

      Option to output raw per-clip per-class recognition scores (only) as is. If set, no post-processing algorithm is applied and all other settings within the 'Output control' group will be ignored.

    - ``--threshold #``

      Suppress detections having scores below this value (valid range: 0.0–1.0). If set, a post-processing algorithm will be applied.

    - ``--reject-class CLASS``

      Name (case sensitive) of the class (like 'Noise' or 'Other') that must be excluded from the recognition results. Can specify multiple (separated by whitespaces).

    - ``--squeeze MIN-DUR``

      If specified, will apply (the non-default) algorithm which 'squeezes together' successive detections from temporally overlapping clips. The 'squeezing' will be restricted to produce detections which are at least 'MIN-DUR' seconds long. MIN-DUR amount specified must be smaller than the duration of the model's input.

    - ``--frequency-info FILE``

      Path to a json file containing a dictionary of per-class frequency bounds. If unspecified, corresponding fields in the outputs (if applicable) will be the same for all classes.

    - ``--raven-combine-outputs``

      Enable this to combine recognition results of processing every file within a directory and write them to a single Raven selection table file. When enabled, the outputs will contain 2 additional fields describing offsets of detections in the corresponding audio files.

Process control
--------------------------------------------------------------------------------
    - ``--threads NUM``

      Number of threads that will fetch audio from files in parallel.

      `Default`: 1

    - ``--batch-size NUM``

      Size to batch audio file's clips into. Increasing this may improve speed on computers with higher RAM.

      `Deafult`: 1

Logging
--------------------------------------------------------------------------------
    - ``--log LOGFILE``

      If set, logging will be enabled and written out to the specified file.

    - ``--loglevel LEVEL``

      Logging level.. Choices: CRITICAL, ERROR, WARNING, INFO, DEBUG.

      `Default`: INFO
