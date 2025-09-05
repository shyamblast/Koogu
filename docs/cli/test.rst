Testing
=======

Apply a trained model to test datasets and generate raw detection scores for subsequent
assessment of performance.

.. code-block:: console

   $> # Test a trained model identified by its name
   $> koogu-test my_project.config my_first_model

This command serves as an intermediate step between model training and performance
evaluation, processing test audio files in the same manner as the training data
preparation but outputting detection scores instead of prepared training clips.

The testing process provides real-time feedback:

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
    - ``<CONFIG FILE>``

      Path to config file.

    - ``<MODEL NAME>``

      Name of the trained model.

Input control
--------------------------------------------------------------------------------
    - ``--filetypes EXTN [EXTN ...]``

      Audio file types to restrict processing to. Can specify multiple types separated by whitespaces. By default, will process all discovered files with the following extensions: [.wav, .WAV, .flac, .aif, .mp3]. When test_audio_annotations_map is available, this option is only considered where audio sources are directories.

    - ``--channels # [# ...]``

      Channels to restrict processing to. List out the desired channel indices, separated by whitespaces. If unspecified, all available channels will be processed. Channel indices must be 0-based.

    - ``--clip-advance SECONDS``

      Override "clip advance". When audio files' contents are broken up into clips, by default the amount of overlap between successive clips is determined by the settings that were in place during model training. Use this flag to override that quantity, by setting a different amount of advance (in seconds) between successive clips.

Process control
--------------------------------------------------------------------------------
    - ``--threads NUM``

      Number of threads that will fetch audio from files in parallel.

      `Default`: 1.

    - ``--batch-size NUM``

      Size to batch audio file's clips into. Increasing this may improve speed on computers with higher RAM.

      `Default`: 1

Logging
--------------------------------------------------------------------------------
    - ``--log LOGFILE``

      If specified, logging will be written out to this file instead of the default.

      `Default`: **PROJECT-LOGS-DIR**/test.log

    - ``--loglevel LEVEL``

      Logging level. Choices: CRITICAL, ERROR, WARNING, INFO, DEBUG.

      `Default`: INFO