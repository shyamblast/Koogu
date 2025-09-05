Data Preparation
================

Convert audio files and annotations into training-ready format by splitting audio into clips, matching annotations, and creating positive/negative class labels.

.. code-block:: console

   $> # Prepare the project's training data
   $> koogu-prepare my_project.config


The preprocessing step will:

    1. **Pre-process audio files**
       by applying necessary filtering and splitting into fixed-duration segments
    2. **Match annotations**
       to segments based on temporal overlap
    3. **Label each segment**
       as positive or negative for each class
    4. **Save prepared segments and annotaions**
       in a format optimized for training


----

Parameters
^^^^^^^^^^

Positional arguments
--------------------------------------------------------------------------------
    - ``<CONFIG FILE>``

      Path to config file.

Input control
--------------------------------------------------------------------------------
    - ``--filetypes EXTN [EXTN ...]``

      Audio file types to restrict processing to. Can specify multiple types separated by whitespaces. By default, will process all discovered files with the following extensions: [.wav, .WAV, .flac, .aif, .mp3]. When train_audio_annotations_map is available, this option is only considered where audio sources are directories.

Process control
--------------------------------------------------------------------------------
    - ``--threads NUM``

      Number of threads to spawn for parallel execution.

      `Default`: as many CPUs

Logging
--------------------------------------------------------------------------------
    - ``--log LOGFILE``

      If specified, logging will be written out to this file instead of the default.

      `Default`: **PROJECT-LOGS-DIR**/prepare.log

    - ``--loglevel LEVEL``

      Logging level. Choices: CRITICAL, ERROR, WARNING, INFO, DEBUG.

      `Default`: INFO
