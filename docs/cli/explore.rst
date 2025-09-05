Dataset Exploration
===================

Explore and analyze your dataset to understand annotation patterns, class distributions, and data quality before proceeding with training. This helps -

    - **validate data quality** before training
    - **identify class imbalances** that may need addressing
    - **understand calls' frequency characteristics** for optimal spectrogram settings
    - **spot annotation inconsistencies** early in the workflow

.. code-block:: console

   $> # Explore and analyze a project's dataset
   $> koogu-explore my_project.config


This command analyzes both training and test annotations (if available) and generates interactive HTML reports:

    - **`training_annotations_stats.html`** - Analysis of training data annotations
    - **`test_annotations_stats.html`** - Analysis of test data annotations

The reports include:

    - **Class distribution** - Number of annotations per class
    - **Duration statistics** - Distribution of per-class call durations
    - **Bandwidth statistics** - Frequency range distribution per class

If ``[data.audio]`` and ``[data.spec]`` are set in the config file, it also prints out
information about model inputs that would result from applying the defined transformations.

----

Parameters
^^^^^^^^^^

Positional arguments
--------------------------------------------------------------------------------
    - ``<CONFIG FILE/ANNOTATIONS DIR>``

      If running within the scope a Koogu project, specify path to the
      project config file (training and test annotations' info will be
      read from the specified config file). Otherwise, specify path to
      the directory containing the set of annotation files you wish to
      analyze (also see additional options under `Direct access`).

Direct access
--------------------------------------------------------------------------------
    These options are only applicable when running outside
    the scope of a Koogu project, i.e., directly accessing
    annotation files by specifying <ANNOTATIONS DIR>. If
    <CONFIG FILE> was specified, all these below options will
    be ignored.

    - ``--reader``

      Set this based on the format of your annotation files.

    - ``--filelist``

      Path to a text file containing one-per-line entries of the
      annotation files to analyze. If not specified, all discoverable
      files under <ANNOTATIONS DIR> will be analyzed.

    - ``--output``

      Analysis outputs (HTML) will be written to this file, if
      specified. Otherwise, outputs will be written to the file
      `annotation_stats.html` in the current directory.

Process control
--------------------------------------------------------------------------------
    - ``--threads NUM``

      Number of threads to spawn for parallel execution.

      `Default`: as many CPUs
