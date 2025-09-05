Performance Assessment
======================

Evaluate your trained model's performance on test data, generating precision-recall
metrics and performance reports in CSV, JSON, or HTML formats.

.. code-block:: console

   $> # Assess model performance
   $> koogu-assess my_project.config my_first_model first_model_performance.csv


Assessment parameters are defined under ``[assess]`` section of your config file.

.. note::

   Before starting assessments, ensure you have already run :doc:`koogu-test <test>` on
   the test dataset.

The output includes precision-recall metrics at different thresholds, helping you
choose optimal operating points for your deployment scenario.



Assessment Types
^^^^^^^^^^^^^^^^

- *Raw assessment* (with `--assess-raw` flag set)
    Determine performance at segment-level (i.e., simply considering scores for
    each segment in isolation). This provides a good assessment of how well a model
    trained.
- *Default assessment*
    Determine performance metrics after applying a post-processing algorithm to
    merge scores from consecutive audio segments. This assessment reflects the
    performance of a model in deployment scenario.


Output Formats
^^^^^^^^^^^^^^

Results can be generated in multiple formats based on file extension:

.. code-block:: console

   $> koogu-assess my_project.config perf.csv    # CSV format
   $> koogu-assess my_project.config perf.json   # JSON format
   $> koogu-assess my_project.config perf.html   # HTML report


----

Parameters
^^^^^^^^^^

Positional arguments
--------------------------------------------------------------------------------
    - ``<CONFIG FILE>``

      Path to config file.

    - ``<MODEL NAME>``

      Name of the trained model.

    - ``<RESULT FILE>``

      Path to output file where results are to be written. Results can be generated in one of the following formats: csv, json, html. Format is inferred from , and where that's not possible, defaults to csv format.

Options
--------------------------------------------------------------------------------
    - ``--assess-raw``

      If set, will assess "raw" clip-level recognition performance. By default, will assess performance from post-processed detections.

Logging
--------------------------------------------------------------------------------
    - ``--log LOGFILE``

      If specified, logging will be written out to this file instead of the default.

      `Default`: **PROJECT-LOGS-DIR**/assess.log

    - ``--loglevel LEVEL``

      Logging level. Choices: CRITICAL, ERROR, WARNING, INFO, DEBUG.

      `Default`: INFO
