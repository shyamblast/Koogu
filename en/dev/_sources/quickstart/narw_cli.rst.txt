Quickstart: CLI Workflow
========================

Here's the complete sequence of commands for the NARW project:

The example uses a light-weight custom
:class:`~koogu.model.architectures.DenseNet` architecture.

Explore the dataset (optional)
------------------------------

.. code-block:: console

   $> koogu-explore narw_project.config

1. Data preparation
-------------------
With audio-annotation mappings and data transformation settings already defined
in the config file, you can simply run the below command to prepare the data for
training.

.. code-block:: console

   $> koogu-prepare narw_project.config

2. Training
-----------
With network architecture and training/hyperparameter settings already defined
in the config file, you can simply run the below command to start training.
Specify a name for the model so you are able to distinguish models from different
training attempts.

.. code-block:: console

   $> koogu-train narw_project.config narw_detector_v1


3. Performance assessment
-------------------------

3.1. Run on test dataset
^^^^^^^^^^^^^^^^^^^^^^^^
# 4. Assess performance

.. code-block:: console

   $> koogu-assess narw_project.config narw_detector_v1_perf.html


# 5. Process field recordings

.. code-block:: console

   $> koogu-recognize narw_detector_v1/ field_audio/ detections/ \
        --threshold 0.75 --recursive --reject-class Other


# 6. Export for Raven Pro

.. code-block:: console

   $> koogu-export narw_project.config raven --model narw_detector_v1 \
        --threshold 0.85 --suppress Other
