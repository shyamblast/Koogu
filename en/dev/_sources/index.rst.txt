Koogu
=====

.. role:: raw-html(raw)
   :format: html

Koogu is a Python package for developing and using Machine Learning (ML) solutions in
Animal Bioacoustics.

   | Koogu (**ಕೂಗು**)
   | :raw-html:`<span style="font-family: arial,sans-serif">/ko͞ogu/</span>`
   |  a word in the Kannada language meaning
   |   - *call*, *utterance* (used as a *noun*)
   |   - *to call* (used as a *verb*)

The package offers tools for -

* preparing audio (pre-process and transform) to form inputs to ML models,
* training ML models,
* assessing their performance, and
* using trained ML models for automating analyses of large datasets.

Koogu offers tools for ML development from the simplest of bioacoustics
applications to more complex scenarios. All stages of the workflow
(data preparation, training, inference, performance assessment) can be performed
independently.

Data preparation
^^^^^^^^^^^^^^^^
Many of the neural networks models used in bioacoustics, especially convolutional
neural networks, expect inputs to be of fixed-dimensions, both during training
and while making inferences. Preparing model inputs involves, at a minimum,
breaking up of long-duration audio files into shorter segments as suitable for
the target sound(s) in consideration. Koogu provides convenient interfaces to
efficiently **batch-process** large collections of acoustic recordings (stored
in various file formats) in preparing model inputs.

Data feeders offer an efficient pipeline for supplying batches of input samples
during model training and validation. They ...

* read pre-processed data from disk,
* apply transformation (if any) to the data,
* perform on-the-fly data augmentations (if any),
* cache, shuffle and batch input samples, and
* present the batched samples to the model.

Performing on-the-fly transformations offers the ability to apply randomized
data augmentations in both time- and spectrotemporal domains independently.
Furthermore, custom user-defined feeders can be implemented by extending one of
the available feeder classes.

Model architectures & Training
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Koogu provides a few varieties of customizable neural network architectures.
Model complexity can be controlled with the customizations offered by the
architecture-specific classes.

User-defined architectures (including pre-trained models) can be implemented
by extending the base class.

Koogu offers a single-point interface for training and evaluating ML models. The
training process can be controlled, along with various hyperparameter and
regularization settings, by assigning appropriate values to the function's
parameters.

Performance assessments
^^^^^^^^^^^^^^^^^^^^^^^

When test datasets are available, Koogu's built-in tools can be used to assess
a trained model's performance thoroughly against the test dataset.

Analyzing field recordings/soundscape data with trained models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Koogu facilitates **batch-processing** of large volumes of field recordings using
a trained model. Currently, automatic recognition outputs from batch-processing are
provided as `Raven Pro <https://ravensoundsoftware.com/software/raven-pro/>`_
selection table files. A user can not only specify the application of detection
thresholds and how outputs are produced, but also control the utilization of
available computational resources.

.. toctree::
   :hidden:
   
   self

.. toctree::
   :hidden:
   
   quickstart

.. toctree::
   :hidden:
   
   advanced/index


.. toctree::
   :hidden:
   
   api/index

.. toctree::
   :hidden:

   integration

.. toctree::
   :hidden:

   genindex
