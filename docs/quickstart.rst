Quick-start guide
=================

We present here a recipe for a full bioacoustics ML workflow, from data
pre-processing to training, to performance assessments, and finally, to
using a trained model for analyzing soundscape/field recordings.

As an example, we considered the North Atlantic Right Whale (NARW) up-call dataset
from the `DCLDE 2013 challenge
<https://doi.org/10.17630/62c3eebc-5574-4ec0-bfef-367ad839fe1a>`_. The dataset
contained 7 days of round-the-clock recordings out of which recordings from the
first 4 days were earmarked as a *training set* and recordings from the remaining
3 days were set aside as a *test set*. Each audio file was 15 minutes in
duration, and files from each day were organized in day-specific subdirectories.
The original dataset contained annotations in the legacy Xbat format, which we
converted to `RavenPro <https://ravensoundsoftware.com/software/raven-pro/>`_
selection table format for compatibility with Koogu. A sample of the dataset,
with converted annotations, can be accessed `here
<https://tinyurl.com/koogu-demo-data>`_.

You may test the below code snippets yourself, using the sample dataset. Once, you
have it working, you could modify the program to suit your own dataset.

The code sections below expect the training and test audio files and corresponding
annotation files to be organized in a directory structure as shown below:

.. code-block:: none

  📁 projects
  └─ 📁 NARW
     └─ 📁 data
        ├─ 📁 train_audio
        ├─ 📁 train_annotations
        ├─ 📁 test_audio
        └─ 📁 test_annotations

Imports
-------

First, import the necessary modules and functions from the Koogu package.

.. literalinclude:: code_samples/full_workflow_example.py
   :end-before: [train_data--start]
   :linenos:
   :lineno-match:

1. Data preparation
----------------------

Point out where to fetch the training dataset from.

We also need to specify which annotation files correspond to which audio files
(or, in this example, to sub-directories containing a collection of files).

.. literalinclude:: code_samples/full_workflow_example.py
   :start-after: [train_data--start]
   :end-before: [train_data--end]
   :linenos:
   :lineno-match:

Define parameters for preparing training audio, and for converting them to
spectrograms.

.. literalinclude:: code_samples/full_workflow_example.py
   :start-after: [data_settings--start]
   :end-before: [data_settings--end]
   :linenos:
   :lineno-match:


1.1 Preprocess
^^^^^^^^^^^^^^

The preprocessing step will split up the audio files into clips (defined by
``data_settings['audio_settings']``), match available annotations to the clips,
and mark each clip to indicate if it matched one or more annotations.

We believe that the available annotations in the training set covered almost
all occurrences of the target `NARW up-calls` in the recordings, with no (or
only a small number of) missed calls. As such, we can consider all un-annotated
time periods in the recordings as inputs for the *negative* class (by setting the
parameter ``negative_class_label``).

.. literalinclude:: code_samples/full_workflow_example.py
   :start-after: [preprocess--start]
   :end-before: [preprocess--end]
   :linenos:
   :lineno-match:

.. seealso::
   Koogu supports annotations in different popular formats, besides the default RavenPro format.
   See :mod:`koogu.data.annotations` for a list of supported formats.

.. seealso::
   If your project does not have annotations, but you have audio files corresponding
   to each species/call type organized under separate directories, you can
   pre-process the data using :func:`~koogu.prepare.from_top_level_dirs`
   instead of :func:`~koogu.prepare.from_annotations`.

You can check how many clips were generated for each class -

.. literalinclude:: code_samples/full_workflow_example.py
   :start-after: [preprocess_output--start]
   :end-before: [preprocess_output--end]
   :linenos:
   :lineno-match:

1.2. Feeder setup
^^^^^^^^^^^^^^^^^

Now, we define a feeder that efficiently feeds all the pre-processed clips, in
batches, to the training/validation pipeline. The feeder is also transforms the
audio clips into spectrograms.

Typically, model training is performed on computers having one or more GPUs.
While the GPUs consume data at extreme speeds during training, it is imperative
that the mechanism to feed the training data doesn't keep the GPUs waiting for
inputs. The feeders provided in Koogu utilize all available CPU cores to ensure
that GPU utilization remains high during training.

.. literalinclude:: code_samples/full_workflow_example.py
   :start-after: [feeder--start]
   :end-before: [feeder--end]
   :linenos:
   :lineno-match:

The considered sample dataset contains very many annotated calls, covering a
reasonable range of input variations. As such, in this example we do not employ
any data augmentation techniques. However, you could easily add some of the
pre-canned :doc:`data augmentations <advanced/data_augmentation>` when you
adopt this example to work with your dataset.

2. Training
-----------
First, describe the architecture of the model that is to be used. With Koogu,
you do not need to write lot's of code to build custom models; simply chose an
exiting/available architecture (e.g., ConvNet, DenseNet) and specify how you'd
want it customized.

In this example, we use a light-weight custom
:class:`~koogu.model.architectures.DenseNet` architecture.

.. literalinclude:: code_samples/full_workflow_example.py
   :start-after: [model_setup--start]
   :end-before: [model_setup--end]
   :linenos:
   :lineno-match:

The training process can be controlled, along with hyperparameter and
regularization settings, by setting appropriate values in the Python
dictionary that is input to :func:`~koogu.train`. See the function API
documentation for all available options.

.. literalinclude:: code_samples/full_workflow_example.py
   :start-after: [training--start]
   :end-before: [training--end]
   :linenos:
   :lineno-match:

You can visualize how well the training progressed by plotting the contents of
the ``history`` variable returned.

.. literalinclude:: code_samples/full_workflow_example.py
   :start-after: [training_output--start]
   :end-before: [training_output--end]
   :linenos:
   :lineno-match:

You may tune the training parameters above and repeat the training step until
the training and validation accuracy/loss reach desired levels.

3. Performance assessment
-------------------------

3.1. Run on test dataset
^^^^^^^^^^^^^^^^^^^^^^^^
If you have a test dataset available for assessing performance, you can easily
run the trained model on that dataset. Simply point out where to fetch the test
dataset from.

Similar to how training annotation data were presented (by associating annotation
files to audio files), we also need to specify which test annotation files
correspond to which test audio files (or, in this example, to sub-directories
containing a collection of test files).

.. literalinclude:: code_samples/full_workflow_example.py
   :start-after: [test_data--start]
   :end-before: [test_data--end]
   :linenos:
   :lineno-match:

Now apply the trained model to this test dataset. During testing, it is useful
to save raw per-clip recognition scores which can be subsequently analyzed for
assessing the model’s recognition performance.

.. literalinclude:: code_samples/full_workflow_example.py
   :start-after: [testing_model--start]
   :end-before: [testing_model--end]
   :linenos:
   :lineno-match:

The :func:`~koogu.recognize` function supports many customizations.
See function API documentation for more details.

3.2. Determine performance
^^^^^^^^^^^^^^^^^^^^^^^^^^
Now, compute performance metrics.

.. literalinclude:: code_samples/full_workflow_example.py
   :start-after: [assess_perf--start]
   :end-before: [assess_perf--end]
   :linenos:
   :lineno-match:

And, visualize the assessments.

.. literalinclude:: code_samples/full_workflow_example.py
   :start-after: [plot_test_results--start]
   :end-before: [plot_test_results--end]
   :linenos:
   :lineno-match:

By analyzing the precision-recall curve, you can pick an operational threshold
that yields the desired precision vs. recall trade-off.

4. Use the trained model
------------------------
Once you are settled on a choice of detection threshold that yields a suitable
precision-recall trade-off, you can apply the trained model on new recordings.

Koogu supports two ways of using a trained model.

4.1. Batch processing
^^^^^^^^^^^^^^^^^^^^^

In most common applications, one would want to be able to **batch process**
large collections of audio files with a trained model.

In this mode, automatic recognition results are written out in `Raven Pro
<https://ravensoundsoftware.com/software/raven-pro/>`_ selection table format
after applying an algorithm to group together detections of the same class across
contiguous clips.

.. literalinclude:: code_samples/full_workflow_example.py
   :start-after: [batch_analyze--start]
   :end-before: [batch_analyze--end]
   :linenos:
   :lineno-match:

The :func:`~koogu.recognize` function supports many customizations.
See function API documentation for more details.

.. _custom_inferences:

4.2 Custom processing
^^^^^^^^^^^^^^^^^^^^^

Sometimes, one may need to process audio data that is not available in the form
of audio files (or in unsupported formats). For example, one may want to apply
a trained model to live-stream acoustic feeds. Koogu facilitates such use of a
trained model via an additional interface in which you implement the task of
preparing the data (breaking up into clips) in the format that a model expects.
Then, you simply pass the clips to :func:`~koogu.inference.analyze_clips`.

.. literalinclude:: code_samples/full_workflow_example.py
   :start-after: [clip_analyze--start]
   :end-before: [clip_analyze--end]

.. seealso::
   :doc:`api/misc/low_inference`
