Quick-start guide
=================

This guide presents a recipe for a full bioacoustics ML workflow, from data
pre-processing to training, to performance assessments, and finally, to
using a trained model for analyzing soundscape/field recordings. End-to-end
examples are presented in both flavors:

- :doc:`CLI workflow <narw_cli>`: a "no code" command-line interface
- :doc:`API workflow <narw_api>`: a programmatic (Python) approach

Both approaches use the same underlying functionality and produce identical results.
You may test them yourself, using the below sample dataset. Once, you
have it working, you could modify the program/config to suit your own dataset.

**Sample dataset**

Both workflows use the North Atlantic Right Whale (NARW) up-call dataset
from the `DCLDE 2013 challenge
<https://doi.org/10.17630/62c3eebc-5574-4ec0-bfef-367ad839fe1a>`_. The dataset
contains 7 days of round-the-clock recordings out of which recordings from the
first 4 days were earmarked as a *training set* and recordings from the remaining
3 days were set aside as a *test set*. Each audio file is 15 minutes in
duration, and files from each day are organized in day-specific subdirectories.
The original dataset contained annotations in the legacy Xbat format, which we
converted to `RavenPro <https://ravensoundsoftware.com/software/raven-pro/>`_
selection table format for compatibility with Koogu. A representative subset of the dataset,
with converted annotations, can be accessed `here
<https://tinyurl.com/koogu-demo-data>`_.

The workflows expect the training and test audio files and corresponding
annotation files to be organized in a directory structure as shown below:

.. code-block:: none

  📁 projects
  └─ 📁 NARW
     └─ 📁 data
        ├─ 📁 train_audio
        ├─ 📁 train_annotations
        ├─ 📁 test_audio
        └─ 📁 test_annotations



.. toctree::
   :hidden:

   CLI workflow <narw_cli>
   API workflow <narw_api>
