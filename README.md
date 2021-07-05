Koogu
=======
A python package for developing and using Machine Learning (ML) solutions in
Animal Bioacoustics.  

Koogu (ಕೂಗು) is a word in the Kannada language and means "call" (used as a
_noun_) or "to call" (used as a _verb_). The package offers tools for -
* preparing and processing audio for training ML models,
* training ML models, and
* using trained ML models for automatic recognition.

Installation
------------

The package is available on PyPI, and can be installed as follows
```
pip install koogu
```

### Dependencies
#### TensorFlow
`koogu` uses `TensorFlow` as backend ML framework. Please ensure that either a
CPU or a GPU version of `TensorFlow` is installed prior to installing `koogu`.
#### librosa
`koogu` uses `librosa` for reading audio files (only). Please refer to [its
GitHub page](https://github.com/librosa/librosa) for details about its
dependencies and how to they may be addressed.
#### Others
Besides `TensorFlow`, all other dependencies will be automatically installed.

How to use Koogu
----------
Koogu offers tools for ML development from the simplest of bioacoustics
applications to more complex scenarios. All stages of the workflow (
preparation, training, inference) can be performed independently.

For simpler forms of supervised learning, data must be available as a
collection of audio files along with the necessary annotations. The
annotations must be in
[Raven Lite](https://ravensoundsoftware.com/software/raven-lite/) /
[RavenPro](https://ravensoundsoftware.com/software/raven-pro/) selection table
format, which is basically a simple tab-delimited text file providing (at a
minimum) the start-time, end-time and label for each event/call. The function
_from_selection_table_map()_ in the module _prepare_data_ helps prepare the
data, with full customization support, for training a model. The _data.feeder_
module provides classes for efficiently feeding prepared data into a
training/evaluation pipeline while the module _train_and_eval_ provides a
convenient and customizable interface for performing the actual training.   

