Koogu
=======
[![DOI](https://zenodo.org/badge/255961543.svg)](https://zenodo.org/badge/latestdoi/255961543)

A python package for developing and using Machine Learning (ML) solutions in
Animal Bioacoustics.

Koogu (ಕೂಗು; <span style="font-family: arial,sans-serif">/ko͞ogu/</span>) is a word in the Kannada language, meaning "call" (used as a
_noun_) or "to call" (used as a _verb_).

The package offers tools for -
* preparing audio (pre-process and transform) to form inputs to ML models,
* training ML models,
* assessing their performance, and
* using trained ML models for automating analyses of large datasets.

Installation
------------

Koogu can be installed, via PyPI, as follows
```bash
pip install koogu
```

### Dependencies
#### TensorFlow
`koogu` uses [`TensorFlow`](https://www.tensorflow.org/) as backend ML framework. Please ensure that either a
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
applications to more complex scenarios. All stages of the workflow 
(input preparation, training, inference and performance assessment) can be performed independently.
An overview of the functionalities is presented in this ***[quick-start guide](HOWTO.md)***.

Technical API documentation is available [here](https://shyamblast.github.io/Koogu/).
