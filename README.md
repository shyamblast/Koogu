Koogu [![DOI](https://zenodo.org/badge/255961543.svg)](https://zenodo.org/badge/latestdoi/255961543)
=======
A python package for developing and using Machine Learning (ML) solutions in
Animal Bioacoustics.  

Koogu (ಕೂಗು) is a word in the Kannada language and means "call" (used as a
_noun_) or "to call" (used as a _verb_). The package offers tools for -
* preparing and processing audio for training ML models,
* training ML models and assessing their performance, and
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
applications to more complex scenarios. All stages of the workflow 
(preparation, training, inference) can be performed independently and
are described ***[here](HOWTO.md)***.
