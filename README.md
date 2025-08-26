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

## Installation

Koogu can be installed directly from **PyPI**.
It supports both **CPU** and **GPU** variants of TensorFlow.
If you have a compatible GPU, you can install the GPU-enabled variant using the `[gpu]` extra 

### CPU Installation

```bash
pip install koogu
```

### GPU Installation

```bash
pip install koogu[gpu]
```

### Dependencies

The following dependencies are installed automatically along with Koogu.

- **TensorFlow**  
  Koogu uses [TensorFlow](https://www.tensorflow.org/) as the ML backend.

- **Audio Libraries**  
  Koogu relies on the following packages for reading and processing audio:
  - [soundfile](https://pypi.org/project/soundfile/)
  - [audioread](https://pypi.org/project/audioread/) (needed for handling mp3 files)
  - [resampy](https://pypi.org/project/resampy)

### TensorFlow GPU option

Installing Koogu with `[gpu]` installs the TensorFlow variant appropriate for your system as described in the below table.

| Platform            | TensorFlow Version | GPU Option | TensorFlow Resolution                                                                                                                         |
|---------------------|:------------------:|:----------:|-----------------------------------------------------------------------------------------------------------------------------------------------|
| **Linux / WSL**     |      \>= 3.14      |   ✅ Yes    | Installation resolves to `tensorflow[and-cuda]` which installs all necessary CUDA support.                                                    |
| **Linux / WSL**     |       < 3.14       |   ✅ Yes    | Installation falls back to TensorFlow's unified CPU+GPU wheel.<br>👉 The user is responsible for setting up CUDA support in the environment. |
| **Windows / macOS** |        any         |    ❌ No    | GPU builds not available. CPU-only TensorFlow will be installed.                                                                              |

> ⚠️ Attempts to install with the `[gpu]` extra on Windows/macOS will be ignored and installation falls back to CPU variant.

---

How to use Koogu
----------
Koogu offers tools for ML development from the simplest of bioacoustics
applications to more complex scenarios. All stages of the workflow 
(input preparation, training, inference and performance assessment) can be performed independently.
An overview of the functionalities is presented in this ***[quick-start guide](HOWTO.md)***.

Technical API documentation is available [here](https://shyamblast.github.io/Koogu/).
