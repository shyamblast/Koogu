Koogu
=======
A python package for developing and using Machine Learning (ML) solutions in
Animal Bioacoustics.  

The package offers tools for -
* preparing and processing audio for training ML models,
* training ML models, and
* using trained ML models for automatic recognition.

How to use
----------
Koogu offers tools for ML development from the simplest of bioacoustics
applications to more complex scenarios. All stages of the workflow 
(preparation, training, inference) can be performed independently and
are described below.

## 1. Data preparation

Imports needed:
```python
from koogu import prepare
```

Describe what kind of processing is needed for you application.

The below example
instructs the ___prepare___ module to break up audio data read from disk into _2 s_
clips with a _75%_ overlap between successive clips. Audio loaded from files will
be resampled to the sampling frequency *desired_fs* Hz if they weren't already at
that sampling frequency.
```python
# Settings for handling raw audio
audio_settings = {
  'clip_length': 2.0,           # in seconds
  'clip_advance': 0.5,          # in seconds
  'desired_fs': 48000           # in Hz
}

# Path to the directory where processed/prepared audio will be written
prepared_audio_dir = '/mnt/projects/dolphins/prepared_clips'
```

Audio data can be organized in one of two ways and the appropriate function can
be invoked.
- _When annotations<sup>†</sup> are available_, place the audio files under a parent
  directory `audio_root` and place the annotations under a common directory
  `annots_root`, then build a Python _list_ `audio_annot_list` containing pairs
  (as 2-element *list*s or *tuple*s) that map an audio file to its corresponding
  annotation file. Audio files and annotation files may be organized into
  subdirectories under `audio_root` and `annots_root`, and the corresponding
  relative paths to the files can be specified in `audio_annot_list`.
  ```python
  # Root directories under which audio & corresponding annotation files are available
  audio_root = '/mnt/projects/dolphins/training_data/audio'
  annots_root = '/mnt/project/dolphins/training_data/annotations'
  
  # Map audio files to crresponding annotation files
  audio_annot_list = [
    ['day1/rec_01.wav', 'day1/rec_01.selections.txt'],
    ['day1/rec_02.wav', 'day1/rec_02.selections.txt'],
    ['day2/rec_10_w_ship_noise.wav', 'day2/rec_10_w_ship_noise.selections.txt'],
    ['day3/clean_recording.wav', 'day3/clean_recording.selections.txt'],
  ]
  
  # Convert audio files into prepared data
  clip_counts = prepare.from_selection_table_map(
    audio_settings, audio_annot_list,
    audio_root, annots_root,
    output_root=prepared_audio_dir
  )
  ```
  
- _When annotations are not available_, place audio files corresponding to
  different classes in their respective subdirectories, then place all the
  class-specific directories under a parent directory `audio_root`. The
  subdirectories' names will be used as class labels.
  ```python
  # Root directories under which audio & corresponding annotation files are available
  audio_root = '/mnt/projects/dolphins/training_data/audio'
  
  # List class-specific subdirectories to process
  class_dirs = ['bottlenose', 'spinner', 'dusky', 'long-beaked_common']
  
  # Convert audio files into prepared data
  clip_counts = prepare.from_top_level_dirs(
    audio_settings, class_dirs,
    audio_root,
    output_root=prepared_audio_dir
  )
  ```
  
_<sup>†</sup>_ Koogu currently supports annotations in
  [Raven Lite](https://ravensoundsoftware.com/software/raven-lite/) /
  [RavenPro](https://ravensoundsoftware.com/software/raven-pro/) selection table
  format, which is basically a simple tab-delimited text file providing (at a
  minimum) the start-time, end-time and label for each event/call. Values in the
  _Tags_ column (must exist) will be used as class labels.

The two functions under ___prepare___ support a few customizations. Resampled
and broken up waveforms and the respective class label info are stored under
*prepared_audio_dir* in compressed `numpy` format. The return value
*clip_counts* is a dictionary indicating the number of clips written for each
class.

___

## 2. Training

Imports needed:
```python
from koogu.model import Architectures
from koogu.data.feeder import SpectralDataFeeder
from koogu import train
```

- The first import provides a few varieties of customizable neural network
  architectures. Model complexity can be controlled with the customizations
  offered by the architecture-specific classes.
  
  User-defined architectures (including pre-trained models) can be
  implemented by extending ***koogu.model.BaseArchitecture***.

- The ***feeder*** module makes available a few varieties of customizable
  Python classes, each offering different capabilities, for efficiently
  feeding "prepared" data into a training/evaluation pipeline. The above
  example imports a feeder that also transforms loaded waveforms into
  spectrograms.
  
  Additional customizations and inclusion of data augmentation operations
  are possible by overriding the classes' ***transform()*** method in an
  inherited class. Furthermore, user-defined feeders can be implemented
  by extending any of the available feeders or by extending 
  ***koogu.data.feeder.BaseFeeder***.

- The training process can be controlled, along with the hyperparameter
  and regularization settings, by setting the appropriate values in the
  _dict_ that input to ***train()***.


A typical training/eval workflow is shown below. 

```python
# Settings describing the transformation of audio clips into
# time-frequency representations (spectrograms).
spec_settings = {
  'win_len': 0.008,                 # in seconds
  'win_overlap_prc': 0.50,          # as a fraction
  'bandwidth_clip': [2000, 45000],  # in Hz
  #'num_mels': 60                   # Uncomment to enable mel-scale conversion
}

# Setup a feeder that
#   i)   loads the prepared audio clips,
#   ii)  transforms the waveform clips into spectrogrms, and
#   iii) feeds them into the training pipeline.
data_feeder = SpectralDataFeeder(
  prepared_audio_dir,
  audio_settings['desired_fs'],
  data_settings,
  validation_split=0.15             # as a fraction
)


# Archecture choice and model customizations
model = Architectures.densenet(
  layers_per_block=[4, 8, 8, 4],
  growth_rate=12
)


# Settings that control the training process
training_settings = {
  'batch_size': 64,
  'epochs': 30,
  'learning_rate': 0.001,           # can set to a 'callable' for variable rate
  #'dropout_rate': 0.05,            # Uncomment to enable
  #'l2_weight_decay': 1e-4,         # Uncomment to enable
  #'optimizer': ['sgd', {}]         # choice & its settings; default is Adam
}

# Combine audio & spectrogram settings into one dict for convenience
data_settings = {
  'audio_settings': audio_settings,
  'spec_settings': spec_settings
}


# Path to the directory where trained model will be saved
model_dir = '/mnt/projects/dolphins/trained_models/DenseNet_1'

# Perform training
history = train(
  data_feeder,
  model_dir,
  data_settings,
  model,
  training_settings
)
```
  
___

## 3. Using a trained model

Imports needed:
```python
from koogu import recognize
```

Automatic recognition results are written out in
[Raven Lite](https://ravensoundsoftware.com/software/raven-lite/) /
[RavenPro](https://ravensoundsoftware.com/software/raven-pro/) selection table
format after applying an algorithm to group together similar successive detections.
Raw per-clip detections can also be saved for later assessments.
The function supports many customizations. See function documentation for details.

```python
# Path to a single audio file or to a directory (can contain subdirectories)
test_audio = '/mnt/projects/dolphins/test_audio/'

# Output directory
detections_dir = '/mnt/projects/dolphins/test_audio_detections'

recognize(
  model_dir,
  test_audio,
  detections_dir,
  reject_class='Noise',   # suppress saving of detections of specific classes
  threshold=0.75,
  #combine_outputs=True,  # combine detections from a directory into single output file
  batch_size=64,
  show_progress=True
)
```  

