How to use
==========
Koogu offers tools for ML development from the simplest of bioacoustics
applications to more complex scenarios. All stages of the workflow 
(data preparation, training, performance assessment, inference) can be performed
independently as outlined below.

If you are interested in a hands-on demo (on Google Colab) with real data,
here is a [video providing an instructional walk-through
](https://youtu.be/3ANAbT90sfo?t=2665) on using the package.

For a more thorough coverage, please refer to Koogu's
[online documentation](https://shyamblast.github.io/Koogu/).

## 1. Data preparation

Imports needed:
```python
from koogu.data import preprocess
from koogu.data import feeder
```

Describe what kind of processing is needed for your application.

The below example
instructs the ___preprocess___ module to break up audio data read from disk into _2 s_
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
  annotation file. Audio files and annotation files may be further organized into
  subdirectories under `audio_root` and `annots_root`, and the corresponding
  relative paths to the files can be specified in `audio_annot_list` (`audio_annot_list`
  may also be set as the path to a csv file that contains the mapping pairs).
  ```python
  # Root directories under which audio & corresponding annotation files are available
  audio_root = '/mnt/projects/dolphins/training_data/audio'
  annots_root = '/mnt/project/dolphins/training_data/annotations'
  
  # Map audio files to corresponding annotation files
  audio_annot_list = [
    ['day1/rec_01.wav', 'day1/rec_01.selections.txt'],
    ['day1/rec_02.wav', 'day1/rec_02.selections.txt'],
    ['day2/rec_10_w_ship_noise.wav', 'day2/rec_10_w_ship_noise.selections.txt'],
    ['day3/clean_recording.wav', 'day3/clean_recording.selections.txt'],
  ]
  
  # Convert audio files into prepared data
  clip_counts = preprocess.from_annotations(
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
  (default), [Audacity](https://www.audacityteam.org/) and 
  [SonicVisualizer](https://www.sonicvisualiser.org/) formats. In the Raven format, by default,
  values in the _Tags_ column (if it exists) will be used as class labels. You
  could also force to look for class labels in a different column by specifying
  the name of the corresponding column.

The two functions under ___preprocess___ support a few customizations. Resampled
and broken up waveforms and the respective class label info are stored under
*prepared_audio_dir* in compressed `numpy` format. The return value
*clip_counts* is a dictionary indicating the number of clips written for each
class.

The ***feeder*** module makes available customizable Python classes that offer
the ability to efficiently feed "prepared" data into a training/evaluation
pipeline.

In common bioacoustic applications, audio clips must be transformed into
spectrograms before they can be fed to a model. For this purpose, we can use the
SpectralDataFeeder which transforms loaded waveforms into spectrograms on-the-fly.

```python
# Settings describing the transformation of audio clips into
# time-frequency representations (spectrograms).
spec_settings = {
  'win_len': 0.008,                 # in seconds
  'win_overlap_prc': 0.50,          # as a fraction
  'bandwidth_clip': [2000, 45000],  # in Hz
  #'num_mels': 60                   # Uncomment to enable mel-scale conversion
}

# Set up a feeder that
#   i)   loads the pre-processed audio clips,
#   ii)  transforms the waveform clips into spectrograms, and
#   iii) feeds them into the training pipeline.
data_feeder = feeder.SpectralDataFeeder(
  prepared_audio_dir,
  audio_settings['desired_fs'],
  spec_settings,
  validation_split=0.15             # as a fraction
)
```

Additional customizations of the transformation operation are possible by
overriding the feeder class' ***transform()*** method in an inherited class.
Performing on-the-fly transformations offers us the ability to apply randomized
data augmentations in both time- and spectrotemporal domains independently.
Data augmentations can be included by overriding the feeder class'
***pre_transform()*** and ***post_transform()*** methods. Furthermore, user-defined
feeders can be implemented by extending any of the available feeders or by
extending ***koogu.data.feeder.BaseFeeder***.

___

## 2. Training

Imports needed:
```python
from koogu.model import architectures
from koogu import train
```

- The first import provides a few varieties of customizable neural network
  architectures. Model complexity can be controlled with the customizations
  offered by the architecture-specific classes.
  
  User-defined architectures (including pre-trained models) can be
  implemented by extending ***koogu.model.BaseArchitecture***.

- The training process can be controlled, along with hyperparameter and
  regularization settings, by assigning appropriate values to the various
  fields in the _dict_ that is input to ***train()***.


A typical training/eval workflow is shown below. 

```python
# Architecture choice and model customizations
model = architectures.DenseNet(
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
  data_feeder,                      # connect to the feeder
  model_dir,
  data_settings,
  model,
  training_settings
)
```
  
___

## 3. Using a trained model on test data

Imports needed:
```python
from koogu import recognize
```

During testing, it is useful to save raw per-clip detections which can be subsequently analyzed
for assessing the model's recognition performance (Step 4).

```python
# Path to a single audio file or to a directory (can contain subdirectories)
test_audio_root = '/mnt/projects/dolphins/test_data/audio'

# Output directory
raw_detections_root = '/mnt/projects/dolphins/test_audio_raw_detections'

recognize(
  model_dir,
  test_audio_root,
  raw_detections_dir=raw_detections_root,
  batch_size=64,    # Increasing this may improve speed on computers having higher resources
  recursive=True,   # Process subdirectories also
  show_progress=True
)
```

The recognize() function supports many customizations. See function documentation for more details.
  
___

## 4. Assessing performance

Imports needed:
```python
from koogu import assessments
```

Similar to how training annotation data were presented in Step 1,
performance assessments also requires annotations corresponding to the test audio
files processed above.

```python
# Root directory under which annotation files (corresponding to the test
# audio files used above) are available.
test_annots_root = '/mnt/project/dolphins/test_data/annotations'

# Map audio files to corresponding annotation files
test_audio_annot_list = [
  ['day7/clean_recording.wav', 'day7/clean_recording.selections.txt'],
  ['day7/rec_01.wav', 'day7/rec_01.selections.txt'],
  ['day8/rec_02.wav', 'day8/rec_02.selections.txt'],
  ['day9/rec_10_w_ship_noise.wav', 'day9/rec_10_w_ship_noise.selections.txt'],
  ['day9/rec_01.wav', 'day9/rec_01.selections.txt'],
]

# Initialize a metric object with the above info
metric = assessments.PrecisionRecall(
  test_audio_annot_list,
  raw_detections_root, test_annots_root)
# The metric supports several options (including setting explicit thresholds).
# Refer to class documentation for more details.

# Run the assessments and gather results
per_class_pr, overall_pr = metric.assess()

# Plot PR curves.
# (Note: the below example code requires the matplotlib package and assumes that
# pyplot was already imported from it as:
#   from matplotlib import pyplot as plt
# )
for class_name, pr in per_class_pr.items():
  print(class_name)
  plt.plot(pr['recall'], pr['precision'], 'rd-')
  plt.xlabel('Recall')
  plt.ylabel('Precision')
  plt.grid()
  plt.show()

# The thresholds at which the different precision-recall values were determined
# can be queried as-
print(metric.thresholds)
```
  
___

## 5. Using the trained model on new recordings

Imports needed:
```python
from koogu import recognize
```

Once you are settled on a choice of detection threshold that yields a desired
precision-recall trade-off, you can apply the trained model on new recordings.
Automatic recognition results are written out in
[Raven Lite](https://ravensoundsoftware.com/software/raven-lite/) /
[RavenPro](https://ravensoundsoftware.com/software/raven-pro/) selection table
format after applying an algorithm to group together similar successive detections.
The function supports many customizations. See function documentation for details.

```python
# Path to a single audio file or to a directory (can contain subdirectories)
new_audio_root = '/mnt/projects/dolphins/new_audio/'

# Output directory
detections_output_dir = '/mnt/projects/dolphins/new_audio_detections'

recognize(
  model_dir,
  new_audio_root,
  output_dir=detections_output_dir,
  reject_class='Noise',   # suppress saving of detections of specific classes
  threshold=0.75,
  #combine_outputs=True,  # combine detections from sub-directory into single annotation files
  batch_size=64,
  recursive=True,         # Process subdirectories also
  show_progress=True
)
```

