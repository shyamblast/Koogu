from koogu.data import feeder
from koogu.model import architectures
from koogu import prepare, train, assessments, recognize

from matplotlib import pyplot as plt           # used for plotting graphs

# [train_data--start]
# The root directories under which the training data (audio files and
# corresponding annotation files) are available.
audio_root = '/home/shyam/projects/NARW/data/train_audio'
annots_root = '/home/shyam/projects/NARW/data/train_annotations'

# Map audio files (or containing folders) to respective annotation files
audio_annot_list = [
    ['NOPP6_EST_20090328', 'NOPP6_20090328_RW_upcalls.selections.txt'],
    ['NOPP6_EST_20090329', 'NOPP6_20090329_RW_upcalls.selections.txt'],
    ['NOPP6_EST_20090330', 'NOPP6_20090330_RW_upcalls.selections.txt'],
    ['NOPP6_EST_20090331', 'NOPP6_20090331_RW_upcalls.selections.txt'],
]
# [train_data--end]

# [data_settings--start]
data_settings = {
    # Settings for handling raw audio
    'audio_settings': {
        'clip_length': 2.0,
        'clip_advance': 0.4,
        'desired_fs': 1000
    },

    # Settings for converting audio to a time-frequency representation
    'spec_settings': {
        'win_len': 0.128,
        'win_overlap_prc': 0.75,
        'bandwidth_clip': [46, 391]
    }
}
# [data_settings--end]

# [preprocess--start]
# Path to the directory where pre-processed data will be written.
# Directory will be created if it doesn't exist.
prepared_audio_dir = '/home/shyam/projects/NARW/prepared_data'

# Convert audio files into prepared data
clip_counts = prepare.from_selection_table_map(
    data_settings['audio_settings'],
    audio_annot_list,
    audio_root, annots_root,
    output_root=prepared_audio_dir,
    negative_class_label='Other')
# [preprocess--end]
# [preprocess_output--start]
# Display counts of how many inputs we got per class
for label, count in clip_counts.items():
    print(f'{label:<10s}: {count:d}')
# [preprocess_output--end]


# [feeder--start]
data_feeder = feeder.SpectralDataFeeder(
    prepared_audio_dir,                        # where the prepared clips are at
    data_settings['audio_settings']['desired_fs'],
    data_settings['spec_settings'],
    validation_split=0.15,                     # set aside 15% for validation
    max_clips_per_class=20000                  # use up to 20k inputs per class
)
# [feeder--end]

# [model_setup--start]
model = architectures.DenseNet(
    [4, 4, 4],                                 # 3 dense-blocks, 4 layers each
    preproc=[ ('Conv2D', {'filters': 16}) ],   # Add a 16-filter pre-conv layer
    dense_layers=[32]                          # End with a 32-node dense layer
)
# [model_setup--end]

# [training--start]
# Settings that control the training process
training_settings = {
    'batch_size': 64,
    'epochs': 50,                              # run for 50 epochs

    # Start with a learning rate of 0.01, and drop it to a tenth of its value,
    # successively, at epochs 20 & 40.
    'learning_rate': 0.01,
    'lr_change_at_epochs': [20, 40],
    'lr_update_factors': [1.0, 1e-1, 1e-2],    # up to 20, beyond 20, beyond 40

    'dropout_rate': 0.05                       # Helps model generalize better
}

# Path to the directory where model files will be written
model_dir = '/home/shyam/projects/NARW/models/my_first_model'

# Perform training
history = train(
    data_feeder,
    model_dir,
    data_settings,
    model,
    training_settings
)
# [training--end]
# [training_output--start]
# Plot training & validation history
fig, ax = plt.subplots(2, sharex=True, figsize=(12, 9))
ax[0].plot(
    history['train_epochs'], history['binary_accuracy'], 'r',
    history['eval_epochs'], history['val_binary_accuracy'], 'g')
ax[0].set_ylabel('Accuracy')
ax[1].plot(
    history['train_epochs'], history['loss'], 'r',
    history['eval_epochs'], history['val_loss'], 'g')
ax[1].set_yscale('log')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Loss')
plt.show()
# [training_output--end]

# [test_data--start]
# The root directories under which the test data (audio files and
# corresponding annotation files) are available.
test_audio_root = '/home/shyam/projects/NARW/data/test_audio'
test_annots_root = '/home/shyam/projects/NARW/data/test_annotations'

# Map audio files to corresponding annotation files
test_audio_annot_list = [
    ['NOPP6_EST_20090401', 'NOPP6_20090401_RW_upcalls.selections.txt'],
    ['NOPP6_EST_20090402', 'NOPP6_20090402_RW_upcalls.selections.txt'],
    ['NOPP6_EST_20090403', 'NOPP6_20090403_RW_upcalls.selections.txt'],
]
# [test_data--end]
# [testing_model--start]
# Directory in which raw detection scores will be saved
raw_detections_root = '/home/shyam/projects/NARW/test_audio_raw_detections'

# Run the model (detector/classifier)
recognize(
    model_dir,
    test_audio_root,
    raw_detections_dir=raw_detections_root,
    batch_size=64,     # Increasing this may improve speed if there's enough RAM
    recursive=True,    # Process subdirectories also
    show_progress=True
)
# [testing_model--end]
# [assess_perf--start]
# Initialize a metric object with the above info
metric = assessments.PrecisionRecall(
    test_audio_annot_list,
    raw_detections_root, test_annots_root)
# The metric supports several options (including setting explicit thresholds).
# Refer to class documentation for more details.

# Run the assessments and gather results
per_class_pr, overall_pr = metric.assess()
# [assess_perf--end]
# [plot_test_results--start]
# Plot PR curves.
for class_name, pr in per_class_pr.items():
    print('-----', class_name, '-----')
    plt.plot(pr['recall'], pr['precision'], 'rd-')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid()
    plt.show()

# Similarly, you could plot the contents of 'overall_pr' too
# [plot_test_results--end]

# [batch_analyze--start]
# Path to directory containing audio files (may contain subdirectories too)
field_recordings_root = '/home/shyam/projects/NARW/field_recordings'
field_rec_detections_root = '/home/shyam/projects/NARW/field_rec_detections'

chosen_threshold = 0.75

recognize(
    model_dir,
    field_recordings_root,
    output_dir=field_rec_detections_root,
    threshold=chosen_threshold,
    reject_class='Other',                      # Only output target class dets
    #clip_advance=0.5,                         # Can use different clip advance
    batch_size=64,                             # Can go higher on good computers
    num_fetch_threads=4,                       # Parallel-process for speed
    recursive=True,                            # Process subdirectories also
    show_progress=True
)
# [batch_analyze--end]


# [clip_analyze--start]
from koogu.model import TrainedModel
from koogu.inference import analyze_clips

# Load the trained model
trained_model = TrainedModel(model_dir)

# Read in the audio samples from a file (using one of SoundFile, AudioRead,
# scipy.io.wavfile, etc.), or buffer-in from a live stream.

# As with the model trained in the above example, you may need to resample the
# new data to 1 kHz, and then break them up into clips of length 2 s to match
# the trained model's input size.

not_end = True

while not_end:

    my_clips = ...
    # say we got 6 clips, making it a 6 x 2000 numpy array

    # Run detections and get per-clip scores for each class
    scores, processing_time = analyze_clips(trained_model, my_clips)
    # Given 6 clips, we get 'scores' to be a 6 x 2 array

    # ... do something with the results
    ...

# [clip_analyze--end]
