import os
import tensorflow as tf
from koogu.data.feeder import BaseFeeder

# Assuming that you have saved 1 second long audio clips each containing
# sounds from one of three species of frogs, and that the audio clips
# are available as .wav files organized under species-specific directories.

fs = 24000        # sampling frequency of the audio files
directories_as_class_names = [
    'Lithobates sylvaticus',
    'Lithobates catesbeianus',
    'Dryophytes versicolor'
]


def read_files(filelist, sp_idx):
    # Utility function (a generator) to read a list of audio files one by one

    # One-hot encoded label for the current species
    label = tf.one_hot(sp_idx, 3)

    for fname in filelist:
        # Read in the audio samples from
        #   directories_as_class_names[sp_idx] + '/' + fname.decode()
        # using one of SoundFile, AudioRead, scipy.io.wavfile, etc.
        clip = ...

        # return the clip and the label
        yield clip, label


class MyFeeder(BaseFeeder):

    def __init__(self):
        """
        Register number of samples available, and decide how to split
        train vs test subsets.
        """

        # Get the list of files available in each directory/class
        self.sp0_files = os.listdir(directories_as_class_names[0])
        self.sp1_files = os.listdir(directories_as_class_names[1])
        self.sp2_files = os.listdir(directories_as_class_names[2])

        # Shuffle the lists' contents as desired
        #  ...

        # File/sample counts per species
        file_counts = [
            len(self.sp0_files),
            len(self.sp1_files),
            len(self.sp2_files)
        ]

        # Earmark 15% for validation; remaining will be used as training samples
        per_class_training_samples = [0, 0, 0]
        per_class_eval_samples = [0, 0, 0]
        for class_idx, fc in enumerate(file_counts):
            per_class_training_samples[class_idx] = int(round(fc * 0.85))
            per_class_eval_samples[class_idx] = \
                fc - per_class_training_samples[class_idx]

        # Invoke the parent constructor
        super(MyFeeder, self).__init__(
            (fs, ),
            per_class_training_samples, per_class_eval_samples,
            directories_as_class_names)

    def make_dataset(self, is_training, batch_size, **kwargs):
        """
        Build a TensorFlow Dataset comprising all training or eval clips
        """

        # Make class-specific datasets
        sp_ds = [None, None, None]
        for sp_idx, sp_files in enumerate(
                [self.sp0_files, self.sp1_files, self.sp2_files]):

            # Restrict which files to read based on train/eval mode
            split_idx = self.training_samples_per_class[sp_idx]
            if is_training:
                filelist = sp_files[:split_idx]
            else:
                filelist = sp_files[split_idx:]

            sp_ds[sp_idx] = tf.data.Dataset.from_generator(
                lambda a, b: read_files(a, b),
                args=(filelist, sp_idx),
                output_signature=(
                    tf.TensorSpec(shape=(fs, ), dtype=tf.float32),  # clip
                    tf.TensorSpec(shape=(3, ), dtype=tf.float32)    # label
                )
            )

        # Concatenate all class-specific data
        dataset = sp_ds[0].concatenate(sp_ds[1]).concatenate(sp_ds[2])

        # Invoke the base class functionality to shuffle & batch, or implement
        # the logic yourself as needed.
        return self._queue_and_batch(dataset, is_training, batch_size, **kwargs)

    def transform(self, sample, label, is_training, **kwargs):
        # Pass as-is, not doing any transformation in this example
        return sample, label

    def pre_transform(self, sample, label, is_training, **kwargs):
        # Pass as-is, not applying any augmentations in this example
        return sample, label

    def post_transform(self, sample, label, is_training, **kwargs):
        # Pass as-is, not applying any augmentations in this example
        return sample, label
# [end-first-example]


class MyFeeder(BaseFeeder):

    # [start-second-example-snippet]
    # --- update the constructor from the above example ---
    def __init__(self):
      ...
      ...
      # Invoke the parent constructor
      super(MyFeeder, ...
      ...
      ...

      # Define the settings for transformation
      spec_settings = {
          'win_len': ...
          # ...
          # see koogu.data.tf_transformations.Audio2Spectral for list of keys
      }

      # Instantiate the transformation object
      self._transform = koogu.data.tf_transformations.Audio2Spectral(fs, spec_settings)

      self._in_shape = (fs, )

      # Update parent's member variable to reflect the transformed output shape
      self._shape = self._transform.compute_output_shape([1] + self._in_shape)[1:]

    def transform(self, clip, label, is_training, **kwargs):

      # Apply the transformation
      output = self._transform(clip)

      return output, label

    def get_shape_transformation_info(self):
      return self._in_shape, self._transform

