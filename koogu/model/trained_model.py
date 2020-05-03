
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras

from koogu.data import AssetsExtraNames


class TrainedModel:

    assets_dirname = 'assets'

    @staticmethod
    def finalize_and_save(classifier, output_dir,
                          input_shape, trans_fn,
                          classes_list, audio_settings):
        """Create a new model encompassing an already-trained 'classifier'."""

        keras.backend.set_learning_phase(0)

        inputs = keras.Input(input_shape, name='inputs')
        class_mask = keras.Input(classifier.output.get_shape().as_list()[1],
                                 name='class_mask', batch_size=1,
                                 dtype=tf.bool)

        # Define input signature
        input_signature = {
            'inputs': inputs,
            'class_mask': class_mask
        }

        # Apply transformation
        new_output = trans_fn(inputs) if trans_fn is not None else inputs

        # Describe outputs
        logits = classifier(new_output)
        probs = logits * tf.cast(class_mask, dtype=logits.dtype)
        new_output = tf.identity(probs, name='scores')

        full_model = tf.keras.Model(input_signature, new_output)
        full_model.trainable = False

        tf.saved_model.save(full_model, output_dir)

        # Write out the list of class names as part of assets
        json.dump(classes_list,
                  open(os.path.join(output_dir, TrainedModel.assets_dirname,
                                    AssetsExtraNames.classes_list), 'w'))

        # Write audio settings (for use during inference) as part of assets
        json.dump(audio_settings,
                  open(os.path.join(output_dir, TrainedModel.assets_dirname,
                                    AssetsExtraNames.audio_settings), 'w'))

    def __init__(self, saved_model_dir):

        # Load model
        self._loaded_model = tf.saved_model.load(saved_model_dir)

        # Load assets
        self._class_names = json.load(
            open(os.path.join(saved_model_dir, TrainedModel.assets_dirname,
                              AssetsExtraNames.classes_list), 'r'))
        self._audio_settings = json.load(
            open(os.path.join(saved_model_dir, TrainedModel.assets_dirname,
                              AssetsExtraNames.audio_settings), 'r'))

        self._default_class_mask = tf.constant(
            np.full((1, len(self._class_names)), True, np.bool),
            tf.bool)

    def infer(self, inputs, class_mask=None):

        assert class_mask is None or len(class_mask) == len(self._class_names)

        res = self._loaded_model.signatures['serving_default'](
            inputs=tf.constant(inputs, tf.float32),
            class_mask=tf.constant(class_mask, tf.bool) \
                if class_mask is not None else self._default_class_mask)

        return res['tf_op_layer_scores'].numpy()

    @property
    def audio_settings(self):
        return self._audio_settings

    @property
    def class_names(self):
        return self._class_names
