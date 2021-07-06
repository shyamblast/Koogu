import os
import json
import tensorflow as tf

from koogu.data import AssetsExtraNames


class TrainedModel:

    assets_dirname = 'assets'
    saved_model_dirname = 'koogu'

    @staticmethod
    def finalize_and_save(classifier, output_dir,
                          input_shape, transformation_info,
                          classes_list, audio_settings,
                          spec_settings=None):
        """Create a new model encompassing an already-trained 'classifier'."""

        classifier.trainable = False

        full_output_dir = os.path.join(output_dir, TrainedModel.saved_model_dirname)

        if transformation_info is not None:
            # If not None, must be a 2-tuple where:
            #   first value is the untransformed input shape
            #   second is the actual transformation function

            class MyModule(tf.Module):
                def __init__(self, base_model):
                    super().__init__()
                    self.model = base_model

                @tf.function(input_signature=[tf.TensorSpec(shape=[None] + input_shape, dtype=tf.float32)])
                def basic(self, inputs):
                    return {'scores': self.model(inputs)}

                @tf.function(input_signature=[tf.TensorSpec(shape=[None] + transformation_info[0], dtype=tf.float32)])
                def with_transformation(self, inputs):
                    outputs = transformation_info[1](inputs)
                    return {'scores': self.model(outputs)}

            model_with_signatures = MyModule(classifier)
            signatures = {'basic': model_with_signatures.basic,
                          'with_transformation': model_with_signatures.with_transformation}

        else:

            class MyModule(tf.Module):
                def __init__(self, base_model):
                    super().__init__()
                    self.model = base_model

                @tf.function(input_signature=[tf.TensorSpec(shape=[None] + input_shape, dtype=tf.float32)])
                def basic(self, inputs):
                    return {'scores': self.model(inputs)}

            model_with_signatures = MyModule(classifier)
            signatures = {'basic': model_with_signatures.basic}

        tf.saved_model.save(model_with_signatures,
                            full_output_dir,
                            signatures=signatures)

        # Write out the list of class names as part of assets
        json.dump(classes_list,
                  open(os.path.join(full_output_dir, TrainedModel.assets_dirname,
                                    AssetsExtraNames.classes_list), 'w'))

        # Write audio settings (for use during inference) as part of assets
        json.dump(audio_settings,
                  open(os.path.join(full_output_dir, TrainedModel.assets_dirname,
                                    AssetsExtraNames.audio_settings), 'w'))

        if spec_settings is not None:
            # Write spec settings (for use during inference) as part of assets
            json.dump(
                spec_settings,
                open(os.path.join(full_output_dir, TrainedModel.assets_dirname,
                                  AssetsExtraNames.spec_settings), 'w'))

    def __init__(self, saved_model_dir):

        full_input_dir = os.path.join(saved_model_dir, TrainedModel.saved_model_dirname)

        # Load model
        self._loaded_model = tf.saved_model.load(full_input_dir)

        self._infer_fns = dict()
        dont_load_spec_settings = False
        for signature in list(self._loaded_model.signatures.keys()):
            infer_fn = self._loaded_model.signatures[signature]
            infer_fn_input_shape = infer_fn.inputs[0].shape.as_list()[1:]

            self._infer_fns[TrainedModel._list2str(infer_fn_input_shape)] = \
                infer_fn

            if signature == 'with_transformation':
                dont_load_spec_settings = True

        # Load assets
        self._class_names = json.load(
            open(os.path.join(full_input_dir, TrainedModel.assets_dirname,
                              AssetsExtraNames.classes_list), 'r'))
        self._audio_settings = json.load(
            open(os.path.join(full_input_dir, TrainedModel.assets_dirname,
                              AssetsExtraNames.audio_settings), 'r'))

        spec_sett_filepath = os.path.join(full_input_dir,
                                          TrainedModel.assets_dirname,
                                          AssetsExtraNames.spec_settings)
        self._spec_settings = None if (dont_load_spec_settings or not os.path.exists(spec_sett_filepath)) \
            else json.load(open(spec_sett_filepath, 'r'))

    def infer(self, inputs):

        infer_fn = self._infer_fns.get(TrainedModel._list2str(inputs.shape[1:]), None)
        if infer_fn is not None:
            return infer_fn(inputs=inputs)['scores'].numpy()

        raise ValueError('Input shape {:s} does not match any existing signatures'.format(repr(inputs.shape)))

    @staticmethod
    def _list2str(in_list):
        return ','.join([repr(d) for d in in_list])

    @property
    def audio_settings(self):
        return self._audio_settings

    @property
    def spec_settings(self):
        return self._spec_settings

    @property
    def class_names(self):
        return self._class_names
