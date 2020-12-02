
import tensorflow as tf
from tensorflow import keras as K
from koogu.model.trained_model import TrainedModel
from koogu.model import architectures


# class _Classifier:
#     """Base class for defining tensorflow classifier models."""
#
#     def __init__(self, model_name, num_classes, estimator_params, estimator_config, data_format='NHWC',
#                  dtype=tf.float32, **kwargs):
#         """Constructor.
#
#         Args:
#             model_name: Name of the model (eg: ResNET)
#             num_classes: Number of classes used as labels
#             estimator_params: The "params" argument passed to the tensorflow estimator.
#             estimator_config: A tensorflow RunConfig instance passed as the "config" argument to the estimator.
#             data_format: One of 'NCHW' (channel first) or 'NHWC' (channel last; default)
#             dtype: Tensorflow type to use for calculations (default: tf.float32)
#
#         Raises:
#             ValueError
#         """
#
#         self.model_name = model_name
#
#     def save_serving_model(self, serving_input_receiver_fn, outdir=None, assets_extra=None):
#         """Save a trained model for (later) Serving.
#         If outdir is not provided, model will be saved under "model dir"/saved_serving_model/"""
#
#         output_dir = os.path.join(self._estimator.config.model_dir, 'saved_serving_model') if outdir is None else outdir
#         os.makedirs(output_dir, exist_ok=True)
#
#         return self._estimator.export_saved_model(output_dir, serving_input_receiver_fn, assets_extra=assets_extra)
#
#     def _conv2d(self, inputs, num_filters, kernel_size, strides, name_suffix, padding='same'):
#         """Apply 2D convolution. A wrapper around tf.layers.conv2d.
#
#         Args:
#             inputs: Input tensor.
#             num_filters: Number of filters/kernels.
#             kernel_size: Height and width of the kernels, as a tuple. If a scalar is given, the kernel will have same
#                 size along both dimensions.
#             strides: Strides the kernel must make, as a tuple. If a scalar is given, the amount of stride will be the
#                 same along both dimensions.
#             name_suffix: A string that is appended to the node name.
#             padding: One of ['same', 'valid', 'valid-cover']. The first two have the same meaning as in Tensorflow. The
#                 last one is useful where just using 'valid' would not cover all the pixels in the inputs.
#         """
#
#         assert padding in ['same', 'valid', 'valid-cover']
#
#         # If scalars, convert to 2-element iterables
#         kernel_size = np.asarray(kernel_size if isinstance(kernel_size, (list, tuple)) else [kernel_size, kernel_size])
#         strides = np.asarray(strides if isinstance(strides, (list, tuple)) else [strides, strides])
#
#         if padding == 'valid-cover':
#             # This is to be used when reducing spatial dimensions as part of convolutions.
#             # It ensures that pixels at boundaries are properly accounted for when stride > 1.
#
#             spatial_dims = inputs.get_shape().as_list()
#             spatial_dims = np.asarray([spatial_dims[self.spatial_axes[0]], spatial_dims[self.spatial_axes[1]]])
#
#             remainders = spatial_dims - ((np.floor((spatial_dims - kernel_size) / strides) * strides) + kernel_size)
#
#             if np.any(remainders):
#                 additional = np.where(remainders, kernel_size - remainders, [0, 0]).astype(np.int)
#                 pad_amt = np.asarray([[0, 0], [0, 0], [0, 0], [0, 0]])
#                 pad_amt[self.spatial_axes[0], 1] = additional[0]
#                 pad_amt[self.spatial_axes[1], 1] = additional[1]
#
#                 inputs = tf.pad(inputs, pad_amt, mode='CONSTANT', constant_values=0)
#
#             padding = 'valid'   # Now set this to what tensorflow can handle
#
#         return tf.layers.conv2d(inputs,
#                                 filters=num_filters,
#                                 kernel_size=kernel_size.tolist(),
#                                 strides=strides.tolist(),
#                                 padding=padding,
#                                 data_format=self.data_format_str,
#                                 use_bias=False,
#                                 kernel_initializer=tf.variance_scaling_initializer(),
#                                 name='conv2d_{}'.format(name_suffix))
#
#     def model_fn(self, features, feature_infos, mode, params, config):
#         """Populates a tensorflow graph that describes the model, and returns
#         an approptiate EstimatorSpec based on mode."""
#
#         if mode == tf.estimator.ModeKeys.TRAIN and self._optimizer is None:
#             raise ValueError('Optimizer not yet set. Set it using SetOptimizer() before attempting training')
#
#         class_mask_multiplier = None
#         if isinstance(features, dict):  # In serving mode. Segregate the input fields
#             class_mask_multiplier = features['class_mask']
#             features = features['feature']
#
#         if class_mask_multiplier is not None:  # In serving mode
#             # Apply the chosen mask to the leaf nodes of the model
#             logits = logits * class_mask_multiplier
#
#         if len(feature_infos.shape) == 1:   # Tracing info not available
#             labels = feature_infos
#         else:
#             labels = tf.squeeze(tf.slice(feature_infos, [0, 0], [-1, 1]), axis=1)   #feature_infos[:, 0]
#
#         # Add evaluation metrics
#         eval_metric_ops = {"accuracy": accuracy}
#         if self.num_classes >= 5:
#             eval_metric_ops['accuracy_top_5'] = tf.metrics.mean(
#                 tf.nn.in_top_k(targets=labels, predictions=logits, k=5, name='top_5_op'))
#         #eval_metric_ops['avg_prec_at_k'] = tf.metrics.average_precision_at_k(
#         #   labels=tf.cast(labels, tf.int64), predictions=prediction_probabilities, k=1, name='avg_prec_at_k_op')
#
#         if self._debug_mode:
#             with tf.variable_scope('debug_summaries'):
#                 # Save top few misclassified features
#                 fp_idxs = tf.squeeze(tf.where(tf.not_equal(labels, predicted_classes)), axis=1)  # False positives
#                 features = tf.gather(features, fp_idxs, axis=0)
#                 if self.data_format == 'NCHW':
#                     features = tf.transpose(features, [0, 2, 3, 1])  # Make channels last
#                 tf.summary.image('features', tf.reverse(1.0 - features, axis=[1]), max_outputs=5)
#                 num_summaries += 1
#
#     # Add hook to build confusion matrix
#     hooks_list.append(
#         SessionRunHooks.ConfusionMatrixHook(output_dir, self.get_global_step() + 1, self.num_classes,
#                                             labels, predicted_classes))
#
#     if self._record_mismatch_prob_thld is not None and \
#             len(feature_infos.shape) > 1:  # Tracing info available
#         if prediction_probabilities is None:
#             prediction_probabilities = tf.nn.softmax(logits, name='probabilities')
#         hooks_list.append(
#             SessionRunHooks.MisclassificationRecorderHook(
#                 output_dir, self.get_global_step() + 1,
#                 self._record_mismatch_prob_thld,
#                 feature_infos, predicted_classes, prediction_probabilities))
#
#
#
# class Classifier:
#     """A container-like interface for all the inherited classes of _Classifier"""
#     # Add below any new models that you include in the current directory.
#
#     from .resnet import ResNET_Classifier as ResNet
#     from .densenet import DenseNET_Classifier as DenseNet
#     from .convnet import ConvNET_Classifier as ConvNet

def get_model(model_cfg,
              input_shape, num_classes,
              data_format='channels_last',
              dtype=tf.float32, **kwargs):

    supported_data_formats = ('channels_first', 'channels_last')
    if data_format not in supported_data_formats:
        raise ValueError('data_format must be one of {}'.format(
            supported_data_formats))

    try:
        arch_submodule = getattr(architectures, model_cfg['arch'])
    except KeyError as _:
        raise ValueError('Architecture {:s} is not available.'.format(
            model_cfg['arch']))

    arch_fn_kwargs = {'data_format': data_format}
    if 'dropout_rate' in kwargs:
        arch_fn_kwargs['dropout_rate'] = kwargs['dropout_rate']

    # Build the functional model here onwards

    inputs = K.Input(shape=input_shape, dtype=dtype)

    outputs = tf.expand_dims(inputs, axis=3 if data_format == 'channels_last' else 1)

    # Do preprocessing feature transformations (if any)
    for trans_layer in model_cfg['preproc']:
        outputs = trans_layer(outputs)

#        # If "pooling" (by splitting of features) is requested
#        fcn_split_num_pieces = [0, 0]
#        feature_shape = features.get_shape().as_list()
#        feature_shape = [feature_shape[self.spatial_axes[0]], feature_shape[self.spatial_axes[1]]]
#        if self._fcn_patch_size[1] is not None and feature_shape[1] > self._fcn_patch_size[1]:
#            # Split along the time axis and concatenate along the batch axis
#            with tf.variable_scope('FCN_split_t'):
#                feature_patches = tf_transformations.spatial_split(features, 2, self._fcn_patch_size[1],
#                                                                   self._fcn_patch_overlap[1],
#                                                                   self.data_format)
#                fcn_split_num_pieces[1] = len(feature_patches)
#                features = tf.concat(feature_patches, axis=0)
#        if self._fcn_patch_size[0] is not None and feature_shape[0] > self._fcn_patch_size[0]:
#            # Split along the frequency axis and concatenate along the batch axis
#            with tf.variable_scope('FCN_split_f'):
#                feature_patches = tf_transformations.spatial_split(features, 1, self._fcn_patch_size[0],
#                                                                   self._fcn_patch_overlap[0],
#                                                                   self.data_format)
#                fcn_split_num_pieces[0] = len(feature_patches)
#                features = tf.concat(feature_patches, axis=0)

    # Build the model network/graph
    outputs = arch_submodule.build_model(outputs, model_cfg['arch_params'],
                                         **arch_fn_kwargs)

#        # If splitting of features (for "FCN") was done above, "undo" it.
#        if fcn_split_num_pieces[0] > 0:
#            # Split along the batch axis and concatenate along the leaf nodes axis
#            with tf.variable_scope('FCN_join_f'):
#                model_outputs = tf.concat(tf.split(model_outputs, fcn_split_num_pieces[0], axis=0), axis=1)
#        if fcn_split_num_pieces[1] > 0:
#            # Split along the batch axis and concatenate along the leaf nodes axis
#            with tf.variable_scope('FCN_join_t'):
#                model_outputs = tf.concat(tf.split(model_outputs, fcn_split_num_pieces[1], axis=0), axis=1)

    # Add dense layers as requested
    for dense_layer_idx, num_nodes in enumerate(model_cfg['dense_layers']):
        outputs = K.layers.Dense(units=num_nodes, use_bias=False,
                                 name='FC-D{:d}'.format(dense_layer_idx + 1)
                                 )(outputs)
        outputs = K.layers.BatchNormalization(
            scale=False,
            name='BatchNorm-D{:d}'.format(dense_layer_idx + 1))(outputs)
        outputs = K.layers.Activation(
            'relu', name='ReLu-D{:d}'.format(dense_layer_idx + 1))(outputs)

    # Classification layer
    activation_type = 'sigmoid' if model_cfg.get('multilabel', False) else 'softmax'
    outputs = K.layers.Dense(units=num_classes,
                             activation=activation_type,
                             name='Logits'
                             )(outputs)

    model = K.Model(inputs, outputs, name=model_cfg['arch'])

    return model


__all__ = ['TrainedModel', 'architectures', 'get_model']
