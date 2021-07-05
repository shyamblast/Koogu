
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as kl
from tensorflow.keras.initializers import VarianceScaling

# from model import _Classifier
#
#
# class DenseNET_Classifier(_Classifier):
#     """DenseNet [http://arxiv.org/abs/1608.06993]. Supports both with and without bottleneck.
#     Model parameters are to be specified as a dict and must include the following fields:
#         version: (integer) Specify 1 (original) or 2 (with Shyam's modifications).
#         with_bottleneck: (boolean) Whether to use regular blocks or blocks with bottleneck.
#         compression: (float 0.0 < compression <= 1.0) Rate at which to compress num features at the end of each dense
#             block. See subsection titled "Compression" under section 3 in the paper.
#         layers_per_block: (integer list) Number of layers within each dense block.
#         growth_rate: (integer) Number of features added at each layer. See subsection titled "Growth rate" under
#             section 3 in the paper.
#         pooling_type: One of 'max' or 'average' (default) to set the type of pooling in transition layers.
#         pool_sizes: (integer list) Pool size for pooling in the transition layers. List must contain same number of
#             items as layers_per_block. If version is set as 2, pooling will be done implicitly within conv2d.
#         pool_strides: (integer list) Stride size for pooling in the transition layers. List must contain same number
#             of items as layers_per_block. If version is set as 2, pooling will be done implicitly within conv2d.
#         first_conv_filters: (integer, optional) Number of filters in the initial convolution.
#         first_conv_size: (integer, optional, default [3, 3]) Initial convolution kernel size.
#         first_conv_strides: (integer, optional, default [1, 1]) Initial convolution strides.
#         first_pool_size: (integer, optional) Pool size for max-pooling in the first layer. No pooling done if
#             zero or if field not provided.
#         first_pool_strides: (integer, optional) Stride size for max-pooling in the first layer. No effect if
#             first_pool_size is not provided.
#         flatten_leaf_nodes: (bool, optional) If set, and set to True, the leaf nodes will be flattened out instead of
#             computing their mean.
#     """
#
#     def __init__(self, model_params, num_classes, estimator_params, estimator_config, data_format='NHWC',
#                  dtype=tf.float32, **kwargs):
#         """Constructor.
#
#         Args:
#             model_params: (dict) Parameters for building the DenseNet model. See class description for dict fields.
#             num_classes: (integer) Number of classes used as labels.
#             estimator_params: The "params" argument passed to the tensorflow estimator.
#             estimator_config: A tensorflow RunConfig instance passed as the "config" argument to the estimator.
#             data_format: One of 'NCHW' (channel first) or 'NHWC' (channel last; default)
#             dtype: Tensorflow type to use for calculations (default: tf.float32)
#
#         Raises:
#             ValueError
#         """
#
#         model_params_req_keys = ['version',
#                                  'with_bottleneck',
#                                  'compression',
#                                  'layers_per_block',
#                                  'growth_rate',
#                                  'pool_sizes',
#                                  'pool_strides' # ,
#                                  # 'pooling_type',
#                                  # 'first_conv_filters',
#                                  # 'first_conv_size',
#                                  # 'first_conv_strides',
#                                  # 'first_pool_size',
#                                  # 'first_pool_strides'
#                                  ]
#         if not all([(req_key in model_params) for req_key in model_params_req_keys]):
#             raise ValueError('One or more required parameters missing in model_params')
#
#         if model_params['version'] not in (1, 2):
#             raise ValueError('DenseNet version can only be either 1 or 2')
#         if len(model_params['pool_sizes']) != len(model_params['layers_per_block']) or \
#                 len(model_params['pool_strides']) != len(model_params['layers_per_block']):
#             raise ValueError('pool_sizes and pool_strides must have same number of items as layers_per_block')
#         if model_params['compression'] <= 0.0 or model_params['compression'] > 1.0:
#             raise ValueError('compression must be in the range 0.0 < compression <= 1.0')
#
#         self.version = int(model_params['version'])
#         self.with_bottleneck = bool(model_params['with_bottleneck'])
#         self.compression = float(model_params['compression'])
#         self.k = int(model_params['growth_rate'])     # growth rate
#         self.num_blocks = len(model_params['layers_per_block'])  # Num blocks
#         self.layers_per_block = model_params['layers_per_block']
#         self.pooling = \
#             tf.layers.max_pooling2d if 'pooling_type' in model_params and model_params['pooling_type'] == 'max' \
#             else tf.layers.average_pooling2d
#         self.pool_sizes = model_params['pool_sizes']
#         self.pool_strides = model_params['pool_strides']
#         if 'first_conv_filters' in model_params:
#             self.first_conv_filters = model_params['first_conv_filters']
#             self.first_conv_size = model_params['first_conv_size'] if 'first_conv_size' in model_params else 3
#             self.first_conv_strides = model_params['first_conv_strides'] if 'first_conv_strides' in model_params else 1
#         else:
#             self.first_conv_filters = None
#             self.first_conv_size = 3
#             self.first_conv_strides = 1
#         self.first_pool_size = model_params['first_pool_size'] if 'first_pool_size' in model_params else None
#         self.first_pool_strides = model_params['first_pool_strides'] if 'first_pool_strides' in model_params else None
#         self._flatten_leaf_nodes = model_params['flatten_leaf_nodes'] if 'flatten_leaf_nodes' in model_params else False
#
#         # If scalars, convert to 2-element iterables
#         if self.first_pool_size is not None:
#             self.first_pool_size = np.asarray(self.first_pool_size if isinstance(self.first_pool_size, (list, tuple))
#                                               else [self.first_pool_size, self.first_pool_size])
#         if self.first_pool_strides is not None:
#             self.first_pool_strides = np.asarray(self.first_pool_strides if isinstance(self.first_pool_strides, (list, tuple))
#                                                  else [self.first_pool_strides, self.first_pool_strides])
#
#         model_name = self.__class__.__name__ + '_v' + str(self.version)
#         model_name_addition = '-'
#         if self.with_bottleneck:
#             model_name_addition += 'B'
#         if self.compression < 1.0:
#             model_name_addition += 'C'
#         model_name += model_name_addition if model_name_addition != '-' else ''
#
#         # Invoke the parent constructor
#         super(DenseNET_Classifier, self).__init__(
#             model_name=model_name,
#             num_classes=num_classes,
#             estimator_params=estimator_params,
#             estimator_config=estimator_config,
#             data_format=data_format,
#             dtype=dtype,
#             **kwargs)
#
#     def _batchnorm_activation_conv2d(self, inputs, is_training, num_filters, kernel_size, strides, padding, name_suffix):
#         outputs = self._batch_norm_and_relu(inputs, is_training, do_relu=True,
#                                             name_suffix=name_suffix)
#         outputs = self._conv2d(outputs, num_filters=num_filters, kernel_size=kernel_size, strides=strides,
#                                padding=padding, name_suffix=name_suffix)
#
#         if self._dropout_rate:
#             outputs = tf.layers.dropout(outputs, rate=self._dropout_rate, training=is_training)
#
#         return outputs
#
#     def _dense_block(self, inputs, num_layers_in_block, is_training, concat_axis):
#
#         outputs = tf.identity(inputs, name='input')
#
#         for layer in range(num_layers_in_block):
#             if self.with_bottleneck:
#                 with tf.variable_scope('bottleneck_%i' % (layer + 1)):
#                     layer_outputs = self._batchnorm_activation_conv2d(outputs, is_training,
#                                                                       num_filters=self.k * 4,
#                                                                       kernel_size=[1, 1], strides=[1, 1],
#                                                                       padding='same',
#                                                                       name_suffix=(layer + 1))
#             else:
#                 layer_outputs = outputs
#
#             with tf.variable_scope('composite_func_%i' % (layer + 1)):
#                 layer_outputs = self._batchnorm_activation_conv2d(layer_outputs, is_training,
#                                                                   num_filters=self.k,
#                                                                   kernel_size=[3, 3], strides=[1, 1],
#                                                                   padding='same',
#                                                                   name_suffix=(layer + 1))
#
#             outputs = tf.concat([outputs, layer_outputs], axis=concat_axis, name='concat_%i' % (layer + 1))
#
#         return outputs
#
#     def _build_model(self, inputs, is_training):
#
#         outputs = inputs
#
#         # Initial convolution, if enabled.
#         if self.first_conv_filters is not None or (self.with_bottleneck and self.compression < 1.0):
#             outputs = self._conv2d(outputs,
#                                    (2 * self.k) if (self.with_bottleneck and self.compression < 1.0) else self.first_conv_filters,
#                                    kernel_size=self.first_conv_size, strides=self.first_conv_strides,
#                                    name_suffix='pre')
#
#         # Initial pooling
#         if self.first_pool_size is not None and self.first_pool_strides is not None:
#             outputs = tf.layers.max_pooling2d(outputs, pool_size=self.first_pool_size,
#                                               strides=self.first_pool_strides, padding='same',
#                                               data_format=self.data_format_str,
#                                               name='initial_pooling')
#
#         chnl_axis_tf = tf.constant(self.channel_axis, dtype=tf.int32, name='channel_axis')
#
#         # Add N dense blocks, succeeded by transition layers as applicable
#         for block_idx in range(self.num_blocks):
#             # Dense block
#             with tf.variable_scope('Dense_block_%i' % (block_idx + 1)):
#                 outputs = self._dense_block(outputs, self.layers_per_block[block_idx], is_training, chnl_axis_tf)
#
#             # Transition layer. If not ver 2, add transition layers for all but the last dense block
#             if block_idx < self.num_blocks - 1 or self.version == 2:
#                 # Transition layer
#                 with tf.variable_scope('Transition_layer_%i' % (block_idx + 1)):
#                     if self.compression < 1.0:  # if compression is enabled
#                         num_features = int(outputs.get_shape().as_list()[self.channel_axis] * self.compression)
#                     else:
#                         num_features = outputs.get_shape().as_list()[self.channel_axis]
#
#                     if self.version == 1:
#                         outputs = self._batchnorm_activation_conv2d(outputs, is_training,
#                                                                     num_filters=num_features,
#                                                                     kernel_size=[1, 1], strides=[1, 1],
#                                                                     padding='valid',
#                                                                     name_suffix=1)
#                         outputs = self.pooling(outputs, pool_size=self.pool_sizes[block_idx],
#                                                strides=self.pool_strides[block_idx], padding='same',
#                                                data_format=self.data_format_str)
#                     else:   # If version is 2, do implicit pooling
#                         outputs = self._batchnorm_activation_conv2d(outputs, is_training,
#                                                                     num_filters=num_features,
#                                                                     kernel_size=self.pool_sizes[block_idx],
#                                                                     strides=self.pool_strides[block_idx],
#                                                                     padding='valid-cover',
#                                                                     name_suffix=1)
#
#         # Final batch_norm & activation
#         outputs = self._batch_norm_and_relu(outputs, is_training, do_relu=True, name_suffix='final')
#
#         # Pooling or flattening
#         if self._flatten_leaf_nodes:    # if flattening is enabled
#             outputs = tf.layers.flatten(outputs, data_format=self.data_format_str)
#         else:
#             # This is the default - take mean
#             outputs = tf.reduce_mean(outputs, axis=self.spatial_axes, keepdims=None)
#
#         return outputs


def build_model(inputs, arch_params, **kwargs):
    """
    Build a DenseNet model, with requested customizations.
    """

    data_format = 'channels_last' if 'data_format' not in kwargs else kwargs['data_format']
    dropout_rate = 0.0 if 'dropout_rate' not in kwargs else kwargs.pop('dropout_rate')

    channel_axis = 3 if data_format == 'channels_last' else 1

    # Parameters configurable as per the DenseNET paper, with some reasonable defaults
    growth_rate = arch_params.get('growth_rate', 12)
    with_bottleneck = arch_params.get('with_bottleneck', False)
    compression = arch_params.get('compression', 1.0)
    # Parameters that are my additions
    quasi_dense = arch_params.get('quasi_dense', False)
    implicit_pooling = arch_params.get('implicit_pooling', False)

    pooling = kl.MaxPooling2D if arch_params.get('pooling_type', 'avg') == 'max' \
        else kl.AveragePooling2D

    pool_sizes = arch_params.get('pool_sizes', [(3, 3)] * (3 + implicit_pooling))
    pool_strides = arch_params.get('pool_strides', pool_sizes)

    def composite_fn(cf_inputs, num_filters, kernel_size, strides, padding, cf_idx, n_pre=''):
        name_prefix = n_pre + 'CF{}_'.format(cf_idx)

        cf_outputs = kl.BatchNormalization(axis=channel_axis, fused=True, scale=False,
                                           name=name_prefix + 'BatchNorm')(cf_inputs)
        cf_outputs = kl.Activation('relu', name=name_prefix + 'ReLu')(cf_outputs)

        if padding == 'valid':
            # Ensure that pixels at boundaries are properly accounted for when stride > 1.
            cf_outputs = pad_for_valid_conv(cf_outputs, kernel_size, strides, data_format)

        cf_outputs = kl.Conv2D(filters=num_filters, kernel_size=kernel_size, strides=strides,
                               padding=padding, use_bias=False, data_format=data_format,
                               kernel_initializer=VarianceScaling(),
                               name=name_prefix + 'Conv2D')(cf_outputs)

        if dropout_rate > 0.0:
            cf_outputs = kl.Dropout(dropout_rate, name=name_prefix + 'Dropout')(cf_outputs)

        return cf_outputs

    def dense_block(db_inputs, num_layers_in_block, b_idx):

        name_prefix = 'B{:d}_'.format(b_idx)

        db_outputs = [db_inputs]

        for layer in range(num_layers_in_block):
            if not quasi_dense and len(db_outputs) > 1:
                db_outputs = [kl.Concatenate(axis=channel_axis,
                                             name=name_prefix + 'Concat{:d}'.format(layer + 1))(db_outputs)]

            if with_bottleneck:
                layer_outputs = composite_fn(db_outputs[-1], growth_rate * 4, [1, 1], [1, 1], 'same',
                                             '-BtlNk{:d}'.format(layer + 1), name_prefix)
            else:
                layer_outputs = db_outputs[-1]

            layer_outputs = composite_fn(layer_outputs, growth_rate, [3, 3], [1, 1], 'same',
                                         (layer + 1), name_prefix)

            db_outputs.append(layer_outputs)

        return kl.Concatenate(axis=channel_axis, name=name_prefix + 'Concat')(db_outputs)

    outputs = inputs

    # Initial convolution, if enabled.
    if arch_params['first_conv_filters'] is not None or (with_bottleneck and compression < 1.0):
        outputs = kl.Conv2D(filters=((2 * growth_rate) if (with_bottleneck and compression < 1.0)
                                     else arch_params['first_conv_filters']),
                            kernel_size=arch_params.get('first_conv_size', 3),
                            strides=arch_params.get('first_conv_strides', 1),
                            padding='same', use_bias=False, data_format=data_format,
                            kernel_initializer=VarianceScaling(),
                            name='PreConv2D')(outputs)

    # Initial pooling
    if arch_params.get('first_pool_size', None) is not None and arch_params.get('first_pool_strides', None) is not None:
        outputs = kl.MaxPooling2D(pool_size=arch_params['first_pool_size'],
                                  strides=arch_params['first_pool_strides'],
                                  padding='same', data_format=data_format,
                                  name='initial_pooling')(outputs)

    # Add N dense blocks, succeeded by transition layers as applicable
    for block_idx, num_layers in enumerate(arch_params['layers_per_block']):
        # Dense block
        outputs = dense_block(outputs, num_layers, block_idx + 1)

        # Transition layer.
        # If implicit_pooling is set, add transition layers for all dense blocks. Otherwise,
        # add transition layers for all but the last dense block.
        if block_idx < len(arch_params['layers_per_block']) - 1 or implicit_pooling:
            # Transition layer
            if compression < 1.0:  # if compression is enabled
                num_features = int(outputs.get_shape().as_list()[channel_axis] * compression)
            else:
                num_features = outputs.get_shape().as_list()[channel_axis]

            if implicit_pooling:    # Achieve pooling by strided convolutions
                outputs = composite_fn(outputs, num_features,
                                       pool_sizes[block_idx],
                                       pool_strides[block_idx],
                                       'valid', '', n_pre='T{:d}_'.format(block_idx + 1))
            else:
                outputs = composite_fn(outputs, num_features, [1, 1], [1, 1], 'valid', '',
                                       n_pre='T{:d}_'.format(block_idx + 1))
                # Ensure that pixels at boundaries are properly accounted for when stride > 1.
                outputs = pad_for_valid_conv(outputs,
                                             pool_sizes[block_idx],
                                             pool_strides[block_idx],
                                             data_format)
                outputs = pooling(pool_size=pool_sizes[block_idx],
                                  strides=pool_strides[block_idx],
                                  padding='valid', data_format=data_format,
                                  name='T{:d}_Pool'.format(block_idx + 1))(outputs)

    # Final batch_norm & activation
    outputs = kl.BatchNormalization(axis=channel_axis, fused=True, scale=False, epsilon=1e-8)(outputs)
    outputs = kl.Activation('relu', name='ReLu')(outputs)

    # Pooling or flattening
    if arch_params.get('flatten_leaf_nodes', False):  # if flattening is enabled, default is False
        outputs = kl.Flatten(data_format=data_format)(outputs)
    else:
        # This is the default - take global mean
        outputs = kl.GlobalAveragePooling2D(data_format=data_format)(outputs)

    return outputs


def pad_for_valid_conv(inputs, kernel_shape, strides, data_format):
    # Ensure that pixels at boundaries are properly accounted for when stride > 1.

    f_axis, t_axis = (1, 2) if data_format == 'channels_last' else (2, 3)

    feature_dims = inputs.get_shape().as_list()
    outputs = inputs

    spatial_dims = np.asarray([feature_dims[f_axis], feature_dims[t_axis]])
    remainders = spatial_dims - (
            (np.floor((spatial_dims - kernel_shape) /
             strides) * strides) +
            kernel_shape)
    if np.any(remainders):
        additional = np.where(remainders, kernel_shape - remainders, [0, 0]).astype(np.int)
        pad_amt = np.asarray([[0, 0], [0, 0], [0, 0], [0, 0]])
        pad_amt[f_axis, 1] = additional[0]
        pad_amt[t_axis, 1] = additional[1]
        #print('Pad amount {} for feature dims {}'.format(pad_amt, feature_dims))
        outputs = tf.pad(outputs, pad_amt, mode='CONSTANT', constant_values=0)

    return outputs
