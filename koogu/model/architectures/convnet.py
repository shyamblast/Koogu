import tensorflow as tf
from . import KooguArchitectureBase


class Architecture(KooguArchitectureBase):
    """
    Boilerplate ConvNet network-building logic that can be used, with
    appropriate customization of parameters, to build networks like LeNet,
    AlexNet, etc.
    """

    def __init__(self, filters_per_layer, **kwargs):

        if not hasattr(filters_per_layer, '__len__'):
            raise ValueError('filters_per_layer must be a list or tuple')

        params = {k: v for k, v in kwargs.items()}   # make a copy

        # Architecture parameters, with some reasonable defaults
        arch_config = dict(
            filters_per_layer=filters_per_layer,
            pool_sizes=params.pop(
                'pool_sizes', [(2, 2)] * (len(filters_per_layer))),

            # Parameters that are my own additions
            add_batchnorm=params.pop('add_batchnorm', False),
            pooling_type=params.pop('pooling_type', 'avg')
        )

        # Derived values
        arch_config['pool_strides'] = params.pop(
            'pool_strides', arch_config['pool_sizes'])

        model_name = params.pop('name', 'ConvNet')
        super(Architecture, self).__init__(
            arch_config, is_2d=True, name=model_name, **params)

    def build_architecture(self, inputs, is_training, data_format, **kwargs):

        dropout_rate = kwargs.get('dropout_rate', 0.0) if is_training else 0.0
        channel_axis = 3 if data_format == 'channels_last' else 1

        arch_config = self.config

        pooling = tf.keras.layers.MaxPooling2D \
            if arch_config['pooling_type'] == 'max' \
            else tf.keras.layers.AveragePooling2D

        outputs = inputs

        for l_idx, (num_filters, pool_size, pool_stride) in \
                enumerate(zip(arch_config['filters_per_layer'],
                              arch_config['pool_sizes'],
                              arch_config['pool_strides'])):
            outputs = tf.keras.layers.Conv2D(
                filters=num_filters, kernel_size=(3, 3), strides=(1, 1),
                padding='same', use_bias=False, data_format=data_format,
                kernel_initializer=tf.keras.initializers.VarianceScaling(),
                name=f'Conv2D_{l_idx + 1:d}')(outputs)

            if dropout_rate > 0.0:
                outputs = tf.keras.layers.Dropout(
                    dropout_rate, name=f'Dropout_{l_idx + 1:d}')(outputs)

            if arch_config['add_batchnorm']:
                outputs = tf.keras.layers.BatchNormalization(
                    axis=channel_axis, fused=True, scale=False,
                    name=f'BatchNorm_{l_idx + 1:d}')(outputs)

            outputs = tf.keras.layers.Activation(
                'relu', name=f'ReLu_{l_idx + 1:d}')(outputs)

            outputs = pooling(pool_size=pool_size,
                              strides=pool_stride,
                              padding='valid', data_format=data_format,
                              name=f'Pool_{l_idx + 1:d}')(outputs)

        outputs = tf.keras.layers.Flatten(data_format=data_format)(outputs)

        return outputs


__all__ = ['Architecture']
