import tensorflow as tf
from . import KooguArchitectureBase


@KooguArchitectureBase.export
class ConvNet(KooguArchitectureBase):
    """
    Boilerplate ConvNet network-building logic that can be used, with
    appropriate customization of parameters, to build networks like LeNet,
    AlexNet, etc.

    :param filters_per_layer: (list/tuple of ints) The length of the list/tuple
        defines the depth of the network and each value in it specifies the
        number of filters at the corresponding level.
    :param pool_sizes: (optional) Must be a list of 2-element tuples (of ints)
        specifying the factors by which to downscale (vertical, horizontal)
        following each convolution. The length of the list must be the same as
        that of ``filters_per_layer``. By default, a pool size of (2, 2) is
        considered throughout.
    :param pool_strides: (optional; defaults to whatever ``pool_sizes`` is) Must
        be of similar structure as ``pool_sizes``, and will define the strides
        that the pooling operation takes along the horizontal and vertical
        directions.

    **Other helpful customizations**

    :param add_batchnorm: (bool; default: False) If True, batch normalization
        layers will be added following each convolution layer.
    :param pooling_type: (optional) By default, average pooling is performed.
        Set to 'max' to use max pooling instead.

    **Koogu-style model customizations**

    :param preproc: (optional) Use this to add pre-convolution operations to the
        model. If specified, must be a list of 2-element tuples, with each tuple
        containing -

        * the name of the operation (either a compatible Keras layer or a
          transformation from :mod:`koogu.data.tf_transformations`).
        * a Python dictionary specifying parameters to the operation.
    :param dense_layers: (optional) Use this to add fully-connected (dense)
        layers to the end of the model network. Can specify a single integer
        (the added layer will have as many nodes) or a list of integers to add
        multiple (connected in sequence) dense layers.
    :param add_dense_layer_nonlinearity: (boolean; default: False) If True, will
        apply ReLU activation to the outputs of the BatchNormalization layer
        following each added dense layer (as per `dense_layers`).
    :param data_format: One of 'channels_last' (default) or 'channels_first'.
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
        super(ConvNet, self).__init__(
            arch_config, is_2d=True, name=model_name, **params)

    def build_architecture(self, inputs, is_training, **kwargs):

        dropout_rate = kwargs.get('dropout_rate', 0.0) if is_training else 0.0
        channel_axis = 3 if self._data_format == 'channels_last' else 1

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
                padding='same', use_bias=False, data_format=self._data_format,
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
                              padding='valid', data_format=self._data_format,
                              name=f'Pool_{l_idx + 1:d}')(outputs)

        outputs = tf.keras.layers.Flatten(
            data_format=self._data_format)(outputs)

        return outputs


__all__ = ['ConvNet']
