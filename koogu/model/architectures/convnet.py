from tensorflow.keras import layers as kl
from tensorflow.keras.initializers import VarianceScaling


def build_model(inputs, arch_params, **kwargs):
    """
    Build a ConvNet model, with requested customizations.
    """

    data_format = kwargs.get('data_format', 'channels_last')
    dropout_rate = kwargs.get('dropout_rate', 0.0)

    channel_axis = 3 if data_format == 'channels_last' else 1

    pooling = kl.AveragePooling2D if arch_params.get('pooling_type', 'max') == 'avg' \
        else kl.MaxPooling2D

    filters_per_layer = arch_params.get('filters_per_layer', [32, 64])
    pool_sizes = arch_params.get('pool_sizes', [(2, 2)] * len(filters_per_layer))
    pool_strides = arch_params.get('pool_strides', pool_sizes)

    add_batchnorm = arch_params.get('add_batchnorm', False)

    outputs = inputs

    for l_idx, (num_filters, pool_size, pool_stride) in enumerate(zip(filters_per_layer, pool_sizes, pool_strides)):
        outputs = kl.Conv2D(
            filters=num_filters, kernel_size=(3, 3), strides=(1, 1),
            padding='same', use_bias=False, data_format=data_format,
            kernel_initializer=VarianceScaling(),
            name='Conv2D_{:d}'.format(l_idx+1))(outputs)

        if dropout_rate > 0.0:
            outputs = kl.Dropout(dropout_rate, name='Dropout_{:d}'.format(l_idx+1))(outputs)

        if add_batchnorm:
            outputs = kl.BatchNormalization(axis=channel_axis, fused=True, scale=False,
                                            name='BatchNorm_{:d}'.format(l_idx+1))(outputs)

        outputs = kl.Activation('relu', name='ReLu_{:d}'.format(l_idx+1))(outputs)

        outputs = pooling(pool_size=pool_size,
                          strides=pool_stride,
                          padding='valid', data_format=data_format,
                          name='Pool_{:d}'.format(l_idx+1))(outputs)

    outputs = kl.Flatten(data_format=data_format)(outputs)

    return outputs
