import tensorflow as tf
from . import KooguArchitectureBase


@KooguArchitectureBase.export
class DenseNet(KooguArchitectureBase):
    """
    DenseNet [http://arxiv.org/abs/1608.06993].
    This implementation supports both with and without bottleneck.


    :param layers_per_block: (list/tuple of ints) The length of the list/tuple
        defines the number of dense-blocks in the network and each value in the
        list specifies the number of composite function layers
        (made up of BatchNorm-Conv-ReLU) in the corresponding dense-block.
    :param growth_rate: (optional; default: 12) Number of composite function
        layers per dense-block.
    :param compression: (optional; default: 1.0) Specifies the rate of
        compression applied in transition blocks. A value of 1.0 means no
        compression; specify value < 1.0 to bring about compression.
    :param with_bottleneck: (bool; default: False) Weather to include bottleneck
        layers.

    **Other helpful customizations**

    :param quasi_dense: (bool; default: False) If True, feed-forward connections
        within a dense-block will be reduced, as described in
        `Madhusudhana et. al. 2021 <https://doi.org/10.1098/rsif.2021.0297>`_.
    :param pooling_type: (optional) By default, average pooling is performed.
        Set to 'max' to use max pooling instead.
    :param pool_sizes: (optional) Must be a list of 2-element tuples (of ints)
        specifying the factors by which to downscale (vertical, horizontal)
        in each transition block. The length of the list must be one less than
        that of ``layers_per_block``. By default, a pool size of (3, 3) is
        considered throughout.
    :param pool_strides: (optional; defaults to whatever ``pool_sizes`` is) Must
        be of similar structure as ``pool_sizes``, and will define the strides
        that the pooling operation takes along the horizontal and vertical
        directions.

    **Koogu-style model customizations**

    :param preproc: (optional) Use this to add pre-convolution operations to the
        model. If specified, must be a list of 2-element tuples, with each tuple
        containing -

        * the name of the operation (either a compatible Keras layer or a
          transformation from :mod:`koogu.data.tf_transformations`.
        * a Python dictionary specifying parameters to the operation.
    :param dense_layers: (optional) Use this to add fully-connected (dense)
        layers to the end of the model network. Can specify a single integer
        (the added layer will have as many nodes) or a list of integers to add
        multiple (connected in sequence) dense layers.
    """

    def __init__(self, layers_per_block, **kwargs):

        if not hasattr(layers_per_block, '__len__'):
            raise ValueError('layers_per_block must be a list or tuple')

        params = {k: v for k, v in kwargs.items()}   # make a copy

        # Architecture parameters as per the DenseNet paper, with some
        # reasonable defaults
        arch_config = dict(
            layers_per_block=layers_per_block,
            growth_rate=params.pop('growth_rate', 12),
            with_bottleneck=params.pop('with_bottleneck', False),
            compression=params.pop('compression', 1.0),

            # Parameters that are my own additions
            quasi_dense=params.pop('quasi_dense', False),
            implicit_pooling=params.pop('implicit_pooling', False),
            pooling_type=params.pop('pooling_type', 'avg'),
            flatten_leaf_nodes=params.pop('flatten_leaf_nodes', False)
        )

        # Derived values
        arch_config['pool_sizes'] = params.pop(
            'pool_sizes', [(3, 3)] * (
                len(layers_per_block) - 1 + arch_config['implicit_pooling']))
        arch_config['pool_strides'] = params.pop(
            'pool_strides', arch_config['pool_sizes'])

        super(DenseNet, self).__init__(
            arch_config, is_2d=True, name='DenseNet', **params)

    def build_architecture(self, inputs, is_training, data_format, **kwargs):

        dropout_rate = kwargs.get('dropout_rate', 0.0) if is_training else 0.0
        channel_axis = 3 if data_format == 'channels_last' else 1

        arch_config = self.config

        pooling = tf.keras.layers.MaxPooling2D \
            if arch_config['pooling_type'] == 'max' \
            else tf.keras.layers.AveragePooling2D

        def composite_fn(cf_inputs, num_filters, kernel_size, strides,
                         padding, cf_idx, n_pre=''):

            name_prefix = n_pre + 'CF{}_'.format(cf_idx)

            cf_outputs = tf.keras.layers.BatchNormalization(
                axis=channel_axis, fused=True, scale=False,
                name=name_prefix + 'BatchNorm')(cf_inputs)
            cf_outputs = tf.keras.layers.Activation(
                'relu', name=name_prefix + 'ReLu')(cf_outputs)

            if padding == 'valid':
                # Ensure that pixels at boundaries are properly accounted for
                # when stride > 1.
                cf_outputs = KooguArchitectureBase.pad_for_valid_conv2d(
                    cf_outputs, kernel_size, strides, data_format)

            cf_outputs = tf.keras.layers.Conv2D(
                filters=num_filters, kernel_size=kernel_size, strides=strides,
                padding=padding, use_bias=False, data_format=data_format,
                kernel_initializer=tf.keras.initializers.VarianceScaling(),
                name=name_prefix + 'Conv2D')(cf_outputs)

            if dropout_rate > 0.0:
                cf_outputs = tf.keras.layers.Dropout(
                    dropout_rate, name=name_prefix + 'Dropout')(cf_outputs)

            return cf_outputs

        def dense_block(db_inputs, num_layers_in_block, b_idx):

            name_prefix = 'B{:d}_'.format(b_idx)

            db_outputs = [db_inputs]

            for layer in range(num_layers_in_block):
                if not arch_config['quasi_dense'] and len(db_outputs) > 1:
                    db_outputs = [
                        tf.keras.layers.Concatenate(
                            axis=channel_axis,
                            name=name_prefix + 'Concat{:d}'.format(layer + 1)
                        )(db_outputs)
                    ]

                if arch_config['with_bottleneck']:
                    layer_outputs = composite_fn(
                        db_outputs[-1], arch_config['growth_rate'] * 4,
                        [1, 1], [1, 1], 'same',
                        '-BtlNk{:d}'.format(layer + 1), name_prefix)
                else:
                    layer_outputs = db_outputs[-1]

                layer_outputs = composite_fn(
                    layer_outputs, arch_config['growth_rate'],
                    [3, 3], [1, 1], 'same', (layer + 1), name_prefix)

                db_outputs.append(layer_outputs)

            return tf.keras.layers.Concatenate(
                axis=channel_axis, name=name_prefix + 'Concat')(db_outputs)

        outputs = inputs

        # Add N dense blocks, succeeded by transition layers as applicable
        for block_idx, num_layers in enumerate(
                arch_config['layers_per_block']):
            # Dense block
            outputs = dense_block(outputs, num_layers, block_idx + 1)

            # Transition layer.
            # If implicit_pooling is set, add transition layers for all dense
            # blocks. Otherwise, add transition layers for all but the last
            # dense block.
            if block_idx < len(arch_config['layers_per_block']) - 1 or \
                    arch_config['implicit_pooling']:

                # Transition layer
                if arch_config['compression'] < 1.0:  # compression is enabled
                    num_features = int(
                        arch_config['compression'] *
                        outputs.get_shape().as_list()[channel_axis])
                else:
                    num_features = outputs.get_shape().as_list()[channel_axis]

                if arch_config['implicit_pooling']:
                    # Achieve pooling by strided convolutions
                    outputs = composite_fn(
                        outputs, num_features,
                        arch_config['pool_sizes'][block_idx],
                        arch_config['pool_strides'][block_idx],
                        'valid', '', n_pre='T{:d}_'.format(block_idx + 1))
                else:
                    outputs = composite_fn(
                        outputs, num_features, [1, 1], [1, 1], 'valid', '',
                        n_pre='T{:d}_'.format(block_idx + 1))
                    # Ensure that pixels at boundaries are properly accounted
                    # for when stride > 1.
                    outputs = KooguArchitectureBase.pad_for_valid_conv2d(
                        outputs,
                        arch_config['pool_sizes'][block_idx],
                        arch_config['pool_strides'][block_idx],
                        data_format)
                    outputs = pooling(
                        pool_size=arch_config['pool_sizes'][block_idx],
                        strides=arch_config['pool_strides'][block_idx],
                        padding='valid', data_format=data_format,
                        name='T{:d}_Pool'.format(block_idx + 1))(outputs)

        # Final batch_norm & activation
        outputs = tf.keras.layers.BatchNormalization(
            axis=channel_axis, fused=True, scale=False, epsilon=1e-8)(outputs)
        outputs = tf.keras.layers.Activation('relu', name='ReLu')(outputs)

        # Pooling or flattening
        if arch_config['flatten_leaf_nodes']:  # if flattening was enabled
            outputs = tf.keras.layers.Flatten(
                data_format=data_format)(outputs)
        else:
            # This is the default - take global mean
            outputs = tf.keras.layers.GlobalAveragePooling2D(
                data_format=data_format)(outputs)

        return outputs


__all__ = ['DenseNet']
