import numpy as np
import tensorflow as tf
import abc
from koogu.data import tf_transformations


class BaseArchitecture(metaclass=abc.ABCMeta):
    """
    Base class for implementing custom user-defined architectures.

    :param multilabel: (bool; default: True) Set appropriately so that the
        loss function and accuracy metrics can be chosen correctly. A multilabel
        model's Logits (final) layer will have Sigmoid activation whereas a
        single-label model's will have SoftMax activation.
    :param dtype: Tensorflow data type of the model's weights (default:
        tf.float32).
    :param name: Name of the model.
    """

    def __init__(self, multilabel=True, dtype=None, name=None):

        self._multilabel = multilabel
        self._dtype = dtype or tf.float32
        self._name = name

    @abc.abstractmethod
    def build_network(self, input_tensor, is_training, **kwargs):
        """
        This method must be implemented in the derived class.

        It should contain logic to construct the desired sequential or
        functional model network starting from the ``input_tensor``.

        .. Note::
            Do not add the Logits layer in your implementation. It will be
            added by internal code.

        :param input_tensor: The Keras tensor to use as the input placeholder
            in model that will be built.
        :param is_training: (boolean) Indicates if operating in training mode.
            Certain elements of the network (e.g., dropout layers) may be
            excluded when not in training mode.

        :return: Must return a Keras tensor corresponding to outputs of the
            architecture.
        """

        raise NotImplementedError(
            'build_network() method not implemented in derived class')

    def __call__(self, input_shape, num_classes, is_training,
                 **kwargs):
        """
        Creates a Keras tensor with input_shape and passes it, along with
        other relevant parameters as-is, to build_network() of inherited
        class. Then adds the final Logits layer before creating the
        tf.keras.Model instance with the full network.

        :param input_shape: Shape of the input tensor (specified without the
            batch dimension here).
        :param num_classes: Number of classes; dictates the number of nodes
            (Logits) that the final layer must have.
        :param is_training: (boolean) Indicates if operating in training mode.

        :return: A tf.keras.Model

        :meta private:
        """

        # -- Build the functional model --

        # Input placeholder
        inputs = tf.keras.Input(shape=input_shape, dtype=self._dtype)
        outputs = inputs

        # Build the desired network in the inherited class
        outputs = self.build_network(outputs, is_training,
                                     **kwargs)

        # Classification layer
        activation_type = 'sigmoid' if self._multilabel else 'softmax'
        outputs = tf.keras.layers.Dense(units=num_classes,
                                        activation=activation_type,
                                        dtype=self._dtype,
                                        name='Logits'
                                        )(outputs)

        model_kwargs = {'name': self._name} if self._name is not None else {}
        return tf.keras.Model(inputs, outputs, **model_kwargs)

    @property
    def multilabel(self):
        return self._multilabel


class KooguArchitectureBase(BaseArchitecture):
    """
    Base class for architectures implemented internally within Koogu.

    Optional parameters

    :param dense_layers: If an integer, a single fully-connected layer with as
        many nodes will be added at the end of the network. If a list, then as
        many fully-connected layers will be added as there are items in the list
        and each layer will have as many nodes as the corresponding integer item
        in the list.
    :param add_dense_layer_nonlinearity: (boolean; default: False) If True, will
        apply ReLU activation to the outputs of the BatchNormalization layer
        following each added dense layer (as per `dense_layers`).
    :param data_format: One of 'channels_last' (default) or 'channels_first'.

    :meta private:
    """

    def __init__(self, arch_config,
                 is_2d=True, multilabel=True, dtype=None, name=None,
                 **kwargs):
        super(KooguArchitectureBase, self).__init__(
            multilabel, dtype, name)

        self._is_2d = is_2d
        self._arch_config = arch_config

        self._dense_layers = \
            kwargs.get('dense_layers', [])  # default to an empty list
        if not hasattr(self._dense_layers, '__len__'):
            self._dense_layers = [self._dense_layers]

        self._dense_layers_activation = \
            bool(kwargs.get('add_dense_layer_nonlinearity', False))

        self._data_format = kwargs.get('data_format', 'channels_last')
        assert self._data_format in ['channels_first', 'channels_last'], \
            '\'data_format\' must be one of \'channels_first\' or ' + \
            '\'channels_last\''

        # Default to an empty list. Otherwise, save instantiated objects of
        # classes that are instances of tf.keras.layers.Layer.
        self._preprocs = [
            self._get_preproc(preproc_item)
            for preproc_item in kwargs.get('preproc', [])]

    @property
    def config(self):
        """Architecture configuration parameters"""
        return {k: v for k, v in self._arch_config.items()}  # return a copy

    def build_network(self, input_tensor, is_training, **kwargs):
        """
        Adds Koogu-specific bells & whistles around the architecture (which
        will be created by the inherited class).
        NOTE: Do not override this method in an inherited class.

        :param input_tensor: The Keras tensor to use as the input placeholder
            in model that will be built.
        :param is_training: Boolean, indicating if operating in training mode.
            Certain elements of the network (e.g., dropout layers) may be
            excluded when in training mode.

        :return: A Keras tensor corresponding to outputs of the architecture.
        """

        outputs = input_tensor

        if self._is_2d:
            # Add the channel axis
            outputs = tf.expand_dims(
                outputs, axis=3 if self._data_format == 'channels_last' else 1)

        # Add preprocessing operations (if any)
        for pp_item in self._preprocs:
            outputs = pp_item(outputs)

        # Build the custom architecture
        outputs = self.build_architecture(outputs, is_training,
                                          **kwargs)

        # Add dense layers as requested
        for dl_idx, num_nodes in enumerate(self._dense_layers):
            outputs = tf.keras.layers.Dense(units=num_nodes, use_bias=False,
                                            dtype=self._dtype,
                                            name=f'FC-D{dl_idx + 1:d}'
                                            )(outputs)
            outputs = tf.keras.layers.BatchNormalization(
                scale=False, epsilon=1e-8,
                dtype=self._dtype,
                name=f'BatchNorm-D{dl_idx + 1:d}')(outputs)

            if self._dense_layers_activation:
                outputs = tf.keras.layers.Activation(
                    'relu', dtype=self._dtype,
                    name=f'ReLu-D{dl_idx + 1:d}')(outputs)

        return outputs

    @abc.abstractmethod
    def build_architecture(self, inputs, is_training, **kwargs):
        """
        :meta private:
        """
        raise NotImplementedError(
            'build_network() method not implemented in derived class')

    def _get_preproc(self, preproc_item):

        if isinstance(preproc_item, tf.keras.layers.Layer):
            # If it was already an instantiated object, return as-is
            return preproc_item
        elif not (
                isinstance(preproc_item, (tuple, list)) and
                len(preproc_item) == 2 and
                isinstance(preproc_item[0], str) and
                isinstance(preproc_item[1], dict)):
            raise ValueError('A \'preproc\' item must be a 2-element ' +
                             'tuple/list specifying a valid preprocessing ' +
                             'operation\'s name and its parameters as a dict.')

        # Instantiate the preprocessing item based on the details provided
        name, params = preproc_item

        fixed_params = {k: v for k, v in params.items()}  # copy

        if 'dtype' not in params:     # Add dtype if missing
            fixed_params['dtype'] = self._dtype

        if name == 'LoG':
            if 'name' not in params:
                fixed_params['name'] = 'PreLoG'
            return tf_transformations.LoG(**fixed_params)

        elif name == 'GaussianBlur':
            if 'name' not in params:
                fixed_params['name'] = 'PreGaussBlur'
            return tf_transformations.GaussianBlur(**fixed_params)

        # If we're here, then it wasn't a Koogu preproc op. Try some compatible
        # keras ones.

        if 'data_format' not in params:     # Add data_format if missing
            fixed_params['data_format'] = self._data_format

        if name == 'Conv2D':
            if 'kernel_size' not in params:
                fixed_params['kernel_size'] = (3, 3)
            if 'strides' not in params:
                fixed_params['strides'] = (1, 1)
            if 'padding' not in params:
                fixed_params['padding'] = 'same'
            if 'use_bias' not in params:
                fixed_params['use_bias'] = False
            if 'kernel_initializer' not in params:
                fixed_params['kernel_initializer'] = \
                    tf.keras.initializers.VarianceScaling()
            if 'name' not in params:
                fixed_params['name'] = 'Pre_Conv'
            return tf.keras.layers.Conv2D(**fixed_params)

        elif name == 'MaxPool2D':
            if 'padding' not in params:
                fixed_params['padding'] = 'same'
            if 'name' not in params:
                fixed_params['name'] = 'Pre_MaxPool'
            return tf.keras.layers.MaxPool2D(**fixed_params)

        elif name == 'AvgPool2D':
            if 'padding' not in params:
                fixed_params['padding'] = 'same'
            if 'name' not in params:
                fixed_params['name'] = 'Pre_AvgPool'
            return tf.keras.layers.AvgPool2D(**fixed_params)

        # Add others here in an if-elif ladder

        # Raise exception if unknown option requested
        raise ValueError(
            'Unknown preproc option requested: {:s}'.format(name))

    def pad_for_valid_conv2d(self, inputs, kernel_shape, strides):
        """
        Utility function to ensure that pixels at boundaries are properly
        accounted for when stride > 1.
        """

        f_axis, t_axis = \
            (1, 2) if self._data_format == 'channels_last' else (2, 3)

        feature_dims = inputs.get_shape().as_list()
        outputs = inputs

        spatial_dims = np.asarray([feature_dims[f_axis],
                                   feature_dims[t_axis]])
        remainders = spatial_dims - (
                (np.floor((spatial_dims - kernel_shape) /
                          strides) * strides) +
                kernel_shape)
        if np.any(remainders):
            additional = np.where(remainders,
                                  kernel_shape - remainders,
                                  [0, 0]).astype(int)
            outputs = tf.keras.layers.ZeroPadding2D(
                padding=((0, additional[0]), (0, additional[1])),
                dtype=self._dtype,
                data_format=self._data_format)(outputs)

        return outputs


from .convnet import ConvNet
from .densenet import DenseNet

__all__ = ['ConvNet', 'DenseNet']

# NOTE: __all__ is accessed directly in /utils/config.py
