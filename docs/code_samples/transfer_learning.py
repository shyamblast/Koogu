import tensorflow as tf
from koogu.data.tf_transformations import Spec2Img
from koogu.model.architectures import BaseArchitecture
from matplotlib import cm


class MyTransferModel(BaseArchitecture):

    def build_network(self, input_tensor, is_training, data_format, **kwargs):

        # Many of the available pre-trained models expect inputs to be of a
        # particular size. The `input_tensor` may not already be in that shape,
        # depending on the chosen data preparation parameters (e.g., with
        # koogu.data.feeder.SpectralDataFeeder). We need to resize the images to
        # match the input shape of the pre-trained model.
        # MobileNetV2 defaults to an input size of 224x224, and also supports a
        # few other sizes. 160x160 is a supported size, and we use that in this
        # example.
        target_img_size = (160, 160)

        # Choose your favourite colorscale from matplotlib or other sources.
        my_cmap = cm.get_cmap('jet')(range(256))
        # `my_cmap` will be a 256 element array of RGB color values from the
        # "Jet" colorscale.

        # First, need to convert input spectrograms to equivalent RGB images.
        # Spec2Img will convert 1-channel spectrograms to 3-channel RGB images
        # (with values in the range [0.0, 1.0]) and resize them as desired.
        to_image = Spec2Img(my_cmap, img_size=target_img_size)

        # The pre-trained MobileNetV2 expects RGB values to be scaled to the
        # range [-1.0, 1.0].
        rescale = tf.keras.layers.Rescaling(2.0, offset=-1.0)
        # NOTE: `Rescaling` was added in TensorFlow v2.6.0. If you use an older
        #       version, you can implement this operation by simply multiplying
        #       the output of to_image() by 2.0 and then subtracting 1.0.

        # Load the pre-trained MobileNetV2 model with ImageNet weights, and
        # without the trailing fully-connected layer.
        pretrained_cnn = tf.keras.applications.MobileNetV2(
            input_shape=target_img_size + (3, ),    # Include RGB dimension
            include_top=False,
            weights='imagenet')
        pretrained_cnn.trainable = False            # Freeze CNN weights

        # Pooling layer
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

        # Put them all together now.
        # NOTE: The "training=False" parameter to `pretrained_cnn` is required
        #       to ensure that BatchNorm layers in the model operate in
        #       inference mode (for more details, see TensorFlow's webpage on
        #       transfer learning).
        output = to_image(input_tensor)
        output = rescale(output)
        output = pretrained_cnn(output, training=False)
        output = global_average_layer(output)

        # NOTE: Do not add the classification layer. It will be added by Koogu's
        #       internal code.

        return output
# [end-implementation]

# [start-create-model]
model = MyTransferModel()
# [end-create-model]