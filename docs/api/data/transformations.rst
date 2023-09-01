Data transformation
===================

Certain data transformations that are unavailable in TensorFlow/Keras are implemented as custom Keras layers in Koogu.

.. automodule:: koogu.data.tf_transformations
   :members: NormalizeAudio, Audio2Spectral, Spec2Img, GaussianBlur, LoG, Linear2dB
   :exclude-members: call, compute_output_shape, get_config

