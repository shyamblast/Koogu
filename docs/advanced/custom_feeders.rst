Custom feeders
==============

While the basic feeders :class:`~koogu.data.feeder.DataFeeder` and :class:`~koogu.data.feeder.SpectralDataFeeder` were designed to work with data that were already pre-processed (resampled, filtered, segmented using the :doc:`../api/prepare` interface) and stored in compressed numpy format, certain applications may require data to be loaded and fed into the training pipeline via other mechanisms/files/formats (for example, feeding directly from audio files or from a database object). Koogu facilitates these by allowing users to define custom feeders that implement their desired logic.

Custom feeders can be defined by extending the abstract class :class:`koogu.data.feeder.BaseFeeder`.

.. note::

   This requires writing code to use the TensorFlow API directly.

The below example shows an implementation which extends the :class:`~koogu.data.feeder.BaseFeeder` class to feed clips by loading directly from audio files.

.. literalinclude:: ../code_samples/custom_feeder.py
   :end-before: [end-first-example]

.. _custom_feeder-psd_transform:

Converting waveforms to spectrograms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the above example, the loaded audio clips will be presented as-is (as waveforms) to the model during training/validation. You can convert the clips into power spectral density spectrograms before they are presented to the model, by implementing the functionality in the :meth:`~koogu.data.feeder.BaseFeeder.transform()` method and overriding the :meth:`~koogu.data.feeder.BaseFeeder.get_shape_transformation_info()` method as shown below.

.. literalinclude:: ../code_samples/custom_feeder.py
   :start-after: [start-second-example-snippet]

