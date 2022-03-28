Data augmentation
=================

On-the-fly data augmentations can be applied during training/validation by implementing the desired augmentation operations in the :meth:`~koogu.data.feeder.BaseFeeder.pre_transform` and :meth:`~koogu.data.feeder.BaseFeeder.post_transform` methods of the classes derived from :class:`koogu.data.feeder.BaseFeeder`. Given that the CNN models used in bioacoustics typically operate on inputs that are :ref:`transformed into 2-dimensional spectrograms <custom_feeder-psd_transform>`, augmentations applicable to time-domain waveforms can be implemented in :meth:`~koogu.data.feeder.BaseFeeder.pre_transform` and augmentations applicable to spectrograms can be implemented in :meth:`~koogu.data.feeder.BaseFeeder.post_transform`.

.. note::

   This requires writing code to use the TensorFlow API directly.

The below example extends :class:`koogu.data.feeder.SpectralDataFeeder` by adding two augmentation operations in the time-domain and one in the spectro-temporal domain. The example also demonstrates the use of a few :doc:`pre-defined & customizable augmentations <../api/data/augmentation>`. You may also add code in these methods to implement your own types of augmentation.

.. literalinclude:: ../code_samples/data_augmentation.py
   :end-before: [end-first-example]

The above example demonstrates finer control in implementing augmentations wherein one may employ branching/looping constructs to combine different augmentations as desired. Sometimes, you may want to simply apply a series of augmentations in a particular order, with respective chosen probabilities. The below code snippet demonstrates the use of :ref:`convenience interface <augmentation-convenience_interface>` to apply chained augmentations. You need not use any TensorFlow API here.

.. literalinclude:: ../code_samples/data_augmentation.py
   :start-after: [start-second-example-snippet]

