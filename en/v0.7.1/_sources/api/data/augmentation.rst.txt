Data augmentation
=================

Koogu supports applying randomized on-the-fly augmentations to input samples during training/validation.

Time-domain augmentations
-------------------------

.. autoclass:: koogu.data.augmentations.Temporal.AddEcho

.. autoclass:: koogu.data.augmentations.Temporal.AddGaussianNoise

.. autoclass:: koogu.data.augmentations.Temporal.RampVolume

.. autoclass:: koogu.data.augmentations.Temporal.ShiftPitch

Spectro-temporal augmentations
------------------------------

.. autoclass:: koogu.data.augmentations.SpectroTemporal.AlterDistance

.. autoclass:: koogu.data.augmentations.SpectroTemporal.SmearFrequency

.. autoclass:: koogu.data.augmentations.SpectroTemporal.SmearTime

.. autoclass:: koogu.data.augmentations.SpectroTemporal.SquishFrequency

.. autoclass:: koogu.data.augmentations.SpectroTemporal.SquishTime


.. _augmentation-convenience_interface:

Convenience interface
---------------------

.. automethod:: koogu.data.augmentations.Temporal.apply_chain

.. automethod:: koogu.data.augmentations.SpectroTemporal.apply_chain

