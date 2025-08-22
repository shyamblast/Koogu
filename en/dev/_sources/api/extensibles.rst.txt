Extensibles
===========


Annotations
-----------

.. autoclass:: koogu.data.annotations.BaseAnnotationReader
   :members: __call__, _fetch

.. autoclass:: koogu.data.annotations.BaseAnnotationWriter
   :members: __call__, _write

Feeder
------

.. autoclass:: koogu.data.feeder.BaseFeeder
   :members:

.. seealso::
   See :doc:`../advanced/custom_feeders` for example pseudo-code.

   Subclasses :class:`koogu.data.feeder.DataFeeder` and
   :class:`koogu.data.feeder.SpectralDataFeeder` are also further extensible.

Augmentations
-------------

.. autoclass:: koogu.data.augmentations.Temporal
   :members: build_graph

.. autoclass:: koogu.data.augmentations.SpectroTemporal
   :members: build_graph

Model architecture
------------------

.. autoclass:: koogu.model.architectures.BaseArchitecture
   :members:

.. seealso::
   Implementing :doc:`../advanced/transfer_learning` models.

Assessment metric
-----------------

.. autoclass:: koogu.utils.assessments.BaseMetric
   :members:
