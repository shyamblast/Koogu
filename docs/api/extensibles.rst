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
   The :class:`koogu.data.feeder.DataFeeder` class is also extensible. See
   :doc:`../advanced/custom_feeders` for example pseudo-code.

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
