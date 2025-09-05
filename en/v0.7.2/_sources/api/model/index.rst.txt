Model
=====

Koogu supports a few ready-to-use :doc:`architectures <architectures>`.

User-defined custom architectures can be created by implementing the abstract base class :class:`koogu.model.architectures.BaseArchitecture`.

----

.. toctree::
   :maxdepth: 1
   :hidden:
   :glob:
   
   Architectures <architectures>

.. autoclass:: koogu.model.TrainedModel
   :members: infer, audio_settings, spec_settings, class_names

.. autoclass:: koogu.model.architectures.BaseArchitecture
   :members:
