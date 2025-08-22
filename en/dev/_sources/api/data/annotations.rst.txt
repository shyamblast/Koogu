Handling Annotations/Detections
===============================

.. automodule:: koogu.data.annotations

Format-specific Readers and Writers for managing annotations and detections.
Currently supported formats include:

* `Audacity`_
* `Raven Pro`_
* `Sonic Visualiser`_

Custom Readers/Writers to handle yet-unsupported formats can be implemented, in
ways compatible for use within the Koogu ecosystem as a drop-in replacement to
existing Readers/Writers, by extending the
:class:`~koogu.data.annotations.BaseAnnotationReader` and
:class:`~koogu.data.annotations.BaseAnnotationWriter` classes.

Audacity
--------

.. autoclass:: koogu.data.annotations.Audacity.Reader
   :members:

.. autoclass:: koogu.data.annotations.Audacity.Writer
   :members:

Raven Pro
---------

.. autoclass:: koogu.data.annotations.Raven.Reader
   :members: get_annotations_from_file

.. autoclass:: koogu.data.annotations.Raven.Writer
   :members:

Sonic Visualiser
----------------

.. autoclass:: koogu.data.annotations.SonicVisualiser.Reader
   :members:
