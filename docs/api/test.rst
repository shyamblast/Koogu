Test a trained model
====================

Apply a trained model on test dataset, and save raw detection scores for subsequent
:doc:`performance assessment(s) <assessments>`. The below functions pre-process
the test dataset recordings with the same parameters (obtained from trained model's
metadata) that were used while preparing the training dataset.

.. automodule:: koogu.test
   :members: