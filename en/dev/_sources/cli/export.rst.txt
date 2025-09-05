Model Export
============

Export your trained models for use with other bioacoustics software platforms like Raven Pro and PAMGuard.

**Raven Pro Export**

    .. code-block:: console

       $> koogu-export my_project.config raven my_first_model \
            --desc "My first Koogu model"


**PAMGuard Export**

    .. code-block:: console

       $> koogu-export my_project.config pamguard my_first_model \
            --desc "My first Koogu model"


----

Parameters
^^^^^^^^^^

Positional arguments
--------------------------------------------------------------------------------
    - ``<CONFIG FILE>``

      Path to config file.

    - ``<MODEL NAME>``

      Name of the trained model instance.

Options
--------------------------------------------------------------------------------
    - ``--prefix``

      String prefix to be added to the output directory/file name.

    - ``--desc``

      Textual description for the model.

Target type (must specify one)
--------------------------------------------------------------------------------
    - ``raven``

      Raven Pro

    - ``pamguard``

      PAMGuard

Raven options
--------------------------------------------------------------------------------
    - ``--threshold``

      Set the default detection threshold within Raven Pro.

      `Default`: 0.90

    - ``--suppress``

      Instruct Raven Pro to suppress outputs for specified class(es).

