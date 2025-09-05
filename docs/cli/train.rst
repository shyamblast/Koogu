Training
========

Train your model using the prepared data and the architecture/hyperparameters specified in your config file.

.. code-block:: console

   $> # Train a model form the project's current settings, give it a name
   $> koogu-train my_project.config my_first_model


The training step will:

    1. **Construct the model** as described in the ``[model]`` section of your config file
    2. **Perform training and validation** based on training/hyperparameter settings in the ``[train]`` section of your config file
    3. **Save trained model** in ``[paths].model``/my_first_model with all necessary files for testing and deployment

👉 Models directory path (``model``) is set under ``[paths]`` section of your config file.

During training, you will see epoch-by-epoch progress as below:

.. code-block:: console

   Epoch 1/50 ████████████████████████ 245/245 - 12s - loss: 0.3421 - binary_accuracy: 0.8234 - val_loss: 0.2891 - val_binary_accuracy: 0.8756


----

Parameters
^^^^^^^^^^

Positional arguments
--------------------------------------------------------------------------------
    - ``<CONFIG FILE>``

      Path to config file.

    - ``<MODEL NAME>``

      An identifier (name) for the model to be trained.

Options
--------------------------------------------------------------------------------
    - ``--seed NUM``

      Seed value (integer) for deterministic shuffling.

Logging
--------------------------------------------------------------------------------
    - ``--log LOGFILE``

      If specified, logging will be written out to this file instead of the default.

      `Default`: **PROJECT-LOGS-DIR**/train.log

    - ``--loglevel LEVEL``

      Logging level. Choices: CRITICAL, ERROR, WARNING, INFO, DEBUG.

      `Default`: INFO
