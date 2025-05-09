==================
Set up environment
==================

Let's now ensure that you have a correct environment set up. 

Development requirements
------------------------

Python >= 3.9 is required.

We strongly recommend using a virtual environment.

.. tab-set::

   .. tab-item:: venv

      Ensure you have `python>=3.9`.

      .. code-block:: console

         $ python -m venv <path-to-new-env>/espresso
         $ source <path-to-new-env>/<env-name>/bin/activate
         $ python -m pip install -r envs/requirements_dev.txt

   .. tab-item:: virtualenv

      .. code-block:: console

         $ virtualenv <path-to-new-env>/<env-name> -p=3.10
         $ source <path-to-new-env>/<env-name>/bin/activate
         $ python -m pip install -r envs/requirements_dev.txt

   .. tab-item::  conda / mamba

      .. code-block:: console

         $ conda env create -f envs/environment_dev.yml
         $ conda activate esp_dev


Espresso core package
---------------------

Install Espresso core library - this enables you to access the base class for an Espresso problem
:code:`EspressoProblem` and some utility functions to help the development.

Since you will be modifying the `espresso` package to create your contribution, you should
install it in 'editable' mode. This means that any changes you make to the code will be
immediately available to you without needing to re-install the package.

Run the following in your terminal, with :code:`<path-to-espresso>/` as your working directory.

.. code-block:: console

   $ pip install -e .
