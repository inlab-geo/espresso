Installation
==========================


Pre-requisite
-------------

Inversion Test Suite requires Python 3.6+, and the following dependencies:

- numpy>=1.18
- scipy>=1.0.0

Install
-------

.. tabs::

   .. tab:: PyPI

       It's optional, but recommended to use a virtual environment::

         conda create -n esp_env scipy jupyterlab matplotlib
         conda activate esp_env

       Install espression with::

         python3 -m pip install espresso


   .. tab:: conda-forge

      Uploading to conda-forge is still work in progress.

      It won't be long!

   .. tab:: from source

      If you'd like to build from source, clone the repository::

        git clone https://github.com/inlab-geo/espresso.git
        cd espresso
        conda create -n esp_env scipy matplotlib
        conda activate esp_env
        pip install build
        python -m build
        pip install dist/*.whl
