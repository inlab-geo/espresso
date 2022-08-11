============
Installation
============

Pre-requisites
--------------

Espresso requires Python 3.6+, and the following dependencies:

- numpy>=1.18
- scipy>=1.0.0
- matplotlib>=3.1


Virtual environment setup
-------------------------

It's optional, but recommended to use a virtual environment.

.. tabbed:: venv

  Ensure you have `python>=3.6`.

  To create:

  .. code-block:: console

    python -m venv <path-to-new-env>/esp_env

  To activate:
  
  .. code-block:: console

    source <path-to-new-env>/esp_env/bin/activate

  To exit:
  
  .. code-block:: console

    deactivate

  To remove:

  .. code-block:: console

    rm -rf <path-to-new-env>/esp_env

.. tabbed:: virtualenv

  To create:

  .. code-block:: console

    virtualenv <path-to-new-env>/esp_env -p=3.10

  To activate:

  .. code-block:: console

    source <path-to-new-env>/esp_env/bin/activate

  To exit:

  .. code-block:: console

    deactivate

  To remove:

  .. code-block:: console

    rm -rf <path-to-new-env>/esp_env

.. tabbed:: conda / mamba

  To create:

  .. code-block:: console

    conda create -n esp_env python=3.10

  To activate:

  .. code-block:: console

    conda activate esp_env

  To exit:

  .. code-block:: console

    conda deactivate

  To remove:
  
  .. code-block:: console

    conda env remove -n esp_env


Install
-------

.. tabbed:: PyPI

  .. code-block:: console
    
    pip install cofi-espresso

.. tabbed:: conda-forge

  Uploading to conda-forge is still work in progress. 
  
  It won't be long!

.. tabbed:: from source

  If you'd like to build from source, clone the repository

  .. code-block:: console

    git clone https://github.com/inlab-geo/espresso.git
    cd espresso

  And use either one of the following command to install

  .. code-block:: console

    pip install .         # library will be copied over to site-packages
    pip install -e .      # developer mode, library will be symbol linked to site-packages
