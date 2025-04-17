================
Folder structure
================

.. figure:: ../_static/espresso_arch.svg
    :align: center

- Folder **src/espresso/** contains Espresso core functions, the base class
    :code:`EspressoProblem`, and the contributions in **src/espresso/contrib**.
    Contributors typically install these functions before they
    start developing a new problem, so that they get access to the base class and utility
    functions.

- If you'd like to improve base class specification or :code:`espresso.utils`,
    this is the place to edit.

- If you'd like to bump the version, change file :code:`src/espresso/version.py`.

- Folder **espresso_machine/** has all the utility scripts to be used by contributors and 
  developers.
