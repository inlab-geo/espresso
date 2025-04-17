================
Contributor FAQs
================

Contents
********

.. contents::
    :local:
    :class: toggle:


I need to include a compiled executable (from C/C++/Fortran source code), how to do it?
---------------------------------------------------------------------------------------

- For ease of maintainability, we recommend you create a separate (pip installable) 
  Python package for your compiled code, e.g. using ``scikit-build`` and ``cmake``.
  This package should be installed as a dependency of your Espresso problem.

I have put together an espresso contribution but I also have this new awesome inference method to solve it
----------------------------------------------------------------------------------------------------------

- An Espresso problem may include an example solution of the related inverse problem in form 
  of a jupyter notebook in the example folder.

The Espresso interface is too restrictive
-----------------------------------------

- While Espresso requires all problems to adhere to the same minimal standard to 
  facilitate testing and experimentation it allows to define optional functions such 
  as `my_unicorn_figure`

.. My forward simulation code is already in a GitHub repository, how to include it?
.. --------------------------------------------------------------------------------

.. - We suggest you to follow the general contributor guide, and include your original
..   repository as a submodule.
.. - Here's a working example that you could refer to: TODO
