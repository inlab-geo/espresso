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

- We suggest you to follow the general contributor guide, and write a ``CMakeLists.txt``
  file so that our build system ``scikit-build`` can pick up the compilation process.
- Here's a working example that you could refer to: 
  `fm_wavefront_tracker <https://github.com/inlab-geo/espresso/tree/main/contrib/fm_wavefront_tracker>`_ 

I need to include a compiled function (written in C/C++/Fortran), how to do it?
-------------------------------------------------------------------------------

- We suggest you to follow the general contributor guide, and write a ``CMakeLists.txt``
  file so that our build system ``scikit-build`` can pick up the compilation process.
- Here's a working example that you could refer to: 
  `receiver_function <https://github.com/inlab-geo/espresso/tree/main/contrib/receiver_function>`_ 


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
