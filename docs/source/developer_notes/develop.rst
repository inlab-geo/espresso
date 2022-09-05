===============
Developer Notes
===============

🚧 This page is under construction. 

Please `contact us <../user_guide/faq.html>`_ directly if you have questions.

Espresso Architecture
=====================

.. figure:: ../_static/espresso_arch.svg
    :align: center

- Folder **contrib/** contains subfolders for each Espresso problem. Each Espresso
  problem has a self-contained subfolder with :code:`problem_name` as the folder name.

- Folder **src/cofi_espresso/** contains Espresso core functions and the base class
  :code:`EspressoProblem`. Contributors typically install these functions before they
  start developing a new problem, so that they get access to the base class and utility
  functions.

  - If you'd like to improve base class specification or :code:`cofi_espresso.utils`,
    this is the place to edit.

  - If you'd like to bump the version, change file :code:`src/cofi_espresso/_version.py`.

- Folder **tools/** has all the utility scripts to be used by contributors and 
  developers.

- Folder **_esp_build/** will contain temporary Python package source files after you
  build :code:`cofi_espresso`.

  - These are built files, so you never have to change the contents under this folder. 
    If you feel something's wrong in this folder, look for the source from the three 
    folders above.

- Folder **docs/** has all the documentation sources.


How-to Guides
=============

.. contents::
    :local:

Build the package locally
-------------------------

Check out the `contributor guide  <../contributor_guide/new_contrib.html>`_.

Build the documentation locally
-------------------------------

Check out README file under docs folder 
`here <https://github.com/jwhhh/espresso/tree/main/docs/README.md>`_.

Modify EspressoProblem class
----------------------------

1. Modify the class in file :code:`src/cofi_espresso/espresso_problem.py`
2. Make sure your changes are backward compatible, otherwise take the responsibility of
   modifying existing contributions under folder :code:`contrib/`
3. Make new contribution generation script compatible with new changes. Check by running 
   file :code:`tools/new_contribution/create_new_contrib.py`. 

   - If generated example doesn't comply with the new specification, potentially you need 
     to edit some files under :code:`tools/new_contribution/_template`. Pay special 
     attention to the following files:

     - :code:`tools/new_contribution/_template/example_name.py`
     - :code:`tools/new_contribution/_template/README.md`

4. Ensure build and validation scripts are compatible with new changes. Check by running:

   - :code:`tools/build_package/validate.py pre`
   - :code:`tools/build_package/build.py`
   - :code:`tools/build_package/validate.py post`
   - :code:`tools/build_package/build_with_checks.py`
   
   Examine reported error (if any) to locate whether to change scripts themselves, or to
   edit the template files under :code:`tools/new_contribution/_template`.

5. Check if you need to change :code:`README.md` file for the repository.

6. Check if you need to update contents in `introduction page <../user_guide/introduction.html>`_.
   If yes, modify file :code:`docs/source/user_guide/introduction.rst`.


Modify build/validation scripts
-------------------------------

1. Navigate to :code:`tools/build_package/` folder, all the scripts are there. Make changes as you need.
2. Ensure the other scripts still work. For example, you might want to change usage of :code:`validate.py`
   inside :code:`build_with_checks.py` after the argument parser is modified. Check by running them on
   your own.
3. Ensure documentations are up to date. The following places need checking:

   - :code:`tools/README.md`
   - :code:`tools/new_contribution/_template/README.md`
   - :code:`docs/source/contributor_guide/new_contrib.rst`
