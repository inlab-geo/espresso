=======
CI / CD
=======

We use GitHub Actions to run all of our automatic workflows. All specification files 
sit under 
`.github/workflows <https://github.com/inlab-geo/espresso/tree/main/.github/workflows>`_.

.. rubric:: Contents

.. contents::
   :local:


build_wheels.yml
----------------

Specification file:
https://github.com/inlab-geo/espresso/blob/main/.github/workflows/build_wheels.yml

This is to test that wheels can be built on Linux and MacOS.

We are not testing Windows here due to problems with linking. Check the yaml files for
detailed reasons.


pr_validation.yml
-----------------

Specification file:
https://github.com/inlab-geo/espresso/blob/main/.github/workflows/pr_validation.yml


This is a validation for new pull requests and merged ones, mainly running the script:

.. code-block:: console

    $ python espresso_machine/build_package/build.py --post

If only a few contribution sub-folders are changed, this workflow will detect a list of
changed contributions and run tests only on them:

.. code-block:: console

    $ python espresso_machine/build_package/build.py --post -f changed_contribs.txt


publish_pypi.yml
----------------

Specification file:
https://github.com/inlab-geo/espresso/blob/main/.github/workflows/publish_pypi.yml

This is the workflow we use for deployment to 
`Test PyPI <https://test.pypi.org/project/geo-espresso/>`_ and 
`PyPI <https://pypi.org/project/geo-espresso/>`_.

It relies on 
`secret tokens <https://github.com/inlab-geo/espresso/settings/secrets/actions>`_ 
from our maintainer accounts on PyPI.

routine_check.yml
-----------------

Specification file:
https://github.com/inlab-geo/espresso/blob/main/.github/workflows/routine_check.yml

This is triggered daily to run all existing problems in the ``main`` branch and 
automatically raises a pull request if there is a change to the 
`active problems <https://github.com/inlab-geo/espresso/blob/main/contrib/active_problems.txt>`_
(for example when a problem fails to run).


update_esp_build.yml
--------------------

https://github.com/inlab-geo/espresso/blob/main/.github/workflows/update_esp_build.yml

The branch ``esp_build`` is reserved for the latest package source code. It is updated 
by this workflow whenever the ``main`` branch is updated.
