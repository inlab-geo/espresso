==========
Deployment
==========

.. 
    Which platforms are we deploying to? (PyPI, conda-forge,...)
    Links to relevant pages
    How does automation work/what needs to be done manually?

Platform
--------

We deploy `geo-espresso` to PyPI: https://pypi.org/project/geo-espresso/

Deploying to conda-forge is work-in-progress.


How to release a new version
----------------------------

We use the workflow `publish_pypi.yml <https://pypi.org/project/geo-espresso/>`_
to deploy to PyPI. This GitHub workflow is triggered by changing the file
``src/espresso/version.py``, or by manually starting (in case the automatic run fails).

In order to release a new version,

1. Update the version number in 
   `src/espresso/version.py <https://github.com/inlab-geo/espresso/blob/main/src/espresso/version.py>`_
2. Update `CHANGELOG.md <https://github.com/inlab-geo/espresso/blob/main/CHANGELOG.md>`_

You can then follow up with the automatic run from the 
`actions page <https://github.com/inlab-geo/espresso/actions/workflows/publish_pypi.yml>`_.
