.. cofi-espresso documentation master file, created by
   sphinx-quickstart on Fri Jun  3 13:47:44 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. title:: Home

====================================
Welcome to Espresso's documentation!
====================================
Espresso (**E**\ arth **S**\ cience **PR**\ oblems for the **E**\ valuation of **S**\ trategies, 
**S**\ olvers and **O**\ ptimizers) is a collection of datasets, and 
associated simulation codes, spanning a wide range of geoscience problems. 
Together they form a suite of real-world test problems that can be used to 
support the development, evaluation and benchmarking of a wide range of tools
and algorithms for inference, inversion and optimisation. All problems are 
designed to share a common interface, so that changing from one test problem
to another requires changing one line of code. 



.. panels::
    :header: text-center text-large
    :card: border-1 m-1 text-center


    **Introduction to Espresso**
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^

    🐣 New to Espresso?

    .. link-button:: user_guide/introduction
        :type: ref
        :text: User Guide
        :classes: btn-outline-primary btn-block stretched-link

    ---

    **Contribute a new problem**
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^

    🍻 Forward codes are always welcomed

    .. link-button:: contributor_guide/ways
        :type: ref
        :text: Contributor Guide
        :classes: btn-outline-primary btn-block stretched-link

    ---


    **Improve the core Espresso package**
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    🛠 Let's build a better Espresso together

    .. link-button:: developer_notes/develop
        :type: ref
        :text: Developer notes
        :classes: btn-outline-primary btn-block stretched-link

    ---

    **Get involved!**
    ^^^^^^^^^^^^^^

    💬 Join our Slack workspace

    .. link-button:: https://join.slack.com/t/inlab-community/shared_invite/zt-1ejny069z-v5ZyvP2tDjBR42OAu~TkHg
        :type: url
        :text: Join the conversation
        :classes: btn-outline-primary btn-block stretched-link


The Espresso project is a community effort - if you think it sounds useful,
please consider contributing an example or two from your own research. The project
is currently being coordinated by `InLab <http://www.inlab.edu.au/>`_, with support from the CoFI development
team.




Table of contents
-----------------

.. toctree::
    :caption: User Guide
    :maxdepth: 1

    user_guide/introduction.rst
    user_guide/installation.rst
    user_guide/usage.rst
    user_guide/contrib/index.rst
    user_guide/api/index.rst
    user_guide/faq.rst

.. toctree::
    :caption: Contributor Guide
    :maxdepth: 1

    contributor_guide/ways.rst
    contributor_guide/new_contrib.rst

.. toctree::
    :caption: Developer Notes
    :maxdepth: 1

    developer_notes/develop.rst
    developer_notes/changelog.md
    developer_notes/licence.rst

.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
