.. geo-espresso documentation master file, created by
   sphinx-quickstart on Fri Jun  3 13:47:44 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. title:: Home

====================
Welcome to Espresso!
====================

.. The following defines the structure of the document. It is hidden so it doesn't
   render into the html, but it cannot be removed!
   We assume that each 'index.rst' document defines its own toctree that can be incorporated.

.. toctree::
    :hidden: 
    :maxdepth: 1

    user_guide/index.rst

.. toctree::
    :hidden:
    :maxdepth: 1

    contributor_guide/index.rst

.. toctree::
    :hidden:
    :maxdepth: 1

    developer_guide/index.rst


.. toctree::
    :hidden:
    
    GitHub repository <https://github.com/inlab-geo/espresso/>
    Issue tracker <https://github.com/inlab-geo/espresso/issues/>


You've just come up with a new optimisation algorithm, inversion strategy, or machine learning-based inference framework. Now you want to see how it performs on a real-world problem... 

Espresso (**E**\ arth **s**\ cience **pr**\ oblems for the **e**\ valuation of **s**\ trategies, **s**\ olvers and **o**\ ptimizers) aims to make this as easy as possible. It provides access to a range of exemplars via a standardized Python interface, including domain-expert--curated datasets and corresponding simulation codes. Swapping from one test problem to the next is just one line of code.

Here's a simple illustration:

.. code-block:: python
    :linenos:
    :class: toggle

    import numpy as np

    # Select the test problem to be used -- change this line to any 
    # other test problem and everything else should still work!
    from espresso import SimpleRegression as test_problem 

    # Create an instance of the test problem class
    tp = test_problem()

    # The test problem provides...
    # ... an initial (null) model vector
    model = tp.starting_model
    # ... the ability to compute simulated data for an arbitrary
    # model vector, and the corresponding jacobian (i.e. derivatives 
    # of data wrt model parameters)
    predictions, G = tp.forward(model, with_jacobian = True)
    # ... a data vector, which matches the output from `tp.forward()`
    residuals = tp.data - predictions 
    # ... and much more!

    # Compute a Gauss-Newton model update
    model += np.linalg.solve(G.T.dot(G), G.T.dot(residuals))
    
    # Compare our result to the answer suggested by Espresso
    print("Our result:", model)
    print("Espresso:", tp.good_model)

    # And let's visualise both to see where the differences are:
    my_fig = tp.plot_model(model)
    espresso_fig = tp.plot_model(tp.good_model)
    data_fig = tp.plot_model(tp.data, tp.forward(model))

If this looks interesting, you can:

- Read the :doc:`user_guide/index` for more information about what Espresso provides and how to use it;
- Contribute your own example problems by following the instructions in the :doc:`contributor_guide/index`;
- Learn about the infrastructure behind Espresso by looking at the :doc:`developer_guide/index`;
- Explore the source code on `GitHub <https://github.com/inlab-geo/espresso/>`_;
- Report a bug or suggest a feature using the `Issue Tracker <https://github.com/inlab-geo/espresso/issues/>`_;
- Join the conversation on `Slack <https://join.slack.com/t/inlab-community/shared_invite/zt-1ejny069z-v5ZyvP2tDjBR42OAu~TkHg>`_.

Espresso is an open-source community effort, currently supported and coordinated by `InLab <http://www.inlab.edu.au/>`_.


.. Espresso (**E**\ arth **S**\ cience **PR**\ oblems for the **E**\ valuation of **S**\ trategies,
.. **S**\ olvers and **O**\ ptimizers) is:

.. - A collection of:

..   - geoscience simulation codes (or 'forward models'), with 
..   - associated datasets;

.. - Designed for researchers, educators and students working in areas such as inference, inversion, optimisation or machine learning. It aims to:

..   - Provide real-world test problems to support algorithm development;
..   - Enable benchmarking and comparison of techniques across a range of application areas;
..   - Support teaching by providing a variety of examples that can be incorporated into lectures, demonstrations and practical exercises.

.. - Accessible via a standardised interface. Testing your algorithm on a new problem means changing just one line of code, and does not require any domain knowledge.

.. - An open-source community effort, currently coordinated by `InLab <http://www.inlab.edu.au/>`_, with support from the CoFI development team.


