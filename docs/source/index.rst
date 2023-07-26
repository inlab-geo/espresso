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
    :maxdepth: 2

    user_guide/index.rst

.. toctree::
    :hidden:
    :maxdepth: 2

    contributor_guide/index.rst

.. toctree::
    :hidden:
    :maxdepth: 2

    developer_guide/index.rst


.. .. toctree::
..     :hidden:
    
..     GitHub repository <https://github.com/inlab-geo/espresso/>
..     Issue tracker <https://github.com/inlab-geo/espresso/issues/>


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
    predictions, G = tp.forward(model, return_jacobian = True)
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

.. grid:: 1 3 3 3
    :gutter: 3
    :padding: 2

    .. grid-item-card::
        :link: user_guide/index.html
        :text-align: center
        :class-card: card-border

        *📙 User Guide*
        ^^^^^^^^^^^^^^^
        Read about what Espresso provides and how to use it

    .. grid-item-card::
        :link: contributor_guide/index.html
        :text-align: center
        :class-card: card-border

        *📗 Contributor Guide*
        ^^^^^^^^^^^^^^^^^^^^^^
        Contribute your own example problems with minimal steps

    .. grid-item-card::
        :link: developer_guide/index.html
        :text-align: center
        :class-card: card-border

        *📘 Developer Guide*
        ^^^^^^^^^^^^^^^^^^^^
        Learn about the infrastructure behind Espresso

    .. grid-item-card::
        :link: https://github.com/inlab-geo/espresso/
        :text-align: center
        :class-card: card-border

        *🐍 GitHub Repository*
        ^^^^^^^^^^^^^^^^^^^^^^
        Explore the source code on GitHub

    .. grid-item-card::
        :link: https://github.com/inlab-geo/espresso/issues/
        :text-align: center
        :class-card: card-border

        *🐛 Issue Tracker*
        ^^^^^^^^^^^^^^^^^^
        Report a bug or suggest a feature by creating an issue

    .. grid-item-card::
        :link: https://join.slack.com/t/inlab-community/shared_invite/zt-1ejny069z-v5ZyvP2tDjBR42OAu~TkHg
        :text-align: center
        :class-card: card-border

        *💬 Join Slack*
        ^^^^^^^^^^^^^^^
        Accept this invitation to join the conversation on Slack


Espresso is an open-source community effort, currently supported and coordinated by `InLab <http://www.inlab.edu.au/>`_.
