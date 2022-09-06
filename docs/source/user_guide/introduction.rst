============
Introduction
============

Espresso is a community driven project to create a large suite of forward simulations 
to enable researchers to get example problems without the need to understand each 
individual forward problem in detail.

Once installed, each test problem can be imported using the following command:

.. code-block:: python

    from cofi_espresso import <testproblem>


Replace ``<testproblem>`` with one of the following currently available problems:

- ``GravityDensity``
- ``SimpleRegression``
- ``XrayTomography``

Once a problem is imported, its main functions can be called using the same 
structure for each problem. For instance:

.. code-block:: python

    from cofi_espresso import GravityDensity

    problem = GravityDensity(example_number=1)
    model = problem.good_model()
    data = problem.data()
    pred = problem.forward(model)
    fig_model = problem.plot_model(model)
    fig_data = problem.plot_data(data, pred)

You can access related metadata programatically:

.. code-block:: python

    print(GravityDensity.problem_title)
    print(GravityDensity.problem_short_description)
    print(GravityDensity.author_names)


Other problem-specific parameters can be accessed through the problem instance. For instance:

.. code-block:: python

    print(problem.m)
    print(problem.rec_coords)


Which additional values are set is highly example-specific and we suggest to 
consult the `Espresso Problems section <contrib/index.html>`_.


Espresso's simple and consistent code structure enables users to access a wide range
of different forward code and contributors to share their solutions with a wider
audience. Having the basic set up and several examples in
place already, we hope to encourage more examples be developed and be contributed.
