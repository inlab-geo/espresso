.. title:: usage

.. highlight:: python

=====
Usage
=====


Espresso is a collection of individual examples, or *test problems*. From a users' perspective, each test problem is a separate Python object (specifically, a subclass of :py:class:`espresso.EspressoProblem`) that is available to be imported:

.. code-block:: python

    from espresso import XrayTomography as testproblem1
    from espresso import SimpleRegression as testproblem2

Then, create an instance of the class to access its functionality:

.. code-block:: python

    tp1 = testproblem1()
    tp2 = testproblem2()

Each test problem defines one or more *examples*. All examples will represent the same fundamental inference problem (i.e. the same physical/conceptual scenario), but may provide access to different datasets or problem configurations --- for full details, consult the :doc:`problem-specific documentation <contrib/index>`. By default, the first example is loaded, but you can select a different example by passing :code:`example_number=<num>` when creating your class instance:

.. code-block:: python
    
    tp1 = testproblem1(example_number = 3)
    tp2 = testproblem2(example_number = 2)

If you wish to use more than one example in your code, you will need to create multiple :code:`testproblem` instances (or re-initialise an existing one). Attempting to load a non-existent example number will raise a :py:class:`espresso.exceptions.InvalidExampleError` exception.

.. code-block:: python
    :class: toggle

    from espresso.exceptions import InvalidExampleError
    from espresso import XrayTomography as xrt

    iexample = 1
    try:
        while True:
            x = xrt(example_number = iexample)
            
            # Do stuff
            # ...
            
            # Now increment the counter and go round again
            iexample += 1
    except InvalidExampleError:
        print("Last valid example:",iexample-1)


Your test problem instance provides access to simulation capabilities, data and other information via a standardised interface. Let's suppose we've imported and instantiated some test problem :code:`<TestProblem>` (this is just a generic placeholder name -- it can be any example from the list).

.. code-block:: python

    from espresso import <TestProblem>
    tp = TestProblem()

All test problems will have at least the following attributes:

- :code:`tp.model_size` -- The number of unknown parameters, :math:`M`, i.e. the dimension of the model vector :math:`\mathbf{m}`.
- :code:`tp.starting_model` -- A :py:class:`numpy.ndarray` with shape :math:`(M,)` containing a null model or other problem-appropriate starting point that can be used to initialise (e.g.) iterative algorithms.
- :code:`tp.good_model` -- A :py:class:`numpy.ndarray` with shape :math:`(M,)` containing a model vector that the problem contributor would regard as one example of a 'good' description of the relevant system. (Problems may be non-unique, and 'good' may involve subjective choices; i.e. this is *one* good model but not necessarily the *best* or *only* good model.) 
- :code:`tp.data_size` -- The dimension, :math:`N`, of the data vector, :math:`\mathbf{d}`
- :code:`tp.data` -- A :py:class:`numpy.ndarray` with shape :math:`(N,)` containing a data vector, :math:`\mathbf{d}`.
- :code:`tp.forward(model)` -- A function that takes one model vector (a :py:class:`numpy.ndarray` of shape :math:`(M,)`) as input, and returns a simulated data vector (:math:`\mathbf{g}(\mathbf{m})`, :py:class:`numpy.ndarray` with shape :math:`(N,)`). The output from :code:`tp.forward` can be assumed to be directly comparable to the data vector :code:`tp.data`. (For some test problems, :code:`tp.forward()` will accept an optional argument :code:`return_jacobian = True`; see below for more details.)

In addition, the following attributes are standardized, but optional:

- :code:`tp.description` -- A :py:class:`str` containing a desciption/summary of the test problem.
- :code:`tp.covariance_matrix` -- A :py:class:`numpy.ndarray` with shape :math:`(N,N)` containing a covariance matrix, :math:`\mathbf{C_d}`, describing the (assumed) uncertainty on :math:`\mathbf{d}`.
- :code:`tp.inverse_covariance_matrix` -- A :py:class:`numpy.ndarray` with shape :math:`(N,N)` containing :math:`\mathbf{C_{d}^{-1}}`.
- :code:`tp.jacobian(model)` -- A function that takes one model vector (a :py:class:`numpy.ndarray` of shape :math:`(M,)`) as input, and returns an :py:class:`numpy.ndarray` of shape :math:`(N,M)` containing :math:`\mathbf{G}` such that :math:`[\mathbf{G}]_{ij} = {\partial[\mathbf{g}(\mathbf{m})]_i}/{\partial [\mathbf{m}]_j}`. Problems that define :code:`tp.jacobian()` will also accept an optional argument :code:`return_jacobian = True` passed to :code:`tp.forward()`, which then returns a :math:`(\mathbf{g}(\mathbf{m}), \mathbf{G})` pair. In some cases this will be computationally more efficient than calling :code:`tp.forward()` and :code:`tp.jacobian()` separately.
- :code:`tp.plot_model(model)` -- A function to visualise a single model-like vector in some problem-appropriate manner.
- :code:`tp.plot_data(data1, data2=None)` -- A function to visualise one or two data-like vectors in a problem-appropriate manner.
- :code:`tp.misfit(data,data2)` -- A function that returns a :py:class:`float` representing a problem-appropriate measure of the disagreement between two data-like vectors (i.e. a value of 0 implies a perfect match).
- :code:`tp.log_likelihood(data,data2)` -- A function that computes a log-likelihood function (:py:class:`float`) between two data-like vectors.
- :code:`tp.log_prior(model)` -- A function that implements a prior distribution in model space, returning the log-probability (:py:class:`float`) for a model-like vector.

If a test problem does not implement a given attribute, attempting to use it will raise a :py:class:`NotImplementedError` exception (which can then be caught and handled as necessary).


All test problems will also define the following basic metadata:

- :code:`tp.problem_title`
- :code:`tp.problem_short_description` -- A few sentences summarising the problem.
- :code:`tp.author_names`
- :code:`tp.contact_name` -- Primary contributor/maintainer for the Espresso test problem.
- :code:`tp.contact_email` 
- :code:`tp.citations` -- List of :code:`(citation, doi)` pairs for any publication(s) that directly describe the test problem;
- :code:`tp.linked_sites` -- List of :code:`(title, address)` pairs for any related resources (e.g. links to data sources or external Github repos).
    
Finally, test problems may expose additional attributes beyond the scope of the Espresso API. For more details see :doc:`api/index` and :doc:`contrib/index`.
    
