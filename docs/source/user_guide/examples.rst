.. highlight:: python

==============
Usage Examples
==============

This page provides some annotated examples showing how Espresso can be used.

Gradient descent
----------------

.. code-block:: python
    :linenos:
    :caption: A generic gradient descent algorithm
    :name: gradient-descent 

    from espresso import <testproblem> as testproblem 

    niterations = 100
    epsilon = 0.01

    tp = testproblem(example_number = 1)

    model = tp.starting_model 
    for k in range(niterations):
        predictions, G = tp.forward(model, return_jacobian = True)
        residuals = tp.data - predictions
        model -= epsilon * G.T.dot(residuals)
    print(model)

The algorithm here is straightforward:

.. math::
    \mathbf{m}^{(k+1)} = \mathbf{m}^{(k)} - \epsilon \mathbf{G^T}\left[\mathbf{d} - \mathbf{g}(\mathbf{m}^{(k)})\right]

with :math:`\mathbf{g}(\mathbf{m})` representing simulated data for model :math:`\mathbf{m}` and :math:`\mathbf{G}` representing the corresponding Jacobian, which has elements given by

.. math::
    G_{ij} = \left.\frac{\partial g_i}{\partial m_j}\right|_{\mathbf{m}={\mathbf{m}^{(k)}}}

Let's go through :ref:`the code <gradient-descent>` in detail and explain the Espresso-specific parts. First we select and import one test problem from Espresso

.. code-block:: python
    :linenos:

    from espresso import <testproblem> as testproblem 

Here ``<testproblem>`` should be replaced by any valid problem name. We then assign some values to variables representing the number of iterations of gradient descent,  and the learning rate :math:`\epsilon`. Next, we instantiate (initialise) the test problem we imported.

.. code-block:: python
    :linenos:
    :lineno-start: 6

    tp = testproblem(example_number = 1)

Individual test problems may contain multiple examples, to provide access to multiple datasets or showcase particular problem characteristics. These can be selected by setting ``example_number`` to the relevant number; consult the documentation for details of what each test problem provides.

Once the instance of :py:class:`EspressoProblem` has been created, it can be used to access various functions and attributes. Each example defines a sensible 'null' or 'initial' model to use for inversion, which we use to initialise our gradient descent algorithm:

.. code-block:: python
    :linenos:
    :lineno-start: 8

    model = tp.starting_model 

We compute simulated data and the Jacobian for our current model estimate, and compare this to the 'data' embedded within our :py:class:`EspressoProblem`.

.. code-block:: python
    :linenos:
    :lineno-start: 10

        predictions, G = tp.forward(model, return_jacobian = True)
        residuals = tp.data - predictions

Finally, we update the model accordingly, and iterate until (hopefully!) a good model is found. 