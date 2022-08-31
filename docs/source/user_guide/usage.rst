===
Usage
===

.. code-block:: python
    from cofi-espresso import <testproblem> as testproblem 

    tp = testproblem(example_number = 1)

    model = tp.starting_model 
    
    predictions, A = tp.forward(model)
    residuals = tp.data - predictions

    model 






