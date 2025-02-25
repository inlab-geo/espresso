# Receiver Function Inversion (Kennett)

This inference problem set uses the forward code by Brian Kennett and adapted
by Lupei Zhu and Sheng Wang.

The original code is written in C, and it's wrapped by Cython so that we can
call the forward function from Python.

## Example models

1. A simple 3-layer example with a synthetic dataset

2. A 9-layer example with a synthetic dataset

3. A field data example [from the Computer Programs in Seismology](https://www.eas.slu.edu/eqc/eqc_cps/TUTORIAL/STRUCT/index.html)

## Using the forward module

The above examples are set up to have a good reference model, observed data and other
reference information (such as model size, data size and starting model) in place. 
If you have your own model, or if you have some data and you'd like to invert from it,
you might want to use the forward function directly.

To access the forward module, instantiate an object of `ReceiverFunctionInversionKnt`.

For example:

```python
import espresso
import matplotlib.pyplot as plt

rf_example = espresso.ReceiverFunctionInversionKnt(example_number=1)

# Choose a model and solve the forward problem

model = rf_example.good_model
data_rf = rf_example.forward(model)

# Plot the model and the predicted data

rf_example.plot_model(model, alpha=1)
plt.plot(data_rf)
```
