# Receiver Function Inversion Knt

This inference problem set uses the forward code by Brian Kennet and adapted
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

To access the forward module, instantiate an object of `ReceiverFunctionInversionKnt` and use the field `.rf`.

For example:

```python
import espresso
rf_example = espresso.ReceiverFunctionInversionKnt()
rf = rf_example.rf

my_model_thicknesses = [10, 20, 0]
my_model_vs = [3.3, 3.4, 4.5]
my_model_vp_vs_ratio = [1.732, 1.732, 1.732]
my_ray_param_s_km = 0.07
my_time_shift = 5
my_time_duration = 50
my_time_sampling_interval = 0.1
my_guass = 1.0

data_rf = rf.rf_calc(
    ps=0, 
    thik=my_model_thicknesses, 
    beta=my_model_vs, 
    kapa=my_model_vp_vs_ratio, 
    p=my_ray_param_s_km, 
    duration=my_time_duration, 
    dt=my_time_sampling_interval, 
    shft=my_time_shift, 
    gauss=my_gauss
)
data_times = np.arange(data_rf.size) * my_time_sampling_interval - my_time_shift

# if you'd like to visualise it
import matplotlib.pyplot as plt

plt.scatter(data_times, data_rf)
```
