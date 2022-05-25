"""Polynomial Linear regression solved by linear system solver

This file sets up an example from setting up problem to running the inversion:
- For the problem: polynomial linear regression,
- Using the tool: linear system solver (scipy.linalg.lstsq)

The function we are going to fit is: y = -6 - 5x + 2x^2 + x^3
    
"""

############# 0. Import modules #######################################################

import numpy as np
import matplotlib.pyplot as plt
from cofi import BaseProblem, InversionOptions, Inversion

np.random.seed(42)

show_plot = True
show_summary = True


############# 1. Define the problem ###################################################

# generate data with random Gaussian noise
def basis_func(x):
    return np.array([x**i for i in range(4)]).T                           # x -> G
_m_true = np.array([-6,-5,2,1])                                           # m

sample_size = 20                                                          # N
x = np.random.choice(np.linspace(-3.5,2.5), size=sample_size)             # x
def forward_func(m):
    return basis_func(x) @ m                                              # m -> y_synthetic
y_observed = forward_func(_m_true) + np.random.normal(0,1,sample_size)    # d

if show_plot:
    _x_plot = np.linspace(-3.5,2.5)
    _G_plot = basis_func(_x_plot)
    _y_plot = _G_plot @ _m_true
    plt.figure(figsize=(12,8))
    plt.plot(_x_plot, _y_plot, color="darkorange", label="true model")
    plt.scatter(x, y_observed, color="lightcoral", label="observed data")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()

# define the problem in cofi
inv_problem = BaseProblem()
inv_problem.name = "Polynomial Regression"
inv_problem.set_dataset(x, y_observed)
inv_problem.set_jacobian(basis_func(x))
if show_summary:
    inv_problem.summary()


############# 2. Define the inversion options #########################################
inv_options = InversionOptions()
inv_options.set_tool("scipy.linalg.lstsq")
if show_summary:
    inv_options.summary()


############# 3. Start an inversion ###################################################
inv = Inversion(inv_problem, inv_options)
inv_result = inv.run()
if show_summary:
    inv_result.summary()


############# 4. Plot result ##########################################################
if show_plot:
    _x_plot = np.linspace(-3.5,2.5)
    _G_plot = basis_func(_x_plot)
    _y_plot = _G_plot @ _m_true
    _y_synth = _G_plot @ inv_result.model
    plt.figure(figsize=(12,8))
    plt.plot(_x_plot, _y_plot, color="darkorange", label="true model")
    plt.plot(_x_plot, _y_synth, color="seagreen", label="least squares solution")
    plt.scatter(x, y_observed, color="lightcoral", label="original data")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()
