"""Seismic waveform inference with Fast Marching and CoFI

Based on original scripts by Andrew Valentine & Malcolm Sambridge -
Research Schoole of Earth Sciences, The Australian National University
Last updated July 2022

"""

import numpy as np

from cofi import BaseProblem, InversionOptions, Inversion
from cofi.utils import QuadraticReg
from cofi_espresso import FmmTomography


# get espresso problem FmmTomography information
fmm = FmmTomography()
model_size = fmm.model_size         # number of model parameters
model_shape = fmm.model_shape       # 2D spatial grids
data_size = fmm.data_size           # number of data points
ref_start_slowness = fmm.starting_model

# define CoFI BaseProblem
fmm_problem = BaseProblem()
fmm_problem.set_initial_model(ref_start_slowness)

# add regularization: damping + smoothing
damping_factor = 100
flattening_factor = 0
smoothing_factor = 5e3
reg_damping = QuadraticReg(damping_factor, model_size, "damping", ref_start_slowness)
reg_flattening = QuadraticReg(flattening_factor, model_shape, "flattening")
reg_smoothing = QuadraticReg(smoothing_factor, model_shape, "smoothing")
reg = reg_damping + reg_flattening + reg_smoothing
fmm_problem.set_regularization(reg)

sigma =  0.00001                   # Noise is 1.0E-4 is ~5% of standard deviation of initial travel time residuals
def objective_func(slowness):
    ttimes = fmm.forward(slowness)
    residual = fmm.data - ttimes
    data_misfit = residual.T @ residual / sigma**2
    model_reg = reg(slowness)
    return  data_misfit + model_reg
def gradient(slowness):
    ttimes, A = fmm.forward(slowness, with_jacobian=True)
    data_misfit_grad = -A.T @ (fmm.data - ttimes) / sigma**2
    model_reg_grad = reg.gradient(slowness)
    return  data_misfit_grad + model_reg_grad
def hessian(slowness):
    A = fmm.jacobian(slowness)
    data_misfit_hess = A.T @ A / sigma**2 
    model_reg_hess = reg.hessian(slowness)
    return data_misfit_hess + model_reg_hess

fmm_problem.set_objective(objective_func)
fmm_problem.set_gradient(gradient)
fmm_problem.set_hessian(hessian)

# Define CoFI InversionOptions
inv_options = InversionOptions()
inv_options.set_tool("scipy.optimize.minimize")
method = "Newton-CG"
inv_options.set_params(method=method, options={"xtol":1e-10, "disp":True})

# Define CoFI Inversion and run
# inv = Inversion(fmm_problem, inv_options)
# inv_result = inv.run()
# fig1 = fmm.plot_model(inv_result.model)
# fig1.savefig(f"figs/fmm_{int(damping_factor)}_{int(smoothing_factor)}_scipy_{method}")

# Run with another approach - iterative Newton full-step updates
num_iterations = 5
m = fmm_problem.initial_model
for i in range(num_iterations):
    print(fmm_problem.objective(m))
    grad = fmm_problem.gradient(m)
    hess = fmm_problem.hessian(m)
    step = -np.linalg.inv(hess).dot(grad)
    m += 0.5*step
fig2 = fmm.plot_model(m)
fig2.savefig(f"figs/fmm_{int(damping_factor)}_{int(flattening_factor)}_{int(smoothing_factor)}_simple_newton")

# # Plot the true model
# fig3 = fmm.plot_model(fmm.good_model)
# fig3.savefig("figs/fmm_true_model")
