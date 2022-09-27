"""Seismic waveform inference with Fast Marching and CoFI

Based on original scripts by Andrew Valentine & Malcolm Sambridge -
Research Schoole of Earth Sciences, The Australian National University
Last updated July 2022

"""

import numpy as np
import findiff

from cofi import BaseProblem, InversionOptions, Inversion
# from cofi.utils import QuadraticReg
from cofi_espresso import FmmTomography

# get espresso problem FmmTomography information
fmm = FmmTomography()
model_size = fmm.model_size
model_shape = (32, 48)      # TODO to be replaced by `fmm.model_shape`
data_size = fmm.data_size
ref_start_model = fmm.starting_model
ref_start_slowness = 1/ref_start_model

# define CoFI BaseProblem
fmm_problem = BaseProblem()
fmm_problem.set_initial_model(ref_start_slowness)

# add regularisation: damping + smoothing
damping_factor = 100.0
smoothing_factor = 5e3

d2_dx2 = findiff.FinDiff(0, 1, 2)
d2_dy2 = findiff.FinDiff(1, 1, 2)
matx = d2_dx2.matrix(model_shape)
maty = d2_dy2.matrix(model_shape)
Dm = np.vstack((matx.toarray(), maty.toarray()))

def damping_reg(slowness):
    slowness_diff = slowness - ref_start_slowness
    return damping_factor * slowness_diff.T @ slowness_diff
def damping_grad(slowness):
    return damping_factor * (slowness - ref_start_slowness)
def damping_hess():
    return damping_factor * np.eye((model_size))

def smoothing_reg(slowness):
    weighted_slowness = Dm @ slowness
    return smoothing_factor * weighted_slowness.T @ weighted_slowness
def smoothing_grad(slowness):
    return smoothing_factor * Dm.T @ Dm @ slowness
def smoothing_hess():
    return smoothing_factor * Dm.T @ Dm

def reg(slowness):
    return damping_reg(slowness) + smoothing_reg(slowness)
def reg_grad(slowness):
    return damping_grad(slowness) + smoothing_grad(slowness)
reg_hess = damping_hess() + smoothing_hess()

fmm_problem.set_regularisation(reg)

# TODO above to be replaced by cofi.utils (below)
# reg_damping = QuadraticReg(damping_factor, model_size, "damping")
# reg_smoothing = QuadraticReg(smoothing_factor, model_size, "smoothing")
# reg = reg_damping + reg_smoothing
# fmm_problem.set_regularisation(reg)

sigma =  0.00001                   # Noise is 1.0E-4 is ~5% of standard deviation of initial travel time residuals
def objective_func(slowness):
    ttimes = fmm.forward(1/slowness)
    residual = fmm.data - ttimes
    return residual.T @ residual / sigma**2 + reg(slowness)
def gradient(slowness):
    ttimes, A = fmm.forward(1/slowness, with_jacobian=True)
    return -A.T @ (fmm.data - ttimes) / sigma**2 + reg_grad(slowness)
def hessian(slowness):
    A = fmm.jacobian(1/slowness)
    return A.T @ A / sigma**2 + reg_hess

fmm_problem.set_objective(objective_func)
fmm_problem.set_gradient(gradient)
fmm_problem.set_hessian(hessian)

# Define CoFI InversionOptions
inv_options = InversionOptions()
inv_options.set_tool("scipy.optimize.minimize")
inv_options.set_params(method="Newton-CG")

# Define CoFI Inversion and run
inv = Inversion(fmm_problem, inv_options)
inv_result = inv.run()
fig1 = fmm.plot_model(1/inv_result.model)
fig1.savefig("tmp1")

num_iterations = 3
m = fmm_problem.initial_model
for i in range(num_iterations):
    print(fmm_problem.objective(m))
    grad = fmm_problem.gradient(m)
    hess = fmm_problem.hessian(m)
    step = -np.linalg.inv(hess).dot(grad)
    m += step
fig2 = fmm.plot_model(1/m)
fig2.savefig("tmp2")
