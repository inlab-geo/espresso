"""Seismic waveform inference with Fast Marching and CoFI

This script searches different damping, flattening and smoothing
factors to get a reasonably good set of hyperparameters.

Based on original scripts by Andrew Valentine & Malcolm Sambridge -
Research Schoole of Earth Sciences, The Australian National University
Last updated July 2022

"""

import numpy as np
import findiff
from cofi import BaseProblem, InversionOptions, Inversion
from cofi_espresso import FmmTomography

fmm = FmmTomography()
model_size = fmm.model_size
model_shape = (32, 48)
data_size = fmm.data_size
ref_start_model = fmm.starting_model
ref_start_slowness = 1 / ref_start_model

fmm_problem = BaseProblem()
fmm_problem.set_initial_model(ref_start_slowness)

d2_dx2 = findiff.FinDiff(0, 1, 2)
d2_dy2 = findiff.FinDiff(1, 1, 2)
matx2 = d2_dx2.matrix(model_shape)
maty2 = d2_dy2.matrix(model_shape)
Dm2 = np.vstack((matx2.toarray(), maty2.toarray()))
d_dx2 = findiff.FinDiff(0, 1, 1)
d_dy2 = findiff.FinDiff(1, 1, 1)
matx = d_dx2.matrix(model_shape)
maty = d_dy2.matrix(model_shape)
Dm1 = np.vstack((matx.toarray(), maty.toarray()))

def damping_reg(slowness):
    slowness_diff = slowness - ref_start_slowness
    return slowness_diff.T @ slowness_diff
def damping_grad(slowness):
    return (slowness - ref_start_slowness)
def damping_hess():
    return np.eye((model_size))
def flattening_reg(slowness):
    weighted_slowness = Dm2 @ slowness
    return weighted_slowness.T @ weighted_slowness
def flattening_grad(slowness):
    return Dm2.T @ Dm2 @ slowness
def flattening_hess():
    return Dm2.T @ Dm2
def smoothing_reg(slowness):
    weighted_slowness = Dm2 @ slowness
    return weighted_slowness.T @ weighted_slowness
def smoothing_grad(slowness):
    return Dm2.T @ Dm2 @ slowness
def smoothing_hess():
    return Dm2.T @ Dm2

def reg(slowness, damping_factor, flattening_factor, smoothing_factor):
    return damping_factor * damping_reg(slowness) + \
        flattening_factor * flattening_reg(slowness) + \
            smoothing_factor * smoothing_reg(slowness)
def reg_grad(slowness, damping_factor, flattening_factor, smoothing_factor):
    return damping_factor * damping_grad(slowness) + \
        flattening_factor * flattening_grad(slowness) + \
            smoothing_factor * smoothing_grad(slowness)
def reg_hess(damping_factor, flattening_factor, smoothing_factor):
    return damping_factor * damping_hess() + \
        flattening_factor * flattening_hess() + \
            smoothing_factor * smoothing_hess()

sigma =  0.00001                   # Noise is 1.0E-4 is ~5% of standard deviation of initial travel time residuals
def objective_func(slowness, d_factor, f_factor, s_factor):
    ttimes = fmm.forward(1/slowness)
    residual = fmm.data - ttimes
    return residual.T @ residual / sigma**2 + reg(slowness, d_factor, f_factor, s_factor)
def gradient(slowness, d_factor, f_factor, s_factor):
    ttimes, A = fmm.forward(1/slowness, with_jacobian=True)
    return -A.T @ (fmm.data - ttimes) / sigma**2 + reg_grad(slowness, d_factor, f_factor, s_factor)
def hessian(slowness, d_factor, f_factor, s_factor):
    A = fmm.jacobian(1/slowness)
    return A.T @ A / sigma**2 + reg_hess(d_factor, f_factor, s_factor)

def run_naive_newton(d_factor, f_factor, s_factor):
    fmm_problem.set_objective(objective_func, args=[d_factor, f_factor, s_factor])
    fmm_problem.set_gradient(gradient, args=[d_factor, f_factor, s_factor])
    fmm_problem.set_hessian(hessian, args=[d_factor, f_factor, s_factor])

    num_iterations = 3
    m = fmm_problem.initial_model.copy()
    for i in range(num_iterations):
        print(fmm_problem.objective(m))
        grad = fmm_problem.gradient(m)
        hess = fmm_problem.hessian(m)
        step = -np.linalg.inv(hess).dot(grad)
        m += step
    fig = fmm.plot_model(1/m)
    fig.savefig(f"figs/fmm_{int(d_factor)}_{int(f_factor)}_{int(s_factor)}_naive_newton")

def run_newton_cg(d_factor, f_factor, s_factor):
    fmm_problem.set_objective(objective_func, args=[d_factor, f_factor, s_factor])
    fmm_problem.set_gradient(gradient, args=[d_factor, f_factor, s_factor])
    fmm_problem.set_hessian(hessian, args=[d_factor, f_factor, s_factor])
    inv_options = InversionOptions()
    inv_options.set_tool("scipy.optimize.minimize")
    inv_options.set_params(method="Newton-CG")
    inv_result = Inversion(fmm_problem, inv_options).run()
    fig = fmm.plot_model(1/inv_result.model)
    fig.savefig(f"figs_newtoncg/fmm_{int(d_factor)}_{int(f_factor)}_{int(s_factor)}_newtom_cg")


# search_range = [1, 100, 500, 1e3, 5e3, 1e4, 5e4]
# for d_factor in search_range:
#     for f_factor in search_range:
#         for s_factor in search_range:
#             run_naive_newton(d_factor, f_factor, s_factor)

search_range = [(1e3,1,1), (1,1e4,1), (1,1,1e4), (1e3,1e4,1), (1e3,1,1e4)]
for (d, f, s) in search_range:
    run_newton_cg(d, f, s)
