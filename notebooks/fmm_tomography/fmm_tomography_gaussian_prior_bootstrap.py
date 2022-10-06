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

# define model covariance matrix
corrx=3.0
corry=3.0
sigma_slowness=0.002
nx, ny = model_shape

Sc=np.zeros([nx*ny,nx*ny])
Cp=np.zeros([nx*ny,nx*ny])
for i1 in range(nx):
    for j1 in range(ny):
        Sc[i1*ny+j1,i1*ny+j1]=sigma_slowness**2
        for i2 in range(nx):
            for j2 in range(ny):
                cx = (((float) (i1 - i2)) / corrx);
                cy = (((float) (j1 - j2)) / corry);
                Cp[i1*ny+j1,i2*ny+j2]=np.exp (-np.sqrt (cx * cx + cy * cy));
                Cp[i1*ny+j1,i2*ny+j2]=np.exp (-np.sqrt (cx * cx + cy * cy));
Cp=((Sc.T).dot(Cp)).dot(Sc)
Cpi=np.linalg.inv(Cp)

# define data covariance matrix
sigma = 0.00001
Cd=np.zeros([data_size,data_size])
np.fill_diagonal(Cd, sigma**2)
Cdi=np.zeros([data_size,data_size])
np.fill_diagonal(Cdi, 1/sigma**2)

# define chi square function
def chi_square(model_slowness, esp_fmm, Cd_inv, Cp_inv):
    pred = esp_fmm.forward(1/model_slowness)
    residual = esp_fmm.data - pred
    model_diff = model_slowness - 1/esp_fmm.starting_model
    return residual.T @ Cd_inv @ residual + model_diff.T @ Cp_inv @ model_diff
def gradient(model_slowness, esp_fmm, Cd_inv, Cp_inv):
    pred, jac = esp_fmm.forward(1/model_slowness, with_jacobian=True)
    residual = esp_fmm.data - pred
    model_diff = model_slowness - 1/esp_fmm.starting_model
    return -jac.T @ Cd_inv @ residual + Cp_inv @ model_diff
def hessian(model_slowness, esp_fmm, Cd_inv, Cp_inv):
    A = esp_fmm.jacobian(1/model_slowness)
    return A.T @ Cd_inv @ A + Cp_inv

# define CoFI BaseProblem
fmm_problem = BaseProblem()
fmm_problem.set_initial_model(ref_start_slowness)
fmm_problem.set_objective(chi_square, args=[fmm, Cdi, Cpi])
fmm_problem.set_gradient(gradient, args=[fmm, Cdi, Cpi])
fmm_problem.set_hessian(hessian, args=[fmm, Cdi, Cpi])

# define CoFI InversionOptions
inv_options = InversionOptions()
inv_options.set_tool("scipy.optimize.minimize")
method = "Newton-CG"
inv_options.set_params(method=method, options={"xtol":1e-16})

# # define CoFI Inversion and run
inv = Inversion(fmm_problem, inv_options)
inv_result = inv.run()
fig1 = fmm.plot_model(1/inv_result.model)
fig1.savefig(f"fmm_gaussian_prior_scipy_{method}")

# alternative approach
def run_naive_newton():
    num_iterations = 3
    m = fmm_problem.initial_model.copy()
    for i in range(num_iterations):
        print(fmm_problem.objective(m))
        grad = fmm_problem.gradient(m)
        hess = fmm_problem.hessian(m)
        step = -np.linalg.inv(hess).dot(grad)
        m += step
    fig2 = fmm.plot_model(1/m)
    fig2.savefig(f"fmm_gaussian_prior_naive_newton")

# run_naive_newton()

# Plot the true model
fig3 = fmm.plot_model(fmm.good_model)
fig3.savefig("fmm_true_model")

# generate Bayesian parametric bootstrap samples
s_ref = ref_start_slowness
t_ref = fmm.data
Lp = np.linalg.cholesky(Cp)
posterior = []
for ii in range(50):
    print("Sample", ii)
    # generate a realisation of the data and prior
    x = np.random.normal(size=model_shape)
    s0 = s_ref + (Lp @ x).reshape(model_shape)
    m0 = 1 / s0

