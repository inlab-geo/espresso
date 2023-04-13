"""Seismic waveform inference with Fast Marching and CoFI

Based on original scripts by Andrew Valentine & Malcolm Sambridge -
Research Schoole of Earth Sciences, The Australian National University
Last updated July 2022

"""

import numpy as np

from cofi import BaseProblem, InversionOptions, Inversion
from espresso import FmWavefrontTracker

# get espresso problem FmWavefrontTracker information
fmm = FmWavefrontTracker()
model_size = fmm.model_size
model_shape = fmm.model_shape
data_size = fmm.data_size
ref_start_slowness = fmm.starting_model

# define model covariance matrix
corrx = 3.0
corry = 3.0
sigma_slowness = 0.002
nx, ny = model_shape

Sc = np.zeros([nx * ny, nx * ny])
Cp = np.zeros([nx * ny, nx * ny])
for i1 in range(nx):
    for j1 in range(ny):
        Sc[i1 * ny + j1, i1 * ny + j1] = sigma_slowness**2
        for i2 in range(nx):
            for j2 in range(ny):
                cx = ((float)(i1 - i2)) / corrx
                cy = ((float)(j1 - j2)) / corry
                Cp[i1 * ny + j1, i2 * ny + j2] = np.exp(-np.sqrt(cx * cx + cy * cy))
                Cp[i1 * ny + j1, i2 * ny + j2] = np.exp(-np.sqrt(cx * cx + cy * cy))
Cp = ((Sc.T).dot(Cp)).dot(Sc)
Cpi = np.linalg.inv(Cp)

# define data covariance matrix
sigma = 0.00001
Cd = np.zeros([data_size, data_size])
np.fill_diagonal(Cd, sigma**2)
Cdi = np.zeros([data_size, data_size])
np.fill_diagonal(Cdi, 1 / sigma**2)


# define chi square function
def chi_square(model_slowness, esp_fmm, Cd_inv, Cp_inv):
    pred = esp_fmm.forward(model_slowness)
    residual = esp_fmm.data - pred
    model_diff = model_slowness - ref_start_slowness
    return residual.T @ Cd_inv @ residual + model_diff.T @ Cp_inv @ model_diff


def gradient(model_slowness, esp_fmm, Cd_inv, Cp_inv):
    pred, jac = esp_fmm.forward(model_slowness, with_jacobian=True)
    residual = esp_fmm.data - pred
    model_diff = model_slowness - ref_start_slowness
    return -jac.T @ Cd_inv @ residual + Cp_inv @ model_diff


def hessian(model_slowness, esp_fmm, Cd_inv, Cp_inv):
    A = esp_fmm.jacobian(model_slowness)
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
inv_options.set_params(method=method, options={"xtol": 1e-12})

# define CoFI Inversion and run
inv = Inversion(fmm_problem, inv_options)
inv_result = inv.run()
fig1 = fmm.plot_model(inv_result.model)
fig1.savefig(f"figs/fmm_gaussian_prior_scipy_{method}")

# Plot the true model
fig3 = fmm.plot_model(fmm.good_model)
fig3.savefig("figs/fmm_true_model")
