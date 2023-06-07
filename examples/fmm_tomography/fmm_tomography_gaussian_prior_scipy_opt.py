"""Seismic waveform inference with Fast Marching and CoFI

Based on original scripts by Andrew Valentine & Malcolm Sambridge -
Research Schoole of Earth Sciences, The Australian National University
Last updated July 2022

"""

import numpy as np

from cofi import BaseProblem, InversionOptions, Inversion
from cofi.utils import GaussianPrior
from espresso import FmmTomography


# get espresso problem FmmTomography information
fmm = FmmTomography()
model_size = fmm.model_size
model_shape = fmm.model_shape
data_size = fmm.data_size
ref_start_slowness = fmm.starting_model

# define regularization (Gaussian Prior)
corrx = 3.0
corry = 3.0
sigma_slowness = 0.002
gaussian_prior = GaussianPrior(
    model_covariance_inv=((corrx, corry), sigma_slowness),
    mean_model=ref_start_slowness.reshape(model_shape)
)

# define data covariance matrix
sigma = 0.00001
Cd = np.zeros([data_size, data_size])
np.fill_diagonal(Cd, sigma**2)
Cdi = np.zeros([data_size, data_size])
np.fill_diagonal(Cdi, 1 / sigma**2)

# define chi square function
def chi_square(model_slowness, esp_fmm, Cd_inv):
    pred = esp_fmm.forward(model_slowness)
    residual = esp_fmm.data - pred
    model_diff = model_slowness - ref_start_slowness
    return residual.T @ Cd_inv @ residual + gaussian_prior(model_slowness)


def gradient(model_slowness, esp_fmm, Cd_inv):
    pred, jac = esp_fmm.forward(model_slowness, with_jacobian=True)
    residual = esp_fmm.data - pred
    model_diff = model_slowness - ref_start_slowness
    return -jac.T @ Cd_inv @ residual + gaussian_prior.gradient(model_slowness)


def hessian(model_slowness, esp_fmm, Cd_inv):
    A = esp_fmm.jacobian(model_slowness)
    return A.T @ Cd_inv @ A + gaussian_prior.hessian(model_slowness)


# define CoFI BaseProblem
fmm_problem = BaseProblem()
fmm_problem.set_initial_model(ref_start_slowness)
fmm_problem.set_objective(chi_square, args=[fmm, Cdi])
fmm_problem.set_gradient(gradient, args=[fmm, Cdi])
fmm_problem.set_hessian(hessian, args=[fmm, Cdi])

# define CoFI InversionOptions
inv_options = InversionOptions()
inv_options.set_tool("scipy.optimize.minimize")
method = "Newton-CG"
inv_options.set_params(method=method, options={"xtol": 1e-12})

# define CoFI Inversion and run
inv = Inversion(fmm_problem, inv_options)
inv_result = inv.run()
ax1 = fmm.plot_model(inv_result.model)
ax1.get_figure().savefig(f"figs/fmm_gaussian_prior_scipy_{method}")

# Plot the true model
ax2 = fmm.plot_model(fmm.good_model)
ax2.get_figure().savefig("figs/fmm_true_model")
