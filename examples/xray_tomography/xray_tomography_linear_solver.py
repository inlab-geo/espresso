"""Xray tomography problem solved with CoFI linear system solver,
with data uncertainty and regularization taken into account.
"""

import numpy as np
from cofi import BaseProblem, InversionOptions, Inversion
from espresso import XrayTracer


# define CoFI BaseProblem
xrt = XrayTracer()
xrt_problem = BaseProblem()
xrt_problem.set_data(xrt.data)
xrt_problem.set_jacobian(xrt.jacobian(xrt.starting_model))
sigma = 0.002
lamda = 50
data_cov_inv = np.identity(xrt.data_size) * (1 / sigma**2)
reg_matrix = lamda * np.identity(xrt.model_size)
xrt_problem.set_data_covariance_inv(data_cov_inv)
xrt_problem.set_regularization(2, 1, reg_matrix)

# define CoFI InversionOptions
my_options = InversionOptions()
my_options.set_tool("scipy.linalg.lstsq")

# define CoFI Inversion and run it
inv = Inversion(xrt_problem, my_options)
inv_result = inv.run()
inv_result.summary()

# plot inferred model
fig = xrt.plot_model(inv_result.model, clim=(1, 1.5))
fig.savefig("xray_tomography_inferred_model")

# plot true model
fig_true = xrt.plot_model(xrt.good_model)
fig_true.savefig("xray_tomography_true_model")
