"""Xray tomography problem solved with CoFI linear system solver,
with data uncertainty and regularisation taken into account.
"""

import numpy as np
from cofi import BaseProblem, InversionOptions, Inversion
from cofi_espresso import XrayTomography

# define CoFI BaseProblem
xrt = XrayTomography()
xrt_problem = BaseProblem()
xrt_problem.set_data(xrt.data)
xrt_problem.set_jacobian(xrt.jacobian(xrt.starting_model))
sigma = 0.01
lamda = 0.5
data_cov = np.identity(xrt.data_size) * sigma
reg_matrix = np.identity(xrt.model_size)
xrt_problem.set_data_covariance(data_cov)
xrt_problem.set_regularisation(2, lamda, reg_matrix)

# define CoFI InversionOptions
my_options = InversionOptions()
my_options.set_tool("scipy.linalg.lstsq")

# define CoFI Inversion and run it
inv = Inversion(xrt_problem, my_options)
inv_result = inv.run()
inv_result.summary()

# plot inferred model
fig = xrt.plot_model(inv_result.model)
fig.savefig("xray_tomography_inferred_model")
