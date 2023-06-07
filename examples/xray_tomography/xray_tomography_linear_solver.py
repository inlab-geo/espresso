"""Xray tomography problem solved with CoFI linear system solver,
with data uncertainty and regularization taken into account.
"""

import numpy as np
from cofi import BaseProblem, InversionOptions, Inversion
from cofi.utils import QuadraticReg
from espresso import XrayTomography


# define CoFI BaseProblem
xrt = XrayTomography()
xrt_problem = BaseProblem()
xrt_problem.set_data(xrt.data)
xrt_problem.set_jacobian(xrt.jacobian(xrt.starting_model))
sigma = 0.002
lamda = 50
data_cov_inv = np.identity(xrt.data_size) * (1 / sigma**2)
xrt_problem.set_data_covariance_inv(data_cov_inv)
xrt_problem.set_regularization(lamda * QuadraticReg(model_shape=(xrt.model_size,)))

# define CoFI InversionOptions
my_options = InversionOptions()
my_options.set_tool("scipy.linalg.lstsq")

# define CoFI Inversion and run it
inv = Inversion(xrt_problem, my_options)
inv_result = inv.run()
inv_result.summary()

# plot inferred model
ax = xrt.plot_model(inv_result.model, clim=(1, 1.5))
ax.get_figure().savefig("xray_tomography_inferred_model")

# plot true model
ax_true = xrt.plot_model(xrt.good_model)
ax_true.get_figure().savefig("xray_tomography_true_model")
