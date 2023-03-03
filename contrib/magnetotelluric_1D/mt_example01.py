import matplotlib.pyplot as plt
import numpy as np

import cofi
from cofi_espresso import Magnetotelluric1D


# Load example
mt = Magnetotelluric1D(example_number=1)

# Plot true model and synthetic response
fig1 = mt.plot_model(mt.good_model, title='True model')
#fig = mt.plot_data(mt.data, Cm = mt.covariance_matrix)
plt.show()

# get problem information from espresso magnetotelluric1D
model_size = mt.model_size         # number of model parameters
data_size = mt.data_size           # number of data points
start_logres = mt.starting_model   # starting model

# define CoFI BaseProblem
mt_problem = cofi.BaseProblem()
mt_problem.set_initial_model(start_logres)

# define regularization: smoothing
smoothing_factor = 1
reg_smoothing = cofi.utils.QuadraticReg(smoothing_factor, model_size = model_size, reg_type = "smoothing")
reg = reg_smoothing
reg.model_size

# Define objective function
def objective_func(logres, reg):
    dpred = mt.forward(logres)
    data_misfit = mt.misfit(mt.data,dpred,mt.inverse_covariance_matrix)
    model_reg = reg(logres)
    return  data_misfit + model_reg

mt_problem.set_objective(objective_func, args=[reg])

# Define the inversion options
my_options = cofi.InversionOptions()
my_options.set_tool("scipy.optimize.minimize")
my_options.set_params(method="L-BFGS-B",options={'ftol':1e-3,'maxiter': 100})

# Run the inversion
inv = cofi.Inversion(mt_problem, my_options)
inv_result = inv.run()

# Plot the results
fig2 = mt.plot_model(inv_result.model, title='Inversion model')
fig3 = mt.plot_data(mt.data, mt.forward(inv_result.model), Cm = mt.covariance_matrix)
nRMSE = np.sqrt(mt.misfit(mt.data, mt.forward(inv_result.model), Cm_inv = mt.inverse_covariance_matrix)/mt.data_size)
print('nRMSE = %.3f'%nRMSE)

plt.show()
