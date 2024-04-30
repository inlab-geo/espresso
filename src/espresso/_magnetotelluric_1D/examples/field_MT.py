import numpy as np
import matplotlib.pyplot as plt

import cofi
from espresso import Magnetotelluric1D
from espresso._magnetotelluric_1D import load_data

mt = Magnetotelluric1D()

#### 0.1 Load the data
filename = '16-A_KN2.dat'
freqs, dobs, derr = load_data(filename, error_floor = 0.05)

#### 0.2 Set-up the data
mt.set_obs_data(dobs, derr, freqs)

#### 0.3. Define a starting 1D mesh and model for the inversion
nLayers, min_thickness, vertical_growth= 100, 3, 1.1
thicknesses = [min_thickness * vertical_growth**i for i in range(nLayers)]
starting_depths = np.cumsum(thicknesses)
starting_model = np.ones((len(starting_depths)+1)) * 2 # 100 ohm.m starting model (log10 scale) 

#### 0.4. Set new starting model and mesh
mt.set_start_model(starting_model)
mt.set_start_mesh(starting_depths)


## 1. Define the problem

# define CoFI BaseProblem
mt_problem = cofi.BaseProblem()
mt_problem.set_initial_model(mt.starting_model)

# add regularization: smoothing
smoothing_factor = 50
reg_smoothing = smoothing_factor * cofi.utils.QuadraticReg("smoothing", (mt.model_size,))
reg = reg_smoothing

def objective_func(model, reg):
    dpred = mt.forward(model)
    data_misfit = mt.misfit(mt.data,dpred,mt.inverse_covariance_matrix)
    model_reg = reg(model)
    return  data_misfit + model_reg

mt_problem.set_objective(objective_func, args=[reg])


## 2. Define the inversion options

my_options = cofi.InversionOptions()
my_options.set_tool("scipy.optimize.minimize")
my_options.set_params(method="L-BFGS-B",options={'ftol':1e-3,'maxiter': 100})


## 3. Start an inversion
print("Running inversion...")
inv = cofi.Inversion(mt_problem, my_options)
inv_result = inv.run()
print("   done!")


## 4. Plotting inversion results

fig1 = mt.plot_model(inv_result.model, max_depth = -500, title='Smooth inversion');            # inverted model
fig = mt.plot_data(mt.data, mt.forward(inv_result.model), Cm = mt.covariance_matrix)
nRMSE = np.sqrt(mt.misfit(mt.data, mt.forward(inv_result.model), Cm_inv = mt.inverse_covariance_matrix)/mt.data_size)
print('nRMSE = %.3f'%nRMSE)

plt.show()

