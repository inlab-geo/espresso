"""
Gravity Density Model
=====================

"""


######################################################################
# .. raw:: html
# 
# 	<badge><a href="https://colab.research.google.com/github/inlab-geo/cofi-examples/blob/main/notebooks/gravity/gravity_density.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></badge>


######################################################################
#    If you are running this notebook locally, make sure you’ve followed
#    `steps
#    here <https://github.com/inlab-geo/cofi-examples#run-the-examples-with-cofi-locally>`__
#    to set up the environment. (This
#    `environment.yml <https://github.com/inlab-geo/cofi-examples/blob/main/environment.yml>`__
#    file specifies a list of packages required to run the notebooks)
# 


######################################################################
# Adapted from `gravity forward
# code <https://github.com/inlab-geo/inversion-test-problems/blob/main/contrib/gravityforward/__init__.py>`__
# written in inversion-test-problems
# 


######################################################################
# --------------
# 
# 0. Import modules
# -----------------
# 

# -------------------------------------------------------- #
#                                                          #
#     Uncomment below to set up environment on "colab"     #
#                                                          #
# -------------------------------------------------------- #

# !pip install -U cofi

######################################################################
#

import numpy as np
import matplotlib.pyplot as plt

from cofi import BaseProblem, InversionOptions, Inversion

from gravity_density_lib import *

np.random.seed(42)

######################################################################
#


######################################################################
# --------------
# 
# 1. Define the problem
# ---------------------
# 
# .. raw:: html
# 
#    <!-- I took out gx, gy for now to make it more straight forward. We can add all kinds of things once it is working.  -->
# 

# Load true model and starting guesses
rec_coords, _, _, z_nodes, model = load_gravity_model()
Starting_model1, Starting_model2, Starting_model3 = load_starting_models()

# Create "observed" data by adding noise to forward solution
noise_level=0.05
gz = forward(model)
dataZ_obs= gz + np.random.normal(loc=0,scale=noise_level*np.max(np.abs(gz)),size=np.shape(gz))  

# Create jacobian
Jz = get_jacobian(model)

# Define depth weighting values
z0=18.6
beta=2
# Define regularization parameter
epsilon=0.2

# Create regularization
# Calculate depth weighting fcn - high values at low z, low values at high z, no zeros.
# Model is [Nx1] with N: no. of cells; W is [NxN] with weighting values on diagonal
W=depth_weight(z_nodes[:,0],z0,beta)
W=np.diag(W)

# Set CoFI problem:
grav_problem = BaseProblem()
grav_problem.name = "Gravity"
grav_problem.set_data(gz)

# Here I linked the function, not the result
grav_problem.set_forward(forward)

# Here I linked to the actual jacobian. Jacobian size is (MxN) with M: receiver and N: model cells
grav_problem.set_jacobian(Jz)

# Set regularization; reg is a function that takes the model as input
grav_problem.set_regularisation(reg_l1, epsilon)

# Use default L2 misfit
grav_problem.set_data_misfit("L2")
grav_problem.set_initial_model(Starting_model3)

# Set gradient, in hope of helping optimisers converge better
def data_misfit_gradient(model):
    return 2* Jz.T @ (forward(model) - gz) / gz.shape[0]
grav_problem.set_gradient(lambda m: data_misfit_gradient(m) + epsilon*reg_gradient_l1(m))

grav_problem.summary()

######################################################################
#


######################################################################
# --------------
# 
# 2. Define the inversion
# -----------------------
# 

inv_options = InversionOptions()
inv_options.set_tool("scipy.optimize.least_squares")

inv_options.summary()

######################################################################
#


######################################################################
# --------------
# 
# 3. Start an inversion runner
# ----------------------------
# 

inv = Inversion(grav_problem, inv_options)
# inv.summary()

######################################################################
#

inv_result = inv.run()
inv_result.summary()

######################################################################
#


######################################################################
# Let’s see the density image from a vertical plane:
# 

result_model = inv_result.model.reshape(12,12,12)

plt.imshow(result_model[::-1,6,:])
plt.colorbar();

######################################################################
#


######################################################################
# From a different angle:
# 

plt.imshow(result_model[6,:,:])
plt.colorbar();

######################################################################
#


######################################################################
# --------------
# 
# Watermark
# ---------
# 

watermark_list = ["cofi", "numpy", "scipy", "matplotlib", "emcee", "arviz"]
for pkg in watermark_list:
    pkg_var = __import__(pkg)
    print(pkg, getattr(pkg_var, "__version__"))

######################################################################
#