"""
Xray Tomography
===============

.. raw:: html

   <!-- Please leave the cell below as it is -->

"""


######################################################################
# Linear travel time tomography based on x-ray tracers. In this notebook,
# we use ``cofi`` to run a linear system solver for this problem.
# 


######################################################################
# {{ badge }}
# 


######################################################################
# .. raw:: html
# 
#    <!-- Again, please don't touch the markdown cell above. We'll generate badge 
#         automatically from the above cell. -->
# 
# .. raw:: html
# 
#    <!-- This cell describes things related to environment setup, so please add more text 
#         if something special (not listed below) is needed to run this notebook -->
# 
# ..
# 
#    If you are running this notebook locally, make sure you’ve followed
#    `steps
#    here <https://github.com/inlab-geo/cofi-examples#run-the-examples-with-cofi-locally>`__
#    to set up the environment. (This
#    `environment.yml <https://github.com/inlab-geo/cofi-examples/blob/main/envs/environment.yml>`__
#    file specifies a list of packages required to run the notebooks)
# 


######################################################################
# 0. Import modules
# -----------------
# 

# -------------------------------------------------------- #
#                                                          #
#     Uncomment below to set up environment on "colab"     #
#                                                          #
# -------------------------------------------------------- #

# !pip install -U cofi
# !pip install -U cofi-espresso

######################################################################
#

import numpy as np
import matplotlib.pyplot as plt

from cofi import BaseProblem, InversionOptions, Inversion
from cofi_espresso import XrayTomography

np.random.seed(42)

######################################################################
#


######################################################################
# 1. Define the problem
# ---------------------
# 

xrt = XrayTomography()

######################################################################
#

xrt_problem = BaseProblem()
xrt_problem.set_data(xrt.data)
xrt_problem.set_jacobian(xrt.jacobian(xrt.starting_model))

######################################################################
#

sigma = 0.01
lamda = 0.5
data_cov = np.identity(xrt.data_size) * sigma
reg_matrix = np.identity(xrt.model_size)

######################################################################
#

xrt_problem.set_data_covariance(data_cov)
xrt_problem.set_regularisation(2, lamda, reg_matrix)

######################################################################
#


######################################################################
# Review what information is included in the ``BaseProblem`` object:
# 

xrt_problem.summary()

######################################################################
#


######################################################################
# 2. Define the inversion options
# -------------------------------
# 

my_options = InversionOptions()
my_options.set_tool("scipy.linalg.lstsq")

######################################################################
#


######################################################################
# Review what’s been defined for the inversion we are about to run:
# 

my_options.summary()

######################################################################
#


######################################################################
# 3. Start an inversion
# ---------------------
# 

inv = Inversion(xrt_problem, my_options)
inv_result = inv.run()
inv_result.summary()

######################################################################
#


######################################################################
# 4. Plotting
# -----------
# 

xrt.plot_model(inv_result.model);

######################################################################
#


######################################################################
# 5. Reflections / Conclusion / Further reading
# ---------------------------------------------
# 


######################################################################
# We can see that…
# 


######################################################################
# --------------
# 
# Watermark
# ---------
# 
# .. raw:: html
# 
#    <!-- Feel free to add more modules in the watermark_list below, if more packages are used -->
# 
# .. raw:: html
# 
#    <!-- Otherwise please leave the below code cell unchanged -->
# 

watermark_list = ["cofi", "numpy", "scipy", "matplotlib", "emcee", "arviz"]
for pkg in watermark_list:
    pkg_var = __import__(pkg)
    print(pkg, getattr(pkg_var, "__version__"))

######################################################################
#