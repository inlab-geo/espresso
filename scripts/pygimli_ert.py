"""
Electrical Resistivity Tomography
=================================

Using the ert solver implemented provided by
`PyGIMLi <https://www.pygimli.org/>`__, we use different ``cofi``
solvers to solve the corresponding inverse problem.

"""


######################################################################
# .. raw:: html
# 
# 	<badge><a href="https://colab.research.google.com/github/inlab-geo/cofi-examples/blob/main/notebooks/pygimli_ert/pygimli_ert.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></badge>


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
# !pip install pygimli

######################################################################
#

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pygimli as pg

from cofi import BaseProblem, InversionOptions, Inversion

np.random.seed(42)

######################################################################
#


######################################################################
# 1. Define the problem
# ---------------------
# 

my_problem = BaseProblem()
# my_problem.set_forward(some_function_here)
# ...

# A list of set methods you can use on BaseProblem can be found here:
# https://cofi.readthedocs.io/en/latest/api/generated/cofi.BaseProblem.html#set-methods

# Feel free to add more cells as needed
# You may also import dataset in this section

######################################################################
#


######################################################################
# Review what information is included in the ``BaseProblem`` object:
# 

my_problem.summary()

######################################################################
#


######################################################################
# 2. Define the inversion options
# -------------------------------
# 

my_options = InversionOptions()
# my_options.set_tool(some_tool_name_here)
# my_options.set_params(some_option = some_value)

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

inv = Inversion(my_problem, my_options)
inv_result = inv.run()
inv_result.summary()

######################################################################
#


######################################################################
# 4. Plotting
# -----------
# 

# Add some plotting here if applicable

######################################################################
#


######################################################################
# 5. Reflections / Conclusion / Further reading
# ---------------------------------------------
# 


######################################################################
# We can see that…
# 