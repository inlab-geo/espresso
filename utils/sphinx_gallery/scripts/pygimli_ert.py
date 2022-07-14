"""
PyGIMLi - Electrical Resistivity Tomography
===========================================

"""


######################################################################
# Using the ERT solver implemented provided by
# `PyGIMLi <https://www.pygimli.org/>`__, we use different ``cofi``
# solvers to solve the corresponding inverse problem.
# 


######################################################################
# .. raw:: html
# 
# 	<badge><a href="https://colab.research.google.com/github/inlab-geo/cofi-examples/blob/main/notebooks/pygimli_ert/pygimli_ert.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></badge>


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
#    `environment.yml <https://github.com/inlab-geo/cofi-examples/blob/main/environment.yml>`__
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

# !MINICONDA_INSTALLER_SCRIPT=Miniconda3-4.5.4-Linux-x86_64.sh
# !MINICONDA_PREFIX=/usr/local
# !wget https://repo.continuum.io/miniconda/$MINICONDA_INSTALLER_SCRIPT
# !chmod +x $MINICONDA_INSTALLER_SCRIPT
# !./$MINICONDA_INSTALLER_SCRIPT -b -f -p $MINICONDA_PREFIX
# !conda install -c gimli pygimli -y

# import sys
# _ = (sys.path.append("/usr/local/lib/python3.7/site-packages"))

######################################################################
#


######################################################################
# We will need the following packages:
# 
# -  ``numpy`` for matrices and matrix-related functions
# -  ``matplotlib`` for plotting
# -  ``pygimli`` for forward modelling of the problem
# -  ``cofi`` for accessing different inference solvers
# 
# Additionally, we wrap some ``pygimli`` code in file
# ``pygimli_ert_lib.py`` and import it here for conciseness.
# 

import numpy as np
import matplotlib.pyplot as plt
import pygimli
from pygimli import meshtools
from pygimli.physics import ert

from cofi import BaseProblem, InversionOptions, Inversion
from cofi.solvers import BaseSolver

from pygimli_ert_lib import *

np.random.seed(42)

######################################################################
#


######################################################################
# 1. Define the problem
# ---------------------
# 


######################################################################
# We first define the true model, the survey and map it on a computational
# mesh designed for the survey and true anomaly.
# 

# PyGIMLi - define measuring scheme, geometry, forward mesh and true model
scheme = survey_scheme()
mesh, rhomap = model_true(scheme)

# plot the true model
ax = pygimli.show(mesh, data=rhomap, label="$\Omega m$", showMesh=True)
ax[0].set_title("True model")

######################################################################
#


######################################################################
# Generate the synthetic data as a container with all the necessary
# information for plotting.
# 

# PyGIMLi - generate data
data, log_data = ert_simulate(mesh, scheme, rhomap)

ax = ert.show(data)
ax[0].set_title("Provided data")

######################################################################
#


######################################################################
# Further, we create a ``pygimli.ert.ERTManager`` instance to keep record
# of problem-specific information like the inversion mesh, and to perform
# forward operation for the inversion solvers.
# 

# create PyGIMLi's ERT manager
ert_manager = ert_manager(data)

######################################################################
#


######################################################################
# The inversion can use a different mesh and the mesh to be used should
# know nothing about the mesh that was designed based on the true model.
# We wrap two kinds of mesh as examples in the library code
# ``pygimli_ert_lib.py``, namely triangular and rectangular mesh.
# 
# Use ``imesh_tri = inversion_mesh(scheme)`` to initialise a triangular
# mesh. This function uses PyGIMLi’s own mesh generator and generates
# triangular mesh automatically from given sensor locations. The resulting
# mesh will have a smaller area as unknowns to be inverted, as well as a
# background part with values prolongated outside from the parametric
# domain by PyGIMLi. You will see an example plot in the code cell below.
# 
# Use ``imesh_rect = inversion_mesh_rect(ert_manager)`` to initislise a
# rectangular mesh. The grid mesh is created from these x and y nodes:
# ``x = np.linspace(start=-5, stop=55, num=61)``, and
# ``y = np.linspace(start=-20,stop=0,num=10)``. And again, there’s a
# triangular background with values prolongated outside from the
# parametric domain by PyGIMLi.
# 
# Here we first demonstrate how to use a *triangular mesh*. Note that this
# makes the inversion problem under-determined.
# 

inv_mesh = inversion_mesh(ert_manager)
ax = pygimli.show(inv_mesh, showMesh=True, markers=True)
ax[0].set_title("Mesh used for inversion")

######################################################################
#


######################################################################
# Check
# `here <https://github.com/inlab-geo/cofi-examples/tree/main/notebooks/pygimli_ert>`__
# for inversion examples using triangular mesh.
# 


######################################################################
# With the inversion mesh created, we now define a starting model, forward
# operator and weighting matrix for regularisation using PyGIMLi.
# 

# PyGIMLi's forward operator (ERTModelling)
forward_oprt = ert_forward_operator(ert_manager, scheme, inv_mesh)

# extract regularisation matrix
Wm = reg_matrix(forward_oprt)

# initialise a starting model for inversion
start_model = starting_model(ert_manager)
ax = pygimli.show(ert_manager.paraDomain, data=start_model, label="$\Omega m$", showMesh=True)
ax[0].set_title("Starting model")

######################################################################
#


######################################################################
# CoFI and other inference packages require a set of functions that
# provide the misfit, the jacobian the residual within the case of scipy
# standardised interfaces. All these functions are defined in the library
# file ``pygimli_ert_lib.py``, so open this file if you’d like to find out
# the details. These functions are:
# 
# -  ``get_response``
# -  ``get_jacobian``
# -  ``get_residuals``
# -  ``get_misfit``
# -  ``get_regularisation``
# -  ``get_gradient``
# -  ``get_hessian``
# 


######################################################################
# With all the above forward operations set up with PyGIMLi, we now define
# the problem in ``cofi`` by setting the problem information for a
# ``BaseProblem`` object.
# 

# hyperparameters
lamda = 0.0005

# CoFI - define BaseProblem
ert_problem = BaseProblem()
ert_problem.name = "Electrical Resistivity Tomography defined through PyGIMLi"
ert_problem.set_forward(get_response, args=[forward_oprt])
ert_problem.set_jacobian(get_jacobian, args=[forward_oprt])
ert_problem.set_residual(get_residual, args=[log_data, forward_oprt])
ert_problem.set_data_misfit(get_data_misfit, args=[log_data, forward_oprt])
ert_problem.set_regularisation(get_regularisation, args=[Wm, lamda])
ert_problem.set_gradient(get_gradient, args=[log_data, forward_oprt, Wm, lamda])
ert_problem.set_hessian(get_hessian, args=[log_data, forward_oprt, Wm, lamda])
ert_problem.set_initial_model(start_model)

######################################################################
#


######################################################################
# Review what information is included in the ``BaseProblem`` object:
# 

ert_problem.summary()

######################################################################
#


######################################################################
# 2. Define the inversion options and run
# ---------------------------------------
# 
# 2.1 SciPy’s optimiser (`TNC <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-tnc.html#optimize-minimize-tnc>`__)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 

ert_problem.suggest_solvers();

######################################################################
#

inv_options_scipy = InversionOptions()
inv_options_scipy.set_tool("scipy.optimize.minimize")
inv_options_scipy.set_params(method="trust-constr")

######################################################################
#


######################################################################
# Review what’s been defined for the inversion we are about to run:
# 

inv_options_scipy.summary()

######################################################################
#

inv = Inversion(ert_problem, inv_options_scipy)
inv_result = inv.run()
inv_result.summary()

######################################################################
#

inv_result.success

######################################################################
#


######################################################################
# Plot the results:
# 

# plot inferred model
inv_result.summary()
ax = pygimli.show(ert_manager.paraDomain, data=inv_result.model, label=r"$\Omega m$")
ax[0].set_title("Inferred model")

######################################################################
#

# plot synthetic data
d = forward_oprt.response(inv_result.model)
ax = ert.showERTData(scheme, vals=d)
ax[0].set_title("Synthetic data from inferred model")

######################################################################
#


######################################################################
# 2.2 A custom `Newton’s optimisation <https://en.wikipedia.org/wiki/Newton%27s_method_in_optimization>`__ approach
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# Now we switch to a Newton’s iterative approach written by ourselves, so
# you’ll have a closer look at what’s happening in the loop.
# 
# First of all, define our own solver.
# 

class GaussNewton(BaseSolver):
    def __init__(self, inv_problem, inv_options):
        __params = inv_options.get_params()
        self._niter = __params.get("niter", 100)
        self._verbose = __params.get("verbose", True)
        self._step = __params.get("step", 1)
        self._model_0 = inv_problem.initial_model
        self._residual = inv_problem.residual
        self._jacobian = inv_problem.jacobian
        self._gradient = inv_problem.gradient
        self._hessian = inv_problem.hessian
        self._misfit = inv_problem.data_misfit if inv_problem.data_misfit_defined else None
        self._reg = inv_problem.regularisation if inv_problem.regularisation_defined else None
        self._obj = inv_problem.objective if inv_problem.objective_defined else None

    def __call__(self):
        current_model = np.array(self._model_0)
        for i in range(self._niter):
            if self._verbose:
                print("-" * 80)
                print(f"Iteration {i+1}")
                print("model min and max:", np.min(current_model), np.max(current_model))
                if self._misfit: print("data misfit:", self._misfit(current_model))
                if self._reg: print("regularisation:", self._reg(current_model))
            term1 = self._hessian(current_model)
            term2 = - self._gradient(current_model)
            model_update = np.linalg.solve(term1, term2) * self._step
            current_model = np.array(current_model + model_update)
        return {"model": current_model, "success": True}

######################################################################
#


######################################################################
# Now, make use of this custom solver and run inversion again:
# 

# hyperparameters
niter = 50
inv_verbose = True
step = 2

# CoFI - define InversionOptions
inv_options_gauss_newton = InversionOptions()
inv_options_gauss_newton.set_tool(GaussNewton)
inv_options_gauss_newton.set_params(niter=niter, verbose=inv_verbose, step=step)

# CoFI - define Inversion, run it
inv = Inversion(ert_problem, inv_options_gauss_newton)
inv_result = inv.run()

######################################################################
#

# plot inferred model
inv_result.summary()
ax = pygimli.show(ert_manager.paraDomain, data=inv_result.model, label=r"$\Omega m$")
ax[0].set_title("Inferred model")

######################################################################
#

# plot synthetic data
d = forward_oprt.response(inv_result.model)
ax = ert.showERTData(scheme, vals=d)
ax[0].set_title("Synthetic data from inferred model")

######################################################################
#


######################################################################
# .. raw:: html
# 
#    <!-- ### 2.3 Bayesian sampling with emcee (exploration)
# 
#    CoFI needs more assumptions about the problem for a sampler to work - these are
#    the log of posterior distribution density and walkers' starting positions.
# 
#    For the log posterior, we define here by specifying `log_prior` and `log_likelihood`.
#    And CoFI will combine them to get the `log_posterior`. -->
# 


######################################################################
# --------------
# 
# Watermark
# ---------
# 

watermark_list = ["cofi", "numpy", "scipy", "pygimli", "matplotlib", "emcee", "arviz"]
for pkg in watermark_list:
    pkg_var = __import__(pkg)
    print(pkg, getattr(pkg_var, "__version__"))

######################################################################
#