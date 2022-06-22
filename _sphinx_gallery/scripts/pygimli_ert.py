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
import pygimli as pg
import pygimli.meshtools as mt
from pygimli.physics import ert

from cofi import BaseProblem, InversionOptions, Inversion

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
scheme = scheme_fwd()
geometry = geometry_true()
fmesh = mesh_fwd(scheme, geometry)
rhomap = markers_to_resistivity()
model_true = model_vec(rhomap, fmesh)

# plot the compuational mesh and the true model
ax=pg.show(fmesh)
ax[0].set_title("Computational Mesh")
ax=pg.show(fmesh,data=model_true,label=r"$\Omega m$")
ax[0].set_title("Resitivity");

######################################################################
#


######################################################################
# Generate the synthetic data as a container with all the necessary
# information for plotting.
# 

# PyGIMLi - generate data
survey = ert.simulate(fmesh, res=rhomap, scheme=scheme)

ax=ert.showERTData(survey,label=r"$\Omega$m")
ax[0].set_title("Aparent Resitivity")

y_obs = np.log(survey['rhoa'].array())

######################################################################
#


######################################################################
# The inversion can use a different mesh and the mesh to be used should
# know nothing about the mesh that was designed based on the true model.
# We wrap two kinds of mesh as examples in the library code
# ``pygimli_ert_lib.py``, namely triangular and rectangular mesh.
# 
# Use ``imesh_tri = mesh_inv_triangular(scheme)`` to initialise a
# triangular mesh, with the following optional arguments and corresponding
# default values:
# 
# -  ``start=[-15, 0]``
# -  ``end=[65, -30]``
# 
# Use ``imesh_rect = mesh_inv_rectangular()`` to initislise a rectangular
# mesh, with the following optional arguments and corresponding default
# values:
# 
# -  ``x_start=-15``
# -  ``x_stop=60``
# -  ``x_num=11``
# -  ``y_start=-30``
# -  ``y_stop=0``
# -  ``y_num=5``
# 
# Here we first demonstrate how to use a *triangular mesh*. Note that this
# makes the inversion problem under-determined.
# 

# PyGIMLi - quick demo of triangular mesh
imesh_tri = mesh_inv_triangular(scheme)

ax=pg.show(imesh_tri)
ax[0].set_title("Inversion Mesh (triangular)");

######################################################################
#


######################################################################
# Check
# `here <https://github.com/inlab-geo/cofi-examples/tree/main/notebooks/pygimli_ert>`__
# for inversion examples using triangular mesh.
# 
# For the purpose of this notebook, we use a *rectangular mesh* for a
# simple demonstration.
# 

# PyGIMLi - create mesh for inversion
imesh = mesh_inv_rectangular()
ax = pygimli.show(imesh)
ax[0].set_title("Inversion Mesh (rectangular)");

######################################################################
#


######################################################################
# With the inversion mesh created, we now define a starting model, forward
# operator and weighting matrix for regularisation using PyGIMLi.
# 

# PyGIMLi - define the starting model on the inversion mesh
model_0 = starting_model(imesh)

# PyGIMLi - set up a forward operator with the inversion mesh
forward_operator = forward_oprt(scheme, imesh)

# PyGIMLi - extract the regularisation weighting matrix
Wm = weighting_matrix(forward_operator, imesh)

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
lamda = 1

# cofi problem definition
ert_problem = BaseProblem()
ert_problem.name = "Electrical Resistivity Tomography defined through PyGIMLi"
ert_problem.set_forward(get_response, args=[forward_operator])
ert_problem.set_jacobian(get_jacobian, args=[forward_operator])
ert_problem.set_residual(get_residuals, args=[y_obs, forward_operator])
ert_problem.set_data_misfit(get_misfit, args=[y_obs, forward_operator, True])
ert_problem.set_regularisation(get_regularisation, lamda=lamda, args=[Wm, True])
ert_problem.set_gradient(get_gradient, args=[y_obs, forward_operator, lamda, Wm])
ert_problem.set_hessian(get_hessian, args=[y_obs, forward_operator, lamda, Wm])
ert_problem.set_initial_model(model_0)

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
# 2.1 SciPy’s optimiser (`L-BFGS-B <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html#optimize-minimize-lbfgsb>`__)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 

ert_problem.suggest_solvers();

######################################################################
#

inv_options = InversionOptions()
inv_options.set_tool("scipy.optimize.minimize")
inv_options.set_params(method="L-BFGS-B")

######################################################################
#


######################################################################
# Review what’s been defined for the inversion we are about to run:
# 

inv_options.summary()

######################################################################
#

inv = Inversion(ert_problem, inv_options)
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

ax=pg.show(
    fmesh,
    data=(model_true),
    label=r"$\Omega m$"
)
ax[0].set_title("True model")

ax=pg.show(
    imesh,
    data=(model_0),
    label=r"$\Omega m$"
)
ax[0].set_title("Starting model")


ax=pg.show(
    imesh,
    data=(inv_result.model),
    label=r"$\Omega m$"
)
ax[0].set_title("Inferred model");

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

from cofi.solvers import BaseSolver

class MyNewtonSolver(BaseSolver):
    def __init__(self, inv_problem, inv_options):
        __params = inv_options.get_params()
        self._niter = __params["niter"]
        self._step = __params["step"]
        self._verbose = __params["verbose"]
        self._model_0 = inv_problem.initial_model
        self._gradient = inv_problem.gradient
        self._hessian = inv_problem.hessian
        self._misfit = inv_problem.data_misfit if inv_problem.data_misfit_defined else None
        self._reg = inv_problem.regularisation if inv_problem.regularisation_defined else None
        self._obj = inv_problem.objective if inv_problem.objective_defined else None
        
    def __call__(self):
        current_model = np.array(self._model_0)
        for i in range(self._niter):
            term1 = self._hessian(current_model)
            term2 = - self._gradient(current_model)
            model_update = np.linalg.solve(term1, term2)
            current_model = np.array(current_model + self._step * model_update)
            if self._verbose:
                print("-" * 80)
                print(f"Iteration {i+1}")
                if self._misfit: self._misfit(current_model)
                if self._reg: self._reg(current_model)
                # if self._obj: print("objective func:", self._obj(current_model))
        return {"model": current_model, "success": True}

######################################################################
#


######################################################################
# Now, make use of this custom solver and run inversion again:
# 

inv_options_own_solver = InversionOptions()
inv_options_own_solver.set_tool(MyNewtonSolver)
inv_options_own_solver.set_params(niter=100, step=1, verbose=True)

inv_own_solver = Inversion(ert_problem, inv_options_own_solver)
inv_own_solver_res = inv_own_solver.run()
inv_own_solver_res.summary()

######################################################################
#


######################################################################
# Plot the results:
# 

ax=pg.show(
    fmesh,
    data=(model_true),
    label=r"$\Omega m$"
)
ax[0].set_title("True model")

ax=pg.show(
    imesh,
    data=(model_0),
    label=r"$\Omega m$"
)
ax[0].set_title("Starting model")


ax=pg.show(
    imesh,
    data=(inv_own_solver_res.model),
    label=r"$\Omega m$"
)
ax[0].set_title("Inferred model");

######################################################################
#


######################################################################
# 2.3 Bayesian sampling with emcee (exploration)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# CoFI needs more assumptions about the problem for a sampler to work -
# these are the log of posterior distribution density and walkers’
# starting positions.
# 
# For the log posterior, we define here by specifying ``log_prior`` and
# ``log_likelihood``. And CoFI will combine them to get the
# ``log_posterior``.
# 

# hyperparameters
nwalkers = 32
nsteps = 10

# define log_likelihood
sigma = 1.0                                     # common noise standard deviation
Cdinv = np.eye(len(y_obs))/(sigma**2)           # inverse data covariance matrix
def log_likelihood(model):
    residual = ert_problem.residual(model)
    return -0.5 * residual @ (Cdinv @ residual).T

# define log_prior
m_lower_bound = np.zeros(model_0.shape)         # lower bound for uniform prior
m_upper_bound = np.ones(model_0.shape) * 250    # upper bound for uniform prior
def log_prior(model):                           # uniform distribution
    for i in range(len(m_lower_bound)):
        if model[i] < m_lower_bound[i] or model[i] > m_upper_bound[i]: return -np.inf
    return 0.0 # model lies within bounds -> return log(1)

# define walkers' starting positions
walkers_start = model_0 + 1e-6 * np.random.randn(nwalkers, model_0.shape[0])

# define them into cofi's BaseProblem object
ert_problem.set_log_likelihood(log_likelihood)
ert_problem.set_log_prior(log_prior)
ert_problem.set_walkers_starting_pos(walkers_start)

######################################################################
#


######################################################################
# As usual, specify how you’d like to run the inversion and run it.
# 

# define inversion options
inv_options_emcee = InversionOptions()
inv_options_emcee.set_tool("emcee")
inv_options_emcee.set_params(nwalkers=nwalkers, nsteps=nsteps, progress=True)

from emcee.moves import GaussianMove
inv_options_emcee.set_params(moves=GaussianMove(1))

# run the inversion
inv_rect_emcee = Inversion(ert_problem, inv_options_emcee)
inv_rect_emcee_res = inv_rect_emcee.run()

######################################################################
#


######################################################################
# Let’s sub-sample 10 models from the posterior ensemble and plot them
# out.
# 
# You’ll see that the results are not as good. That’s because we’ve used
# only 32 walkers and 10 sampling steps.
# 

sampler = inv_rect_emcee_res.sampler

######################################################################
#

flat_samples = sampler.get_chain(discard=5, flat=True)
indices = np.random.randint(len(flat_samples), size=10) # get a random selection from posterior ensemble
for idx in indices:
    ax=pg.show(
        imesh,
        data=(flat_samples[idx]),
        label=r"$\Omega m$"
    )
    ax[0].set_title(f"Inferred model - sample {idx}");

######################################################################
#


######################################################################
# Not satisfied with the results? Go back to the code cell under 2.3 and
# try with bigger numbers of walkers and steps 😉
# 


######################################################################
# --------------
# 
# Watermark
# =========
# 

watermark_list = ["cofi", "numpy", "scipy", "pygimli", "matplotlib", "emcee", "arviz"]
for pkg in watermark_list:
    pkg_var = __import__(pkg)
    print(pkg, getattr(pkg_var, "__version__"))

######################################################################
#