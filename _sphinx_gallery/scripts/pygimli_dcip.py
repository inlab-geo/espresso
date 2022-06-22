"""
PyGIMLi - Complex valued ERT
============================

"""


######################################################################
# Using the ERT solver implemented provided by
# `PyGIMLi <https://www.pygimli.org/>`__, we use different ``cofi``
# solvers to solve the corresponding inverse problem.
# 
# .. raw:: html
# 
#    <!-- Please leave the cell below as it is -->
# 


######################################################################
# .. raw:: html
# 
# 	<badge><a href="https://colab.research.google.com/github/inlab-geo/cofi-examples/blob/main/notebooks/pygimli_dcip/pygimli_dcip.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></badge>


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
# ``pygimli_dcip_lib.py`` and import it here for conciseness.
# 

import numpy as np
import matplotlib.pyplot as plt

from cofi import BaseProblem, InversionOptions, Inversion

from pygimli_dcip_lib import *

np.random.seed(42)

######################################################################
#


######################################################################
# 1. Define the problem
# ---------------------
# 
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
ax=pygimli.show(fmesh)
ax[0].set_title("Computational Mesh")
ax=pygimli.show(fmesh, data=np.abs(model_true), label=r"$\Omega m$")
ax[0].set_title("Resitivity")
ax=pygimli.show(fmesh, data=np.arctan2(np.imag(model_true), np.real(model_true)) * 1000,label=r"mrad",)
ax[0].set_title("Chargeability") 

######################################################################
#


######################################################################
# Generate the synthetic data as a container with all the necessary
# information for plotting.
# 

# PyGIMLi - generate data
survey = ert.simulate(fmesh, res=rhomap, scheme=scheme)

y_obs = survey['rhoa'].array() * np.exp(1j * survey['phia'].array())

ax=ert.showERTData(survey, vals=np.real(y_obs),label=r"$\Omega$m")
ax[0].set_title("Aparent Resitivity")
ax=ert.showERTData(survey, vals=np.arctan2(np.imag(y_obs), np.real(y_obs)) * 1000, label=r"mrad")
ax[0].set_title("Aparent Chargeability");

######################################################################
#


######################################################################
# The inversion can use a different mesh and the mesh to be used should
# know nothing about the mesh that was designed based on the true model.
# We wrap two kinds of mesh as examples in the library code
# ``pygimli_dcip_lib.py``, namely triangular and rectangular mesh.
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

ax=pygimli.show(imesh_tri)
ax[0].set_title("Inversion Mesh (triangular)");

######################################################################
#


######################################################################
# Check
# `here <https://github.com/inlab-geo/cofi-examples/tree/main/notebooks/pygimli_dcip>`__
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
ert_problem.name = "Complex valued ERT defined through PyGIMLi"
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
# 2. Define the inversion options
# -------------------------------
# 
# As what
# `pygimli_ert.ipynb <https://github.com/inlab-geo/cofi-examples/blob/main/notebooks/pygimli_ert/pygimli_ert.ipynb>`__
# example does, we use a Newton’s iterative approach written by ourselves,
# so you’ll have a closer look at what’s happening in the loop.
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
# Now, make use of this custom solver to define inversion options.
# 

inv_options_own_solver = InversionOptions()
inv_options_own_solver.set_tool(MyNewtonSolver)
inv_options_own_solver.set_params(niter=20, step=1, verbose=True)

######################################################################
#


######################################################################
# Review what’s been defined for the inversion we are about to run:
# 

inv_options_own_solver.summary()

######################################################################
#


######################################################################
# 3. Start an inversion
# ---------------------
# 

inv_own_solver = Inversion(ert_problem, inv_options_own_solver)
inv_own_solver_res = inv_own_solver.run()
inv_own_solver_res.summary()

######################################################################
#


######################################################################
# 4. Plotting
# -----------
# 


ax=pygimli.show(
    fmesh,
    data=np.log(np.abs(model_true)),
    label=r"$\Omega m$"
)
ax[0].set_title("True model")

ax=pygimli.show(
    fmesh,
    data=np.arctan2(np.imag(model_true), np.real(model_true)) * 1000,
    label=r"mrad"
)
ax[0].set_title("True model")

ax=pygimli.show(
    imesh,
    data=(model_0.real),
    label=r"$\Omega m$"
)
ax[0].set_title("Starting model")

ax=pygimli.show(
    imesh,
    data=np.arctan2(np.imag(model_0), np.real(model_0)) * 1000,
    label=r"mrad"
)
ax[0].set_title("Starting model")

ax=pygimli.show(
    imesh,
    data=(inv_own_solver_res.model.real),
    label=r"$\Omega m$"
)
ax[0].set_title("Inferred model")

ax=pygimli.show(
    imesh,
    data=np.arctan2(np.imag(inv_own_solver_res.model), np.real(inv_own_solver_res.model)) * 1000,
    label=r"mrad"
)
ax[0].set_title("Inferred model")


######################################################################
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

watermark_list = ["cofi", "numpy", "scipy", "pygimli", "matplotlib"]
for pkg in watermark_list:
    pkg_var = __import__(pkg)
    print(pkg, getattr(pkg_var, "__version__"))

######################################################################
#