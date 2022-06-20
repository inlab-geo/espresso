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

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pygimli as pg
import pygimli.meshtools as mt
from pygimli.physics import ert

from cofi import BaseProblem, InversionOptions, Inversion

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

# Generate the survey
scheme = ert.createData(elecs=np.linspace(start=0, stop=50, num=51),schemeName='dd')
# Generate true model
world = mt.createWorld(start=[-55, 0], end=[105, -80], worldMarker=True)
conductive_anomaly = mt.createCircle(pos=[10, -7], radius=5, marker=2)
plc = mt.mergePLC((world, conductive_anomaly))
# Generate forward model mesh
for s in scheme.sensors():
    plc.createNode(s + [0.0, -0.2])
mesh_coarse = mt.createMesh(plc, quality=33)
fmesh = mesh_coarse.createH2()
# link markers to resistivity
rhomap = [[1, 200],
          [2,  50],]
# create model vector
model_true = pg.solver.parseArgToArray(rhomap, fmesh.cellCount(), fmesh)

# plot the compuational mesh and the true model
ax=pg.show(fmesh)
ax[0].set_title("Computational Mesh")
ax=pg.show(fmesh,data=model_true,label=r"$\Omega m$")
ax[0].set_title("Resitivity")

######################################################################
#


######################################################################
# Generate the synthetic data as a container with all the necessary
# information for plotting.
# 

survey = ert.simulate(
        fmesh,
        res=rhomap,
        scheme=scheme,
)
ax=ert.showERTData(survey,label=r"$\Omega$m")
ax[0].set_title("Aparent Resitivity")

y_obs=np.log(survey['rhoa'].array())

######################################################################
#


######################################################################
# The inversion can use a different mesh and the mesh to be used should
# know nothing about the mesh that was designed based on the true model.
# Here we first use a triangular mesh for the inversion, which makes the
# problem underdetermined.
# 

world = mt.createWorld(
    start=[-15, 0], end=[65, -30], worldMarker=False, marker=2)

# local refinement of mesh near electrodes
for s in scheme.sensors():
    world.createNode(s + [0.0, -0.4])

mesh_coarse = mt.createMesh(world, quality=33)
imesh = mesh_coarse.createH2()
for nr, c in enumerate(imesh.cells()):
    c.setMarker(nr)
ax=pg.show(imesh)
ax[0].set_title("Inversion Mesh")

######################################################################
#


######################################################################
# Define the starting model on the inversion mesh.
# 

model_0 = np.ones(imesh.cellCount()) * 80.0

######################################################################
#


######################################################################
# Setup a forward operator with the inversion mesh.
# 

forward_operator = ert.ERTModelling(
    sr=False,
    # verbose=True,
)
forward_operator.setComplex(False)
forward_operator.setData(scheme)
forward_operator.setMesh(imesh, ignoreRegionManager=True)

######################################################################
#


######################################################################
# CoFI and other inference packages require a set of functions that
# provide the misfit, the jacobian the resiudal with in the case of scipy
# standardised interfaces.
# 

def get_response(model, y_obs, forward_operator):
    y_synth = np.array(forward_operator.response(model))
    return np.log(y_synth)

def get_jacobian(model, y_obs, forward_operator):
    y_synth_log = get_response(model, y_obs, forward_operator)
    forward_operator.createJacobian(model)
    J0 = np.array(forward_operator.jacobian())
    model_log = np.log(model)
    J = J0 / np.exp(y_synth_log[:, np.newaxis]) * np.exp(model_log)[np.newaxis, :]
    return J

def get_residuals(model, y_obs, forward_operator):
    y_synth_log = get_response(model, y_obs, forward_operator)
    return y_obs - y_synth_log

def get_misfit(model, y_obs, forward_operator, print_progress=False):
    res = get_residuals(model, y_obs, forward_operator)
    phi = np.abs(np.dot(res, res))
    if print_progress: print("data misfit:", phi)
    return phi

######################################################################
#


######################################################################
# Define the regularisation by PyGIMLi.
# 

region_manager = forward_operator.regionManager()
region_manager.setMesh(imesh) 
# region_manager.setVerbose(True)
region_manager.setConstraintType(2)
Wm = pg.matrix.SparseMapMatrix()
region_manager.fillConstraints(Wm)
Wm = pg.utils.sparseMatrix2coo(Wm)

def get_regularisation(model, Wm, print_progress=False):
    weighted_model = Wm @ model
    reg = weighted_model.T @ weighted_model
    if print_progress: print("raw regularisation:", reg)
    return reg

######################################################################
#


######################################################################
# To help optimisers terminate quicker, we provide gradient and hessian as
# well.
# 

def get_gradient(model, y_obs, forward_operator, lamda, Wm):
    res = get_residuals(model, y_obs, forward_operator)
    jac = get_jacobian(model, y_obs, forward_operator)
    grad = - res @ jac + lamda * Wm.T @ Wm @ model
    return grad

def get_hessian(model, y_obs, forward_operator, lamda, Wm):
    jac = get_jacobian(model, y_obs, forward_operator)
    hess = jac.T @ jac + lamda * Wm.T @ Wm
    return hess

######################################################################
#


######################################################################
# With all the above forward operations set up with PyGIMLi, we now define
# the problem in ``cofi`` by setting the problem information for a
# ``BaseProblem`` object.
# 

# hyperparameters
lamda = 10

# cofi problem definition
ert_problem = BaseProblem()
ert_problem.name = "Electrical Resistivity Tomography"
ert_problem.set_forward(get_response, args=[y_obs, forward_operator])
ert_problem.set_jacobian(get_jacobian, args=[y_obs, forward_operator])
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

ert_problem.suggest_solvers();

######################################################################
#

inv_options = InversionOptions()
inv_options.set_tool("scipy.optimize.minimize")
inv_options.set_params(method="Newton-CG")

######################################################################
#


######################################################################
# Review what’s been defined for the inversion we are about to run:
# 

inv_options.summary()

######################################################################
#


######################################################################
# 3. Start an inversion
# ---------------------
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
# 4. Plotting
# -----------
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
ax[0].set_title("Inferred model")

######################################################################
#


######################################################################
# 5. Use a custom solver
# ----------------------
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
inv_options_own_solver.set_params(niter=30, step=1, verbose=True)

inv_own_solver = Inversion(ert_problem, inv_options_own_solver)
inv_own_solver_res = inv_own_solver.run()
inv_own_solver_res.summary()

######################################################################
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
ax[0].set_title("Inferred model")

######################################################################
#


######################################################################
# 6. Rectangular Mesh
# -------------------
# 
# Now lets do this on a rectangular mesh in the region of interests for
# the inversion with boundary represented using triangles.
# 

imesh=pg.createGrid(x=np.linspace(start=-15, stop=60, num=11),
                                y=np.linspace(start=-30, stop=0, num=5),
                                marker=2)
imesh = pg.meshtools.appendTriangleBoundary(imesh, marker=1,
                                           xbound=50, ybound=50)

for nr, c in enumerate(imesh.cells()):
    c.setMarker(nr)
ax=pg.show(imesh)
ax[0].set_title("Inversion Mesh")

######################################################################
#

ax=pg.show(imesh,data=np.linspace(1,imesh.cellCount(),imesh.cellCount()))
ax[0].set_title("Cell indices")

######################################################################
#


######################################################################
# Again, set the starting model value.
# 

model_0 = np.ones(imesh.cellCount()) * 80.0

######################################################################
#


######################################################################
# Set up the operator and regularisation for the new mesh
# 

# Forward operator
forward_operator2 = ert.ERTModelling(
    sr=False,
    # verbose=True,
)
forward_operator2.setComplex(False)
forward_operator2.setData(scheme)
forward_operator2.setMesh(imesh, ignoreRegionManager=True)
# Regularisation
rm2 = forward_operator2.regionManager()
rm2.setMesh(imesh) 
# rm2.setVerbose(True)
rm2.setConstraintType(2)
Wm2 = pg.matrix.SparseMapMatrix()
rm2.fillConstraints(Wm2)
Wm2 = pg.utils.sparseMatrix2coo(Wm2)

######################################################################
#


######################################################################
# Define a new problem instance with the new forward operator.
# 

# hyperparameters
lamda2 = 0.1

# cofi problem definition
ert_problem_rect = BaseProblem()
ert_problem_rect.name = "Electrical Resistivity Tomography"
ert_problem_rect.set_forward(get_response, args=[y_obs, forward_operator2])
ert_problem_rect.set_jacobian(get_jacobian, args=[y_obs, forward_operator2])
ert_problem_rect.set_residual(get_residuals, args=[y_obs, forward_operator2])
ert_problem_rect.set_data_misfit(get_misfit, args=[y_obs, forward_operator2])
ert_problem_rect.set_regularisation(get_regularisation, lamda=lamda2, args=[Wm2])
ert_problem_rect.set_gradient(get_gradient, args=[y_obs, forward_operator2, lamda2, Wm2])
ert_problem_rect.set_hessian(get_hessian, args=[y_obs, forward_operator2, lamda2, Wm2])
ert_problem_rect.set_initial_model(model_0)

######################################################################
#


######################################################################
# Run the inversion with the same inversion options.
# 

inv_rect = Inversion(ert_problem_rect, inv_options)
inv_rect_res = inv_rect.run()
inv_rect_res.summary()

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
    data=(inv_rect_res.model),
    label=r"$\Omega m$"
)
ax[0].set_title("Inferred model")

######################################################################
#


######################################################################
# 6. Rectangular mesh with ``MyNewtonSolver`` and ``emcee``
# ---------------------------------------------------------
# 
# 6.1. ``MyNewtonSolver``
# ~~~~~~~~~~~~~~~~~~~~~~~
# 

inv_rect_own_solver = Inversion(ert_problem_rect, inv_options_own_solver)
inv_rect_own_solver_res = inv_rect_own_solver.run()
inv_rect_own_solver_res.summary()

######################################################################
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
    data=(inv_rect_own_solver_res.model),
    label=r"$\Omega m$"
)
ax[0].set_title("Inferred model")

######################################################################
#


######################################################################
# 6.2 ``emcee``
# ~~~~~~~~~~~~~
# 
# CoFI needs more assumptions about the problem for a sampler to work -
# these are the log of posterior distribution density and walkers’
# starting positions.
# 
# For the log posterior, we define here by specifying ``log_prior`` and
# ``log_likelihood``. And CoFI will combine them to get the
# ``log_posterior``.
# 

# define log_likelihood
sigma = 1.0                                     # common noise standard deviation
Cdinv = np.eye(len(y_obs))/(sigma**2)           # inverse data covariance matrix
def log_likelihood(model):
    residual = ert_problem_rect.residual(model)
    return -0.5 * residual @ (Cdinv @ residual).T

# define log_prior
m_lower_bound = np.zeros(model_0.shape)         # lower bound for uniform prior
m_upper_bound = np.ones(model_0.shape) * 250    # upper bound for uniform prior
def log_prior(model):                           # uniform distribution
    for i in range(len(m_lower_bound)):
        if model[i] < m_lower_bound[i] or model[i] > m_upper_bound[i]: return -np.inf
    return 0.0 # model lies within bounds -> return log(1)

# define walkers' starting positions
nwalkers = 32
nsteps = 1000
walkers_start = model_0 + 1e-6 * np.random.randn(nwalkers, model_0.shape[0])

# define them into cofi's BaseProblem object
ert_problem_rect.set_log_likelihood(log_likelihood)
ert_problem_rect.set_log_prior(log_prior)
ert_problem_rect.set_walkers_starting_pos(walkers_start)

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
inv_rect_emcee = Inversion(ert_problem_rect, inv_options_emcee)
inv_rect_emcee_res = inv_rect_emcee.run()

######################################################################
#


######################################################################
# --------------
# 
# Watermark
# ---------
# 

# In case watermark doesn't work (e.g. sphinx-gallery)
watermark_list = ["cofi", "numpy", "scipy", "matplotlib", "emcee", "arviz", "pygimli"]
for pkg in watermark_list:
    pkg_var = __import__(pkg)
    print(pkg, getattr(pkg_var, "__version__"))

######################################################################
#