"""2D Gravity Inversion

Gravity inversion using the SimPEG kernels.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from discretize import TensorMesh
from SimPEG.utils import plot2Ddata, surface2ind_topo, model_builder
from SimPEG.potential_fields import gravity
from SimPEG import (
    maps,
    data,
    data_misfit,
    inverse_problem,
    regularization,
    optimization,
    directives,
    inversion,
    utils,
)

from cofi import BaseProblem, InversionOptions, Inversion
from cofi.tools import BaseInferenceTool


############# Define the true model with SimPEG #######################################
# define the tensor mesh
dh = 10.0
hx = [(dh, 1, -1.3), (dh, 10), (dh, 1, 1.3)]
hy = [(dh, 1, -1.3), (dh, 1), (dh, 1, 1.3)]
hz = [(dh, 1, -1.3), (dh, 8)]
mesh = TensorMesh([hx, hy, hz], "CCN")
xyz_topo=np.array([[-50, -50, 0],
                      [-50,  50, 0],
                      [ 50, -50, 0],
                      [ 50,  50, 0]])

# define density constrast values for each unit in g/cc
background_density = 0.0
block_density = -0.2
sphere_density = 0.2

# find the indices of the active cells in forward model (ones below surface)
ind_active = surface2ind_topo(mesh, xyz_topo)
nC = int(ind_active.sum())

# define model (models in SimPEG are vector arrays)
true_model = background_density * np.ones(nC)

# You could find the indicies of specific cells within the model and change their
# value to add structures.
ind_block = (
    (mesh.gridCC[ind_active, 0] > -50.0)
    & (mesh.gridCC[ind_active, 0] < -20.0)
    & (mesh.gridCC[ind_active, 1] > -15.0)
    & (mesh.gridCC[ind_active, 1] < 15.0)
    & (mesh.gridCC[ind_active, 2] > -50.0)
    & (mesh.gridCC[ind_active, 2] < -30.0)
)
true_model[ind_block] = block_density

# You can also use SimPEG utilities to add structures to the model more concisely
ind_sphere = model_builder.getIndicesSphere(np.r_[35.0, 0.0, -40.0], 15.0, mesh.gridCC)
ind_sphere = ind_sphere[ind_active]
true_model[ind_sphere] = sphere_density

nC = int(ind_active.sum())
model_map = maps.IdentityMap(nP=nC)  # model consists of a value for each active cell

# Plot True Model
fig = plt.figure(figsize=(9, 4))
plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)

ax1 = fig.add_axes([0.1, 0.1, 0.73, 0.8])
mesh.plot_slice(
    plotting_map * true_model,
    normal="Y",
    ax=ax1,
    ind=int(mesh.shape_cells[1] / 2),
    grid=True,
    clim=(np.min(true_model), np.max(true_model)),
    pcolor_opts={"cmap": "viridis"},
)
ax1.set_title("Model slice at y = 0 m")

ax2 = fig.add_axes([0.85, 0.1, 0.05, 0.8])
norm = mpl.colors.Normalize(vmin=np.min(true_model), vmax=np.max(true_model))
cbar = mpl.colorbar.ColorbarBase(
    ax2, norm=norm, orientation="vertical", cmap=mpl.cm.viridis, format="%.1e"
)
cbar.set_label("$g/cm^3$", rotation=270, labelpad=15, size=12)

plt.savefig("figs/model_slice_at_y_0m")


############# Define the survey with SimPEG ###########################################
nx, ny = (10, 10)
receiver_locations=[]
for x in np.linspace(-80, 80, nx):
     for y in np.linspace(-80, 80, ny):
        receiver_locations.append([x,y,0])
receiver_locations=np.array(receiver_locations)

# Define the receivers. The data consist of vertical gravity anomaly measurements.
# The set of receivers must be defined as a list.
receiver_list = gravity.receivers.Point(receiver_locations, components="gz")
receiver_list = [receiver_list]

# Define the source field
source_field = gravity.sources.SourceField(receiver_list=receiver_list)

# Define the survey
survey = gravity.survey.Survey(source_field)

# Define the forward simulation. By setting the 'store_sensitivities' keyword
# argument to "forward_only", we simulate the data without storing the sensitivities
simulation = gravity.simulation.Simulation3DIntegral(
    survey=survey,
    mesh=mesh,
    rhoMap=model_map,
    actInd=ind_active,
    store_sensitivities="forward_only",
)

# Compute predicted data for some model
# SimPEG uses right handed coordinate where Z is positive upward.
# This causes gravity signals look "inconsistent" with density values in visualization.
y_obs = simulation.dpred(true_model)

# Plot
fig = plt.figure(figsize=(7, 5))

ax1 = fig.add_axes([0.1, 0.1, 0.75, 0.85])
plot2Ddata(receiver_list[0].locations, y_obs, ax=ax1, contourOpts={"cmap": "bwr"})
ax1.set_title("Gravity Anomaly (Z-component)")
ax1.set_xlabel("x (m)")
ax1.set_ylabel("y (m)")

ax2 = fig.add_axes([0.82, 0.1, 0.03, 0.85])
norm = mpl.colors.Normalize(vmin=-np.max(np.abs(y_obs)), vmax=np.max(np.abs(y_obs)))
cbar = mpl.colorbar.ColorbarBase(
    ax2, norm=norm, orientation="vertical", cmap=mpl.cm.bwr, format="%.1e"
)
cbar.set_label("$mgal$", rotation=270, labelpad=15, size=12)

plt.savefig("figs/gravity_anomaly_z")

observations = data.Data(survey, dobs=y_obs)
observations.relative_error = 0.05
observations.noise_floor = 1e-5


############# Define starting model with SimPEG #######################################

# Define density contrast values for each unit in g/cc. Don't make this 0!
# Otherwise the gradient for the 1st iteration is zero and the inversion will
# not converge.
background_density = 1e-6

# Find the indecies of the active cells in forward model (ones below surface)
ind_active = surface2ind_topo(mesh, xyz_topo)

# Define mapping from model to active cells
nC = int(ind_active.sum())
model_map = maps.IdentityMap(nP=nC)  # model consists of a value for each active cell

# Define and plot starting model
starting_model = background_density * np.ones(nC)


############# Define forward operator / regularization with SimPEG ####################

simulation = gravity.simulation.Simulation3DIntegral(
    survey=survey, mesh=mesh, rhoMap=model_map, actInd=ind_active,
)

# Define the regularization (model objective function).
reg = regularization.Sparse(mesh, indActive=ind_active, mapping=model_map)
reg.norms = np.c_[0, 2, 2, 2]


############# Extra info from SimPEG, required by our own solver ######################
def get_jacobian(model, simulation):
    # J = simulation.getJ(model)
    return simulation.G

def get_response(model, simulation):
    y_pred = simulation.dpred(model)
    return y_pred

def get_residuals(model, y_obs, simulation):
    y_pred = simulation.dpred(model)
    return y_obs - y_pred

def get_misfit(model, y_obs, simulation):
    residual = get_residuals(model, y_obs, simulation)
    phi = np.abs(np.dot(residual, residual))
    return phi

def get_regularization(model, Wm, lamda):
    return lamda * (Wm @ model).T @ (Wm @ model)  

def get_gradient(model, y_obs, simulation, Wm, lamda):
    J = get_jacobian(model, simulation)
    residual = get_residuals(model, y_obs, simulation) 
    data_misfit_grad = -residual @ J
    regularization_grad = lamda * Wm.T @ Wm @ model
    return data_misfit_grad + regularization_grad

def get_hessian(model, simulation, Wm, lamda):
    J = get_jacobian(model, simulation)
    hess = J.T @ J + lamda * Wm.T @ Wm
    return hess


############# Inverted by our Gauss-Newton algorithm ##################################

class GaussNewton(BaseInferenceTool):
    def __init__(self, inv_problem, inv_options):
        __params = inv_options.get_params()
        self._niter = __params["niter"]
        self._verbose = __params["verbose"]
        self._model_0 = inv_problem.initial_model
        self._residual = inv_problem.residual
        self._jacobian = inv_problem.jacobian
        self._gradient = inv_problem.gradient
        self._hessian = inv_problem.hessian
        self._misfit = inv_problem.data_misfit if inv_problem.data_misfit_defined else None
        self._reg = inv_problem.regularization if inv_problem.regularization_defined else None
        self._obj = inv_problem.objective if inv_problem.objective_defined else None

    def __call__(self):
        current_model = np.array(self._model_0)
        for i in range(self._niter):
            if self._verbose:
                print("-" * 80)
                print(f"Iteration {i+1}")
                if self._obj: print(self._obj(current_model))
            term1 = self._hessian(current_model)
            term2 = - self._gradient(current_model)
            model_update = np.linalg.solve(term1, term2)
            current_model = np.array(current_model + model_update)
        return {"model": current_model, "success": True}

# hyper parameters
lamda = 10
niter = 1
inv_verbose = True

# CoFI - define BaseProblem
grav_problem = BaseProblem()
grav_problem.name = "2D Gravity Inversion using SimPEG kernels"
grav_problem.set_forward(get_response, args=[simulation])
grav_problem.set_jacobian(get_jacobian, args=[simulation])
grav_problem.set_residual(get_residuals, args=[y_obs, simulation])
grav_problem.set_data_misfit(get_misfit, args=[y_obs, simulation])
grav_problem.set_regularization(get_regularization, args=[reg.W, lamda])
grav_problem.set_gradient(get_gradient, args=[y_obs, simulation, reg.W, lamda])
grav_problem.set_hessian(get_hessian, args=[simulation, reg.W, lamda])
grav_problem.set_initial_model(starting_model)

# define CoFI InversionOptions, Inversion and run it
inv_options = InversionOptions()
inv_options.set_tool(GaussNewton)
inv_options.set_params(niter=niter, verbose=inv_verbose)
inv = Inversion(grav_problem, inv_options)
inv_result = inv.run()
model = inv_result.model
inv_result.summary()

# Plot Recovered Model
fig = plt.figure(figsize=(9, 4))
plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)

ax1 = fig.add_axes([0.1, 0.1, 0.73, 0.8])
mesh.plot_slice(
    plotting_map * model,
    normal="Y",
    ax=ax1,
    ind=int(mesh.nCy / 2),
    grid=True,
    clim=(np.min(model), np.max(model)),
    pcolor_opts={"cmap": "viridis"},
)
ax1.set_title("Model slice at y = 0 m")

ax2 = fig.add_axes([0.85, 0.1, 0.05, 0.8])
norm = mpl.colors.Normalize(vmin=np.min(true_model), vmax=np.max(true_model))
cbar = mpl.colorbar.ColorbarBase(
    ax2, norm=norm, orientation="vertical", cmap=mpl.cm.viridis, format="%.1e"
)
cbar.set_label("$g/cm^3$", rotation=270, labelpad=15, size=12)

plt.savefig("figs/inverted_by_gauss_newton")
