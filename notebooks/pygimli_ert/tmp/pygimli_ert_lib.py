
"""Wrappers around PyGIMLi library for ERT problem

The file name should end with "_lib.py", otherwise our bot may fail when generating
scripts for Sphinx Gallery. Furthermore, we recommend the file name to start with your
forward problem name, to align well with the naming of Jupyter notebook.
"""

import numpy as np

import pygimli
from pygimli import meshtools
from pygimli.physics import ert


############# Helper functions from PyGIMLi ###########################################

# Dipole Dipole (dd) measuring scheme
def survey_scheme(start=0, stop=50, num=51, schemeName="dd"):
    scheme = ert.createData(elecs=np.linspace(start=start, stop=stop, num=num),schemeName=schemeName)
    return scheme

# true geometry, forward mesh and true model
def model_true(scheme, start=[-55, 0], end=[105, -80], anomaly_pos=[10,-7], anomaly_rad=5):
    world = meshtools.createWorld(start=start, end=end, worldMarker=True)
    for s in scheme.sensors():          # local refinement 
        world.createNode(s + [0.0, -0.1])
    conductive_anomaly = meshtools.createCircle(pos=anomaly_pos, radius=anomaly_rad, marker=2)
    geom = world + conductive_anomaly
    rhomap = [[1, 200], [2,  50],]
    mesh = meshtools.createMesh(geom, quality=33)
    return mesh, rhomap

# generate synthetic data
def ert_simulate(mesh, scheme, rhomap):
    data = ert.simulate(mesh, scheme=scheme, res=rhomap, noiseLevel=1,
                        noiceAbs=1e-6, seed=42)
    data.remove(data["rhoa"] < 0)
    log_data = np.log(data["rhoa"].array())
    return data, log_data

# PyGIMLi ert.ERTManager
def ert_manager(data, verbose=False):
    return ert.ERTManager(data, verbose=verbose, useBert=True)

# inversion mesh
def inversion_mesh(ert_manager):
    inv_mesh = ert_manager.createMesh(ert_manager.data)
    ert_manager.setMesh(inv_mesh)
    return inv_mesh

# PyGIMLi ert.ERTModelling
def ert_forward_operator(ert_manager, scheme, inv_mesh):
    forward_operator = ert_manager.fop
    forward_operator.setComplex(False)
    forward_operator.setData(scheme)
    forward_operator.setMesh(inv_mesh, ignoreRegionManager=True)
    return forward_operator

# regularisation matrix
def reg_matrix(forward_oprt):
    region_manager = forward_oprt.regionManager()
    region_manager.setConstraintType(2)
    Wm = pygimli.matrix.SparseMapMatrix()
    region_manager.fillConstraints(Wm)
    Wm = pygimli.utils.sparseMatrix2coo(Wm)
    return Wm

# initialise model
def starting_model(ert_manager, val=None):
    data = ert_manager.data
    start_val = val if val else np.median(data['rhoa'].array())     # this is how pygimli initialises
    start_model = np.ones(ert_manager.paraDomain.cellCount()) * start_val
    return start_model

# convert model to numpy array
def model_vec(rhomap, fmesh):
    model_true = pygimli.solver.parseArgToArray(rhomap, fmesh.cellCount(), fmesh)
    return model_true


############# Functions provided to CoFI ##############################################

def get_response(model, forward_operator):
    return np.log(np.array(forward_operator.response(model)))

def get_residual(model, log_data, forward_operator):
    response = get_response(model, forward_operator)
    residual = log_data - response
    return residual

def get_jacobian(model, forward_operator):
    response = get_response(model, forward_operator)
    forward_operator.createJacobian(model)
    J = np.array(forward_operator.jacobian())
    jac = J / np.exp(response[:, np.newaxis]) * np.exp(np.log(model))[np.newaxis, :]
    return jac

def get_jac_residual(model, log_data, forward_operator):
    response = get_response(model, forward_operator)
    residual = log_data - response
    forward_operator.createJacobian(model)
    J = np.array(forward_operator.jacobian())
    jac = J / np.exp(response[:, np.newaxis]) * np.exp(np.log(model))[np.newaxis, :]
    return jac, residual

def get_data_misfit(model, log_data, forward_operator):
    residual = get_residual(model, log_data, forward_operator)
    return np.abs(residual.T @ residual)

def get_regularisation(model, Wm, lamda):
    return lamda * (Wm @ model).T @ (Wm @ model)

def get_objective(model, log_data, forward_operator, Wm, lamda):
    data_misfit = get_data_misfit(model, log_data, forward_operator)
    regularisation = get_regularisation(model, Wm, lamda)
    obj = data_misfit + regularisation
    return obj

def get_gradient(model, log_data, forward_operator, Wm, lamda):
    jac, residual = get_jac_residual(model, log_data, forward_operator)
    data_misfit_grad =  - residual @ jac
    regularisation_grad = lamda * Wm.T @ Wm @ model
    return data_misfit_grad + regularisation_grad

def get_hessian(model, log_data, forward_operator, Wm, lamda):
    jac = get_jacobian(model, forward_operator)
    hess = jac.T @ jac + lamda * Wm.T @ Wm
    return hess
