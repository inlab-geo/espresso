"""Wrappers around PyGIMLi library

The file name should end with "_lib.py", otherwise our bot may fail when generating
scripts for Sphinx Gallery. Furthermore, we recommend the file name to start with your
forward problem name, to align well with the naming of Jupyter notebook.

"""

import numpy as np
import pygimli
from pygimli import meshtools
from pygimli.physics import ert


############# Helper functions from PyGIMLi ###########################################

def scheme_fwd():                   # Dipole Dipole (dd) measuring scheme
    scheme = ert.createData(elecs=np.linspace(start=0, stop=50, num=51),schemeName='dd')
    return scheme

def geometry_true():                # piecewise linear complex
    world = meshtools.createWorld(start=[-55, 0], end=[105, -80], worldMarker=True)
    conductive_anomaly = meshtools.createCircle(pos=[10, -7], radius=5, marker=2)
    plc = meshtools.mergePLC((world, conductive_anomaly))    
    return plc

def mesh_fwd(scheme, plc):          # forward mesh
    # local refinement of mesh near electrodes
    for s in scheme.sensors():
        plc.createNode(s + [0.0, -0.2])
    mesh_coarse = meshtools.createMesh(plc, quality=33)
    fmesh = mesh_coarse.createH2()
    return fmesh

def markers_to_resistivity():       # set resistivity values in each region
    rhomap = [[1, 200],
              [2,  50],]
    return rhomap

def model_vec(rhomap, fmesh):       # create true model vector
    model_true = pygimli.solver.parseArgToArray(rhomap, fmesh.cellCount(), fmesh)
    return model_true

def mesh_inv_triangular(scheme):    # inverse mesh (triangular)
    world = meshtools.createWorld(start=[-15, 0], end=[65, -30], worldMarker=False, marker=2)
    # local refinement of mesh near electrodes
    for s in scheme.sensors():
        world.createNode(s + [0.0, -0.4])
    mesh_coarse = meshtools.createMesh(world, quality=33)
    imesh = mesh_coarse.createH2()
    for nr, c in enumerate(imesh.cells()):
        c.setMarker(nr)
    return imesh

def mesh_inv_rectangular():         # inverse mesh (rectangular)
    imesh = pygimli.createGrid(x=np.linspace(start=-15, stop=60, num=11),
                                y=np.linspace(start=-30, stop=0, num=5),
                                marker=2)
    imesh = pygimli.meshtools.appendTriangleBoundary(imesh, marker=1,
                                            xbound=50, ybound=50)
    for nr, c in enumerate(imesh.cells()):
        c.setMarker(nr)
    return imesh

def starting_model(imesh):
    return np.ones(imesh.cellCount()) * 80.0

def forward_oprt(scheme, imesh):
    forward_operator = ert.ERTModelling(sr=False, verbose=False)
    forward_operator.setComplex(False)
    forward_operator.setData(scheme)
    forward_operator.setMesh(imesh, ignoreRegionManager=True)
    return forward_operator

def weighting_matrix(forward_operator, imesh):
    region_manager = forward_operator.regionManager()
    region_manager.setMesh(imesh) 
    # region_manager.setVerbose(True)
    region_manager.setConstraintType(2)
    Wm = pygimli.matrix.SparseMapMatrix()
    region_manager.fillConstraints(Wm)
    Wm = pygimli.utils.sparseMatrix2coo(Wm)
    return Wm


############# Functions provided to CoFI ##############################################

def get_response(model, forward_operator):
    y_synth = np.array(forward_operator.response(model))
    return np.log(y_synth)

def get_jacobian(model, forward_operator):
    y_synth_log = get_response(model, forward_operator)
    forward_operator.createJacobian(model)
    J0 = np.array(forward_operator.jacobian())
    model_log = np.log(model)
    J = J0 / np.exp(y_synth_log[:, np.newaxis]) * np.exp(model_log)[np.newaxis, :]
    return J

def get_residuals(model, y_obs, forward_operator):
    y_synth_log = get_response(model, forward_operator)
    return y_obs - y_synth_log

def get_misfit(model, y_obs, forward_operator, print_progress=False):
    res = get_residuals(model, y_obs, forward_operator)
    phi = np.abs(np.dot(res, res))
    if print_progress: print("data misfit:", phi)
    return phi

def get_regularisation(model, Wm, print_progress=False):
    weighted_model = Wm @ model
    reg = weighted_model.T @ weighted_model
    if print_progress: print("raw regularisation:", reg)
    return reg

def get_gradient(model, y_obs, forward_operator, lamda, Wm):
    res = get_residuals(model, y_obs, forward_operator)
    jac = get_jacobian(model, forward_operator)
    grad = - res @ jac + lamda * Wm.T @ Wm @ model
    return grad

def get_hessian(model, y_obs, forward_operator, lamda, Wm):
    jac = get_jacobian(model, forward_operator)
    hess = jac.T @ jac + lamda * Wm.T @ Wm
    return hess
