"""Wrappers around PyGIMLi library for DCIP problem

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
def scheme_fwd(start=0, stop=50, num=51, schemeName="dd"):
    scheme = ert.createData(elecs=np.linspace(start=start, stop=stop, num=num),schemeName=schemeName)
    return scheme

# piecewise linear complex
def geometry_true(start=[-55, 0], end=[105, -80], anomalies_pos=[[10,-7],[40,-7]], anomalies_rad=[5,5]):
    world = meshtools.createWorld(start=start, end=end, worldMarker=True)
    anomalies = []
    for i, (pos, rad) in enumerate(zip(anomalies_pos, anomalies_rad)):
        anomaly_i = meshtools.createCircle(pos=pos, radius=rad, marker=i+2)
        anomalies.append(anomaly_i)
    all_geometry = [world] + anomalies
    plc = meshtools.mergePLC(all_geometry)
    return plc

# forward mesh
def mesh_fwd(scheme, plc):
    # local refinement of mesh near electrodes
    for s in scheme.sensors():
        plc.createNode(s + [0.0, -0.2])
    mesh_corase = meshtools.createMesh(plc, quality=33)
    fmesh = mesh_corase.createH2()
    return fmesh

# set resistivity values in each region
def markers_to_resistivity():
    rhomap = [[1, pygimli.utils.complex.toComplex(100, 0 / 1000)],
                # Magnitude: 50 ohm m, Phase: -50 mrad
                [2, pygimli.utils.complex.toComplex(50, 0 / 1000)],
                [3, pygimli.utils.complex.toComplex(100, -50 / 1000)],]
    return rhomap

# create true model vector
def model_vec(rhomap, fmesh):
    model_true = pygimli.solver.parseArgToArray(rhomap, fmesh.cellCount(), fmesh)
    return model_true

# inverse mesh (triangular)
def mesh_inv_triangular(scheme, start=[-15, 0], end=[65, -30]):
    world = meshtools.createWorld(start=start, end=end, worldMarker=False, marker=2)
    # local refinement of mesh near electrodes
    for s in scheme.sensors():
        world.createNode(s + [0.0, -0.4])
    mesh_coarse = meshtools.createMesh(world, quality=33)
    imesh = mesh_coarse.createH2()
    for nr, c in enumerate(imesh.cells()):
        c.setMarker(nr)
    return imesh

# inverse mesh (rectangular)
def mesh_inv_rectangular(x_start=-15, x_stop=60, x_num=11, y_start=-30, y_stop=0, y_num=5):
    imesh = pygimli.createGrid(x=np.linspace(start=x_start, stop=x_stop, num=x_num),
                                y=np.linspace(start=y_start, stop=y_stop, num=y_num),
                                marker=2)
    imesh = pygimli.meshtools.appendTriangleBoundary(imesh, marker=1,
                                            xbound=50, ybound=50)
    for nr, c in enumerate(imesh.cells()):
        c.setMarker(nr)
    return imesh

# initialisation
def starting_model(imesh, val=80.0):
    return np.ones(imesh.cellCount()) * pygimli.utils.complex.toComplex(80, -0.01 / 1000)

# forward operator (PyGIMLi's ert.ErtModelling object)
def forward_oprt(scheme, imesh):
    forward_operator = ert.ERTModelling(sr=False, verbose=False)
    forward_operator.setComplex(True)
    forward_operator.setData(scheme)
    forward_operator.setMesh(imesh, ignoreRegionManager=True)
    return forward_operator

# for regularisation
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
    x_re_im=pygimli.utils.squeezeComplex(model)
    f_0 = np.array(forward_operator.response(x_re_im))
    return np.log(pygimli.utils.toComplex(f_0))

def get_jacobian(model, forward_operator):
    x_log = np.log(model)
    x_re_im=pygimli.utils.squeezeComplex(model)
    f_0 = np.array(forward_operator.response(x_re_im))
    y_synth_log = np.log(pygimli.utils.toComplex(f_0))
    J_block = forward_operator.createJacobian(x_re_im)
    J_re = np.array(J_block.mat(0))
    J_im = np.array(J_block.mat(1))
    J0 = J_re + 1j * J_im
    J = J0 / np.exp(y_synth_log[:, np.newaxis]) * np.exp(x_log)[np.newaxis, :]
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
