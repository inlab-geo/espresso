"""Wrappers around PyGIMLi library for DCIP problem

The file name should end with "_lib.py", otherwise our bot may fail when generating
scripts for Sphinx Gallery. Furthermore, we recommend the file name to start with your
forward problem name, to align well with the naming of Jupyter notebook.

"""

import numpy as np
import matplotlib.pyplot as plt

import pygimli
from pygimli import meshtools
from pygimli.physics import ert


############# Helper functions from PyGIMLi ###########################################

# Dipole Dipole (dd) measuring scheme
def survey_scheme(start=0, stop=50, num=51, schemeName="dd"):
    scheme = ert.createData(elecs=np.linspace(start=start, stop=stop, num=num),schemeName=schemeName)
    # Not strictly required, but we switch potential electrodes to yield
    # positive geometric factors. Note that this was also done for the
    # synthetic data inverted later.
    m = scheme['m']
    n = scheme['n']
    scheme['m'] = n
    scheme['n'] = m
    scheme.set('k', [1 for x in range(scheme.size())])
    return scheme

# true geometry, forward mesh and true model
def model_true(scheme, start=[-55, 0], end=[105, -80], anomalies_pos=[[10,-7],[40,-7]], anomalies_rad=[5,5]):
    world = meshtools.createWorld(start=start, end=end, worldMarker=True)
    for s in scheme.sensors():          # local refinement
        world.createNode(s + [0.0, -0.2])
    anomalies = []
    for i, (pos, rad) in enumerate(zip(anomalies_pos, anomalies_rad)):
        anomaly_i = meshtools.createCircle(pos=pos, radius=rad, marker=i+2)
        anomalies.append(anomaly_i)
    geom = meshtools.mergePLC([world] + anomalies)
    rhomap = [[1, pygimli.utils.complex.toComplex(100, 0 / 1000)],
                # Magnitude: 50 ohm m, Phase: -50 mrad
                [2, pygimli.utils.complex.toComplex(50, 0 / 1000)],
                [3, pygimli.utils.complex.toComplex(100, -50 / 1000)],]
    mesh = meshtools.createMesh(geom, quality=33)
    return mesh, rhomap

# generate synthetic data
def ert_simulate(mesh, scheme, rhomap):
    data = ert.simulate(mesh, scheme=scheme, res=rhomap, noiseLevel=1,
                        noiceAbs=1e-6, seed=42)
    data["ip"] = - data["phia"] * 1000
    data_obs = data['rhoa'].array() * np.exp(1j * data['phia'].array())
    return data, data_obs

# PyGIMLi ert.ERTManager
def ert_manager(data, verbose=False):
    mgr = ert.ERTManager(data, verbose=verbose, useBert=True)
    mgr.fop.setComplex(True)
    return mgr

# inversion mesh
def inversion_mesh(ert_manager):
    inv_mesh = ert_manager.createMesh(ert_manager.data)
    print("model size", inv_mesh.cellCount())   # 1031
    ert_manager.setMesh(inv_mesh)
    return inv_mesh

# inversion mesh rectangular (the above is by default triangular)
def inversion_mesh_rect(ert_manager):
    x = np.linspace(start=-5, stop=55, num=40)
    y = np.linspace(start=-20,stop=0,num=10)
    inv_mesh = pygimli.createGrid(x=x, y=y, marker=2)
    inv_mesh = pygimli.meshtools.appendTriangleBoundary(inv_mesh, marker=1, xbound=50, ybound=50)
    print("model size", inv_mesh.cellCount())    # 1213
    ert_manager.setMesh(inv_mesh)
    return inv_mesh

# PyGIMLi ert.ERTModelling
def ert_forward_operator(ert_manager, scheme, inv_mesh):
    forward_operator = ert_manager.fop
    forward_operator.setComplex(True)
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
def starting_model(ert_manager):
    data = ert_manager.data
    rhoa = data["rhoa"]
    phia = data["phia"]
    data_vals = pygimli.utils.squeezeComplex(pygimli.utils.toComplex(rhoa, phia))
    start_model = ert_manager.fop.createStartModel(data_vals)
    return start_model

# convert model to numpy array
def model_vec(rhomap, fmesh):
    model_true = pygimli.solver.parseArgToArray(rhomap, fmesh.cellCount(), fmesh)
    return model_true


############# Plotting functions from PyGIMLi #########################################

def plot_model(mesh, model):
    fig, axes = plt.subplots(1, 2)
    pygimli.show(mesh, data=np.log(np.abs(model)), showMesh=True,
                label=r"$log_{10}(|\rho|~[\Omega m])$", ax=axes[0])
    axes[0].set_title("Resitivity")
    pygimli.show(mesh, data=np.arctan2(np.imag(model), np.real(model)) * 1000,
                label=r"$\phi$ [mrad]", showMesh=True, cMap='jet_r', ax=axes[1])
    axes[1].set_title("Chargeability")
    return fig

def plot_synth_data(data, r_complex):
    fig, axes = plt.subplots(2, 2)
    ert.showERTData(data, vals=data["rhoa"], ax=axes[0, 0])
    # phia is stored in radians, but usually plotted in milliradians
    ert.showERTData(data, vals=data["phia"]*1000, label=r"$\phi$ [mrad]", ax=axes[0, 1])
    ert.showERTData(data, vals=np.real(r_complex), label=r"$Z'$~[$\Omega$m", ax=axes[1, 0])
    ert.showERTData(data, vals=np.imag(r_complex), label=r"$Z''$~[$\Omega$]", ax=axes[1, 1])
    fig.tight_layout()
    return fig


############# Functions provided to CoFI ##############################################

def get_response(model, forward_operator):
    x_re_im=pygimli.utils.squeezeComplex(model)
    f = np.array(forward_operator.response(x_re_im))
    return np.log(pygimli.utils.toComplex(f))

def get_residual(model, data_obs, forward_operator):
    # response = get_response(model, forward_operator)
    response = np.array(forward_operator.response(model))
    return np.log(response)-data_obs
    return data_obs - response

def get_jacobian(model, forward_operator, return_response=False):
    x_re_im=pygimli.utils.squeezeComplex(model)
    f = np.array(forward_operator.response(x_re_im))
    response = np.log(pygimli.utils.toComplex(f))
    J_block = forward_operator.createJacobian(x_re_im)
    J_re = np.array(J_block.mat(0))
    J_im = np.array(J_block.mat(1))
    J = J_re + 1j * J_im
    jac = J / np.exp(response[:, np.newaxis]) * np.exp(np.log(model))[np.newaxis, :]
    if return_response: return jac, response
    return jac

def get_jac_residual(model, data_obs, forward_operator):
    jac, response = get_jacobian(model, forward_operator, True)
    return jac, data_obs - response

def get_data_misfit(model, data_obs, forward_operator):
    residual = get_residual(model, data_obs, forward_operator)
    return np.abs(residual.T @ residual)

def get_regularisation(model, Wm, lamda):
    return lamda * (Wm @ model).T @ (Wm @ model)

def get_objective(model, data_obs, forward_operator, Wm, lamda):
    data_misfit = get_data_misfit(model, data_obs, forward_operator)
    regularisation = get_regularisation(model, Wm, lamda)
    obj = data_misfit + regularisation
    return obj

def get_gradient(model, data_obs, forward_operator, Wm, lamda):
    jac, residual = get_jac_residual(model, data_obs, forward_operator)
    data_misfit_grad =  - residual @ jac
    regularisation_grad = lamda * Wm.T @ Wm @ model
    return data_misfit_grad + regularisation_grad

def get_hessian(model, log_data, forward_operator, Wm, lamda):
    jac = get_jacobian(model, forward_operator)
    hess = jac.T @ jac + lamda * Wm.T @ Wm
    return hess
