import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

import pygimli
from pygimli import meshtools
from pygimli.physics import ert


world = meshtools.createWorld(start=[-55,0], end=[105,-80], worldMarker=True)
conductive_anomaly = meshtools.createCircle(pos=[10,-7], radius=5, marker=2)
geom = world + conductive_anomaly
ax = pygimli.show(geom)
ax[0].figure.savefig("figs/true_geometry")

scheme = ert.createData(elecs=np.linspace(start=0, stop=50, num=51), schemeName="dd")
for s in scheme.sensors():
    geom.createNode(s + [0.0, -0.2])

rhomap = [[1, 200], [2,  50],]

mesh = meshtools.createMesh(geom, quality=33)
ax = pygimli.show(mesh, data=rhomap, label="$\Omega m$", showMesh=True)
ax[0].figure.savefig("figs/true_model_coarse")

# mesh = mesh.createH2()
# ax = pygimli.show(mesh, data=rhomap, label="$\Omega m$", showMesh=True)
# ax[0].figure.savefig("figs/true_model")

data = ert.simulate(mesh, scheme=scheme, res=rhomap, noiseLevel=1,
                    noiseAbs=1e-6, seed=42)

data.remove(data['rhoa'] < 0)
log_data = np.log(data['rhoa'].array())
ax = ert.show(data)
ax[0].figure.savefig("figs/data")

############# Inverted by PyGIMLi solvers #############################################

# mgr = ert.ERTManager(data)
# inv = mgr.invert(lam=20, verbose=True)
# fig = mgr.showResultAndFit()
# fig.savefig("figs/invert_result")


############# Inverted by scipy.optimize.minimize #####################################

forward_operator = ert.ERTModelling(sr=False, verbose=False)
forward_operator.setComplex(False)
forward_operator.setData(scheme)
forward_operator.setMesh(mesh, ignoreRegionManager=True)

start_model = np.ones(mesh.cellCount()) * 80.0

region_manager = forward_operator.regionManager()
region_manager.setMesh(mesh)
region_manager.setConstraintType(2)
Wm = pygimli.matrix.SparseMapMatrix()
region_manager.fillConstraints(Wm)
Wm = pygimli.utils.sparseMatrix2coo(Wm)

lamda = 20

def get_response(model, forward_operator):
    return np.log(np.array(forward_operator.response(model)))

def get_residual(model, log_data, forward_operator):
    response = get_response(model, forward_operator)
    return response - log_data

def get_jacobian(model, forward_operator):
    response = get_response(model, forward_operator)
    forward_operator.createJacobian(model)
    J = np.array(forward_operator.jacobian())
    jac = J / np.exp(response[:, np.newaxis]) * np.exp(np.log(model))[np.newaxis, :]
    return jac

def get_jac_residual(model, log_data, forward_operator):
    response = get_response(model, forward_operator)
    residual = response - log_data
    forward_operator.createJacobian(model)
    J = np.array(forward_operator.jacobian())
    jac = J / np.exp(response[:, np.newaxis]) * np.exp(np.log(model))[np.newaxis, :]
    return jac, residual

def get_data_misfit(model, log_data, forward_operator):
    residual = get_residual(model, log_data, forward_operator)
    return residual.T @ residual

def get_regularisation(model, Wm, lamda):
    return lamda * (Wm @ model).T @ (Wm @ model)

def get_objective(model, log_data, forward_operator, Wm, lamda):
    data_misfit = get_data_misfit(model, log_data, forward_operator)
    regularisation = get_regularisation(model, Wm, lamda)
    obj = data_misfit + regularisation
    print(obj)
    return obj

def get_gradient(model, log_data, forward_operator, Wm, lamda):
    jac, residual = get_jac_residual(model, log_data, forward_operator)
    data_misfit_grad =  - 2 * residual @ jac
    regularisation_grad = 2 * lamda * Wm.T @ Wm @ model
    return data_misfit_grad + regularisation_grad

def get_hessian(model, log_data, forward_operator, Wm, lamda):
    jac = get_jacobian(model, forward_operator)
    hess = jac.T @ jac + lamda * Wm.T @ Wm
    return hess


opt_result = minimize(
    fun=get_objective,
    args=(log_data, forward_operator, Wm, lamda),
    x0=start_model,
    jac=get_gradient,
    hess=get_hessian,
    method="Newton-CG"
)
print(opt_result.x)
print(opt_result.success)
print(opt_result.status)
print(opt_result.message)
print(opt_result.fun)
print(opt_result.jac)
# print(opt_result.hess)
print(opt_result.nit)
