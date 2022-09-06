import os
from tkinter import W
import numpy as np
import matplotlib.pyplot as plt
import pygimli
from pygimli.physics import ert
from cofi import BaseProblem, InversionOptions, Inversion

from pygimli_dcip_lib import (
    survey_scheme,
    model_true,
    model_vec,
    ert_simulate,
    ert_manager,
    inversion_mesh,
    ert_forward_operator,
    reg_matrix,
    starting_model,
    get_response,
    get_residual,
    get_jacobian,
    get_data_misfit,
    get_regularisation,
    get_gradient,
    get_hessian,
)

if not os.path.exists("figs/tri_mesh"): os.makedirs("figs/tri_mesh")


############# DCIP Modelling with PyGIMLi #############################################

# measuring scheme
scheme = survey_scheme()

# create simulation mesh and true model
mesh, rhomap = model_true(scheme)
true_model = model_vec(rhomap, mesh)
fig, axes = plt.subplots(1, 2)
pygimli.show(mesh, data=np.log(np.abs(true_model)), 
                label=r"$log_{10}(|\rho|~[\Omega m])$", showMesh=True, ax=axes[0])
pygimli.show(mesh, data=np.arctan2(np.imag(true_model), np.real(true_model))*1000, 
                label=r"$\phi$ [mrad]", showMesh=True, ax=axes[1], cMap="jet_r")
fig.savefig("figs/tri_mesh/tri_scipy_opt_model_true")

# generate data
data, r_complex, r_complex_log, data_cov_inv = ert_simulate(mesh, scheme, rhomap)
fig, axes = plt.subplots(2, 2)
ert.showERTData(data, vals=data["rhoa"], ax=axes[0,0])
ert.showERTData(data, vals=data["phia"]*1000, label=r"$\phi$ [mrad]", ax=axes[0,1])
ert.showERTData(data, vals=np.real(r_complex), label=r"$Z'$~[$\Omega$m]", ax=axes[1,0])
ert.showERTData(data, vals=np.imag(r_complex), label=r"$Z''$~[$\Omega$]", ax=axes[1,1])
fig.tight_layout()
fig.savefig("figs/tri_mesh/tri_scipy_opt_data")

# create PyGIMLi's ERT manager
mgr = ert_manager(data)

# create inversion mesh
inv_mesh = inversion_mesh(mgr)
ax = pygimli.show(inv_mesh, showMesh=True, markers=True)
ax[0].set_title("Mesh used for inversion")
ax[0].figure.savefig("figs/tri_mesh/tri_scipy_opt_inv_mesh")

# PyGIMLi's forward operator (ERTModelling)
forward_oprt = ert_forward_operator(mgr, scheme, inv_mesh)

# extract regularisation matrix
Wm = reg_matrix(forward_oprt)
