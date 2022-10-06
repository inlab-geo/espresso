import os
import numpy as np
import matplotlib.pyplot as plt
import pygimli
from pygimli.physics import ert

from pygimli_dcip_lib import (
    inversion_mesh_rect,
    survey_scheme,
    model_true,
    model_vec,
    ert_simulate,
    ert_manager,
    reg_matrix,
    get_data_misfit,
    get_regularization,
)

if not os.path.exists("figs/rect_mesh"): os.makedirs("figs/rect_mesh")


############# ERT Modelling with PyGIMLi ##############################################

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
fig.savefig("figs/rect_mesh/rect_inbuilt_solver_model_true")

# generate data
data, r_complex, r_complex_log, data_cov_inv = ert_simulate(mesh, scheme, rhomap, noise_level=1)
fig, axes = plt.subplots(2, 2)
ert.showERTData(data, vals=data["rhoa"], ax=axes[0,0])
ert.showERTData(data, vals=data["phia"]*1000, label=r"$\phi$ [mrad]", ax=axes[0,1])
ert.showERTData(data, vals=np.real(r_complex), label=r"$Z'$~[$\Omega$m]", ax=axes[1,0])
ert.showERTData(data, vals=np.imag(r_complex), label=r"$Z''$~[$\Omega$]", ax=axes[1,1])
fig.tight_layout()
fig.savefig("figs/rect_mesh/rect_inbuilt_solver_data")

# create PyGIMLi's ERT manager
mgr = ert_manager(data)

# create rectangular inversion mesh
inv_mesh = inversion_mesh_rect(mgr)
ax = pygimli.show(inv_mesh, showMesh=True, markers=True)
ax[0].figure.savefig("figs/rect_mesh/rect_inbuilt_solver_mesh")


# ############# Inverted by PyGIMLi solvers #############################################

inv = mgr.invert(verbose=True)
# fig = mgr.showResultAndFit()
# fig.savefig("figs/rect_mesh/inbuilt_solver_result")

# plot inferred model
fig, axes = plt.subplots(1, 2)
pygimli.show(mgr.paraDomain, data=np.log(np.abs(inv)), 
                label=r"$log_{10}(|\rho|~[\Omega m])$", showMesh=True, ax=axes[0])
pygimli.show(mgr.paraDomain, data=np.arctan2(np.imag(inv), np.real(inv))*1000, 
                label=r"$\phi$ [mrad]", showMesh=True, ax=axes[1], cMap="jet_r")
fig.savefig("figs/inbuilt_solver_inferred_model")

# plot synthetic data
data, r_complex, r_complex_log, data_cov_inv = ert_simulate(mgr.paraDomain, scheme, inv)
fig, axes = plt.subplots(2, 2)
ert.showERTData(data, vals=data["rhoa"], ax=axes[0,0])
ert.showERTData(data, vals=data["phia"]*1000, label=r"$\phi$ [mrad]", ax=axes[0,1])
ert.showERTData(data, vals=np.real(r_complex), label=r"$Z'$~[$\Omega$m]", ax=axes[1,0])
ert.showERTData(data, vals=np.imag(r_complex), label=r"$Z''$~[$\Omega$]", ax=axes[1,1])
fig.tight_layout()
fig.savefig("figs/inbuilt_solver_inferred_data")

# print data misfit and regularization term for inversion result
Wm = reg_matrix(mgr.fop)
print("data misfit:", get_data_misfit(inv, r_complex, mgr.fop))
print("regularization:", get_regularization(inv, Wm, 20))
