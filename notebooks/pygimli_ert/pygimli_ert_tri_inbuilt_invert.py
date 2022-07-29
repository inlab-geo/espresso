import numpy as np
import pygimli
from pygimli.physics import ert

from pygimli_ert_lib import (
    survey_scheme,
    model_true,
    ert_simulate,
    ert_manager,
    reg_matrix,
    get_data_misfit,
    get_regularisation,
)

############# ERT Modelling with PyGIMLi ##############################################

# measuring scheme
scheme = survey_scheme()

# create simulation mesh and true model
mesh, rhomap = model_true(scheme)
ax = pygimli.show(mesh, data=rhomap, label="$\Omega m$", showMesh=True)
ax[0].figure.savefig("figs/tri_mesh/tri_inbuilt_solver_model_true")

# generate data
data, log_data, data_cov_inv = ert_simulate(mesh, scheme, rhomap)
ax = ert.show(data)
ax[0].figure.savefig("figs/tri_mesh/tri_inbuilt_solver_data")

# create PyGIMLi's ERT manager
mgr = ert_manager(data, verbose=True)


############# Inverted by PyGIMLi solvers #############################################

inv = mgr.invert(lam=20, verbose=True)
# fig = mgr.showResultAndFit()
# fig.savefig("figs/tri_mesh/tri_inbuilt_solver_result")

Wm = reg_matrix(mgr.fop)
print("data misfit:", get_data_misfit(np.log(inv), log_data, mgr.fop, data_cov_inv))
print("regularisation:", get_regularisation(np.log(inv), Wm, 0.0005))

# plot inferred model
ax = pygimli.show(mgr.paraDomain, data=inv, label=r"$\Omega m$")
ax[0].set_title("Inferred model")
ax[0].figure.savefig("figs/tri_mesh/tri_inbuilt_solver_inferred_model")

# plot synthetic data
d = mgr.fop.response(inv)
ax = ert.showERTData(scheme, vals=d, cMin=np.min(data["rhoa"]), cMax=np.max(data["rhoa"]))
ax[0].set_title("Synthetic data from inferred model")
ax[0].figure.savefig("figs/tri_mesh/tri_inbuilt_solver_inferred_data")
