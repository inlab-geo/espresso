import pygimli
from pygimli.physics import ert

from pygimli_ert_lib import (
    survey_scheme,
    model_true,
    ert_simulate,
    ert_manager,
)

############# ERT Modelling with PyGIMLi ##############################################

# measuring scheme
scheme = survey_scheme()

# create simulation mesh and true model
mesh, rhomap = model_true(scheme)
ax = pygimli.show(mesh, data=rhomap, label="$\Omega m$", showMesh=True)
ax[0].figure.savefig("figs/inbuilt_solver_model_true")

# generate data
data, log_data = ert_simulate(mesh, scheme, rhomap)
ax = ert.show(data)
ax[0].figure.savefig("figs/inbuilt_solver_data")

# create PyGIMLi's ERT manager
mgr = ert_manager(data, verbose=True)


############# Inverted by PyGIMLi solvers #############################################

inv = mgr.invert(lam=20, verbose=True)
fig = mgr.showResultAndFit()
fig.savefig("figs/inbuilt_solver_result")
