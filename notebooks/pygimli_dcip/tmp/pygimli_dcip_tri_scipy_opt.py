import numpy as np
import matplotlib.pyplot as plt
import pygimli
from pygimli.physics import ert
from cofi import BaseProblem, InversionOptions, Inversion

from pygimli_dcip_lib import (
    survey_scheme,
    model_true,
    ert_simulate,
    ert_manager,
)


############# DCIP Modelling with PyGIMLi #############################################

# measuring scheme
scheme = survey_scheme()

# create simulation mesh and true model
mesh, rhomap = model_true(scheme)
fig, axes = plt.subplots(2, 1, figsize=(16 / 2.54, 16 / 2.54))
ert.show(mesh, data=np.real(rhomap), label=r"$log_{10}(|\rho|~[\Omega m])$", ax=axes[0])
ert.show(mesh, data=)
