import os
import numpy as np
import pygimli
from pygimli.physics import ert
from cofi import BaseProblem, InversionOptions, Inversion
from cofi.tools import BaseInferenceTool

from pygimli_ert_lib import (
    survey_scheme,
    model_true,
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
    get_regularization,
    get_gradient,
    get_hessian,
)

if not os.path.exists("figs/tri_mesh"): os.makedirs("figs/tri_mesh")


############# ERT Modelling with PyGIMLi ##############################################

# measuring scheme
scheme = survey_scheme() #start=-25,stop=75,num=101,schemeName="dd")

# create simulation mesh and true model
mesh, rhomap = model_true(scheme)
ax = pygimli.show(mesh, data=rhomap, label="$\Omega m$", showMesh=True)
ax[0].set_title("True model")
ax[0].figure.savefig("figs/tri_mesh/tri_gauss_newton_armijo_linesearch_model_true")

# generate data
data, log_data, data_cov_inv = ert_simulate(mesh, scheme, rhomap)
ax = ert.show(data)
ax[0].set_title("Provided data")
ax[0].figure.savefig("figs/tri_mesh/tri_gauss_newton_armijo_linesearch_data")

# create PyGIMLi's ERT manager
ert_manager = ert_manager(data)

# create inversion mesh
inv_mesh = inversion_mesh(ert_manager)
ax = pygimli.show(inv_mesh, showMesh=True, markers=True)
ax[0].set_title("Mesh used for inversion")
ax[0].figure.savefig("figs/tri_mesh/tri_gauss_newton_armijo_linesearch_inv_mesh")

# PyGIMLi's forward operator (ERTModelling)
forward_oprt = ert_forward_operator(ert_manager, scheme, inv_mesh)

# extract regularization matrix
Wm = reg_matrix(forward_oprt)

# initialise a starting model for inversion
start_model, start_model_log = starting_model(ert_manager)
ax = pygimli.show(ert_manager.paraDomain, data=start_model, label="$\Omega m$", showMesh=True)
ax[0].set_title("Starting model")
ax[0].figure.savefig("figs/tri_mesh/tri_gauss_newton_armijo_linesearch_model_start")


############# Inverted by our Gauss-Newton algorithm ##################################


class GaussNewtonArmjioLineaSearch(BaseInferenceTool):
    required_in_problem = {"initial_model", "residual", "jacobian", "gradient"}
    optional_in_problem = dict()
    required_in_options = set()
    optional_in_options = {"niter": 100, "verbose": True, "tau_tol": 1e-5, "update_tol": 1e-5}
    def __init__(self, inv_problem, inv_options):
        __params = inv_options.get_params()
        self._niter = __params.get("niter", 100)
        self._verbose = __params.get("verbose", True)
        self._tau_tol = __params.get("tau_tol", 1e-5)
        self._update_tol = __params.get("update_tol", 1e-5)
        self._model_0 = inv_problem.initial_model
        self._residual = inv_problem.residual
        self._jacobian = inv_problem.jacobian
        self._gradient = inv_problem.gradient
        self._hessian = inv_problem.hessian
        self._misfit = inv_problem.data_misfit if inv_problem.data_misfit_defined else None
        self._reg = inv_problem.regularization if inv_problem.regularization_defined else None
        self._obj = inv_problem.objective if inv_problem.objective_defined else None

    def __call__(self):
        current_model = np.array(self._model_0)
        tau = 1

        for i in range(self._niter):
            tau = 1
            if self._verbose:
                print("-" * 80)
                print(f"Iteration {i+1}")
                print("model min and max:", np.min(current_model), np.max(current_model))
                if self._misfit: print("data misfit:", self._misfit(current_model))
                if self._reg: print("regularization:", self._reg(current_model))
            term1 = self._hessian(current_model)
            term2 = - self._gradient(current_model)
            model_update = np.linalg.solve(term1, term2)
        
            current_obj = self._obj(current_model)
            print("line search")

            # Reduce tau to ensure positivity (the following block is problem specific)
            while (tau > self._tau_tol):
                trial_model = current_model + model_update*tau
                if (np.min(np.exp(trial_model))>1.0):
                    break
                tau = tau * 0.5

            # Armjio Line search
            while (tau > self._tau_tol):          
                trial_model = current_model + model_update*tau
                trial_obj = self._obj(trial_model)
                print(f"tau {tau} current_obj {current_obj} trial_obj {trial_obj} |dm| {np.linalg.norm(model_update*tau,2)}")
                if trial_obj<current_obj:
                    current_model = trial_model
                    break
                tau = tau * 0.5

            # the following block is problem specific
            if (np.linalg.norm(model_update*tau,2) < self._update_tol) or (tau < self._tau_tol):
                break

        return {"model": current_model, "success": True}


# hyperparameters
lamda = 0.0005
inv_verbose = True
tau_tol = 1e-2              # smaller tolerance is needed, try a smaller value
update_tol = 1e-1

# CoFI - define BaseProblem
ert_problem = BaseProblem()
ert_problem.name = "Electrical Resistivity Tomography defined through PyGIMLi"
ert_problem.set_forward(get_response, args=[forward_oprt])
ert_problem.set_jacobian(get_jacobian, args=[forward_oprt])
ert_problem.set_residual(get_residual, args=[log_data, forward_oprt])
ert_problem.set_data_misfit(get_data_misfit, args=[log_data, forward_oprt, data_cov_inv])
ert_problem.set_regularization(get_regularization, args=[Wm, lamda])
ert_problem.set_gradient(get_gradient, args=[log_data, forward_oprt, Wm, lamda, data_cov_inv])
ert_problem.set_hessian(get_hessian, args=[log_data, forward_oprt, Wm, lamda, data_cov_inv])
ert_problem.set_initial_model(start_model_log)

# CoFI - define InversionOptions
inv_options = InversionOptions()
inv_options.set_tool(GaussNewtonArmjioLineaSearch)
inv_options.set_params(verbose=inv_verbose, tau_tol=tau_tol, update_tol=update_tol)

# CoFI - define Inversion, run it
inv = Inversion(ert_problem, inv_options)
inv_result = inv.run()
inv_result.summary()
model = np.exp(inv_result.model)

# plot inferred model
ax = pygimli.show(ert_manager.paraDomain, data=model, label=r"$\Omega m$")
ax[0].set_title("Inferred model")
ax[0].figure.savefig("figs/tri_mesh/tri_gauss_newton_armijo_linesearch_inferred_model")

# plot synthetic data
d = forward_oprt.response(model)
ax = ert.showERTData(scheme, vals=d, cMin=np.min(data["rhoa"]), cMax=np.max(data["rhoa"]))
ax[0].set_title("Synthetic data from inferred model")
ax[0].figure.savefig("figs/tri_mesh/tri_gauss_newton_armjio_linesearch_inferred_data")
