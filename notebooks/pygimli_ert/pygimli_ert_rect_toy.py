import numpy as np
import pygimli
from pygimli.physics import ert
from cofi import BaseProblem, InversionOptions, Inversion
from cofi.solvers import BaseSolver

from pygimli_ert_lib import (
    inversion_mesh_rect_toy,
    survey_scheme,
    model_true,
    ert_simulate,
    ert_manager,
    ert_forward_operator,
    reg_matrix,
    starting_model,
    get_response,
    get_residual,
    get_gradient,
    get_hessian,
    get_jacobian,
    get_data_misfit,
    get_regularisation,
)


############# ERT Modelling with PyGIMLi ##############################################

# measuring scheme
scheme = survey_scheme()

# create simulation mesh and true model
mesh, rhomap = model_true(scheme)
ax = pygimli.show(mesh, data=rhomap, label="$\Omega m$", showMesh=True)
ax[0].figure.savefig("figs/toy_rect_mesh/model_true")

# generate data
data, log_data, data_cov_inv = ert_simulate(mesh, scheme, rhomap)
ax = ert.show(data)
ax[0].figure.savefig("figs/toy_rect_mesh/data")

# create PyGIMLi's ERT manager
mgr = ert_manager(data)

# create rectangular inversion mesh
inv_mesh = inversion_mesh_rect_toy(mgr)
ax = pygimli.show(inv_mesh, showMesh=True, markers=True)
ax[0].figure.savefig("figs/toy_rect_mesh/mesh")

# PyGIMLi's forward operator (ERTModelling)
forward_oprt = ert_forward_operator(mgr, scheme, inv_mesh)

# extract regularisation matrix
Wm = reg_matrix(forward_oprt)

# initialise a starting model for inversion
start_model, start_model_log = starting_model(mgr)
ax = pygimli.show(mgr.paraDomain, data=start_model, label="$\Omega m$", showMesh=True)
ax[0].set_title("Starting model")
ax[0].figure.savefig("figs/toy_rect_mesh/model_start")


############# Inverted by PyGIMLi solvers #############################################

# inv = mgr.invert(lam=20, verbose=False)

# Wm = reg_matrix(mgr.fop)
# print("data misfit:", get_data_misfit(np.log(inv), log_data, mgr.fop, data_cov_inv))
# print("regularisation:", get_regularisation(np.log(inv), Wm, 0.0005))

# # plot inferred model
# ax = pygimli.show(mgr.paraDomain, data=inv, label=r"$\Omega m$")
# ax[0].set_title("Inferred model")
# ax[0].figure.savefig("figs/toy_rect_mesh/inferred_model_inbuilt_solver")

# # plot synthetic data
# d = mgr.fop.response(inv)
# ax = ert.showERTData(scheme, vals=d, cMin=np.min(data["rhoa"]), cMax=np.max(data["rhoa"]))
# ax[0].set_title("Synthetic data from inferred model")
# ax[0].figure.savefig("figs/toy_rect_mesh/inferred_data_inbuilt_solver")


############# Define CoFI BaseProblem ################################

# hyperparameters
lamda = 0.000001

# CoFI - define BaseProblem
ert_problem = BaseProblem()
ert_problem.name = "Electrical Resistivity Tomography defined through PyGIMLi"
ert_problem.set_forward(get_response, args=[forward_oprt])
ert_problem.set_jacobian(get_jacobian, args=[forward_oprt])
ert_problem.set_residual(get_residual, args=[log_data, forward_oprt])
ert_problem.set_data_misfit(get_data_misfit, args=[log_data, forward_oprt, data_cov_inv])
ert_problem.set_regularisation(get_regularisation, args=[Wm, lamda])
ert_problem.set_gradient(get_gradient, args=[log_data, forward_oprt, Wm, lamda, data_cov_inv])
ert_problem.set_hessian(get_hessian, args=[log_data, forward_oprt, Wm, lamda, data_cov_inv])
ert_problem.set_initial_model(start_model_log)


############# Inverted by SciPy optimiser through CoFI ################################

# CoFI - define InversionOptions
inv_options_scipy = InversionOptions()
inv_options_scipy.set_tool("scipy.optimize.minimize")
inv_options_scipy.set_params(method="trust-exact")     # L-BFGS-B Newton-CG trust-exact

# # CoFI - define Inversion, run it
# inv = Inversion(ert_problem, inv_options_scipy)
# inv_result = inv.run()
# inv_result.summary()
# model = np.exp(inv_result.model)

# # plot inferred model
# ax = pygimli.show(mgr.paraDomain, data=model, label=r"$\Omega m$")
# ax[0].set_title("Inferred model")
# ax[0].figure.savefig("figs/toy_rect_mesh/inferred_model_scipy_opt")

# # plot synthetic data
# d = forward_oprt.response(model)
# ax = ert.showERTData(scheme, vals=d, cMin=np.min(data["rhoa"]), cMax=np.max(data["rhoa"]))
# ax[0].set_title("Synthetic data from inferred model")
# ax[0].figure.savefig("figs/toy_rect_mesh/inferred_data_scipy_opt")


############# Inverted by our Gauss-Newton algorithm ##################################

# reference: 
# Carsten Rücker, Thomas Günther, Florian M. Wagner,
# pyGIMLi: An open-source library for modelling and inversion in geophysics,
# https://doi.org/10.1016/j.cageo.2017.07.011.

class GaussNewton(BaseSolver):
    def __init__(self, inv_problem, inv_options):
        __params = inv_options.get_params()
        self._niter = __params.get("niter", 100)
        self._verbose = __params.get("verbose", True)
        self._step = __params.get("step", 1)
        self._model_0 = inv_problem.initial_model
        self._residual = inv_problem.residual
        self._jacobian = inv_problem.jacobian
        self._gradient = inv_problem.gradient
        self._hessian = inv_problem.hessian
        self._misfit = inv_problem.data_misfit if inv_problem.data_misfit_defined else None
        self._reg = inv_problem.regularisation if inv_problem.regularisation_defined else None
        self._obj = inv_problem.objective if inv_problem.objective_defined else None

    def __call__(self):
        current_model = np.array(self._model_0)
        for i in range(self._niter):
            if self._verbose:
                print("-" * 80)
                print(f"Iteration {i+1}")
                print("model min and max:", np.min(current_model), np.max(current_model))
                if self._misfit: print("data misfit:", self._misfit(current_model))
                if self._reg: print("regularisation:", self._reg(current_model))
            term1 = self._hessian(current_model)
            term2 = - self._gradient(current_model)
            model_update = np.linalg.solve(term1, term2) * self._step
            current_model = current_model + model_update
        return {"model": current_model, "success": True}

# hyperparameters
niter = 200
inv_verbose = True
step = 0.005

# CoFI - define InversionOptions
inv_options = InversionOptions()
inv_options.set_tool(GaussNewton)
inv_options.set_params(niter=niter, verbose=inv_verbose, step=step)

# # CoFI - define Inversion, run it
# inv = Inversion(ert_problem, inv_options)
# inv_result = inv.run()
# inv_result.summary()
# model = np.exp(inv_result.model)

# # plot inferred model
# ax = pygimli.show(mgr.paraDomain, data=model, label=r"$\Omega m$")
# ax[0].set_title("Inferred model")
# ax[0].figure.savefig("figs/toy_rect_mesh/inferred_model_gauss_newton")

# # plot synthetic data
# d = forward_oprt.response(model)
# ax = ert.showERTData(scheme, vals=d, cMin=np.min(data["rhoa"]), cMax=np.max(data["rhoa"]))
# ax[0].set_title("Synthetic data from inferred model")
# ax[0].figure.savefig("figs/toy_rect_mesh/inferred_data_gauss_newton")


############# Inverted by our Gauss-Newton algorithm ##################################

class GaussNewtonArmjioLineaSearch(BaseSolver):
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
        self._reg = inv_problem.regularisation if inv_problem.regularisation_defined else None
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
                if self._reg: print("regularisation:", self._reg(current_model))
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
inv_verbose = True
tau_tol = 1e-10
update_tol = 1e-10

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
ax = pygimli.show(mgr.paraDomain, data=model, label=r"$\Omega m$")
ax[0].set_title("Inferred model")
ax[0].figure.savefig("figs/toy_rect_mesh/inferred_model_gauss_newton_armjio_linesearch")

# plot synthetic data
d = forward_oprt.response(model)
ax = ert.showERTData(scheme, vals=d, cMin=np.min(data["rhoa"]), cMax=np.max(data["rhoa"]))
ax[0].set_title("Synthetic data from inferred model")
ax[0].figure.savefig("figs/toy_rect_mesh/inferred_data_gauss_newton_armjio_linesearch")
