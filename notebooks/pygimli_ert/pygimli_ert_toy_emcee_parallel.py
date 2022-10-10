import numpy as np
import pygimli
from pygimli.physics import ert
import emcee
from cofi import BaseProblem, InversionOptions, Inversion

from pygimli_ert_lib import (
    survey_scheme,
    model_true,
    ert_simulate,
    ert_manager,
    inversion_mesh_rect_toy,
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


############# ERT Modelling with PyGIMLi ##############################################

# measuring scheme
scheme = survey_scheme()

# create simulation mesh and true model
mesh, rhomap = model_true(scheme)
ax = pygimli.show(mesh, data=rhomap, label="$\Omega m$", showMesh=True)
ax[0].set_title("True model")
ax[0].figure.savefig("figs/emcee_toy_model_true")

# generate data
data, log_data, data_cov_inv = ert_simulate(mesh, scheme, rhomap)
ax = ert.show(data)
ax[0].set_title("Provided data")
ax[0].figure.savefig("figs/emcee_data")

# create PyGIMLi's ERT manager
ert_manager = ert_manager(data)

# create inversion mesh
inv_mesh = inversion_mesh_rect_toy(ert_manager)
ax = pygimli.show(inv_mesh, showMesh=True, markers=True)
ax[0].set_title("Mesh used for inversion")
ax[0].figure.savefig("figs/emcee_toy_inv_mesh")

# PyGIMLi's forward operator (ERTModelling)
forward_oprt = ert_forward_operator(ert_manager, scheme, inv_mesh)

# extract regularization matrix
Wm = reg_matrix(forward_oprt)

# initialise a starting model for inversion
start_model, start_model_log = starting_model(ert_manager)
ax = pygimli.show(ert_manager.paraDomain, data=start_model, label="$\Omega m$", showMesh=True)
ax[0].set_title("Starting model")
ax[0].figure.savefig("figs/emcee_toy_model_start")


############# Define CoFI BaseProblem #################################################

# hyperparameters
lamda = 0.0005

# CoFI - define BaseProblem
ert_problem = BaseProblem()
ert_problem.name = "Electrical Resistivity Tomography defined through PyGIMLi"
# ert_problem.set_forward(get_response, args=[forward_oprt])
# ert_problem.set_jacobian(get_jacobian, args=[forward_oprt])
# ert_problem.set_residual(get_residual, args=[log_data, forward_oprt])
ert_problem.set_data_misfit(get_data_misfit, args=[log_data, forward_oprt, data_cov_inv])
# ert_problem.set_regularization(get_regularization, args=[Wm, lamda])
# ert_problem.set_gradient(get_gradient, args=[log_data, forward_oprt, Wm, lamda, data_cov_inv])
# ert_problem.set_hessian(get_hessian, args=[log_data, forward_oprt, Wm, lamda, data_cov_inv])
# ert_problem.set_initial_model(start_model_log)


############# Further information defined for emcee ###################################

# for emcee - hyperparameters
nwalkers = 512
nsteps = 10000

# for emcee - define log_likelihood
def log_likelihood(model):
    return -0.5 * ert_problem.data_misfit(model)

# for emcee - define log_prior
m_lower_bound = np.ones(start_model.shape) * 3    # lower bound for uniform prior
m_upper_bound = np.ones(start_model.shape) * 6    # upper bound for uniform prior
def log_prior(model):                           # uniform distribution
    for i in range(len(m_lower_bound)):
        if model[i] < m_lower_bound[i] or model[i] > m_upper_bound[i]: return -np.inf
    return 0.0 # model lies within bounds -> return log(1)

# for emcee - define walkers' starting positions
walkers_start = start_model_log + 1e-3 * np.random.randn(nwalkers, start_model.shape[0])

# CoFI - define them into cofi's BaseProblem object
ert_problem.set_log_likelihood(log_likelihood)
ert_problem.set_log_prior(log_prior)
ert_problem.set_walkers_starting_pos(walkers_start)


############# Sampled by emcee ########################################################
# CoFI - define inversion options
from multiprocessing import Pool, cpu_count

ncpu = cpu_count()
print("{0} CPUs".format(ncpu))

with Pool() as pool:
    inv_options_emcee = InversionOptions()
    inv_options_emcee.set_tool("emcee")
    inv_options_emcee.set_params(nwalkers=nwalkers, nsteps=nsteps, progress=True, pool=pool)

    # emcee - set backend to store progress
    filename = f"pygimli_ert_toy_emcee_{nsteps}.h5"
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, start_model_log.shape[0])
    inv_options_emcee.set_params(backend=backend)

    # CoFI - run the inversion
    inv_rect_emcee = Inversion(ert_problem, inv_options_emcee)
    inv_rect_emcee_res = inv_rect_emcee.run()

# sample 10 models from the posterior ensemble and plot
sampler = inv_rect_emcee_res.sampler

flat_samples = sampler.get_chain(discard=5, flat=True)
indices = np.random.randint(len(flat_samples), size=10) # get a random selection from posterior ensemble
for idx in indices:
    ax=pygimli.show(
        ert_manager.paraDomain,
        data=(flat_samples[idx]),
        label=r"$\Omega m$"
    )
    ax[0].set_title(f"Inferred model - sample {idx}")
    ax[0].figure.savefig(f"figs/emcee_samples/{idx}")
