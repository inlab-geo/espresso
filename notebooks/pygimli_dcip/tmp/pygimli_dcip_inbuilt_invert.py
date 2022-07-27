from pygimli_dcip_lib import (
    survey_scheme,
    model_true,
    model_vec,
    ert_simulate,
    ert_manager,
    reg_matrix,
    get_data_misfit,
    get_regularisation,
    plot_model,
    plot_synth_data,
)

############# ERT Modelling with PyGIMLi ##############################################

# measuring scheme
scheme = survey_scheme()

# create simulation mesh and true model
mesh, rhomap = model_true(scheme)
fig = plot_model(mesh, model_vec(rhomap, mesh))
fig.savefig("figs/inbuilt_solver_model_true")

# generate data
data, r_complex = ert_simulate(mesh, scheme, rhomap)
fig = plot_synth_data(data, r_complex)
fig.savefig("figs/inbuilt_solver_data")

# create PyGIMLi's ERT manager
mgr = ert_manager(data, verbose=True)


# ############# Inverted by PyGIMLi solvers #############################################

inv = mgr.invert(verbose=True)
fig = mgr.showResultAndFit()
fig.savefig("figs/inbuilt_solver_result")

# plot inferred model
fig = plot_model(mgr.paraDomain, inv)
fig.savefig("figs/inbuilt_solver_inferred_model")

# plot synthetic data
data, r_complex = ert_simulate(mgr.paraDomain, scheme, inv)
fig = plot_synth_data(data, r_complex)
fig.savefig("figs/inbuilt_solver_inferred_data")

# print data misfit and regularisation term for inversion result
Wm = reg_matrix(mgr.fop)
print("data misfit:", get_data_misfit(inv, r_complex, mgr.fop))
print("regularisation:", get_regularisation(inv, Wm, 20))
