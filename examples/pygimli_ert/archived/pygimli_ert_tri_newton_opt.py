"""Eletrical Resistivity Tomography Inversion with PyGIMLi + CoFI

This script runs:
- ERT problem (triangular mesh) defined with PyGIMLi, and
- Newton's optimization method with CoFI


To run this script, refer to the following examples:

- `python pygimli_ert_tri_newton_opt.py` for a simple run, with all the figures saved to
  current directory by default

- `python pygimli_ert_tri_newton_opt.py -o figs` for the same run as above, with all the
  figures saved to subfolder `figs`

- `python pygimli_ert_tri_newton_opt.py -h` to see all available options

"""

############# 0. Import modules #######################################################

import numpy as np
import matplotlib.pyplot as plt
import pygimli
from pygimli.physics import ert

from cofi import BaseProblem, InversionOptions, Inversion
from cofi.tools import BaseInferenceTool

from pygimli_ert_lib import *


############# Configurations ##########################################################

np.random.seed(42)

save_plot = True
show_plot = False
show_summary = True

_problem_name = "pygimli_ert"
_solver_name = "scipiy_opt"
_file_prefix = f"{_problem_name}_{_solver_name}"
_figs_prefix = f"./{_file_prefix}"


############# Define a custom solver (Newton's optimization method) ###################

class MyNewtonSolver(BaseInferenceTool):
    def __init__(self, inv_problem, inv_options):
        __params = inv_options.get_params()
        self._niter = __params["niter"]
        self._step = __params["step"]
        self._verbose = __params["verbose"]
        self._model_0 = inv_problem.initial_model
        self._gradient = inv_problem.gradient
        self._hessian = inv_problem.hessian
        self._misfit = inv_problem.data_misfit if inv_problem.data_misfit_defined else None
        self._reg = inv_problem.regularization if inv_problem.regularization_defined else None
        self._obj = inv_problem.objective if inv_problem.objective_defined else None
        
    def __call__(self):
        current_model = np.array(self._model_0)
        for i in range(self._niter):
            term1 = self._hessian(current_model)
            term2 = - self._gradient(current_model)
            model_update = np.linalg.solve(term1, term2)
            current_model = np.array(current_model + self._step * model_update)
            if self._verbose:
                print("-" * 80)
                print(f"Iteration {i+1}")
                if self._misfit: self._misfit(current_model)
                if self._reg: self._reg(current_model)
                # if self._obj: print("objective func:", self._obj(current_model))
        return {"model": current_model, "success": True}


############# Plotting functions ######################################################

def _post_plot(ax, title):
    ax[0].set_title(title)
    if save_plot:
        ax[0].figure.savefig(f"{_figs_prefix}_{title}")
    if show_plot:
        plt.show()

def plot_mesh(mesh, title):
    ax = pygimli.show(mesh)
    _post_plot(ax, title)
    
def plot_model(mesh, model, label, title):
    ax = pygimli.show(mesh, data=model, label=label)
    _post_plot(ax, title)

def plot_data(survey, label, title):
    ax=ert.showERTData(survey,label=label)
    _post_plot(ax, title)


############# Main ####################################################################

def main():

    ######### 1. Define the problem ###################################################

    # PyGIMLi - define measuring scheme, geometry, forward mesh and true model
    scheme = scheme_fwd()
    geometry = geometry_true()
    fmesh = mesh_fwd(scheme, geometry)
    rhomap = markers_to_resistivity()
    model_true = model_vec(rhomap, fmesh)

    # PyGIMLi - plot the compuational mesh and the true model
    plot_mesh(fmesh, "Computational Mesh")
    plot_model(fmesh, model_true, "$\Omega m$", "Resistivity")

    # PyGIMLi - generate data
    survey = ert.simulate(fmesh, res=rhomap, scheme=scheme)
    plot_data(survey, "$\Omega m$", "Aparent Resistivity")
    y_obs = np.log(survey['rhoa'].array())

    # PyGIMLi - create mesh for inversion
    imesh_tri = mesh_inv_triangular(scheme)
    plot_mesh(imesh_tri, "Inversion Mesh")

    # PyGIMLi - define starting model, forward operator and Wm
    model_0 = starting_model(imesh_tri)
    forward_operator = forward_oprt(scheme, imesh_tri)
    Wm = weighting_matrix(forward_operator, imesh_tri)

    # hyperparameters
    lamda = 2
    niter = 100
    learning_step = 1
    inv_verbose = True

    # CoFI - define BaseProblem
    ert_problem = BaseProblem()
    ert_problem.name = "Electrical Resistivity Tomography defined through PyGIMLi"
    ert_problem.set_forward(get_response, args=[forward_operator])
    ert_problem.set_jacobian(get_jacobian, args=[forward_operator])
    ert_problem.set_residual(get_residuals, args=[y_obs, forward_operator])
    ert_problem.set_data_misfit(get_misfit, args=[y_obs, forward_operator, True])
    ert_problem.set_regularization(get_regularization, lamda=lamda, args=[Wm, True])
    ert_problem.set_gradient(get_gradient, args=[y_obs, forward_operator, lamda, Wm])
    ert_problem.set_hessian(get_hessian, args=[y_obs, forward_operator, lamda, Wm])
    ert_problem.set_initial_model(model_0)

    if show_summary:
        ert_problem.summary()


    ######### 2. Define the inversion options #########################################
    inv_options = InversionOptions()
    inv_options.set_tool(MyNewtonSolver)
    inv_options.set_params(niter=niter, step=learning_step, verbose=inv_verbose)

    if show_summary:
        inv_options.summary()


    ######### 3. Run the inversion ####################################################
    inv = Inversion(ert_problem, inv_options)
    inv_result = inv.run()

    if show_summary:
        inv_result.summary()


    ######### 4. Plot the results #####################################################
    plot_model(fmesh, model_true, r"$\Omega m$", "True model")
    plot_model(imesh_tri, model_0, r"$\Omega m$", "Starting model")
    plot_model(imesh_tri, inv_result.model, r"$\Omega m$", "Inferred model")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='Polynomial Linear regression solved by linear system solver'
    )
    parser.add_argument("--output-dir", "-o", type=str, help="output folder for figures")
    parser.add_argument("--show-plot", dest="show_plot", action="store_true", default=None)
    parser.add_argument("--no-show-plot", dest="show_plot", action="store_false", default=None)
    parser.add_argument("--save-plot", dest="save_plot", action="store_true", default=None)
    parser.add_argument("--no-save-plot", dest="save_plot", action="store_false", default=None)
    parser.add_argument("--show-summary", dest="show_summary", action="store_true", default=None)
    parser.add_argument("--no-show-summary", dest="show_summary", action="store_false", default=None)
    args = parser.parse_args()
    output_dir = args.output_dir or "."
    if output_dir.endswith("/"):
        output_dir = output_dir[:-1]
    show_plot = show_plot if args.show_plot is None else args.show_plot
    save_plot = save_plot if args.save_plot is None else args.save_plot
    show_summary = show_summary if args.show_summary is None else args.show_summary

    _figs_prefix = f"{output_dir}/{_file_prefix}"
    main()
