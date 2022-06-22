"""Gravity Inversion Problem with CoFI

This script runs:
- Inversion on gravity density model, and
- Scipy's optimisation with CoFI


To run this script, refer to the following examples:

- `python gravity_density_scipy_opt.py` for a simple run, with all the figures saved to
  current directory by default

- `python gravity_density_scipy_opt.py -o figs` for the same run as above, with all the
  figures saved to subfolder `figs`

- `python gravity_density_scipy_opt.py -h` to see all available options

"""

############# 0. Import modules #######################################################

import numpy as np
import matplotlib.pyplot as plt

from cofi import BaseProblem, InversionOptions, Inversion

from gravity_density_lib import *


############# Configurations ##########################################################

np.random.seed(42)

save_plot = True
show_plot = False
show_summary = True

_problem_name = "gravity_density"
_solver_name = "scipiy_opt"
_file_prefix = f"{_problem_name}_{_solver_name}"
_figs_prefix = f"./{_file_prefix}"


############# Main ####################################################################

def main():

    ######### 1. Define the problem ###################################################
    
    # Load true model and starting guesses
    rec_coords, _, _, z_nodes, model = load_gravity_model()
    Starting_model1, Starting_model2, Starting_model3 = load_starting_models()

    # Create "observed" data by adding noise to forward solution
    noise_level=0.05
    gz = forward(model)
    dataZ_obs= gz + np.random.normal(loc=0,scale=noise_level*np.max(np.abs(gz)),size=np.shape(gz))  

    # Create jacobian
    Jz = get_jacobian(model)

    # Define depth weighting values
    z0=18.6
    beta=2
    # Define regularization parameter
    epsilon=0.2

    # Create regularization
    # Calculate depth weighting fcn - high values at low z, low values at high z, no zeros.
    # Model is [Nx1] with N: no. of cells; W is [NxN] with weighting values on diagonal
    W=depth_weight(z_nodes[:,0],z0,beta)
    W=np.diag(W)

    # Set CoFI problem:
    grav_problem = BaseProblem()
    grav_problem.name = "Gravity"
    grav_problem.set_data(gz)

    # Here I linked the function, not the result
    grav_problem.set_forward(forward)

    # Here I linked to the actual jacobian. Jacobian size is (MxN) with M: receiver and N: model cells
    grav_problem.set_jacobian(Jz)

    # Set regularization; reg is a function that takes the model as input
    grav_problem.set_regularisation(reg_l1, epsilon, args=[W])

    # Use default L2 misfit
    grav_problem.set_data_misfit("L2")
    grav_problem.set_initial_model(Starting_model3)

    # Set gradient, in hope of helping optimisers converge better
    def data_misfit_gradient(model):
        return 2* Jz.T @ (forward(model) - gz) / gz.shape[0]
    grav_problem.set_gradient(lambda m: data_misfit_gradient(m) + epsilon*reg_gradient_l1(m, W))

    if show_summary:
        grav_problem.summary()
        

    ######### 2. Define the inversion options #########################################
    inv_options = InversionOptions()
    inv_options.set_tool("scipy.optimize.least_squares")

    if show_summary:
        inv_options.summary()


    ######### 3. Run the inversion ####################################################
    inv = Inversion(grav_problem, inv_options)
    inv_result = inv.run()

    if show_summary:
        inv_result.summary()


    ######### 4. Plot the results #####################################################
    if save_plot or show_plot:
        result_model = inv_result.model.reshape(12,12,12)

        plt.imshow(result_model[::-1,6,:])
        plt.colorbar()
        if save_plot:
            plt.savefig(f"{_figs_prefix}_vertical_plane")

        plt.imshow(result_model[6,:,:])
        plt.colorbar()
        if save_plot:
            plt.savefig(f"{_figs_prefix}_horizontal_plane")

        if show_plot:
            plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Gravity Inversion Problem with CoFI"
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
