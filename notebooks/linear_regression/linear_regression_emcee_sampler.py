"""Polynomial Linear regression solved by sampler with emcee

This file sets up an example from setting up problem to running the inversion:
- For the problem: polynomial linear regression,
- Using the tool: Bayesian sampler (emcee)

The function we are going to fit is: y = -6 - 5x + 2x^2 + x^3

We may also write the polynomial curves in this form: y = sum(m_n * x^n), n=0,1,2,3,
where: m_n, n=0,1,2,3 are the model coefficients.

If we consider N data points and M=3 model parameters, then N equations like above 
yields a linear operation: d = Gm,
where: d refers to data observations (y_1, y_2, ..., y_N).T
       G refers to basis matrix: ( (1, x_1, x_1^2, x_1^3)
                                   (1, x_2, x_2^2, x_2^3)
                                   ...
                                   (1, x_N, x_N^2, x_N^3) )
       m refers to the unknown model parameters (m_0, m_1, m_2, m_3)

Note that G matrix is here equivalent to the Jacobian as it entries G(i,j) are the 
first derivative of the i-th datum d(i) with respect to the j-th model parameter m(j). 
We here refer to the function that calculates the G matrix given a set of model 
parameters as the basis function.

"""

############# 0. Import modules #######################################################

import numpy as np
import arviz as az
import matplotlib.pyplot as plt
from cofi import BaseProblem, InversionOptions, Inversion

np.random.seed(42)

save_plot = True
show_plot = False
show_summary = True

_problem_name = "linear_reg"
_solver_name = "emcee"
_file_prefix = f"{_problem_name}_{_solver_name}"

def main(output_dir="."):
    _figs_prefix = f"{output_dir}/{_file_prefix}"

    ############# 1. Define the problem ###############################################

    # generate data with random Gaussian noise
    def basis_func(x):
        return np.array([x**i for i in range(4)]).T                           # x -> G
    _m_true = np.array([-6,-5,2,1])                                           # m

    sample_size = 20                                                          # N
    x = np.random.choice(np.linspace(-3.5,2.5), size=sample_size)             # x
    def forward_func(m):
        return basis_func(x) @ m                                              # m -> y_synthetic
    y_observed = forward_func(_m_true) + np.random.normal(0,1,sample_size)    # d

    sigma = 1.0                                     # common noise standard deviation
    Cdinv = np.eye(len(y_observed))/(sigma**2)      # inverse data covariance matrix
    m_lower_bound = np.ones(4) * (-10.)             # lower bound for uniform prior
    m_upper_bound = np.ones(4) * 10                 # upper bound for uniform prior

    nwalkers = 32
    ndim = 4
    nsteps = 5000
    walkers_start = np.array([0.,0.,0.,0.]) + 1e-4 * np.random.randn(nwalkers, ndim)

    if save_plot or show_plot:
        _x_plot = np.linspace(-3.5,2.5)
        _G_plot = basis_func(_x_plot)
        _y_plot = _G_plot @ _m_true
        plt.figure(figsize=(12,8))
        plt.plot(_x_plot, _y_plot, color="darkorange", label="true model")
        plt.scatter(x, y_observed, color="lightcoral", label="observed data")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        if save_plot:
            plt.savefig(f"{_figs_prefix}_problem")

    # define functions for Bayesian sampling
    def log_prior(model):    # uniform distribution
        for i in range(len(m_lower_bound)):
            if model[i] < m_lower_bound[i] or model[i] > m_upper_bound[i]: return -np.inf
        return 0.0 # model lies within bounds -> return log(1)

    def log_likelihood(model):
        y_synthetics = forward_func(model)
        residual = y_observed - y_synthetics
        return -0.5 * residual @ (Cdinv @ residual).T


    # define the problem in cofi
    inv_problem = BaseProblem()
    inv_problem.name = "Polynomial Regression"
    inv_problem.set_log_prior(log_prior)
    inv_problem.set_log_likelihood(log_likelihood)
    inv_problem.set_walkers_starting_pos(walkers_start)
    if show_summary:
        inv_problem.summary()


    ############# 2. Define the inversion options #####################################
    inv_options = InversionOptions()
    inv_options.set_tool("emcee")
    inv_options.set_params(nwalkers=nwalkers, nsteps=nsteps)
    if show_summary:
        inv_options.summary()


    ############# 3. Start an inversion ###############################################
    inv = Inversion(inv_problem, inv_options)
    inv_result = inv.run()
    if show_summary:
        inv_result.summary()


    ############# 4. Plot result ######################################################
    sampler = inv_result.sampler
    az_idata = inv_result.to_arviz()
    labels = ["m0", "m1", "m2","m3"]

    if save_plot or show_plot:
        # plot sampling performance
        az.plot_trace(az_idata)
        if save_plot:
            plt.savefig(f"{_figs_prefix}_trace")

        # autocorrelation analysis
        tau = sampler.get_autocorr_time()
        print(f"autocorrelation time: {tau}")

        # corner plot after thinning by about half the autocorrelation time
        az.plot_pair(
            az_idata.sel(draw=slice(300,None)), 
            marginals=True, 
            reference_values=dict(zip([f"var_{i}" for i in range(4)], _m_true.tolist()))
        )
        if save_plot:
            plt.savefig(f"{_figs_prefix}_corner")

        # sub-sample of 100 predicted curves from the posterior ensemble
        flat_samples = sampler.get_chain(discard=300, thin=30, flat=True)
        inds = np.random.randint(len(flat_samples), size=100) # get a random selection from posterior ensemble
        _x_plot = np.linspace(-3.5,2.5)
        _G_plot = basis_func(_x_plot)
        _y_plot = _G_plot @ _m_true
        plt.figure(figsize=(12,8))
        sample = flat_samples[0]
        _y_synth = _G_plot @ sample
        plt.plot(_x_plot, _y_synth, color="seagreen", label="Posterior samples",alpha=0.1)
        for ind in inds:
            sample = flat_samples[ind]
            _y_synth = _G_plot @ sample
            plt.plot(_x_plot, _y_synth, color="seagreen", alpha=0.1)
        plt.plot(_x_plot, _y_plot, color="darkorange", label="true model")
        plt.scatter(x, y_observed, color="lightcoral", label="observed data")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        if save_plot:
            plt.savefig(f"{_figs_prefix}_sample_curves")

        if show_plot:
            plt.show()
        

    if show_summary:
        flat_samples = sampler.get_chain(discard=300, thin=30, flat=True)
        # uncertainties - 16th, 50th, and 84th percentiles of the samples in the marginalized distributions
        solmed = np.zeros(4)
        for i in range(ndim):
            mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
            solmed[i] = mcmc[1]
            q = np.diff(mcmc)
            print(f"{labels[i]} = {round(mcmc[1],3)}, (-{round(q[0],3)}, +{round(q[1],3)})")
        
        # posterior model covariance matrix
        CMpost = np.cov(flat_samples.T)
        CM_std= np.std(flat_samples,axis=0)
        print('Posterior model covariance matrix\n',CMpost)
        print('\nPosterior estimate of model standard deviations in each parameter')
        for i in range(ndim):
            print("    {} {:7.4f}".format(labels[i],CM_std[i]))

        # solution and 95% credible intervals
        print("\n Solution and 95% credible intervals ")
        for i in range(ndim):
            mcmc = np.percentile(flat_samples[:, i], [5, 50, 95])
            print(" {} {:7.3f} [{:7.3f}, {:7.3f}]".format(labels[i],mcmc[1],mcmc[0],mcmc[2]))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='Polynomial Linear regression solved by linear system solver'
    )
    parser.add_argument("--output_dir", "-o", type=str, help="output folder for figures")
    parser.add_argument("--show-plot", dest="show_plot", action="store_true", default=None)
    parser.add_argument("--no-show-plot", dest="show_plot", action="store_false", default=None)
    parser.add_argument("--save-plot", dest="save_plot", action="store_true", default=None)
    parser.add_argument("--no-save-plot", dest="save_plot", action="store_false", default=None)
    parser.add_argument("--show-summary", dest="show_summary", action="store_true", default=None)
    parser.add_argument("--no-show-summary", dest="show_summary", action="store_false", default=None)
    args = parser.parse_args()
    output_dir = args.output_dir or "."
    show_plot = show_plot if args.show_plot is None else args.show_plot
    save_plot = save_plot if args.save_plot is None else args.save_plot
    show_summary = show_summary if args.show_summary is None else args.show_summary

    main(output_dir)
