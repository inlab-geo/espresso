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

Note that G matrix can also be called the Jacobian as it is the first derivative of
forward operator with respect to the unknown model. We refer to the function that 
calculates G matrix given a set of x as the basis function.

"""

############# 0. Import modules #######################################################

import numpy as np
import arviz as az
import matplotlib.pyplot as plt
from cofi import BaseProblem, InversionOptions, Inversion

np.random.seed(42)

show_plot = False
show_summary = True

def main():

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

    if show_plot:
        _x_plot = np.linspace(-3.5,2.5)
        _G_plot = basis_func(_x_plot)
        _y_plot = _G_plot @ _m_true
        plt.figure(figsize=(12,8))
        plt.plot(_x_plot, _y_plot, color="darkorange", label="true model")
        plt.scatter(x, y_observed, color="lightcoral", label="observed data")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.show()

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

    if show_plot:
        # plot sampling performance
        az.plot_trace(az_idata)

        # autocorrelation analysis
        tau = sampler.get_autocorr_time()
        print(f"autocorrelation time: {tau}")

        # corner plot after thinning by about half the autocorrelation time
        az.plot_pair(
            az_idata.sel(draw=slice(300,None)), 
            marginals=True, 
            reference_values=dict(zip([f"var_{i}" for i in range(4)], _m_true.tolist()))
        )

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
    main()
