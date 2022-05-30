"""Polynomial Linear regression solved by sampler with emcee

This file sets up an example from setting up problem to running the inversion:
- For the problem: polynomial linear regression,
- Using the tool: Bayesian sampler (emcee)

The function we are going to fit is: y = -6 - 5x + 2x^2 + x^3

"""

############# 0. Import modules #######################################################

import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import corner
from cofi import BaseProblem, InversionOptions, Inversion

np.random.seed(42)

show_plot = False
show_summary = True


############# 1. Define the problem ###################################################

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


############# 2. Define the inversion options #########################################
inv_options = InversionOptions()
inv_options.set_tool("emcee")
inv_options.set_params(nwalkers=nwalkers, nsteps=nsteps)
if show_summary:
    inv_options.summary()


############# 3. Start an inversion ###################################################
inv = Inversion(inv_problem, inv_options)
inv_result = inv.run()
if show_summary:
    inv_result.summary()


############# 4. Plot result ##########################################################
sampler = inv_result.sampler
print(type(sampler.get_blobs()))
az_idata = inv_result.to_arviz()

# if show_plot:
# plot sampling performance
az.plot_trace(az_idata)
# autocorrelation analysis
tau = sampler.get_autocorr_time()
print(f"autocorrelation time: {tau}")

plt.show()


if show_plot:
    # plot sampling performance
    samples = sampler.get_chain()
    labels = ["m0", "m1", "m2","m3"]
    fig, axes = plt.subplots(len(labels), figsize=(10, 7), sharex=True)
    for i in range(len(labels)):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel("step number")

    # autocorrelation analysis
    tau = sampler.get_autocorr_time()
    print(f"autocorrelation time: {tau}")

    # thin by about half the autocorrelation time, and flatten the chain
    flat_samples = sampler.get_chain(discard=300, thin=30, flat=True)

    # corner plot
    fig = corner.corner(flat_samples, labels=labels, truths=_m_true.tolist())

    # sub-sample of 100 predicted curves from the posterior ensemble
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
    plt.legend()
    plt.show()
