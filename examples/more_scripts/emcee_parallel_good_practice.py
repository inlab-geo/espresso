"""
This script demonstrates the "good" practice for a simple parallel example.

Code in this file is adapted from emcee documentation. And we intend to deal with the
exact issue described here:
https://emcee.readthedocs.io/en/stable/tutorials/parallel/#pickling-data-transfer-arguments

Compared to the file emcee_parallel_bad_practice.py (in the same folder), the "data" 
argument is set to be global instead of being an argument of log probability function.

As you run this file and you will see, the speed increases when it's in parallel.
"""

import os
import time
from multiprocessing import Pool, cpu_count, set_start_method
import numpy as np
import cofi

# some setup
def log_prob_data(theta):
    global data
    a = data[0]  # Use the data somehow...
    t = time.time() + np.random.uniform(0.005, 0.008)
    while True:
        if time.time() >= t:
            break
    return -0.5 * np.sum(theta**2)

# more setup
set_start_method("fork")
os.environ["OMP_NUM_THREADS"] = "1"
np.random.seed(42)
initial = np.random.randn(32, 5)
nwalkers, ndim = initial.shape
nsteps = 100
data = np.random.randn(5000, 200)

# cofi BaseProblem setup
inv_problem = cofi.BaseProblem()
inv_problem.set_log_posterior(log_prob_data)
inv_problem.set_model_shape(ndim)

# serial experiment
inv_options_serial = cofi.InversionOptions()
inv_options_serial.set_tool("emcee")
inv_options_serial.set_params(
    nwalkers=nwalkers,
    nsteps=nsteps,
    progress=True,
    initial_state=initial,
)
inv_serial = cofi.Inversion(inv_problem, inv_options_serial)
start = time.time()
inv_result_serial = inv_serial.run()
end = time.time()
serial_data_time = end - start
print("Serial took {0:.1f} seconds".format(serial_data_time))

# parallel experiment
with Pool() as pool:
    inv_options_parallel = cofi.InversionOptions()
    inv_options_parallel.set_tool("emcee")
    inv_options_parallel.set_params(
        nwalkers=nwalkers,
        nsteps=nsteps,
        progress=True,
        pool=pool,
        initial_state=initial,
    )
    inv_parallel = cofi.Inversion(inv_problem, inv_options_parallel)
    start = time.time()
    inv_result_parallel = inv_parallel.run()
    end = time.time()
    multi_data_time = end - start
    print("Multiprocessing took {0:.1f} seconds".format(multi_data_time))
    print(
        "{0:.1f} times faster than serial".format(
            serial_data_time / multi_data_time
        )
    )

# print number of cores
ncpu = cpu_count()
print("{0} CPUs".format(ncpu))
