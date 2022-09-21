"""Seismic waveform inference with Fast Marching and CoFI

Based on original scripts by Andrew Valentine & Malcolm Sambridge -
Research Schoole of Earth Sciences, The Australian National University
Last updated July 2022

"""

import numpy as np
import matplotlib.pyplot as plt
import findiff
from scipy.interpolate import interp2d
import random

from cofi import BaseProblem, InversionOptions, Inversion
from cofi_espresso import FmmTomography

# define CoFI BaseProblem
fmm = FmmTomography()
fmm_problem = BaseProblem()
fmm_problem.set_data(fmm.data)
fmm_problem.set_forward(fmm.forward)
fmm_problem.set_data_misfit("L2")

# add regularisation: damping + smoothing
# to be replaced by cofi.utils
damping_factor = 100.0
smoothing_factor = 5e3


