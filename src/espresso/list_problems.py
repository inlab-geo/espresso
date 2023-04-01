_all_problems = []
_capability_matrix = dict()

def list_problem_names(capabilities: list = None):
    """Returns a list of all Espresso problem names"""
    _problems = list_problems(capabilities)
    _all_names = [p.__name__ for p in _problems]
    return _all_names

def list_problems(capabilities: list = None):
    """Returns a list of all Espresso problem classes"""
    if capabilities is None:
        return _all_problems
    else:
        _problem_names = []
        for p, c in _capability_matrix.items():
            ok = True
            for to_check in capabilities:
                if not (to_check in c and c[to_check] == 1):
                    ok = False
                    break
            if ok:
                _problem_names.append(p)
        _problems = []
        for p in _all_problems:
            if p.__name__ in _problem_names:
                _problems.append(p)
        return _problems

def list_capability(problem_names: list) -> dict:
    return {k:v for k,v in _capability_matrix.items() if k in problem_names}

# from .example_name import ExampleName

# _all_problems = [ ExampleName, ]

# _capability_matrix = {
#     "ExampleName": {
#         "model_size": 1,
#         "data_size": 1,
#         "starting_model": 1,
#         ...
#     },
#     ...
# }

from ._fmm_tomography import FmmTomography
from ._pumping_test import PumpingTest
from ._simple_regression import SimpleRegression
from ._slug_test import SlugTest
from ._xray_tomography import XrayTomography
from ._gravity_density import GravityDensity
from ._magnetotelluric_1D import Magnetotelluric1D

_all_problems = [
    FmmTomography,
    PumpingTest,
    SimpleRegression,
    SlugTest,
    XrayTomography,
    GravityDensity,
    Magnetotelluric1D,
]

_capability_matrix = {
    "FmmTomography": {
        "model_size": 1,
        "data_size": 1,
        "good_model": 1,
        "starting_model": 1,
        "data": 1,
        "forward": 1,
        "description": 0,
        "covariance_matrix": 0,
        "inverse_covariance_matrix": 0,
        "jacobian": 1,
        "plot_model": 1,
        "plot_data": 0,
        "misfit": 0,
        "log_likelihood": 0,
        "log_prior": 0,
        "exe_fm2dss": 1,
        "tmp_paths": 1,
        "tmp_files": 1,
        "clean_tmp_files": 1
    },
    "PumpingTest": {
        "model_size": 1,
        "data_size": 1,
        "good_model": 1,
        "starting_model": 1,
        "data": 1,
        "forward": 0,
        "description": 1,
        "covariance_matrix": 1,
        "inverse_covariance_matrix": 1,
        "jacobian": 0,
        "plot_model": 1,
        "plot_data": 0,
        "misfit": 0,
        "log_likelihood": 1,
        "log_prior": 0
    },
    "SimpleRegression": {
        "model_size": 1,
        "data_size": 1,
        "good_model": 1,
        "starting_model": 1,
        "data": 1,
        "forward": 1,
        "description": 1,
        "covariance_matrix": 1,
        "inverse_covariance_matrix": 1,
        "jacobian": 1,
        "plot_model": 0,
        "plot_data": 0,
        "misfit": 0,
        "log_likelihood": 0,
        "log_prior": 0
    },
    "SlugTest": {
        "model_size": 1,
        "data_size": 1,
        "good_model": 1,
        "starting_model": 1,
        "data": 1,
        "forward": 0,
        "description": 1,
        "covariance_matrix": 1,
        "inverse_covariance_matrix": 1,
        "jacobian": 0,
        "plot_model": 1,
        "plot_data": 0,
        "misfit": 0,
        "log_likelihood": 1,
        "log_prior": 0
    },
    "XrayTomography": {
        "model_size": 1,
        "data_size": 1,
        "good_model": 1,
        "starting_model": 1,
        "data": 1,
        "forward": 0,
        "description": 1,
        "covariance_matrix": 0,
        "inverse_covariance_matrix": 0,
        "jacobian": 0,
        "plot_model": 1,
        "plot_data": 0,
        "misfit": 0,
        "log_likelihood": 0,
        "log_prior": 0
    },
    "GravityDensity": {
        "model_size": 1,
        "data_size": 1,
        "good_model": 1,
        "starting_model": 1,
        "data": 1,
        "forward": 1,
        "description": 0,
        "covariance_matrix": 1,
        "inverse_covariance_matrix": 0,
        "jacobian": 1,
        "plot_model": 1,
        "plot_data": 1,
        "misfit": 0,
        "log_likelihood": 0,
        "log_prior": 0
    },
    "Magnetotelluric1D": {
        "model_size": 1,
        "data_size": 1,
        "good_model": 1,
        "starting_model": 1,
        "data": 1,
        "forward": 1,
        "description": 1,
        "covariance_matrix": 1,
        "inverse_covariance_matrix": 1,
        "jacobian": 1,
        "plot_model": 1,
        "plot_data": 1,
        "misfit": 1,
        "log_likelihood": 0,
        "log_prior": 0,
        "set_start_model": 1,
        "set_obs_data": 1,
        "set_start_mesh": 1
    }
}