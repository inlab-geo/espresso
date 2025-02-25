"""
=======================
Understand capabilities
=======================

Submodule for listing capabilities in Espresso problems.

To programatically see what problems are available in Espresso, these functions are 
helpful:

"""

import typing

_all_problems = []
_capability_matrix = dict()


def list_problem_names(capabilities: list = None):
    """Returns a list of all Espresso problem names

    Parameters
    ----------
    capabilities : list
        a list of strings to filter the problem names, default to None

    Examples
    --------
    >>> import espresso
    >>> problem_names = espresso.list_problem_names()
    """
    _problems = list_problems(capabilities)
    _all_names = [p.__name__ for p in _problems]
    return _all_names


def list_problems(capabilities: list = None):
    """Returns a list of all Espresso problem classes

    Parameters
    ----------
    capabilities : list
        a list of strings to filter the problem classes, default to None

    Examples
    --------
    >>> import espresso
    >>> problems = espresso.list_problems()
    >>> problems_with_model_plotting = epsresso.list_problems(['plot_model']])
    """
    if capabilities is None:
        return _all_problems
    elif not isinstance(capabilities, (list, set, tuple)):
        raise ValueError(
            "pass in a list of capabilities, e.g. "
            "`espresso.list_problems(['plot_model'])"
        )
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


def list_capabilities(problem_names: typing.Union[list, str] = None) -> dict:
    """Returns a dictionary of capabilities filtered by problem names

    Parameters
    ----------
    problem_names : list
        a list of strings of problem names, default to None

    Examples
    --------
    >>> import espresso
    >>> capabilities = espresso.list_capabilities(['SimpleRegression'])
    """
    all_capabilities = dict()
    for problem, report in _capability_matrix.items():
        all_capabilities[problem] = [k for k, v in report.items() if v == 1]
    if problem_names is None:
        return all_capabilities
    else:
        problem_names = (
            [problem_names] if isinstance(problem_names, str) else problem_names
        )
        filtered_capabilities = {
            k: v for k, v in all_capabilities.items() if k in problem_names
        }
        return filtered_capabilities


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

from ._receiver_function_inversion_knt import ReceiverFunctionInversionKnt
from ._slug_test import SlugTest
from ._fmm_tomography import FmmTomography
from ._gravity_inversion import GravityInversion
from ._receiver_function_inversion_shibutani import ReceiverFunctionInversionShibutani
from ._magnetotelluric_1D import Magnetotelluric1D
from ._pumping_test import PumpingTest
from ._xray_tomography import XrayTomography
from ._surface_wave_tomography import SurfaceWaveTomography
from ._simple_regression import SimpleRegression

_all_problems = [
    ReceiverFunctionInversionKnt,
    SlugTest,
    FmmTomography,
    GravityInversion,
    ReceiverFunctionInversionShibutani,
    Magnetotelluric1D,
    PumpingTest,
    XrayTomography,
    SurfaceWaveTomography,
    SimpleRegression,
]
del ReceiverFunctionInversionKnt
del SlugTest
del FmmTomography
del GravityInversion
del ReceiverFunctionInversionShibutani
del Magnetotelluric1D
del PumpingTest
del XrayTomography
del SurfaceWaveTomography
del SimpleRegression


_capability_matrix = {
    "ReceiverFunctionInversionKnt": {
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
        "plot_data": 1,
        "misfit": 0,
        "log_likelihood": 0,
        "log_prior": 0,
        "list_capabilities": 1
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
        "plot_data": 1,
        "misfit": 0,
        "log_likelihood": 1,
        "log_prior": 0,
        "list_capabilities": 1
    },
    "FmmTomography": {
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
        "plot_data": 0,
        "misfit": 0,
        "log_likelihood": 0,
        "log_prior": 0,
        "tmp_files": 1,
        "call_wavefront_tracker": 1,
        "list_capabilities": 1
    },
    "GravityInversion": {
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
        "log_prior": 0,
        "list_capabilities": 1
    },
    "ReceiverFunctionInversionShibutani": {
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
        "plot_data": 1,
        "misfit": 1,
        "log_likelihood": 1,
        "log_prior": 1,
        "list_capabilities": 1
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
        "set_start_mesh": 1,
        "set_obs_data": 1,
        "list_capabilities": 1
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
        "plot_data": 1,
        "misfit": 0,
        "log_likelihood": 1,
        "log_prior": 0,
        "list_capabilities": 1
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
        "log_prior": 0,
        "list_capabilities": 1
    },
    "SurfaceWaveTomography": {
        "model_size": 1,
        "data_size": 1,
        "good_model": 1,
        "starting_model": 1,
        "data": 1,
        "forward": 1,
        "description": 1,
        "covariance_matrix": 0,
        "inverse_covariance_matrix": 0,
        "jacobian": 1,
        "plot_model": 1,
        "plot_data": 0,
        "misfit": 1,
        "log_likelihood": 0,
        "log_prior": 0,
        "parameterization": 1,
        "example_dict": 1,
        "list_capabilities": 1
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
        "log_prior": 0,
        "list_capabilities": 1
    }
}