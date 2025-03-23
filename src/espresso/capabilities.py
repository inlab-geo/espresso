"""
=======================
Understand capabilities
=======================

Submodule for listing capabilities in Espresso problems.

To programatically see what problems are available in Espresso, these functions are 
helpful:

"""

import typing
from .contrib import _all_problems

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
