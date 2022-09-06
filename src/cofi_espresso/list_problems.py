_all_problem_names = []
_all_problems = []

def list_problem_names():
    """Returns a list of all Espresso problem names"""
    return _all_problem_names

def list_problems():
    """Returns a list of all Espresso problem classes"""
    return _all_problems


# from .example_name import ExampleName

# __all_problem_names__ = [ "ExampleName", ]
# __all_problems__ = [ ExampleName, ]

from .xray_tomography import XrayTomography
from .simple_regression import SimpleRegression
from .gravity_density import GravityDensity

_all_problem_names = [
	'XrayTomography',
	'SimpleRegression',
	'GravityDensity',
]
_all_problems = [
	XrayTomography,
	SimpleRegression,
	GravityDensity,
]