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

from ._simple_regression import SimpleRegression
from ._slug_test import SlugTest
from ._fmm_tomography import FmmTomography
from ._xray_tomography import XrayTomography
from ._gravity_density import GravityDensity
from ._pumping_test import PumpingTest

_all_problem_names = [
	'SimpleRegression',
	'SlugTest',
	'FmmTomography',
	'XrayTomography',
	'GravityDensity',
	'PumpingTest',
]
_all_problems = [
	SimpleRegression,
	SlugTest,
	FmmTomography,
	XrayTomography,
	GravityDensity,
	PumpingTest,
]