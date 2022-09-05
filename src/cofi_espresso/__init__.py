from .espresso_problem import EspressoProblem
from .exceptions import EspressoError, InvalidExampleError


try:
    from . import _version

    __version__ = _version.__version__
except ImportError:
    pass


__all__ = [
    "EspressoProblem",
    "EspressoError",
    "InvalidExampleError",
]

# from .example_name import ExampleName
# __all__.append("ExampleName")

from .xray_tomography import XrayTomography
from .simple_regression import SimpleRegression
from .gravity_density import GravityDensity

__additional_all__ = [
	'XrayTomography',
	'SimpleRegression',
	'GravityDensity',
]
__all__ += __additional_all__