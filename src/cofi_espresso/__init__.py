r"""

EspressoProblem
---------------

The essential idea of Espresso is to have exactly same interface for different 
inversion test problems, so we list all the standard methods and attributes once in 
this API reference page. Check out documentation for :class:`EspressoProblem` for 
details.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    EspressoProblem

List the problems
-----------------

To programatically see what problems are available in Espresso, these functions are 
helpful:

.. autofunction:: list_problems
.. autofunction:: list_problem_names

Utility functions
-----------------

Some utility functions are there to help contributors load data and calculate things.
Check out documentation for submodule :mod:`cofi_espresso.utils` for details.

.. autosummary::
    :toctree: generated/
    :nosignatures:
    
    cofi_espresso.utils

EspressoError
-------------

Additionally, Espresso's own exception classes:

.. autosummary::
    :toctree: generated/
    :nosignatures:

    EspressoError

"""

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

# from .list_problems import list_problem_names, list_problems
# __all__ += list_problem_names()
# __all__ += ["list_problem_names", "list_problems"]

from .fmm_tomography import FmmTomography
from .xray_tomography import XrayTomography
from .gravity_density import GravityDensity
from .simple_regression import SimpleRegression

from .list_problems import list_problem_names, list_problems
__all__ += list_problem_names()
__all__ += ['list_problem_names', 'list_problems']
