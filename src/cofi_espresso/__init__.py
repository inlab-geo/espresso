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

"""

from ._espresso_problem import EspressoProblem


from ._version import __version__


__all__ = [
    "EspressoProblem",
]

# from .example_name import ExampleName

# from .list_problems import list_problem_names, list_problems
# __all__ += list_problem_names()
# __all__ += ["list_problem_names", "list_problems"]

from ._simple_regression import SimpleRegression
from ._slug_test import SlugTest
from ._fmm_tomography import FmmTomography
from ._xray_tomography import XrayTomography
from ._gravity_density import GravityDensity
from ._pumping_test import PumpingTest

from .list_problems import list_problem_names, list_problems
__all__ += list_problem_names()
__all__ += ['list_problem_names', 'list_problems']
