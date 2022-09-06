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

TODO:
√ Add content into this docstring
√ Change list_problem_names & list_problems from lambda to named functions
√ Add docstring for 2 functions above
√ Add docstring for EspressoProblem class
√ Check formatting of the docstrings for all in espresso_problem.py
√ Add docstring for EspressoError class
√ Check formatting of the docstrings for all in exceptions.py
8. Add content into utils docstring
9. Add docstrings for utils functions

Check example here: https://github.com/inlab-geo/cofi/blob/main/src/cofi/base_problem.py
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
