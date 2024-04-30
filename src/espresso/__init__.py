r"""

EspressoProblem
---------------

The essential idea of Espresso is to have exactly same interface for different 
inversion test problems, so we list all the standard methods and attributes once in 
this API reference page.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    EspressoProblem


Utility functions
-----------------

Some utility functions are there to help contributors load data and calculate things.

.. autosummary::
    :toctree: generated/
    :nosignatures:
    
    espresso.utils


List the problems & capabilities
--------------------------------

To programatically see what problems are available in Espresso, these functions are 
helpful:

.. autosummary::
    :toctree: generated/
    :nosignatures:
    
    espresso.capabilities

Espresso exceptions
-------------------

.. autosummary::
    :toctree: generated/
    :nosignatures:
    
    espresso.exceptions

"""

from ._espresso_problem import EspressoProblem

from . import utils
from . import exceptions


from ._version import __version__


__all__ = [
    "EspressoProblem",
]

# from .example_name import ExampleName

# from .capabilities import list_problem_names, list_problems
# __all__ += list_problem_names()
# __all__ += ["list_problem_names", "list_problems", "list_capabilities"]

from ._receiver_function_inversion_shibutani import ReceiverFunctionInversionShibutani

from .capabilities import list_problem_names, list_problems, list_capabilities

__all__ += list_problem_names()
__all__ += ['list_problem_names', 'list_problems', 'list_capabilities']
