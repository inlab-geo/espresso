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

# __all_problem_names__ = [ "ExampleName", ]
# __all__ += __all_problem_names__
# list_problem_names = lambda: __all_problem_names__

# __all_problems__ = [ ExampleName, ]
# list_problems = lambda: __all_problems__
