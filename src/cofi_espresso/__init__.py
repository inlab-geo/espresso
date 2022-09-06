r"""

TODO:
1. Add content into this docstring
2. Change list_problem_names & list_problems from lambda to named functions
3. Add docstring for 2 functions above
4. Add docstring for EspressoProblem class
5. Check formatting of the docstrings for all in espresso_problem.py
6. Add docstring for EspressoError class
7. Check formatting of the docstrings for all in exceptions.py
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
