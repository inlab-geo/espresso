"""Module initialisation file

Code inside this file will be executed when this submodule is imported.

For contributors: feel free to add your initialisation code here if needed.
"""

from .gravity_density import *

__all__ = [
    "set_example_number",
    "suggested_model",
    "data",
    "forward",
    "jacobian",
    "plot_model",
    "plot_data",
]
