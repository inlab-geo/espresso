r"""Submodule for utility functions in cofi_espresso.

Some of these functions are used by contributors to conveniently implement their
problem class. Others can be utilised by users to perform analysis on inversion
results.

.. important::

    Please always use :func:`absolute_path` or :func:`loadtxt` when trying to load data
    from a relative path into an Espresso problem class.

"""

from .data_loader import loadtxt, absolute_path
from .file_handler import silent_remove

__all__ = [
    "loadtxt",
    "absolute_path",
    "silent_remove",
]
