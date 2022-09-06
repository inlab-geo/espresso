from pathlib import Path
import inspect
import numpy as np


def _absolute_path(relative_path):
    # get caller file path
    caller_frame = inspect.stack()[2]
    caller_filename_full = caller_frame.filename
    # combine path
    file_path = (Path(caller_filename_full).parent / relative_path).resolve()
    return file_path

def absolute_path(relative_path):
    r"""Returns the absolute path of a data file

    Please always use this function or :func:`loadtxt` when trying to load data from 
    a relative path into an Espresso problem class.

    Parameters
    ----------
    relative_path : str
        The relative path of the file you want the absoluate path for

    Examples
    --------

    >>> from cofi_espresso.utils import absolute_path
    >>> from PIL import Image
    >>> png = Image.open(absolute_path("data/csiro_logo.png"))

    """
    return _absolute_path(relative_path)

def loadtxt(relative_path, *args, **kwargs):
    r"""Wrapper of :func:`numpy.loadtxt` given a relative path

    Please always use this function or :func:`absolute_path` when trying to load data 
    from a relative path into an Espresso problem class.

    Parameters
    ----------
    relative_path : str
        The relative path of the file you want to load
    *args : any
        Passed directly into :func:`numpy.loadtxt`
    **kwargs : any
        Passed directly into :func:`numpy.loadtxt`

    Examples
    --------

    >>> from cofi_espresso.utils import loadtxt
    >>> data = loadtxt("data/example1.dat")
    >>> data
    array([[0.    , 0.0323, 5.9987, 1.    , 0.0323, 2.057 ],
        [0.    , 0.0323, 6.3879, 1.    , 0.0645, 2.1543],
        [0.    , 0.0323, 1.8923, 1.    , 0.0968, 0.6279],
        ...,
        [1.    , 0.    , 4.1087, 0.    , 1.    , 0.5583],
        [1.    , 0.    , 3.9118, 1.    , 0.    , 3.9118],
        [1.    , 0.    , 9.2198, 1.    , 1.    , 3.2873]])

    """
    abs_path = _absolute_path(relative_path)
    return np.loadtxt(abs_path, *args, **kwargs)
