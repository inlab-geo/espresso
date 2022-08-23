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
    return _absolute_path(relative_path)

def loadtxt(relative_path, *args, **kwargs):
    abs_path = _absolute_path(relative_path)
    return np.loadtxt(abs_path, *args, **kwargs)
