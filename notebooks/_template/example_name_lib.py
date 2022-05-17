"""Template library file(s)

Please put all the library code that will be needed in the notebook in this file.

The file name should end with "_lib.py", otherwise our bot may fail when generating
scripts for Sphinx Gallery. Furthermore, we recommend the file name to start with your
forward problem name, to align well with the naming of Jupyter notebook.

"""

from numbers import Number
import numpy as np

def forward(model: np.ndarray) -> Number:
    raise NotImplementedError
