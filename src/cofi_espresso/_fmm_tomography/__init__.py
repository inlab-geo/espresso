"""Module initialisation file

It's NOT recommended to put extra code inside this file. Code inside this file will be
executed when this submodule is imported, so adding things in this file can slow down
the importing process of `cofi_espresso`.

For contributors, add any intialisation code for your problem into _fmm_tomography.py, 
under the method `__init__()` of the class `FmmTomography`.

Don't touch this file unless you know what you are doing :)
"""

from ._fmm_tomography import *

__all__ = [ "FmmTomography" ]
