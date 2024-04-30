"""Module initialisation file

It's NOT recommended to put extra code inside this file. Code inside this file will be
executed when this submodule is imported, so adding things in this file can slow down
the importing process of `espresso`.

For contributors, add any intialisation code for your problem into _magnetotelluric_1D.py, 
under the method `__init__()` of the class `Magnetotelluric1D`.

Don't touch this file unless you know what you are doing :)
"""

from ._magnetotelluric_1D import *

__all__ = [ "Magnetotelluric1D" ]
