from .fmm_tomography import FmmTomography
from .gravity_inversion import GravityInversion
from .magnetotelluric_1D import Magnetotelluric1D
from .pumping_test import PumpingTest
from .receiver_function_inversion_knt import ReceiverFunctionInversionKnt
from .receiver_function_inversion_shibutani import ReceiverFunctionInversionShibutani
from .simple_regression import SimpleRegression
from .slug_test import SlugTest
from .surface_wave_tomography import SurfaceWaveTomography
from .xray_tomography import XrayTomography

_all_problems = [
    FmmTomography,
    GravityInversion,
    Magnetotelluric1D,
    PumpingTest,
    ReceiverFunctionInversionKnt,
    ReceiverFunctionInversionShibutani,
    SimpleRegression,
    SlugTest,
    SurfaceWaveTomography,
    XrayTomography,
]

__all__ = [p.__name__ for p in _all_problems]