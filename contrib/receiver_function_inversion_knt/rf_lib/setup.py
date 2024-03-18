from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

ext = Extension(
    "rf", 
    sources=[
        "rf.pyx", 
        "src/Complex.c", 
        "src/matrix.c", 
        "src/knt_mini.c", 
        "src/fft.c", 
    ],
    include_dirs=[numpy.get_include()]
)

setup(
    name="rf",
    ext_modules=cythonize([ext])
)
