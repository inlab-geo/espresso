# CoFI Test Suite


CoFi TestSuite is a collection of clearly defined, stand-alone forward and inversion problems that each can be solved using the same code structure. The aim is to create a minimalistic framework that is easily understandable while still being capable of hosting a wide range of inversion problems. Information will be conveyed via a sensible, consistent naming convention and informative output at the end consisting of text and visualisations. Extended documentation will be available on Github. 


# Installation

It is recommended to use a clean virtual environment for the install: 

```console
conda create -n cofi_testsuite_env python=3.8 scipy jupyterlab numpy
conda activate cofi_testsuite_env
```

CoFi Test Suite is available on the PyIP test server and can be installed using this command:

```console
python3 -m pip install --index-url https://test.pypi.org/simple/ cofitestsuite-h-hollmann
```

# Basic usage

Once installed, each CoFI test problem can be imported using the following command structure:

```console
from cofitestsuite.testproblem import testproblem as tp
```

Replace ``testproblem`` with one of the following currently available problems:

- ``gravityforward``: 
- ``xraytomorgraphy``: 
- ``earthquakebootstrap``: 
- ``earthquakeleastsquares``: 
  

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/inlab-geo/inversion-test-problems/HEAD)
