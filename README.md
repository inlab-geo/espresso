# Inversion Test Problems


Inversion Test Problems is a collection of clearly defined, stand-alone forward codes of a variety of different geophysical problems... This suite of forward problems enables users .... and has an easily understandable, consistent code structure that is capable of hosting a wide range of typical geophysical research problems. Information will be conveyed via a sensible, consistent naming convention and an informative output that can consist of text and/or visualisations.


# Installation

It is recommended to use a clean virtual environment for the install:

```console
conda create -n ITP_env python=3.8 scipy jupyterlab numpy matplotlib
conda activate ITP_env
```

`Inversion Test Problems` is available on PyIP and can be installed using this command:

Linux/MacOS
```console
python3 -m pip install --index-url https://test.pypi.org/simple/ inversiontestproblems-h-hollmann
```

Windows:
```console
py -m pip install --index-url https://test.pypi.org/simple/ inversiontestproblems-h-hollmann
```

# Basic usage

Once installed, each test problem can be imported using the following command:

```console
from inversiontestproblems import testproblem
```

Replace ``testproblem`` with one of the following currently available problems:

- ``gravityforward``:

Once a problem is imported, it's main functions can be called using the same structure for each problem. For more information for each function please use help():

```console

from inversiontestproblems import testproblem

tp = testproblem()

model=tp.get_model()

data=tp.get_data()

synthetic_data = tp.forward(model)

jacobian=tp.gradient(model)

tp.plot_model()

```

Other problem-specific values can be accessed through the 'testproblem' object, such as:

```console

tp.rec_coords # coordinates of recording locations

tp.x_nodes # x-coordinates of all nodes of the model

```

The set values are set is highly problem-specific and we suggest to use 'help(tp)' or 'dir(tp)' to quickly see what is available. Alternatively, consult the Inversion Test Suite Documentation (later).
