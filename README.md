# Espresso

**E**arth **S**cience **PR**oblems for the **E**valuation of **S**trategies, **S**olvers and **O**ptimizers (Espresso) is a collection of clearly defined forward codes that simulate a wide range of geophysical processes. The goal of Espresso is to bring together people developing physical simulations with those who need them. Espresso's simple and consistent code structure enables users to acces a wide range of different forward code and contributers to share their solutions with a wider audience. For more information, please visit our documentation (coming soon).


# Installation

It is recommended to use a clean virtual environment for the install:

```console
conda create -n esp_env scipy jupyterlab numpy matplotlib
conda activate esp_env
```

`Earth Science PRoblems for the Evaluation of Strategies, Solvers and Optimizers` is available on PyPI and can be installed using this command:

Linux/MacOS
```console
python3 -m pip install espresso
```

Windows:
```console
py -m pip install espresso
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

from espresso import testproblem

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

Which additional values are set is highly example-specific and we suggest to use 'help(tp)' or 'dir(tp)' to quickly see what is available, or consult the Inversion Test Suite Documentation (coming soon).
