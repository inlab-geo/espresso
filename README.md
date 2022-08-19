# Espresso

[![PyPI version](https://img.shields.io/pypi/v/cofi-espresso?logo=pypi&style=flat-square&color=bde0fe)](https://pypi.org/project/cofi-espresso/)
[![Documentation Status](https://img.shields.io/readthedocs/cofi-espresso?logo=readthedocs&style=flat-square&color=faedcd)](https://cofi-espresso.readthedocs.io/en/latest/?badge=latest)
[![Slack](https://img.shields.io/badge/Slack-inlab-4A154B?logo=slack&style=flat-square&color=cdb4db)](https://inlab-geo.slack.com)

## Introduction

**E**arth **S**cience **PR**oblems for the **E**valuation of **S**trategies, 
**S**olvers and **O**ptimizers (Espresso) is a collection of clearly defined forward 
codes that simulate a wide range of geophysical processes. The goal of Espresso is to 
bring together people developing physical simulations with those who need them. 
Espresso's simple and consistent code structure enables users to access a wide range 
of different forward code and contributers to share their solutions with a wider 
audience. For more information, please visit our documentation (coming soon).


## Installation

```console
$ pip install cofi-espresso
```

Check Espresso documentation - 
[installation page](https://cofi-espresso.readthedocs.io/en/latest/installation.html) 
for details on dependencies and setting up with virtual environments.

## Basic usage

Once installed, each test problem can be imported using the following command:

```python
from cofi_espresso import <testproblem>
```

Replace ``<testproblem>`` with one of the following currently available problems:

- ``gravity_density``

Once a problem is imported, its main functions can be called using the same 
structure for each problem. For instance:

```python
from cofi_espresso import GravityDensity

grav = GravityDensity(example_number=1)
grav_model = grav.suggested_model()
grav_data = grav.data()
grav_synthetics = grav.forward(grav_model)
grav_jacobian = grav.jacobian(grav_model)
grav.plot_model(grav_model)
grav.plot_data(grav_data)
```

Other problem-specific parameters can be accessed through the problem instance. For instance:

```python
print(grav.params.keys())
# dict_keys(['m', 'rec_coords', 'x_nodes', 'y_nodes', 'z_nodes', 'lmx', 'lmy', 'lmz', 'lrx', 'lry'])
print(grav.m)
print(grav.rec_coords)
```

Which additional values are set is highly example-specific and we suggest to 
consult the [Espresso Documentation](https://cofi-espresso.readthedocs.io).


## Contributing

Interested in contributing? Please check out our [contributor's guide](https://cofi-espresso.readthedocs.io/en/latest/contribute.html).


## Licence

Espresso is a community driven project to create a large suite of forward
simulations to enable researchers to get example data without the need to 
understand each individual problem in detail.

Licensing is done individually by each contributor. If a contributor wants to freely share their code example we recommend the MIT licence or a 
2-clause BSD licence. To determine the licence of an existing Espresso
problem, please consult the documentation section of that problem.

All the other core functions of Espresso written by InLab Espresso developer
team are distributed under a 2-clause BSD licence. A copy of this licence is
provided with distributions of the software.
