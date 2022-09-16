# Espresso

[![PyPI version](https://img.shields.io/pypi/v/cofi-espresso?logo=pypi&style=flat-square&color=bde0fe)](https://pypi.org/project/cofi-espresso/)
[![build](https://img.shields.io/github/workflow/status/inlab-geo/espresso/Build?logo=githubactions&style=flat-square&color=ccd5ae)](https://github.com/inlab-geo/espresso/actions/workflows/build_wheels.yml)
[![Documentation Status](https://img.shields.io/readthedocs/cofi-espresso?logo=readthedocs&style=flat-square&color=faedcd)](https://cofi-espresso.readthedocs.io/en/latest/?badge=latest)
[![Slack](https://img.shields.io/badge/Slack-inlab-4A154B?logo=slack&style=flat-square&color=cdb4db)](https://join.slack.com/t/inlab-community/shared_invite/zt-1ejny069z-v5ZyvP2tDjBR42OAu~TkHg)

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

- `GravityDensity`
- `SimpleRegression`
- `XrayTomography`

Once a problem is imported, its main functions can be called using the same 
structure for each problem. For instance:

```python
from cofi_espresso import GravityDensity

problem = GravityDensity(example_number=1)
model = problem.good_model()
data = problem.data()
pred = problem.forward(model)
fig_model = problem.plot_model(model)
fig_data = problem.plot_data(data, pred)
```

You can access related metadata programatically:

```python
print(GravityDensity.problem_title)
print(GravityDensity.problem_short_description)
print(GravityDensity.author_names)
```

Other problem-specific parameters can be accessed through the problem instance. For instance:

```python
print(problem.m)
print(problem.rec_coords)
```

Which additional values are set is highly probl-specific and we suggest to 
consult the 
[Espresso Documentation on the problems](https://cofi-espresso.readthedocs.io/en/latest/user_guide/contrib/index.html).


## Contributing

Interested in contributing? Please check out our [contributor's guide](https://cofi-espresso.readthedocs.io/en/latest/contributor_guide/ways.html).


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
