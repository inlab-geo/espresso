# Espresso

[![PyPI version](https://img.shields.io/pypi/v/geo-espresso?logo=pypi&style=flat-square&color=bde0fe&labelColor=f8f9fa)](https://pypi.org/project/geo-espresso/)
[![build](https://img.shields.io/github/actions/workflow/status/inlab-geo/espresso/build_wheels.yml?branch=main&logo=githubactions&style=flat-square&color=ccd5ae&labelColor=f8f9fa)](https://github.com/inlab-geo/espresso/actions/workflows/build_wheels.yml)
[![Documentation Status](https://img.shields.io/readthedocs/geo-espresso?logo=readthedocs&style=flat-square&color=fed9b7&labelColor=f8f9fa&logoColor=eaac8b)](https://geo-espresso.readthedocs.io/en/latest/?badge=latest)
[![Slack](https://img.shields.io/badge/Slack-InLab_community-4A154B?logo=slack&style=flat-square&color=cdb4db&labelColor=f8f9fa&logoColor=9c89b8)](https://join.slack.com/t/inlab-community/shared_invite/zt-1ejny069z-v5ZyvP2tDjBR42OAu~TkHg)

> Related repositories by [InLab](https://inlab.edu.au/community/)
> - [CoFI](https://github.com/inlab-geo/cofi)
> - [CoFI Examples](https://github.com/inlab-geo/cofi-examples)

## Introduction

**E**arth **S**cience **PR**oblems for the **E**valuation of **S**trategies, 
**S**olvers and **O**ptimizers (Espresso) is a collection of datasets, and 
associated simulation codes, spanning a wide range of geoscience problems. 
Together they form a suite of real-world test problems that can be used to 
support the development, evaluation and benchmarking of a wide range of tools
and algorithms for inference, inversion and optimisation. All problems are 
designed to share a common interface, so that changing from one test problem
to another requires changing one line of code. 

The Espresso project is a community effort - if you think it sounds useful,
please consider contributing an example or two from your own research. The project
is currently being coordinated by InLab, with support from the CoFI development
team.

For more information, please visit [our documentation](geo-espresso.readthedocs.io).


## Installation

```console
$ pip install geo-espresso
```

Check Espresso documentation - 
[installation page](https://geo-espresso.readthedocs.io/en/latest/user_guide/installation.html) 
for details on dependencies and setting up with virtual environments.

## Basic usage

Once installed, each test problem can be imported using the following command:

```python
from espresso import <testproblem>
```

Replace ``<testproblem>`` with an actual problem class in Espresso, such as
`SimpleRegression` and `FmmTomography`. See 
[here](https://geo-espresso.readthedocs.io/en/latest/user_guide/contrib/index.html) 
for a full list of problems Espresso currently includes.

Once a problem is imported, its main functions can be called using the same 
structure for each problem. For instance:

```python
from espresso import FmmTomography

problem = FmmTomography(example_number=1)
model = problem.good_model
data = problem.data
pred = problem.forward(model)
fig_model = problem.plot_model(model)
```

You can access related metadata programatically:

```python
print(FmmTomography.metadata["problem_title"])
print(FmmTomography.metadata["problem_short_description"])
print(FmmTomography.metadata["author_names"])
```

Other problem-specific parameters can be accessed through the problem instance. For instance:

```python
print(problem.extent)
print(problem.model_shape)
```

Which additional values are set is highly problem-specific and we suggest to 
consult the 
[Espresso Documentation on the problems](https://geo-espresso.readthedocs.io/en/latest/user_guide/contrib/index.html).


## Contributing

Interested in contributing? Please check out our [contributor's guide](https://geo-espresso.readthedocs.io/en/latest/contributor_guide/index.html).


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
