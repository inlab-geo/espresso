# <img src="https://raw.githubusercontent.com/inlab-geo/cofi/main/docs/source/_static/latte_art_cropped.png" width="5%" style="vertical-align:bottom"/> CoFI Examples

[![Open In Colab](https://img.shields.io/badge/open%20in-Colab-b5e2fa?logo=googlecolab&style=flat-square&color=ffd670&labelColor=f8f9fa)](https://colab.research.google.com/github/inlab-geo/cofi-examples/blob/main/index.ipynb)
[![badge](https://img.shields.io/badge/launch-Binder-E66581.svg?logo=data:image/png;base64,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&style=flat-square&labelColor=f8f9fa&color=ffafcc)](https://mybinder.org/v2/gh/inlab-geo/cofi-examples/main?filepath=index.ipynb)
[![CoFI docs](https://img.shields.io/badge/CoFI%20docs-Example%20Gallery-bbd0ff?style=flat-square&labelColor=f8f9fa&logo=readthedocs&logoColor=bbd0ff)](https://cofi.readthedocs.io/en/latest/examples/generated/index.html)


> Related repositories by [InLab](https://inlab.edu.au/community/):
> - [CoFI](https://github.com/inlab-geo/cofi)
> - [Espresso](https://github.com/inlab-geo/espresso)

CoFI (Common Framework for Inference) is an open-source 
initiative for interfacing between generic inference algorithms and specific geoscience problems.
Read [CoFI's documentation](https://cofi.readthedocs.io/en/latest/) for more details.

This repository contains examples for running inversion algorithms using CoFI with increasing complexity in problems.

## Run the examples

- To run the examples interactively without any local setup, click on the "Colab" 
  (recommended) or "binder" badges above
- To view the examples and sample output (without interaction, no local setup), click 
  on the "Example Gallery"
- To install `cofi` and run the examples locally, follow the instructions below


### Run the examples with `cofi` locally

#### Step 1. Get `cofi`

(Strongly recommended) Create a virtual environment to avoid conflicts with your other projects:

```console
$ conda env create -f envs/environment.yml
$ conda activate cofi_env
```

> For MacOS M1 users, unfortunately `tetgen` (which is a dependency of `pygimli`) doesn't support
> ARM machines yet. Please use `conda env create -f envs/environment_arm.yml` instead, and use
> another X86 machine to run the notebooks that make use of `pygimli`.

Otherwise (skip this if you've followed above), ensure you have `scipy` and `jupyter-lab` in your environment and then install `cofi` with:

```console
$ pip install cofi
```

#### Step 2. Get the examples

Clone this repository:

```console
$ git clone https://github.com/inlab-geo/cofi-examples.git
```

#### Step 3. Run the examples

Open up Jupyter-lab:

```console
$ cd cofi-examples
$ jupyter-lab
```

Run through examples and have fun :tada:! We recommend opening up the `index.ipynb` at root folder to decide where to start.

## Contribution

Thanks for contributing! Please refer to our [Contributor's Guide](CONTRIBUTING.md) for
details.

## Troubleshooting for interactive lab
If you've followed the [instructions on running locally](#run-the-examples-with-cofi-locally)
above, and are still having trouble ***displaying the ipython widgets***, then hopefully 
[this StackOverflow thread](https://stackoverflow.com/questions/36351109/ipython-notebook-ipywidgets-does-not-show) 
will help you. 
