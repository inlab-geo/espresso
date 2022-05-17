# CoFI examples

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/inlab-geo/cofi-examples/blob/main/index.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/inlab-geo/cofi-examples/main?filepath=index.ipynb)
[![CoFI docs](https://img.shields.io/badge/CoFI%20docs-Example%20Gallery-B5E2FA)](https://cofi.readthedocs.io/en/latest/cofi-examples/generated/index.html)

[CoFI](https://github.com/inlab-geo/cofi) (Common Framework for Inference) is an open-source 
initiative for interfacing between generic inference algorithms and specific geoscience problems.

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
$ conda env create -f environment.yml
$ conda activate cofi_env
```

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

## Useful resources
- InLab [website](http://www.inlab.edu.au/)
- CoFI [documentation](https://cofi.readthedocs.io/en/latest/index.html) (under construction)
- CoFI [GitHub repository](https://github.com/inlab-geo/cofi) (under construction)

## Troubleshooting for interactive lab
If you've followed the [instructions on running locally](README.md#run-the-examples-with-cofi-locally)
above, and are still having trouble ***displaying the ipython widgets***, then hopefully 
[this StackOverflow thread](https://stackoverflow.com/questions/36351109/ipython-notebook-ipywidgets-does-not-show) 
will help you. 
