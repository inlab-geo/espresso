# CoFI examples

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/inlab-geo/cofi-examples.git/HEAD)

[CoFI](https://github.com/inlab-geo/cofi) (Common Framework for Inference) is an open-source 
initiative for interfacing between generic inference algorithms and specific geoscience problems.

This repository contains examples for running inversion algorithms using CoFI.

## Getting started

### Step 1. Get `cofi`

(Strongly recommended) Create a virtual environment to avoid conflicts with your other projects:

```console
$ conda env create -f environment.yml
$ conda activate cofi_env
```

Otherwise (if you've followed above then skip this), ensure you have `scipy` in your environment and then install `cofi` with:

```console
$ pip install cofi
```

### Step 2. Get the examples

Clone this repository:

```console
$ git clone https://github.com/inlab-geo/cofi-examples.git
```

### Step 3. Run it!

Open up Jupyter-lab:

```console
$ cd cofi-examples
$ jupyter-lab
```

Run through examples and have fun!

## Contribution

Read `CoFI`'s documentation "Advanced Usage" section for how you can add your own forward examples.

To report bugs or typos, please head to either [GitHub issues](https://github.com/inlab-geo/cofi-examples/issues) 
or our [Slack workspace](https://inlab-geo.slack.com/).

## Useful resources
- InLab [website](http://www.inlab.edu.au/)
- CoFI [documentation](https://cofi.readthedocs.io/en/latest/index.html) (under construction)
- CoFI [GitHub repository](https://github.com/inlab-geo/cofi) (under construction)

## Troubleshooting for interactive lab
If you've followed the [getting started section](README.md#getting-started) above, and are still 
having trouble displaying the ipython widgets, then hopefully 
[this StackOverflow thread](https://stackoverflow.com/questions/36351109/ipython-notebook-ipywidgets-does-not-show) 
will help you. 
