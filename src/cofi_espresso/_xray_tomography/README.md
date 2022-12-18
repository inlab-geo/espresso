# Xray Tomography

<!-- Please write anything you'd like to explain about the forward problem here -->

Welcome to your new Espresso example!

## Pre-requisites

Make sure you have Python>=3.6 installed in your system. 

[`mamba`](https://mamba.readthedocs.io/en/latest/) is recommended, and we provide
instructions that work for both `conda` and `mamba` below. Check contributor's guide in 
[cofi-espresso documentation](https://cofi-espresso.readthedocs.io/en/latest/index.html) 
for other options.

1. Install required Python packages for contributing to `cofi-espresso`. Run the following
   commands with the project root level as working directory:
   ```console
   $ conda env create -f envs/environment_contrib.yml
   $ conda activate esp_contrib
   ```
2. Install `cofi-espresso` base package
   ```console
   $ pip install .
   ```

## Getting started

To complete this contribution, here are some ideas on what to do next:

- [ ] **Modify [README.md](README.md)**. Document anything you'd like to add for this problem
  (in this README.md file). Some recommended parts include:
   - What this test problem is about
   - What you would recommend inversion practitioners to notice
   - etc.
- [ ] **Modify [LICENCE](LICENCE)**. The default one we've used is a 2-clauss BSD licence. 
   Feel free to replace the content with a licence that suits you best.
- [ ] **Write code in [xray_tomography.py](xray_tomography.py) (and [__init__.py](__init__.py) if
   necessary)**. Some basic functions have been defined in the template - these are the
   standard interface we'd like to enforce in Espresso. You'll see
   clearly some functionalities that are required to implement and others that are
   optional.
- [ ] **Validate and build your contribution locally**. We have seperate scripts for 
   validation and packaging.
   ```console
   $ python tools/build_package/validate.py         # to validate your contribution
   $ python tools/build_package/build.py            # to install updated Espresso in your environment
   $ python tools/build_package/validate_build.py   # to run both of above together
   ```
- [ ] **Delete / comment out these initial instructions**. They are for your own reference
   so feel free to delete them or comment them out once you've finished the above
   checklist.


## How to test your code

> **Note that you cannot test your code directly inside your example subfolder**, if you
> have any relative import inside the contribution file. Check the following for details.

***In order to test your code in that case***, use `contrib` as your working directory and 
import your contribution in the following ways.

(Python interactive mode)
```python
$ pwd                            # check you are in the right folder
<path-to-espresso>/contrib
$ python
>>> from xray_tomography import ExampleName   # import it this way
```

(Creating temporary Python file)
```python
# file contrib/tmp.py            # create tmp file in the right folder
from xray_tomography import ExampleName       # import it this way
```
