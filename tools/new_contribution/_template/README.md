# Example Name Title

<!-- Please write anything you'd like to explain about the forward problem here -->

Welcome to your new Espresso example!

## Pre-requisites

Make sure you have Python>=3.6 and [required packages](../../envs/environment_contrib.yml) 
installed in your system. 

<details>

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

</details>

## Getting started

To complete this contribution, here are some ideas on what to do next:

- [ ] **Modify [README.md](README.md)**. Document anything you'd like to add for this problem
  (in this README.md file). Some recommended parts include:
   - What this test problem is about
   - What you would recommend inversion practitioners to notice
   - etc.
- [ ] **Modify [LICENCE](LICENCE)**. The default one we've used is a 2-clauss BSD licence. 
   Feel free to replace the content with a licence that suits you best.
- [ ] **Write code in [example_name.py](example_name.py) (and [\_\_init\_\_.py](__init__.py) if
   necessary)**. Some basic functions have been defined in the template - these are the
   standard interface we'd like to enforce in Espresso. You'll see
   clearly some functionalities that are required to implement and others that are
   optional.
- [ ] **Validate and build your contribution locally**. We have seperate scripts for 
   validation and packaging. Check 
   [how to test building your contribution](README.md#how-to-test-building-your-contribution-with-cofi-espresso) 
   for details.
- [ ] **Delete / comment out these initial instructions**. They are for your own reference
   so feel free to delete them or comment them out once you've finished the above
   checklist.


## How to unit test your code

When developing your contribution, it's sometimes useful for yourself to try running
your code. Normally, creating a temporary test file or running Python interactively
within the contribution subfolder is sufficient. 

<details>
   <summary>Some instructions when you have relative import in the 
files</summary>

> **Note that you cannot test your code directly inside your example subfolder**, if you
> have any relative import (e.g. `from .lib import *`) inside the contribution file. 
> Check the following for details.

***In order to test your code in that case***, use `contrib` as your working directory and 
import your contribution in the following ways.

(Python interactive mode)
```python
$ pwd                            # check you are in the right folder
<path-to-espresso>/contrib
$ python
>>> from example_name import ExampleName   # import it this way
```

(Creating temporary Python file)
```python
# file contrib/tmp.py            # create tmp file in the right folder
from example_name import ExampleName       # import it this way
```

</details>

## How to test building your contribution with `cofi-espresso`

1. To **validate your contribution** when almost finished, run

   ```console
   $ python tools/build_package/validate.py
   ```

2. To **build your contribution into cofi-espresso**, run

   ```console
   $ python tools/build_package/build.py
   ```

3. To **validate your built contribution** after running the build script above, run

   ```console
   $ python tools/build_package/validate.py post
   ```

4. To do **pre-build validation**, **build**, **post-build validation** (1-3 above) all together at once,
run

   ```console
   $ python tools/build_package/build_with_checks.py
   ```
