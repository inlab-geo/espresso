# Example Name Title

<!-- Please write anything you'd like to explain about the forward problem here -->

Welcome to your new Espresso example!

## Pre-requisites

Make sure you have Python>=3.6 and [required packages](../../envs/environment_contrib.yml) 
installed in your system. 

<details>

[`mamba`](https://mamba.readthedocs.io/en/latest/) is recommended, and we provide
instructions that work for both `conda` and `mamba` below. Check contributor's guide in 
[geo-espresso documentation](https://geo-espresso.readthedocs.io/en/latest/index.html) 
for other options.

1. Install required Python packages for contributing to `geo-espresso`. Run the following
   commands with the project root level as working directory:
   ```console
   $ conda env create -f envs/environment_contrib.yml
   $ conda activate esp_contrib
   ```
2. Install `geo-espresso` core package
   ```console
   $ pip install .
   ```

</details>

## Checklist

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
   - If you would like to load data from files, please use our 
     [utility functions](https://geo-espresso.readthedocs.io/en/latest/user_guide/api/generated/espresso.utils.html) 
     to get absoluate path before calling your load function.
- [ ] **Validate and build your contribution locally**. We have seperate scripts for 
   validation and packaging. Check 
   [how to test building your contribution](README.md#how-to-test-building-your-contribution-with-geo-espresso) 
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

## How to test building your contribution with `geo-espresso`

The **recommended way** is this:

```console
$ python espresso_machine/build_package/build.py --validate
```

Read on if you are looking for further details:

1. To **validate your contribution** when almost finished, run the following (replacing `<example_name>` with your problem name, e.g. `simple_regression`)

   ```console
   $ python espresso_machine/build_package/validate.py -c <example_name1>
   ```

   Or the following for more than one contributions (replacing `<example_name_1>` and `<example_name_2>` with your problem names)

   ```console
   $ python espresso_machine/build_package/validate.py -c <example_name_1> -c <example_name_2>
   ```

   Or the following for all existing contributions

   ```console
   $ python espresso_machine/build_package/validate.py --all
   ```

2. To **build your contribution into geo-espresso**, run

   ```console
   $ python espresso_machine/build_package/build.py
   ```

3. To **validate your built contribution** after running the build script above, run the following ()

   ```console
   $ python espresso_machine/build_package/validate.py --post -c <example_name1>
   ```

   Or the following for more than one contributions (replacing `<example_name_1>` and `<example_name_2>` with your problem names)

   ```console
   $ python espresso_machine/build_package/validate.py --post -c <example_name_1> -c <example_name_2>
   ```

   Or the following for all existing contributions

   ```console
   $ python espresso_machine/build_package/validate.py --post --all
   ```

4. To do **pre-build validation**, **build**, **post-build validation** (1-3 above) all together at once,
run

   ```console
   $ python espresso_machine/build_package/build.py --validate
   ```
