# Gravity Density

<!-- Please write anything you'd like to explain about the forward problem here -->

Welcome to your new Espresso example!

To complete this contribution, here are some ideas on what to do next:

- [ ] **Modify [README.md](README.md)**. Replace the title above with your test problem name,
   and document anything you'd like to add for this problem. Some recommended parts
   include:
   - What this test problem is about
   - What you would recommend inversion practitioners to notice
   - etc.
- [ ] **Modify [metadata.yml](metadata.yml)**. As the name suggests, this file contains basic
   information about the problem itself, authors, citations, example information and
   other extra information you'd like to include. It's a yaml file so that we can 
   render some of the information in a more structured manner.
- [ ] **Modify [LICENCE](LICENCE)**. The default one we've used is a 2-clauss BSD licence. 
   Feel free to replace the content with a licence that suits you best.
- [ ] **Write code in [gravity_density.py](gravity_density.py) (and [__init__.py](__init__.py) if
   necessary)**. Some basic functions have been defined in the template - these are the
   standard interface we'd like to enforce in Espresso. You'll see
   clearly some functionalities that are required to implement and others that are
   optional.
- [ ] **Validate and build your contribution locally**. We have seperate scripts for 
   validation and packaging.
   ```console
   $ python utils/build_package/validate.py         # to validate your contribution
   $ python utils/build_package/build.py            # to install updated Espresso in your environment
   $ python utils/build_package/validate_build.py   # to run both of above together
   ```
- [ ] **Delete / comment out these initial instructions**. They are for your own reference
   so feel free to delete them or comment them out once you've finished the above
   checklist.

## How to test your code

***In order to test your code***, use `contrib` as your working directory and import your contribution
in the following ways.

(Python interactive mode)
```python
$ pwd                            # check you are in the right folder
<path-to-espresso>/contrib
$ python
>>> from example_name import *   # import it this way
```

(Creating temporary Python file)
```python
# file contrib/tmp.py            # create tmp file in the right folder
from example_name import *       # import it this way
```
