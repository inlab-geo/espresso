# Developer notes for utility scripts

This folder contains some utilities for updating notebooks and CoFI documentations.

All the commands below assume you are now in the project root as working directory.

## Create new example (for future contribution)

```console
python tools/generate_example/create_new_example.py <example-name>
```

## Run all the notebooks (for github view & colab)

```console
python tools/run_notebooks/run_notebooks.py all
```

## Convert notebooks to scripts (for ***CoFI documentation***)

This includes converting ``{{badge}}`` in each notebook into the actual badge,
and converting the notebooks to Sphinx-Gallery compatible scripts.

Both can be done automatically with GitHub actions 
`.github/workflows/gen_gallery_scripts.yml`. So just merge into main branch or push 
directly to main branch and wait for a while. 

Alternatively, replace the badges yourself and run the following script:

```console
python tools/sphinx_gallery/ipynb_to_gallery.py all
```

Note that the GitHub action is "scheduled every day" instead of "triggered immediately".
This is to avoid authentication issue when we merge pull requests from other 
contributors.

## Run all Sphinx-Gallery scripts (for ***CoFI documentation***)

Some notebooks are computationally expensive so it's a better idea to run and store
the cache locally. Otherwise readthedocs may time out.

```console
cd $COFI/docs
make html
cd cofi-examples
git commit -am "chore: sphinx gallery cache"
cd ..
git commit -am "chore: update cofi-examples"
```


## Run all example scripts (for validation)

Note that "example scripts" here are different from the Sphinx-Gallery scripts above.
These are working examples for users to refer to.

This validation step serves as an integration and regression test for changes in CoFI.

```console
python tools/validation/test_all_notebooks_scripts.py
python tools/validation/output_to_validation.py    # when necessary
```
