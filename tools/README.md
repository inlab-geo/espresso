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

## Run all example scripts (for validation)

Note that "example scripts" here are different from the Sphinx-Gallery scripts above.
These are working examples for users to refer to.

This validation step serves as an integration and regression test for changes in CoFI.

```console
python tools/validation/test_all_notebooks_scripts.py
python tools/validation/output_to_validation.py    # when necessary
```
