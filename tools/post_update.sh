#!/bin/bash

# activate environment
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate cofi_dev
pwd

# run all the examples and tutorials
python tools/run_notebooks/run_notebooks.py all

# test all scripts under examples/
python tools/validation/test_all_notebooks_scripts.py
