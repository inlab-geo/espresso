#!/bin/bash

# activate environment
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate cofi_dev
pwd

# run all the notebooks
python tools/run_notebooks/run_notebooks.py all

# run all sphinx-gallery scripts
cd ..
make html

# commit all changes (cache)
cd cofi-examples
git commit -am "chore: outputs cleanup"
cd ..
git commit -am "chore: update cofi-examples and docs cache"
