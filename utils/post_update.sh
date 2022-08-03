#!/bin/bash

# activate environment
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate cofi_dev
pwd

# run all the notebooks
python utils/run_notebooks/run_notebooks.py all

# convert all notebooks to scripts 
python utils/sphinx_gallery/ipynb_to_gallery.py all

# run all sphinx-gallery scripts
cd ..
make html

# commit all changes (cache)
cd cofi-examples
git commit -am "chore: sphinx gallery cache"
cd ..
git commit -am "chore: update cofi-examples"
