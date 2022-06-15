import sys
from glob import glob

import papermill as pm

NOTEBOOKS_FOLDER = "notebooks"


def execute(input, output, params=None):
    pm.execute_notebook(input, output, parameters=params)


if __name__ == '__main__':
    # collect all notebooks
    if sys.argv[-1] == "all":
        all_notebooks = glob(f"{NOTEBOOKS_FOLDER}/*/*.ipynb")
    else:
        all_notebooks = [sys.argv[-1]]
    # execute listed notebooks
    print("Executing notebooks...")
    for nb in all_notebooks:
        print(f"file: {nb}")
        execute(nb, nb)
