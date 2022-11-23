import sys
from glob import glob
from pathlib import Path

import papermill as pm

current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent.parent
NOTEBOOKS = "notebooks"
NOTEBOOKS_DIR = str(root_dir / NOTEBOOKS)


def execute(input, output, cwd=None, params=None):
    pm.execute_notebook(input, output, cwd=cwd, parameters=params)


if __name__ == '__main__':
    # collect all notebooks
    if sys.argv[-1] == "all":
        all_notebooks = glob(f"{NOTEBOOKS_DIR}/*/*.ipynb")
    else:
        all_notebooks = sys.argv[2:]
        all_notebooks = [file for file in all_notebooks if file.endswith(".ipynb")]
    # execute listed notebooks
    print("Executing notebooks...")
    for nb in all_notebooks:
        print(f"file: {nb}")
        path = nb[:nb.rfind("/")]
        execute(nb, nb, cwd=path)
