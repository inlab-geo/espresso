import sys
from glob import glob
from pathlib import Path

import papermill as pm

current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent.parent
EXAMPLES = "examples"
EXAMPLES_DIR = str(root_dir / EXAMPLES)
TUTORIALS = "tutorials"
TUTORIALS_DIR = str(root_dir / TUTORIALS)


def execute(input, output, cwd=None, params=None):
    pm.execute_notebook(input, output, cwd=cwd, parameters=params)


if __name__ == '__main__':
    # collect all examples & tutorials
    if sys.argv[-1] == "all":
        all_examples = glob(f"{EXAMPLES_DIR}/*/*.ipynb")
        all_examples.extend(glob(f"{TUTORIALS_DIR}/*.ipynb"))
    else:
        all_examples = sys.argv[1:]
        all_examples = [file for file in all_examples if file.endswith(".ipynb")]
    print(all_examples)
    # execute listed examples & tutorials
    print("Executing examples...")
    for nb in all_examples:
        print(f"file: {nb}")
        path = nb[:nb.rfind("/")]
        execute(nb, nb, cwd=path)
