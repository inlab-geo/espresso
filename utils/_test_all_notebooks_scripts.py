"""This script tests running all Python files under notebooks/ folder.
"""


import subprocess
from glob import glob

NOTEBOOKS_FOLDER = "notebooks"
PYTHON = "python"

if __name__ == "__main__":
    # collect all scripts under notebooks/ folder
    all_scripts = glob(f"{NOTEBOOKS_FOLDER}/*/*.py")
    all_scripts = [name for name in all_scripts if "lib.py" not in name]
    # run them
    for script in all_scripts:
        subprocess.call([PYTHON, script])
