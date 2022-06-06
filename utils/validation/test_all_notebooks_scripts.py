"""This script tests running all Python files under notebooks/ folder.
"""


import subprocess
from glob import glob

NOTEBOOKS_FOLDER = "notebooks"
PYTHON = "python"

class bcolors:
    PASSED   = '\033[92m'
    WARNING  = '\033[95m'
    FAILED   = '\033[91m'
    MISSING = '\033[96m'
    ENDC     = '\033[0m'
    BOLD     = '\033[1m'

def listdir_nohidden():
    for dir in glob(f"{NOTEBOOKS_FOLDER}/*/"):
        if not dir.startswith("."):
            yield dir

def listpy_nohidden_notlib(dir):
    for script in glob(f"{dir}/*.py"):
        if not dir.startswith(".") and "lib.py" not in script:
            yield script


def main():
    # collect all scripts under notebooks/ folder
    all_scripts = glob(f"{NOTEBOOKS_FOLDER}/*/*.py")
    all_scripts = [name for name in all_scripts if "lib.py" not in name]
    # run them
    for script in all_scripts:
        subprocess.call([PYTHON, script])


if __name__ == "__main__":
    main()
