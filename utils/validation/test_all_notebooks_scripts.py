"""This script tests running all Python files under notebooks/ folder.
"""


import subprocess
from glob import glob
import os

NOTEBOOKS_FOLDER = "notebooks"
OUTPUT_FOLDER = "utils/validation/_output"
VALIDATION_FOLDER = "utils/validation/_validation_output"

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
    # # collect all scripts under notebooks/ folder
    # all_scripts = glob(f"{NOTEBOOKS_FOLDER}/*/*.py")
    # all_scripts = [name for name in all_scripts if "lib.py" not in name]
    # run them
    # for script in all_scripts:
    #     subprocess.call([PYTHON, script])
    print("Validation starts.")
    subprocess.run(["rm", "-rf", OUTPUT_FOLDER])
    os.mkdir(OUTPUT_FOLDER)

    for example_dir in listdir_nohidden():
        print(f"\nTesting example - {example_dir} ...")
        out_dir = example_dir.replace(NOTEBOOKS_FOLDER, OUTPUT_FOLDER)
        os.mkdir(out_dir)
        val_dir = example_dir.replace(NOTEBOOKS_FOLDER, VALIDATION_FOLDER)
        for script in listpy_nohidden_notlib(example_dir):
            script_name = script.split("/")[-1]
            out_subdir = f"{out_dir}{script_name[:-3]}"
            os.mkdir(out_subdir)
            val_subdir = f"{val_dir}{script_name[:-3]}"
            print(f"script: {script}")
            print(f"\tsaving outputs to folder {out_subdir}")
            res = subprocess.run([
                PYTHON, 
                script, 
                "--no-show-plot", 
                "--save-plot", 
                "--show-summary",
                "--output-dir",
                out_subdir,
            ], stdout=subprocess.PIPE, text=True)
            with open(f"{out_subdir}/log.txt", "w") as log_file:
                log_file.write(res.stdout)
            print(f"\tcomparing outputs with folder {val_subdir}")
        try:   # remove empty folders
            os.rmdir(out_dir)
        except:
            pass

    print("\nOK.")


if __name__ == "__main__":
    main()
