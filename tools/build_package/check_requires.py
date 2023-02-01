"""Check if all requirements are listed in pyproject.toml

This script assumes you have cofi-espresso installed via:
$ python tools/build_package/build.py

"""

import sys
import pytest
from stdlib_list import stdlib_list


def strip_pkg(modules):
    res = set()
    for mod in modules:
        pkg = mod.split(".")[0]
        if not pkg.startswith("_"):
            res.add(pkg)
    return res

def get_inbuilt_pkg():
    return strip_pkg(stdlib_list("3.7"))

def get_known_depended_pkg():
    known_dependencies = [
        "numpy",
        "scipy",
        "scipy.stats",
        "scipy.sparse",
        "scipy.interpolate",
        "scipy.constants",
        "matplotlib",
        "matplotlib.pyplot",
        "tqdm",
    ]
    for pkg in known_dependencies:
        __import__(pkg)
    return strip_pkg(set(sys.modules.keys()))

def get_imported_pkg():
    from run_examples import run_problems
    run_problems(True)
    return strip_pkg(set(sys.modules.keys()))

def get_requirements():
    inbuilt = get_inbuilt_pkg()
    known_depended = get_known_depended_pkg()
    all_imported = get_imported_pkg()
    return inbuilt, known_depended, all_imported

def get_extra_requirements():
    inbuilt, known_depended, all_imported = get_requirements()
    to_exclude = {"cofi_espresso", "run_examples"}
    new_dependencies = all_imported - known_depended - inbuilt
    not_listed = new_dependencies - to_exclude
    return not_listed

def test_requires():
    not_listed = get_extra_requirements()
    assert len(not_listed) == 0, \
        f"new dependency to be listed: {not_listed}"
    print("√ Passed requirements test.")

if __name__ == "__main__":
    test_requires()
