"""Check if all requirements are listed in pyproject.toml

This script assumes you have geo-espresso installed via:
$ python espresso_machine/build_package/build.py

"""

import sys
import pathlib
import pytest
from stdlib_list import stdlib_list

import validate


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

args = validate.args

def _pre_build():
    return args.pre or (not args.pre and not args.post)

def _all_contribs():
    pre = _pre_build()
    problems = run_examples.problems_to_run(args.contribs)
    print("🥃 Running " + ("pre-" if pre else "post-") + "build tests for the following contributions:")
    print("- " + "\n- ".join([c[0] for c in problems]) + "\n")
    results = run_examples.run_problems(problems, pre_build=pre)
    return results

def _strip_pkg(modules):
    res = set()
    for mod in modules:
        pkg = mod.split(".")[0]
        if not pkg.startswith("_"):
            res.add(pkg)
    return res

def _get_inbuilt_pkg():
    return _strip_pkg(stdlib_list("3.7"))

def _get_known_depended_pkg():
    for pkg in known_dependencies:
        __import__(pkg)
    return _strip_pkg(set(sys.modules.keys()))

def _get_imported_pkg():
    import run_examples
    run_examples.main(args.contribs)
    return _strip_pkg(set(sys.modules.keys()))

def _get_requirements():
    inbuilt = _get_inbuilt_pkg()
    known_depended = _get_known_depended_pkg()
    all_imported = _get_imported_pkg()
    return inbuilt, known_depended, all_imported

def get_extra_requirements():
    inbuilt, known_depended, all_imported = _get_requirements()
    to_exclude = {"espresso", "run_examples"}
    new_dependencies = all_imported - known_depended - inbuilt
    not_listed = new_dependencies - to_exclude
    return not_listed

def test_requires():
    not_listed = get_extra_requirements()
    assert len(not_listed) == 0, \
        f"new dependency to be listed: {not_listed}"
    print("√ Passed requirements test.")

if __name__ == "__main__":
    pytest.main([pathlib.Path(__file__)])
