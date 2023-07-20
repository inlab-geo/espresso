"""Check if all requirements are listed in pyproject.toml

This script assumes you have geo-espresso installed via:
$ python espresso_machine/build_package/build.py

Usage: python check_requires.py [-c <example_name] [--file <file_name>]
"""

import sys
import pathlib
import pytest
from stdlib_list import stdlib_list

import _utils

known_dependencies = {
    "numpy",
    "scipy",
    "scipy.stats",
    "scipy.sparse",
    "scipy.interpolate",
    "scipy.constants",
    "matplotlib",
    "matplotlib.pyplot",
    "tqdm",
    "mpmath",
    "seislib", 
    "requests", 
    "socks", 
    "lxml", 
    "shapefile", 
    "pkg_resources", 
    "idna", 
    "urllib3", 
    "certifi", 
    "brotli", 
    "decorator", 
    "pyproj", 
    "charset_normalizer", 
    "shapely", 
    "obspy", 
    "cartopy", 
}
to_exclude = {"espresso", "run_examples"}


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
        try:
            __import__(pkg)
        except:
            pass
    return _strip_pkg(set(sys.modules.keys()))


def _get_imported_pkg(problem_specified):
    import run_examples

    run_examples.main([problem_specified], _utils.DEFAULT_TIMEOUT)
    return _strip_pkg(set(sys.modules.keys()))


def _get_requirements(problem_specified):
    inbuilt = _get_inbuilt_pkg()
    known_depended = _get_known_depended_pkg()
    all_imported = _get_imported_pkg(problem_specified)
    return inbuilt, known_depended, all_imported


def get_extra_requirements(problem_specified):
    inbuilt, known_depended, all_imported = _get_requirements(problem_specified)
    new_dependencies = all_imported - known_depended - inbuilt
    not_listed = new_dependencies - to_exclude
    return not_listed


def test_requires(contrib):
    not_listed = get_extra_requirements(contrib[0])
    assert len(not_listed) == 0, f"new dependency to be listed: {not_listed}"
    print("√ Passed requirements test.")


if __name__ == "__main__":
    pytest.main([pathlib.Path(__file__)])
