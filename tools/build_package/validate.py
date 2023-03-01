"""Validate a few / all contributions by 
1. Running the problem(s) and checking outputs
   - according to criteria.py
   - tested by test_examples.py
2. If post-build, checking all required packages are specified
   - tested by check_requires.py

Usage: 
python validate.py [-h] [--contrib CONTRIBS] [--all] [--pre] [--post]
"""

import sys
import pathlib
import argparse
import pytest


# --> define arguments to be parsed with Python command
def setup_parser():
    parser = argparse.ArgumentParser(
        description="Script to validate specified or all contributions for Espresso, pre/post build for the package"
    )
    parser.add_argument(
        "--contrib", "-c", "--contribution", 
        dest="contribs", action="append", 
        help="Specify which contribution to validate")
    parser.add_argument(
        "--all", "-a", default=None,
        dest="all", action="store_true")
    parser.add_argument(
        "--pre", dest="pre", action="store_true", default=None,
        help="Run tests before building the package")
    parser.add_argument(
        "--post", dest="post", action="store_true", default=None,
        help="Run tests after building the package " + 
            "(we assume you've built the package so won't build it for you; " + 
            "otherwise please use `python build.py` beforehand)")
    return parser

args = setup_parser().parse_args()

def _pre_build():
    return args.pre or (not args.pre and not args.post)


# --> main test
def main():
    # preparation
    pre = _pre_build()
    dir_parent = pathlib.Path(__file__).parent.resolve()
    py_test_examples = dir_parent / "test_examples.py"
    py_check_requires = dir_parent / "check_requires.py"
    # test all examples
    exit_status_test_examples = pytest.main([py_test_examples])
    if pre:
        sys.exit(exit_status_test_examples)
    # test requirements
    else:
        if exit_status_test_examples != pytest.ExitCode.OK:
            sys.exit(exit_status_test_examples)
        exit_status_check_requires = pytest.main([py_check_requires])
        sys.exit(exit_status_check_requires)

if __name__ == "__main__":
    main()
