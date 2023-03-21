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
import pytest

import _utils


# --> main test
def main():
    # preparation
    pre = _utils.pre_build()
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
