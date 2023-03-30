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
import subprocess
import pathlib
import pytest

import _utils


def test_all_examples(pre, dir_parent):
    py_test_examples = dir_parent / "test_examples.py"
    pytest_cmd = [sys.executable, "-m", "pytest", str(py_test_examples)]
    if not pre:
        pytest_cmd.append("--post")
    exit_status_test_examples = subprocess.run(pytest_cmd).returncode
    if exit_status_test_examples != pytest.ExitCode.OK:
        sys.exit(exit_status_test_examples)

def test_requirements(pre, dir_parent):
    if not pre:
        py_check_requires = dir_parent / "check_requires.py"
        pytest_cmd = [sys.executable, "-m", "pytest", str(py_check_requires)]
        exit_status_check_requires = subprocess.run(pytest_cmd).returncode
        if exit_status_check_requires != pytest.ExitCode.OK:
            sys.exit(exit_status_check_requires)

# --> main test
def main(pre_build=None):
    pre = _utils.pre_build() if pre_build is None else pre_build
    dir_parent = pathlib.Path(__file__).parent.resolve()
    test_all_examples(pre, dir_parent)
    test_requirements(pre, dir_parent)

if __name__ == "__main__":
    main()
