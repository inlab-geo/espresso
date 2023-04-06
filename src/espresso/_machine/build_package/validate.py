"""Validate a few / all contributions by 
1. Running the problem(s) and checking outputs
   - according to criteria.py
   - tested by test_examples.py
2. If post-build, checking all required packages are specified
   - tested by check_requires.py

Usage: 
python validate.py [-h] [--all] [--pre] [--post] [--contrib CONTRIBS] [--file <file_name>]
"""

import sys
import subprocess
import pathlib
import pytest

import _utils


def prep_params(pre_build=None):
    pre = _utils.pre_build() if pre_build is None else pre_build
    dir_parent = pathlib.Path(__file__).parent.resolve()
    specified_contribs = _utils.problems_to_run_names_only()
    extra_args = []
    if specified_contribs is not None:
        for contrib in specified_contribs:
            extra_args.append("--contribution")
            extra_args.append(contrib)
    return pre, dir_parent, extra_args


def test_all_examples(pre, dir_parent, extra_args):
    py_test_examples = dir_parent / "test_examples.py"
    pytest_cmd = [sys.executable, "-m", "pytest", str(py_test_examples)]
    pytest_cmd.extend(extra_args)
    if not pre:
        pytest_cmd.append("--post")
    exit_status_test_examples = subprocess.run(pytest_cmd).returncode
    if exit_status_test_examples != pytest.ExitCode.OK:
        sys.exit(exit_status_test_examples)


def test_requirements(pre, dir_parent, extra_args):
    if not pre:
        py_check_requires = dir_parent / "check_requires.py"
        pytest_cmd = [sys.executable, "-m", "pytest", str(py_check_requires)]
        pytest_cmd.extend(extra_args)
        exit_status_check_requires = subprocess.run(pytest_cmd).returncode
        if exit_status_check_requires != pytest.ExitCode.OK:
            sys.exit(exit_status_check_requires)


# --> main test
def main(pre_build=None):
    pre, dir_parent, extra_args = prep_params(pre_build)
    test_all_examples(pre, dir_parent, extra_args)
    test_requirements(pre, dir_parent, extra_args)


if __name__ == "__main__":
    main()
