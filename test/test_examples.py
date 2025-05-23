"""Test all / specified contributed problems, and this depends on:
1. results generated by: run_examples.py
2. criteria specified in: criteria.py
3. problems identified in: conftest.py

Usage: python test_examples.py [--post] [--contribution <example_name>] [--file <file_name>]
"""

import pathlib
import pytest
import json
import re

import espresso.utils.report as report
import _utils


# Note: `contrib` fixture is defined in conftest.py
def test_contrib(contrib):
    _report = report.compliance_report([contrib[0]])
    for _r in _report.values():  # always only one iteration
        report.pprint_compliance_report(_report)
        if isinstance(_r.api_compliance, Exception):
            raise _r.api_compliance
        assert _r.api_compliance, "Not API-compliant. Check report above for details."


pass_pattern = r"\(([\w\s]+)\) is API-compliant"


def write_active_list():
    # read report
    with open(".report.json", "r") as f:
        report = json.load(f)
    # analyse report
    active_list = []
    for r in report["tests"]:
        match = re.search(pass_pattern, r["call"]["stdout"])
        if match:
            active_list.append(match.group(1))
    # write to file
    active_list.sort()
    with open(_utils.ACTIVE_LIST, "w") as f:
        f.write(f"# augo generated by {__file__}\n")
        f.write("\n".join(active_list))


def main():
    this_file = pathlib.Path(__file__)
    pytest.main([this_file, "--json-report", "--json-report-indent=2"])
    write_active_list()


if __name__ == "__main__":
    main()
