"""Test all / specified contributed problems, and this depends on:
1. results generated by: run_examples.py
2. criteria specified in: criteria.py
3. problems identified in: conftest.py

Usage: python test_examples.py [--post] [--contribution <example_name>]
"""

import pathlib
import pytest

import report


@pytest.fixture
def pre_build(request):
    return not request.config.getoption("--post")

# Note: `contrib` fixture is defined in conftest.py
def test_contrib(contrib, pre_build):
    _report = report.compliance_report([contrib[0]], pre_build)
    for _r in _report.values():     # always only one iteration
        report.pprint_compliance_report(_report)
        if isinstance(_r["api_compliance"], Exception):
            raise _r["api_compliance"]
        assert _r["api_compliance"], \
            "Not API-compliant. Check report above for details."

def main():
    pytest.main([pathlib.Path(__file__)])

if __name__ == "__main__":
    main()
