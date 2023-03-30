"""Test all / specified contributed problems, and this depends on:
1. results generated by: run_examples.py
2. criteria specified in: criteria.py
"""

import pathlib
import pytest

import report
import _utils


@pytest.fixture
def pre_build(request):
    return not request.config.getoption("--post")

def _all_contribs():
    problems = _utils.problems_to_run()
    print("🥃 Running tests for the following contributions:")
    print("- " + "\n- ".join([c[0] for c in problems]) + "\n")
    return problems

@pytest.fixture(params=_all_contribs())
def contrib(request):
    return request.param

def test_contrib(contrib, pre_build):
    _report = report.compliance_report([contrib[0]], pre_build)
    # raise RuntimeError(str(contrib) + str(pre_build))
    for _r in _report.values():
        report.pprint_compliance_report(_report)
        if isinstance(_r["api_compliance"], Exception):
            raise _r["api_compliance"]
        assert _r["api_compliance"], \
            "Not API-compliant. Check report above for details."

def main():
    pytest.main([pathlib.Path(__file__)])

if __name__ == "__main__":
    main()
