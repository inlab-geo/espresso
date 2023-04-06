import _utils


def pytest_addoption(parser):
    parser.addoption("--post", action="store_true", default=False)
    parser.addoption(
        "--timeout",
        dest="timeout",
        action="store",
        default=999,
        type=int,
        help="Specify the number of seconds as timeout limit for each attribute",
    )
    parser.addoption(
        "--contrib",
        "--contribution",
        dest="contribs",
        action="append",
        default=None,
        help="Specify which contribution to validate",
    )


def pytest_generate_tests(metafunc):
    if "contrib" in metafunc.fixturenames:
        specified = metafunc.config.getoption("contribs")
        problems = _utils.problems_to_run(specified)
        metafunc.parametrize("contrib", set(problems))
