import errno
import os
import signal
import functools
import contextlib
import sys
import tqdm
import argparse
import warnings
import pathlib
import typing


# ------------------------------- constants -------------------------------------------
PKG_NAME = "espresso"
ROOT = str(pathlib.Path(__file__).resolve().parent.parent.parent)
CONTRIB_FOLDER = ROOT + "/contrib"
ACTIVE_LIST = CONTRIB_FOLDER + "/active_problems.txt"

DEFAULT_TIMEOUT = 60
DEFAULT_TIMEOUT_SHORT = 1


# ------------------------------- argument parser -------------------------------------
def setup_parser():
    parser = argparse.ArgumentParser(
        description="Script to build Espresso, with/without pre/post-build validation"
    )
    parser.add_argument(
        "-v",
        "--checks",
        "--post",
        "--validate",
        dest="post",
        action="store_true",
        default=False,
        help="Run tests after building the package",
    )
    parser.add_argument(
        "--pre",
        dest="pre",
        action="store_true",
        default=False,
        help="Run tests before building the package",
    )
    parser.add_argument(
        "--timeout",
        "-t",
        dest="timeout",
        action="store",
        default=999,
        type=int,
        help="Specify the number of seconds as timeout limit for each attribute",
    )
    parser.add_argument(
        "--contrib",
        "-c",
        "--contribution",
        dest="contribs",
        action="append",
        default=None,
        help="Specify which contribution to validate",
    )
    parser.add_argument(
        "--file",
        "-f",
        dest="file",
        action="store",
        default=None,
        help="Specify which contributions to validate in a txt file",
    )
    parser.add_argument("--all", "-a", default=None, dest="all", action="store_true")
    parser.add_argument(
        "--no-install", default=False, dest="dont_install", action="store_true"
    )
    return parser


def args():
    args, unknown = setup_parser().parse_known_args()
    return args


def pre_build():
    _args = args()
    return _args.pre or (not _args.pre and not _args.post)


# ------------------------------- running timeout -------------------------------------
# Thanks to: https://stackoverflow.com/a/2282656
def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        if seconds is None:

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

        else:

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                signal.signal(signal.SIGALRM, _handle_timeout)
                signal.alarm(seconds)
                try:
                    result = func(*args, **kwargs)
                finally:
                    signal.alarm(0)
                return result

        return wrapper

    return decorator


# ------------------------------- hide console output ---------------------------------
# Thanks to:
# https://stackoverflow.com/a/25061573
# https://stackoverflow.com/a/72617364
@contextlib.contextmanager
def suppress_stdout():
    def tqdm_replacement(iterable_object, *args, **kwargs):
        return iterable_object

    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        tqdm_copy = tqdm.tqdm  # store it if you want to use it later
        tqdm.tqdm = tqdm_replacement
        try:
            yield
        finally:
            sys.stdout = old_stdout
            tqdm.tqdm = tqdm_copy


# ------------------------------- contribution processing -----------------------------
def problem_name_to_class(problem_name):  # e.g. "xray_tracer" -> "XrayTomography"
    return problem_name.title().replace("_", "")


def get_folder_content(folder_name) -> typing.Tuple[typing.List[str], typing.List[str]]:
    names = [name for name in os.listdir(folder_name)]
    paths = [f"{folder_name}/{name}" for name in names]
    return names, paths


def problems_specified_from_args():
    contribs_file = args().file
    if contribs_file is not None:
        with open(contribs_file, "r") as f:
            lines = f.readlines()
        contribs = []
        for line in lines:
            if line.startswith("#"):
                continue
            contribs.append(line.strip())
    else:
        contribs = args().contribs
    return contribs

def problems_to_run(problems_specified=None) -> typing.List[typing.Tuple[str, str]]:
    if problems_specified is None:
        problems_specified = problems_specified_from_args()
    all_problems = get_folder_content(CONTRIB_FOLDER)
    all_problems_zipped = list(zip(*all_problems))
    all_problems_zipped = [c for c in all_problems_zipped if "." not in c[0]]
    if problems_specified is None:
        return all_problems_zipped
    else:  # filter by specified problems list
        problems = [c for c in all_problems_zipped if c[0] in problems_specified]
        problems_not_in_folder = [
            c for c in problems_specified if c not in all_problems[0]
        ]
        if problems_not_in_folder:
            warnings.warn(
                "these examples are not detected in 'contrib' folder: "
                + ", ".join(problems_not_in_folder)
            )
        return problems

def problems_to_run_names_only(problems_specified=None) -> typing.List[str]:
    problems = problems_to_run(problems_specified)
    problem_names = [c[0] for c in problems]
    return problem_names
