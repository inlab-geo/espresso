"""Run all problems and associated examples in espresso

This script assumes you have geo-espresso installed via:
$ python espresso_machine/build_package/build.py

"""

import sys
import os
import typing
import types
import pathlib
import subprocess
import dataclasses

import _utils

try:
    from espresso.exceptions import InvalidExampleError
    from espresso import EspressoProblem
except ModuleNotFoundError as e:
    e.msg += (
        "\n\nNote: To run pre-build validation, please firstly install "
        "`espresso` core module by running the following from the root"
        "level of the project\n  $ pip install ."
    )
    raise e


@dataclasses.dataclass
class ResultsFromExample:
    prob_instance_str: str  # e.g. "SimpleRegression(2)""
    prob_instance: EspressoProblem  # e.g. actual instance of str above
    i: int  # e.g. 2
    results: dict

    def error_in_init(
        self,
    ) -> typing.Tuple[str, typing.Union[Exception, EspressoProblem]]:
        return isinstance(self.prob_instance, Exception), self.prob_instance

    def __getattr__(self, key):
        if key in self.results:
            return self.results[key]
        else:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{key}'"
            )


@dataclasses.dataclass
class ResultsFromProblem:
    parent_module: types.ModuleType
    problem_name: str
    problem_class: typing.Type
    problem_class_str: str
    problem_path: str
    problem_results_generator: typing.Iterator[ResultsFromExample]


class _ProblemModule:
    """get the parent module of a problem class
    two different cases:
    1. run the examples from contrib/ (pre-build)
    2. run the examples from a built espresso package (post-build)
    """

    def __init__(self, pre_build, problem_name):
        self._pre_build = pre_build
        self._problem_name = problem_name

    def __enter__(self):
        if self._pre_build:
            sys.path.insert(1, _utils.CONTRIB_FOLDER)
            return __import__(self._problem_name)
        else:
            return __import__(_utils.PKG_NAME)

    def __exit__(self, exc_type, exc_value, traceback):
        _to_del = set()
        for key in sys.modules.keys():
            if self._problem_name in key:
                _to_del.add(key)
        for m in _to_del:
            del sys.modules[m]


# For each of the below methods / properties,
# 1. Try to get / access
# 2. If not implemented, assign None to `output_name_for_testing_purpose`
# 3. If implemeneted but got error, assign the error to `output_name_for_testing_purpose`
# 4. If implemeneted and no error, assign the output to `output_name_for_testing_purpose`

prob_methods = [
    # (output_name_for_testing_purpose, how_to_get_it)
    ("synth1", lambda p: p.forward(p.good_model)),
    ("jac1", lambda p: p.jacobian(p.good_model)),
    ("synth2", lambda p: p.forward(p.good_model, True)[0]),
    ("jac2", lambda p: p.forward(p.good_model, True)[1]),
    ("fig_model", lambda p: p.plot_model(p.good_model)),
    ("fig_data", lambda p: p.plot_data(p.data)),
    ("misfit", lambda p: p.misfit(p.data, p.data)),
    ("log_likelihood", lambda p: p.log_likelihood(p.data, p.data)),
    ("log_prior", lambda p: p.log_prior(p.good_model)),
]
prob_properties = [
    # (output_name_for_testing_purpose, how_to_access_it)
    ("nmodel", "model_size"),
    ("ndata", "data_size"),
    ("model", "good_model"),
    ("null_model", "starting_model"),
    ("data", "data"),
    ("description", "description"),
    ("cov", "covariance_matrix"),
    ("inv_cov", "inverse_covariance_matrix"),
]


def _get_result(prob_instance, how_to_get, is_method=True, timeout=None):
    @_utils.timeout(seconds=timeout)
    def _run_attr(prob_instance, how_to_get, is_method=True):
        if is_method:
            return how_to_get(prob_instance)
        return getattr(prob_instance, how_to_get)

    try:
        return _run_attr(prob_instance, how_to_get, is_method)
    except NotImplementedError:
        return None
    except Exception as e:
        return e


def instantiate_example(problem_class, i):
    try:
        prob_instance_i = problem_class(i)
    except Exception as e:
        if not isinstance(e, InvalidExampleError):
            prob_instance_i = e
        else:
            raise e
    return prob_instance_i


def collect_methods_outputs(prob_instance_i, all_outputs, timeout=None):
    for (output_name, how) in prob_methods:
        all_outputs[output_name] = _get_result(prob_instance_i, how, True, timeout)


def collect_properties(prob_instance_i, all_outputs, timeout=None):
    for (output_name, prop) in prob_properties:
        all_outputs[output_name] = _get_result(prob_instance_i, prop, False, timeout)


def run_example(
    problem_class, problem_class_str, i, timeout=None
) -> ResultsFromExample:
    # prepare
    all_outputs = dict()
    prob_instance_i = instantiate_example(problem_class, i)
    results_from_example = ResultsFromExample(
        prob_instance_str=f"{problem_class_str}({i})",
        prob_instance=prob_instance_i,
        i=i,
        results=all_outputs,
    )
    if not results_from_example.error_in_init()[0]:  # collect results
        collect_methods_outputs(prob_instance_i, all_outputs, timeout)
        collect_properties(prob_instance_i, all_outputs, timeout)
    return results_from_example


def run_problem(
    problem_class, problem_class_str, timeout=None
) -> typing.Iterator[dict]:
    if isinstance(problem_class, Exception):
        return []
    i = 1
    while True:
        if i > 99:
            raise ValueError(
                "Reached example 100: aborting."
            )  # Guard against silliness
        try:
            example_res = run_example(problem_class, problem_class_str, i, timeout)
        except InvalidExampleError:
            if i == 1:
                raise ValueError("Ensure there are at least one examples")
            return
        if example_res.error_in_init()[0]:
            raise example_res.error_in_init()[1]
        i += 1
        yield example_res


def run_cmake_if_needed(prob_path, pre_build):
    if pre_build and "CMakeLists.txt" in os.listdir(prob_path):
        build_dir = pathlib.Path(prob_path) / "build"
        build_dir.mkdir(exist_ok=True)
        res1 = subprocess.call(["cmake", ".."], cwd=build_dir)
        if res1:
            raise ChildProcessError(f"`cmake ..` failed in {prob_path}/build")
        res2 = subprocess.call(["make"], cwd=build_dir)
        if res2:
            raise ChildProcessError(f"`make` failed in {prob_path}/build")


def run_problems(
    problems, pre_build, timeout=None
) -> typing.Iterator[ResultsFromProblem]:
    for (prob_name, prob_path) in problems:
        prob_class_str = _utils.problem_name_to_class(prob_name)
        run_cmake_if_needed(prob_path, pre_build)
        try:
            with _ProblemModule(pre_build, prob_name) as parent_module:
                try:
                    prob_class = getattr(parent_module, prob_class_str)
                except Exception as e:
                    prob_class = e
                yield ResultsFromProblem(
                    parent_module=parent_module,
                    problem_name=prob_name,
                    problem_class=prob_class,
                    problem_class_str=prob_class_str,
                    problem_path=prob_path,
                    problem_results_generator=run_problem(
                        prob_class, prob_class_str, timeout
                    ),
                )
        except Exception as e:
            yield ResultsFromProblem(
                parent_module=e,
                problem_name=prob_name,
                problem_class=None,
                problem_class_str=prob_class_str,
                problem_path=prob_path,
                problem_results_generator=[],
            )


def main(problems_specified=None, timeout=_utils.DEFAULT_TIMEOUT, verbose=False):
    problems = _utils.problems_to_run(problems_specified)
    results = run_problems(problems, pre_build=True, timeout=timeout)
    for res in results:
        if verbose:
            print(res.problem_class)
        for prob_out_i in res.problem_results_generator:
            if verbose:
                print(prob_out_i.keys())


if __name__ == "__main__":
    main()
