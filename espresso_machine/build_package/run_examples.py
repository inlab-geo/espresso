"""Run all problems and associated examples in espresso

This script assumes you have geo-espresso installed via:
$ python espresso_machine/build_package/build.py

"""

import sys
import os
import warnings
import pathlib
import typing

try:
    import espresso
    from espresso.exceptions import InvalidExampleError
except ModuleNotFoundError as e:
    e.msg += "\n\nNote: To run pre-build validation, please firstly install " \
             "`espresso` core module by running the following from the root" \
             "level of the project\n  $ pip install ."
    raise e


PKG_NAME = "espresso"
ROOT = str(pathlib.Path(__file__).resolve().parent.parent.parent)
CONTRIB_FOLDER = ROOT + "/contrib"

def _problem_name_to_class(problem_name):   # e.g. "xray_tomography" -> "XrayTomography"
    return problem_name.title().replace("_", "")

def get_folder_content(folder_name):
    names = [name for name in os.listdir(folder_name)]
    paths = [f"{folder_name}/{name}" for name in names]
    return names, paths

def problems_to_run(problems_specified: typing.Optional[list] = None):
    all_problems = get_folder_content(CONTRIB_FOLDER)
    all_problems_zipped = list(zip(*all_problems))
    if problems_specified is None:
        return all_problems_zipped
    else:       # filter by specified problems list
        problems = [c for c in all_problems_zipped if c[0] in problems_specified]
        problems_not_in_folder = [
            c for c in problems_specified if c not in all_problems[0]
        ]
        if problems_not_in_folder:
            warnings.warn(
                "these examples are not detected in 'contrib' folder: " + 
                ", ".join(problems_not_in_folder)
            )
        return problems

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
            sys.path.insert(1, CONTRIB_FOLDER)
            return __import__(self._problem_name)
        else:
            return __import__(PKG_NAME)
        
    def __exit__(self, exc_type, exc_value, traceback):
        # print(sys.modules)
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

def collect_methods_outputs(prob_instance_i, all_outputs):
    for (output_name, how_to_get) in prob_methods:
        try:
            all_outputs[output_name] = how_to_get(prob_instance_i)
        except NotImplementedError:
            all_outputs[output_name] = None
        except Exception as e:
            all_outputs[output_name] = e

def collect_properties(prob_instance_i, all_outputs):
    for (output_name, prop) in prob_properties:
        try:
            all_outputs[output_name] = getattr(prob_instance_i, prop)
        except NotImplementedError:
            all_outputs[output_name] = None
        except Exception as e:
            all_outputs[output_name] = e

def run_example(problem_class, problem_class_str, i) -> dict:
    # prepare
    all_outputs = dict()
    try:
        prob_instance_i = problem_class(i)
    except Exception as e:
        if not isinstance(e, InvalidExampleError):
            prob_instance_i = e
        else:
            raise e
    else:  # collect results
        collect_methods_outputs(prob_instance_i, all_outputs)
        collect_properties(prob_instance_i, all_outputs)
    all_outputs["prob_instance_str"] = f"{problem_class_str}({i})"
    all_outputs["prob_instance"] = prob_instance_i
    all_outputs["i"] = i
    return all_outputs

def run_problem(problem_class, problem_class_str) -> typing.Iterator[dict]:
    if isinstance(problem_class, Exception): return []
    i = 1
    while True:
        if i > 99: raise ValueError("Reached example 100: aborting.") # Guard against silliness
        try:
            example_res = run_example(problem_class, problem_class_str, i)
        except InvalidExampleError:
            if i == 1: raise ValueError("Ensure there are at least one examples")
            return
        i += 1
        yield example_res

def run_problems(problems, pre_build):
    for (prob_name, prob_path) in problems:
        prob_class_str = _problem_name_to_class(prob_name)
        try:
            with _ProblemModule(pre_build, prob_name) as parent_module:
                try:
                    prob_class = getattr(parent_module, prob_class_str)
                except Exception as e:
                    prob_class = e
                yield {
                    "parent module": parent_module,
                    "problem class": prob_class, 
                    "problem class str": prob_class_str, 
                    "problem path": prob_path, 
                    "problem results generator": run_problem(prob_class, prob_class_str),
                }
        except Exception as e:
            yield {
                "parent module": e,
                "problem class str": prob_class_str,
                "problem path": prob_path,
                "problem results generator": [],
            }


def main(problems_specified=None):
    _you_want_to_print_something = False
    problems = problems_to_run(problems_specified)
    results = run_problems(problems, pre_build=True)
    for res in results:
        if _you_want_to_print_something: print(res["problem class"])
        for prob_out_i in res["problem results generator"]:
            if _you_want_to_print_something: print(prob_out_i.keys())

if __name__ == "__main__":
    main()
