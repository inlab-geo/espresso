"""Run all problems and associated examples in espresso

This script assumes you have cofi-espresso installed via:
$ python tools/build_package/build.py

"""

import sys
import os
import warnings
import pathlib
import typing

try:
    import cofi_espresso
except ModuleNotFoundError as e:
    e.msg += "\n\nNote: To run pre-build validation, please firstly install " \
             "`cofi_espresso` core module by running the following from the root" \
             "level of the project\n  $ pip install ."
    raise e


PKG_NAME = "cofi_espresso"
ROOT = str(pathlib.Path(__file__).resolve().parent.parent.parent)
CONTRIB_FOLDER = ROOT + "/contrib"

def problem_name_to_class(problem_name):   # e.g. "xray_tomography" -> "XrayTomography"
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

# two cases:
# 1. run the examples from contrib/ (pre-build)
# 2. run the examples from a built espresso package (post-build)

def problem_module_pre_build(problem_name: str):
    sys.path.insert(1, CONTRIB_FOLDER)
    return __import__(problem_name)

def problem_module_post_build():
    importlib = __import__("importlib")
    return importlib.import_module(PKG_NAME)


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
    ("misfit", lambda p: p.plot_data(p.data, p.data)),
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

def run_example(problem_class, problem_class_str, i):
    all_outputs = dict()
    prob_instance_i = problem_class(i)
    all_outputs["prob_instance_str"] = f"{problem_class_str}({i})"
    for (output_name, prop) in prob_properties:
        try:
            all_outputs[output_name] = getattr(prob_instance_i, prop)
        except NotImplementedError:
            all_outputs[output_name] = None
        except Exception as e:
            all_outputs[output_name] = e
    for (output_name, how_to_get) in prob_methods:
        try:
            all_outputs[output_name] = how_to_get(prob_instance_i)
        except NotImplementedError:
            all_outputs[output_name] = None
        except Exception as e:
            all_outputs[output_name] = e
    return all_outputs

def run_problem(problem_class, problem_class_str):
    i = 1
    while True:
        if i > 99: raise ValueError("Reached example 100: aborting.") # Guard against silliness
        try:
            yield i, run_example(problem_class, problem_class_str, i)
        except cofi_espresso.exceptions.InvalidExampleError:
            assert i-1 > 0, "ensure there are at least one examples"
            break
        i += 1

def run_problems(pre_build, problems_specified = None):
    problems = problems_to_run(problems_specified)
    if not pre_build:
        parent_module = problem_module_post_build()
    for (prob_name, prob_path) in problems:
        if pre_build:
            parent_module = problem_module_pre_build(prob_name)
        prob_class_str = problem_name_to_class(prob_name)
        prob_class = getattr(parent_module, prob_class_str)
        yield prob_class, prob_class_str, run_problem(prob_class, prob_class_str)

def main():
    for prob_class, prob_class_str, prob_out_gen in run_problems(True):
        print(prob_class)
        for i, prob_out_i in prob_out_gen:
            print(prob_out_i.keys())

if __name__ == "__main__":
    main()
