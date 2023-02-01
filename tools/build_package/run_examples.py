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

def run_example(problem_class, problem_class_str, i):
    prob_instance_i = problem_class(i)
    _prob_instance_i_str = f"{problem_class_str}({i})"
    _nmodel = prob_instance_i.model_size
    _ndata = prob_instance_i.data_size
    _model = prob_instance_i.good_model
    _null_model = prob_instance_i.starting_model
    _data = prob_instance_i.data
    _synth1 = prob_instance_i.forward(_model)
    try: _jac1 = prob_instance_i.jacobian(_model)
    except NotImplementedError: _jac1 = None
    try: _synth2, _jac2 = prob_instance_i.forward(_model, True)
    except NotImplementedError: _synth2, _jac2 = None, None
    try: _fig_model = prob_instance_i.plot_model(_model)
    except NotImplementedError: _fig_model = None
    try: _fig_data = prob_instance_i.plot_data(_data)
    except NotImplementedError: _fig_data = None
    try: _misfit = prob_instance_i.misfit(_data, _data)
    except NotImplementedError: _misfit = None
    try: _log_likelihood = prob_instance_i.log_likelihood(_data, _data)
    except NotImplementedError: _log_likelihood = None
    try: _log_prior = prob_instance_i.log_prior(_model)
    except NotImplementedError: _log_prior = None
    try: _description = prob_instance_i.description
    except NotImplementedError: _description = None
    try: _cov = prob_instance_i.covariance_matrix
    except NotImplementedError: _cov = None
    try: _inv_cov = prob_instance_i.inverse_covariance_matrix
    except NotImplementedError: _inv_cov = None
    return (
        _prob_instance_i_str,
        _nmodel,
        _ndata,
        _model,
        _null_model,
        _data,
        _synth1,
        _jac1,
        _synth2,
        _jac2,
        _fig_model,
        _fig_data,
        _misfit,
        _log_likelihood,
        _log_prior,
        _description,
        _cov,
        _inv_cov,
    )

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
        for prob_out_i in prob_out_gen:
            print(prob_out_i[0])

if __name__ == "__main__":
    main()
