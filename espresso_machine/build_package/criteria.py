"""Helper functions to analyse the base class `EspressoProblem`

This script assumes you have geo-espresso installed via:
    $ python espresso_machine/build_package/build.py
Or the core package via:
    $ pip install .

The following two functions are exposed to other scripts by this file:
- criteria_for_problems
- criteria_for_example

Check comments for these functions for what criteria are implemented.
"""

import os
import numpy as np
from matplotlib.axes import Axes

import run_examples
import _utils

try:
    import espresso
except ModuleNotFoundError as e:
    e.msg += (
        "\n\nNote: To run pre-build validation, please firstly install "
        "`espresso` core module by running the following from the root"
        "level of the project\n  $ pip install ."
    )
    raise e


# --------------> validate a contributed problem

# 1. Contribution folder name matches the main Python file name
def _check_folder_file_names(prob_name, prob_path, names_in_folder):
    assert f"{prob_name}.py" in names_in_folder, (
        "the contribution folder name should match the main Python file name, "
        f"so you should have this file: contrib/{prob_name}/{prob_name}.py"
    )


# 2. These file exist: README.md, LICENCE, __init__.py
def _check_required_files(prob_path, names_in_folder):
    required_files = ["README.md", "LICENCE", "__init__.py"]
    for file in required_files:
        assert (
            file in names_in_folder
        ), f"{file} is required but you don't have it in {prob_path}"


# 3. LICENCE file is not empty
def _check_licence_nonempty(prob_path):
    assert (
        os.stat(f"{prob_path}/LICENCE").st_size != 0
    ), "ensure the LICENCE file is not empty"


# 4. The __init__.py contains `__all__` variable and has the class name in it
def _check_init_all(parent_mod, prob_name, prob_class_name):
    _class_not_in_init_all_msg = (
        "make sure you include "
        f"`__all__ = ['{prob_class_name}']` in the file "
        f"contrib/{prob_name}/__init__.py"
    )
    try:
        _init_all = parent_mod.__all__
    except:
        raise AssertionError(_class_not_in_init_all_msg)
    else:
        assert prob_class_name in _init_all, (
            _class_not_in_init_all_msg
            + f". We found {_init_all} but {prob_class_name} is not in there"
        )


# 5. The class is a subclass of EspressoProblem
def _check_subclass(prob_class, prob_class_name):
    assert issubclass(prob_class, espresso.EspressoProblem), (
        f"make sure your problem class `{prob_class_name}` is a subclass of "
        "`espresso.EspressoProblem`, by defining it with:\n\n-------\n"
        "from espresso import EspressoProblem\n\n"
        f"class {prob_class_name}(EspressoProblem):\n"
        "\tdef __init__(self, example_number=1):\n"
        "\t\tsuper().__init__(example_number)\n\t\t...\n\n-------"
    )


# 6. The following metadata are defined properly:
#     - problem_title
#     - problem_short_description
#     - author_names
#     - contact_name
#     - contact_email
#     - [optional] citations -> []
#     - [optional] linked_sites -> [(name, link)]
def _check_metadata(prob_class, prob_class_name):
    class_metadata = prob_class.metadata
    # problem_title
    assert (
        type(class_metadata["problem_title"]) is str
        and len(class_metadata["problem_title"]) > 0
    ), (
        f"check class attribute `{prob_class_name}.metadata['problem_title']`"
        " is present and is a non-empty string"
    )
    # problem_short_description
    assert (
        type(class_metadata["problem_short_description"]) is str
    ), (  # Allow empty field
        "check class attribute "
        f"`{prob_class_name}.metadata['problem_short_description']`"
        " is present and is a string"
    )
    # author_names
    assert type(class_metadata["author_names"]) is list, (
        "check class attribute "
        f"`{prob_class_name}.metadata['author_names']`"
        " is present and is a list"
    )
    assert len(class_metadata["author_names"]) > 0, (
        "check class attribute "
        f"`{prob_class_name}.metadata['author_names']` is not empty"
    )
    for author in class_metadata["author_names"]:
        assert type(author) is str and len(author) > 0, (
            "check elements of class attribute "
            f"`{prob_class_name}.metadata['author_names`] are non-empty strings"
        )
    # contact_name
    assert (
        type(class_metadata["contact_name"]) is str
        and len(class_metadata["contact_name"]) > 0
    ), (
        "check class attribute "
        f"`{prob_class_name}.metadata['contact_name']`"
        " is present and is a non-empty string"
    )
    # contact_email
    assert (
        type(class_metadata["contact_email"]) is str
        and "@" in class_metadata["contact_email"]
    ), (
        "check class attribute "
        f"`{prob_class_name}.metadata['contact_email']`"
        " is present and is a valid email address string"
    )
    # citations
    assert type(class_metadata["citations"]) is list, (
        "check class attribute "
        f"`{prob_class_name}.metadata['citations']`"
        " is present and is a list"
    )
    for citation in class_metadata["citations"]:
        assert type(citation) is tuple and len(citation) == 2, (
            "check elements of class attribute "
            f"`{prob_class_name}.metadata['citations`] are 2-element tuples"
        )
        for field in citation:
            assert type(field) is str, (
                "check elements of class attribute "
                f"`{prob_class_name}.metadata['citations`] have tuples of "
                "strings"
            )
    # linked_sites
    assert type(class_metadata["linked_sites"]) is list, (
        "check class attribute "
        f"`{prob_class_name}.metadata['linked_sites']`"
        " is present and is a list"
    )
    for site in class_metadata["linked_sites"]:
        assert type(site) is tuple and len(site) == 2, (
            "check elements of class attribute "
            f"`{prob_class_name}.metadata['linked_sites`] are 2-element tuples"
        )
        for field in site:
            assert type(field) is str, (
                "check elements of class attribute "
                f"`{prob_class_name}.metadata['linked_sites`] have tuples of "
                "strings"
            )


def criteria_for_problem(prob_class, prob_class_name, prob_path, parent_mod):
    """validate a contributed problem

    This function checks the folder structure and metadata of specified problem:
    1. Contribution folder name matches the main Python file name
    2. These file exist: README.md, LICENCE, __init__.py
    3. LICENCE file is not empty
    4. The __init__.py contains `__all__` variable and has the class name in it
    5. The class is a subclass of EspressoProblem
    6. The following metadata are defined properly:
        - problem_title
        - problem_short_description
        - author_names
        - contact_name
        - contact_email
        - [optional] citations -> []
        - [optional] linked_sites -> [(name, link)]

    Note: a problem will have several examples, but we will put all the criteria for
    the examples to function `criteria_for_example`.
    """
    # preparation
    names_in_folder, paths_in_folder = _utils.get_folder_content(prob_path)
    prob_name = prob_path.split("/")[-1]

    # checking
    _check_folder_file_names(prob_name, prob_path, names_in_folder)
    _check_required_files(prob_path, names_in_folder)
    _check_licence_nonempty(prob_path)
    _check_init_all(parent_mod, prob_name, prob_class_name)
    _check_subclass(prob_class, prob_class_name)
    _check_metadata(prob_class, prob_class_name)


# --------------> validate an example of problem


def _inspect_espresso_problem():
    all_props = dir(espresso.EspressoProblem)
    abs_meta = espresso.EspressoProblem.__abstract_metadata_keys__
    abs_props = espresso.EspressoProblem.__abstractmethods__
    opt_props = [p for p in all_props if p not in abs_meta and p not in abs_props]
    return {
        "required meta keys": abs_meta,
        "required attributes": abs_props,
        "optional attributes": opt_props,
    }


def _check_is_number(all_results: run_examples.ResultsFromExample, obj, obj_str):
    assert isinstance(obj, (float, int)), f"ensure {obj_str} is a number"


def _check_is_str(all_results: run_examples.ResultsFromExample, obj, obj_str):
    assert isinstance(obj, str), f"ensure {obj_str} is a string"


def _check_is_axes(all_results: run_examples.ResultsFromExample, obj, obj_str):
    assert isinstance(obj, Axes) or \
        (isinstance(obj, list) or isinstance(obj, np.ndarray) \
            and isinstance(obj[0], Axes)), \
                f"ensure {obj_str} returns an instance of matplotlib.axes.Axes"


def _check_1d_array_like(all_results: run_examples.ResultsFromExample, obj, obj_str):
    assert np.ndim(obj) == 1, f"ensure {obj_str} is a flat array"


def _check_2d_array_like(all_results: run_examples.ResultsFromExample, obj, obj_str):
    assert np.ndim(obj) == 2, f"ensure {obj_str} is a 2d array"


def _check_shape_factory(get_shape, get_shape_str):
    def _check_shape(all_results: run_examples.ResultsFromExample, obj, obj_str):
        expected_shape = get_shape(all_results)
        expected_shape_str = get_shape_str(all_results)
        assert (
            np.shape(obj) == expected_shape
        ), f"ensure {obj_str} has shape {expected_shape_str}, i.e. {expected_shape}"

    return _check_shape


def _check_shape_config_models():
    get_shape = lambda all_results: (all_results.nmodel,)
    get_shape_str = lambda all_results: f"({all_results.prob_instance_str}.model_size,)"
    return get_shape, get_shape_str


_check_model_shape = _check_shape_factory(*_check_shape_config_models())


def _check_shape_config_data():
    get_shape = lambda all_results: (all_results.ndata,)
    get_shape_str = lambda all_results: f"({all_results.prob_instance_str}.data_size,)"
    return get_shape, get_shape_str


_check_data_shape = _check_shape_factory(*_check_shape_config_data())


def _check_shape_config_cov():
    get_shape = lambda all_results: (all_results.ndata, all_results.ndata)
    get_shape_str = (
        lambda all_results: f"({all_results.prob_instance_str}.data_size, {all_results.prob_instance_str}.data_size)"
    )
    return get_shape, get_shape_str


_check_cov_shape = _check_shape_factory(*_check_shape_config_cov())


def _check_shape_config_jac():
    get_shape = lambda all_results: (all_results.ndata, all_results.nmodel)
    get_shape_str = (
        lambda all_results: f"({all_results.prob_instance_str}.data_size, {all_results.prob_instance_str}.model_size)"
    )
    return get_shape, get_shape_str


_check_jac_shape = _check_shape_factory(*_check_shape_config_jac())

attributes_to_check = [
    # ( output dict key,
    #   attribute str,
    #   whether required,
    #   how to validate given the attribute value/result and str )
    ("nmodel", "model_size", True, [_check_is_number]),
    ("ndata", "data_size", True, [_check_is_number]),
    ("model", "good_model", True, [_check_1d_array_like, _check_model_shape]),
    ("null_model", "starting_model", True, [_check_model_shape]),
    ("data", "data", True, [_check_1d_array_like, _check_data_shape]),
    ("description", "description", False, [_check_is_str]),
    ("cov", "covariance_matrix", False, [_check_2d_array_like, _check_cov_shape]),
    (
        "inv_cov",
        "inverse_covariance_matrix",
        False,
        [_check_2d_array_like, _check_cov_shape],
    ),
    (
        "synth1",
        "forward(prob.good_model)",
        True,
        [_check_1d_array_like, _check_data_shape],
    ),
    (
        "jac1",
        "jacobian(prob.good_model)",
        False,
        [_check_2d_array_like, _check_jac_shape],
    ),
    (
        "synth2",
        "forward(prob.good_model, True)[0]",
        False,
        [_check_1d_array_like, _check_data_shape],
    ),
    (
        "jac2",
        "forward(prob.good_model, True)[1]",
        False,
        [_check_2d_array_like, _check_jac_shape],
    ),
    ("fig_model", "plot_model(prob.good_model)", False, [_check_is_axes]),
    ("fig_data", "plot_data(prob.data)", False, [_check_is_axes]),
    ("misfit", "misfit(prob.data, prob.data)", False, [_check_is_number]),
    (
        "log_likelihood",
        "log_likelihood(prob.data, prob.data)",
        False,
        [_check_is_number],
    ),
    ("log_prior", "log_prior(prob.good_model)", False, [_check_is_number]),
]


def criteria_for_example(all_results):
    """validate an example under a specific problem

    This function checks that the required attributes are defined, and that if anything
    is defined then it's done properly (including the types and shapes):
    1. Required methods/properties are implemented and can run
    2. Optional methods/properties are either not implemented or have no error
    3. Any methods/properties, if implemented, are done propertly
    """
    for attr_check in attributes_to_check:
        attr_key, attr_str, required, to_check = attr_check
        obj = all_results[attr_key]
        obj_str = f"{all_results['prob_instance_str']}.{attr_str}"
        if isinstance(obj, Exception):
            raise obj
        if obj is None and required:
            raise NotImplementedError(
                f"{obj_str} is required but you haven't implemented it"
            )
        if obj is not None:
            for check_func in to_check:
                check_func(all_results, obj, obj_str)


def main():
    problems = _utils.problems_to_run(problems_specified=["testtest"])
    results = run_examples.run_problems(problems, pre_build=True, timeout=_utils.DEFAULT_TIMEOUT)
    for res in results:
        criteria_for_problem(
            res.problem_class,
            res.problem_class_str,
            res.problem_path,
            res.parent_module,
        )
        for prob_out_i in res.problem_results_generator:
            criteria_for_example(prob_out_i)


if __name__ == "__main__":
    main()
