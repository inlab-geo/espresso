"""Validate a new contribution from the following aspects:

1. Contribution folder name matches the main Python file name

2. The following files exist:
    - README.md
    - LICENCE
    - __init__.py

3. There is an `__all__` variable in __init__.py with one class name exposed

4. The contribution provides access to the correct metadata, including:
    - problem_title
    - problem_short_description
    - author_names
    - contact_name
    - contact_email
    - [optional] citations -> []
    - [optional] linked_sites -> [(name, link)]

5. Required methods/properties are implemented and can run in each example
    - __init__(self, example_number) -> None
    - model_size
    - data_size
    - good_model: flat array like
    - starting_model: flat array like
    - data: flat array like
    - forward(self, model, with_jacobian=False) -> flat array like
    * Check array like datatypes with `np.ndim(m) != 0`

6. Optional functions, if implemented, have the correct signatures
    - forward(self, model, with_jacobian=True) -> tuple
    - jacobian(self, model) -> numpy.ndarray | pandas.Series | list
    - plot_model(self, model) -> matplotlib.figure.Figure
    - plot_data(self, model) -> matplotlib.figure.Figure
    - covariance_matrix
    - inverse_covariance_matrix
    - misfit
    - log_likelihood
    - log_prior

7. LICENCE file is not empty


NOTE: To use this script, run `python validate.py pre` for pre build validation, 
    and run `python validate.py post` for post build validation.

"""

import os
import sys
from pathlib import Path
import pytest
import numpy as np
from matplotlib.figure import Figure
import subprocess
import argparse
import warnings


# --> constants
PKG_NAME = "cofi_espresso"
ROOT = str(Path(__file__).resolve().parent.parent.parent)
CONTRIB_FOLDER = ROOT + "/contrib"


# --> define arguments to be parsed with Python command
parser = argparse.ArgumentParser(
    description="Script to validate specified or all contributions for Espresso, pre/post build for the package"
)
parser.add_argument(
    "--contrib", "-c", "--contribution", 
    dest="contribs", action="append", 
    help="Specify which contribution to validate")
parser.add_argument(
    "--all", "-a", default=None,
    dest="all", action="store_true")
parser.add_argument(
    "--pre", dest="pre", action="store_true", default=None,
    help="Run tests before building the package")
parser.add_argument(
    "--post", dest="post", action="store_true", default=None,
    help="Run tests after building the package " + 
        "(we assume you've built the package so won't build it for you; " + 
        "otherwise please use `python build.py` beforehand)")
args = parser.parse_args()

def _pre_build():
    return args.pre or (not args.pre and not args.post)

@pytest.fixture()
def pre_build():
    return _pre_build()


# --> helper methods
def get_folder_content(folder_name):
    names = [name for name in os.listdir(folder_name)]
    paths = [f"{folder_name}/{name}" for name in names]
    return names, paths

def all_contribs():
    _contribs = args.contribs
    _all = args.all
    all_contribs = get_folder_content(CONTRIB_FOLDER)
    all_contribs_zipped = list(zip(*all_contribs))
    if (not _contribs) or _all:    # if no "contrib" is specified, or "all" is explicitly set
        contribs = all_contribs_zipped
    else:
        contribs = [c for c in all_contribs_zipped if c[0] in _contribs]
        contribs_not_in_folder = [c for c in _contribs if c not in all_contribs[0]]
        if contribs_not_in_folder:
            warnings.warn(
                "these examples are not detected in 'contrib' folder: " + ", ".join(contribs_not_in_folder)
            )
    return contribs

@pytest.fixture(params=all_contribs())
def contrib(request):
    return request.param

def _flat_array_like(obj):
    return np.ndim(obj) == 1

def _2d_array_like(obj):
    return np.ndim(obj) == 2

# --> main test (once for each contribution)
def test_contrib(pre_build, contrib):
    contrib_name, contrib_sub_folder = contrib
    contrib_name_capitalised = contrib_name.title().replace("_", " ")
    contrib_name_class = contrib_name_capitalised.replace(" ", "")
    _pre_post = "pre" if pre_build else "post"
    print(f"\n🔍 Performing {_pre_post}-build test on '{contrib_name}' at {contrib_sub_folder}...")
    print("❗️ We assume you've built the package so won't build it for you, otherwise please use `python tools/build_package/build.py` beforehand")
    names, paths = get_folder_content(contrib_sub_folder)
    
    # 1 - contribution folder name matches the main Python file name
    assert f"{contrib_name}.py" in names

    # 2 - files exist: README.md, LICENCE, metadata.yml, __init__.py
    required_files = ["README.md", "LICENCE", "__init__.py"]
    for file in required_files:
        assert file in names, \
            f"{file} is required but you don't have it in {contrib_sub_folder}"
    
    # 3 - __all__ includes standard functions exposed to users
    if pre_build:
        sys.path.insert(1, CONTRIB_FOLDER)
        parent_mod = __import__(contrib_name)
    else:
        importlib = __import__('importlib')
        parent_mod = importlib.import_module(PKG_NAME)
    assert contrib_name_class in parent_mod.__all__
    contrib_class = getattr(parent_mod, contrib_name_class)
    exception_class = getattr(parent_mod, "InvalidExampleError")
    
    # 4 - Check metadata is present within class
    class_metadata = contrib_class.metadata
    assert type(class_metadata["problem_title"]) is str and len(class_metadata["problem_title"])>0
    assert type(class_metadata["problem_short_description"]) is str # Allow empty field
    assert type(class_metadata["author_names"]) is list
    assert len(class_metadata["author_names"])>0
    for author in class_metadata["author_names"]: assert type(author) is str and len(author)>0
    assert type(class_metadata["contact_name"]) is str and len(class_metadata["contact_name"])>0
    assert type(class_metadata["contact_email"]) is str and "@" in class_metadata["contact_email"]
    assert type(class_metadata["citations"]) is list
    for citation in class_metadata["citations"]: 
        assert type(citation) is tuple and len(citation)==2
        for field in citation: assert type(field) is str
    assert type(class_metadata["linked_sites"]) is list
    for site in class_metadata["linked_sites"]:
        assert type(site) is tuple and len(site)==2
        for field in site: assert type(field) is str

    # We don't know how many examples there. We start at 1 and work up until it breaks.
    i = 0
    while True:
        i+=1 # Example numbering starts at 1
        if i > 99: raise ValueError("Reached example 100: aborting.") # Guard against silliness
        # 5 - functions/properties are defined:
        #    model_size, data_size, good_model, starting_model, data, forward
        try:
            contrib_instance = contrib_class(i)
        except exception_class:
            # Assume that we've found all the examples
            n_examples = i-1
            assert n_examples > 0
            break
        _nmodel = contrib_instance.model_size
        _ndata = contrib_instance.data_size
        _model = contrib_instance.good_model
        _null_model = contrib_instance.starting_model
        _data = contrib_instance.data
        # _cov = contrib_instance.covariance_matrix
        _synthetics = contrib_instance.forward(_model)
        assert _flat_array_like(_model) and np.shape(_model) == (_nmodel,)
        assert _flat_array_like(_null_model) and np.shape(_null_model) == (_nmodel,)
        assert _flat_array_like(_data) and np.shape(_data) ==  (_ndata,)
        # assert _2d_array_like(_cov) and np.shape(_cov) == (_ndata, _ndata)
        assert _flat_array_like(_synthetics) and np.shape(_synthetics) == (_ndata,)

        # 6 - optional functions have correct signatures:
        #    description, covariance_matrix, inverse_covariance_matrix, jacobian, 
        #    plot_model, plot_data, misfit, log_likelihood, log_prior
        try: _description = contrib_instance.description
        except NotImplementedError: pass
        else: assert type(_description) is str
        _cov = None
        _inv_cov = None
        try: _cov = contrib_instance.covariance_matrix
        except NotImplementedError: pass
        else: assert _2d_array_like(_cov) and np.shape(_cov) == (_ndata, _ndata)
        try: _inv_cov = contrib_instance.inverse_covariance_matrix
        except NotImplementedError: pass
        else: assert _2d_array_like(_inv_cov) and np.shape(_inv_cov) == (_ndata, _ndata)
        if _cov is not None and _inv_cov is not None: np.allclose(np.dot(_cov, _inv_cov), np.eye(_ndata))
        try: _synthetics, _jacobian = contrib_instance.forward(_model, with_jacobian=True)
        except NotImplementedError: pass # Note that we've already tested the case `with_jacobian=False`
        else:
            assert _flat_array_like(_synthetics) and np.shape(_synthetics) == (_ndata,)
            assert _2d_array_like(_jacobian) and np.shape(_jacobian) == (_ndata, _nmodel)
        try: _jacobian = contrib_instance.jacobian(_model)
        except NotImplementedError: pass
        else: assert _2d_array_like(_jacobian) and np.shape(_jacobian) == (_ndata, _nmodel)
        try: _fig_model = contrib_instance.plot_model(_model)
        except NotImplementedError: pass
        else: assert isinstance(_fig_model, Figure)
        try: _fig_data = contrib_instance.plot_data(_data)
        except NotImplementedError: pass
        else: assert isinstance(_fig_data, Figure)
        try: _misfit = contrib_instance.misfit(_data,_data)
        except NotImplementedError: pass 
        else: assert type(_misfit) is float and _misfit==0.
        try: _log_likelihood = contrib_instance.log_likelihood(_data,_data)
        except NotImplementedError: pass
        else: assert type(_log_likelihood) is float
        try: _log_prior = contrib_instance.log_prior(_model)
        except NotImplementedError: pass
        else: assert type(_log_prior) is float

    # 7 - LICENCE file not empty
    assert os.stat(f"{contrib_sub_folder}/LICENCE").st_size != 0, \
        "LICENCE file shouldn't be empty"

    print(f"✔️ Passed")


def main():
    contribs = all_contribs()
    pre = _pre_build()
    print("🥃 Running " + ("pre-" if pre else "post-") + "build tests for the following contributions:")
    print("- " + "\n- ".join([c[0] for c in contribs]) + "\n")
    if not pre:
        print("❗️ We assume you've built the package so won't build it for you, otherwise please use `python tools/build_packge/build.py` beforehand")
    if (pre):
        subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y", PKG_NAME])
        subprocess.call([sys.executable, "-m", "pip", "install", "."], cwd=ROOT)
    sys.exit(pytest.main([Path(__file__)]))

if __name__ == "__main__":
    main()
