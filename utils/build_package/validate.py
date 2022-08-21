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
    - [optional] citations -> doi
    - [optional] extra_websites -> name, link 

5. Required methods are implemented and can run in each example
    - __init__(self, example_number) -> None
    - suggested_model(self) -> numpy.ndarray | pandas.Series | list
    - data(self) -> numpy.ndarray | pandas.Series | list
    - forward(self, model, with_jacobian=False) -> numpy.ndarray | pandas.Series | list
    * Check array like datatypes with `np.ndim(m) != 0`

6. Optional functions, if implemented, have the correct signatures
    - forward(self, model, with_jacobian=True) -> tuple
    - jacobian(self, model) -> numpy.ndarray | pandas.Series | list
    - plot_model(self, model) -> matplotlib.figure.Figure
    - plot_data(self, model) -> matplotlib.figure.Figure

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
import yaml


MODULE_NAME = "cofi_espresso"
CONTRIB_FOLDER = str(Path(__file__).parent.parent.parent / "contrib")

def get_folder_content(folder_name):
    names = [name for name in os.listdir(folder_name)]
    paths = [f"{folder_name}/{name}" for name in names]
    return names, paths

def all_contribs():
    return list(zip(*get_folder_content(CONTRIB_FOLDER)))

@pytest.fixture(params=all_contribs())
def contrib(request):
    return request.param

@pytest.fixture
def pre_build():
    pre_post = "pre"
    if len(sys.argv) > 1:
        pre_post = sys.argv[-1]
        if pre_post not in ["pre", "post"]:
            raise ValueError("Please either pass `pre` or `post` as the only argument")
    return pre_post == "pre"

def _flat_array_like(obj):
    return np.ndim(obj) == 1

def _2d_array_like(obj):
    return np.ndim(obj) == 2

def test_contrib(contrib, pre_build):
    contrib_name, contrib_sub_folder = contrib
    contrib_name_capitalised = contrib_name.title().replace("_", " ")
    contrib_name_class = contrib_name_capitalised.replace(" ", "")
    print(f"\n🔍 Checking '{contrib_name}' at {contrib_sub_folder}...")
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
        contrib_mod = __import__(contrib_name)
    else:
        importlib = __import__('importlib')
        contrib_mod = importlib.import_module(MODULE_NAME)
    assert contrib_name_class in contrib_mod.__all__
    
    # 4 - Check metadata is present within module
    assert type(contrib_mod.problem_title) is str and len(contrib_mod.problem_title)>0
    assert type(contrib_mod.problem_short_description) is str # Allow empty field
    assert type(contrib_mod.author_names) is list
    assert len(contrib_mod.author_names)>0
    for author in contrib_mod.author_names: assert type(author) is str and len(author)>0
    assert type(contrib_mod.contact_name) is str and len(contrib_mod.contact_name)>0
    assert type(contrib_mod.contact_email) is str and "@" in contrib_mod.contact_email
    assert type(contrib_mod.citations) is list
    for citation in contrib_mod.citations: 
        assert type(citation) is tuple and len(citation)==2
        for field in citation: assert type(field) is str
    assert type(contrib_mod.linked_sites) is list
    for site in contrib_mod.linked_sites:
        assert type(site) is tuple and len(site)==2
        for field in site: assert type(field) is str

    # We don't know how many examples there. We start at 1 and work up until it breaks.
    i = 0
    while True:
        i+=1 # Example numbering starts at 1
        if i > 99: raise ValueError("Reached example 100: aborting.") # Guard against silliness
        # 5 - functions are defined: set_example_number, suggested_model, data, forward
        contrib_class = getattr(contrib_mod, contrib_name_class)
        try:
            contrib_instance = contrib_class(i)
        except ValueError:
            # Assume that we've found all the examples
            n_examples = i-1
            assert n_examples > 0
            break
        _nmodel = contrib_instance.model_size
        _ndata = contrib_instance.data_size
        _model = contrib_instance.good_model
        _null_model = contrib_instance.starting_model
        _data = contrib_instance.data
        _cov = contrib_instance.covariance_matrix
        _synthetics = contrib_instance.forward(_model)
        assert _flat_array_like(_model) and np.shape(_model) == (_nmodel,)
        assert _flat_array_like(_null_model) and np.shape(_null_model) == (_nmodel,)
        assert _flat_array_like(_data) and np.shape(_data) ==  (_ndata,)
        assert _2d_array_like(_cov) and np.shape(_cov) == (_ndata, _ndata)
        assert _flat_array_like(_synthetics) and np.shape(_synthetics) == (_ndata,)

        # 6 - optional functions have correct signatures
        try: _description = contrib_instance.description
        except NotImplementedError: pass
        else: assert type(_description) is str
        try: _inv_cov = contrib_instance.inverse_covariance_matrix
        except NotImplementedError: pass
        else: assert _2d_array_like(_inv_cov) and np.shape(_inv_cov) == (_ndata, _ndata) and np.allclose(np.dot(_cov, _inv_cov), np.eye(_ndata))
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
    return pytest.main([Path(__file__)])

if __name__ == "__main__":
    main()
