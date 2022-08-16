"""Validate a new contribution from the following aspects:

1. Contribution folder name matches the main Python file name

2. The following files exist:
    - README.md
    - LICENCE
    - metadata.yml
    - __init__.py

3. There is an `__all__` variable in __init__.py with one class name exposed

4. The metadata.yml file can be parsed and has the following keys:
    - name
    - short_description
    - authors
    - examples -> description, model_dimension, data_dimension
    - [optional] citations -> doi
    - [optional] contacts -> name, email, website
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

def _array_like(obj):
    return np.ndim(obj) != 0

def test_contrib(contrib, pre_build):
    contrib_name, contrib_sub_folder = contrib
    contrib_name_capitalised = contrib_name.title().replace("_", " ")
    contrib_name_class = contrib_name_capitalised.replace(" ", "")
    print(f"\n🔍 Checking '{contrib_name}' at {contrib_sub_folder}...")
    names, paths = get_folder_content(contrib_sub_folder)
    
    # 1 - contribution folder name matches the main Python file name
    assert f"{contrib_name}.py" in names

    # 2 - files exist: README.md, LICENCE, metadata.yml, __init__.py
    required_files = ["README.md", "LICENCE", "metadata.yml", "__init__.py"]
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
        print(contrib_mod.__path__)
        with open(Path(__file__).parent.parent.parent / "_esp_build" / "src" / "cofi_espresso" / "__init__.py") as f:
            print(f.read())
    print(contrib_mod.__all__)
    assert contrib_name_class in contrib_mod.__all__
    
    # 4 - metadata.yml can be parsed and has necessary keys
    with open(f"{contrib_sub_folder}/metadata.yml", "r") as stream:
        meta_data = yaml.safe_load(stream)
    for k in ["name", "short_description", "authors", "examples"]:
        assert k in meta_data
    n_examples = len(meta_data["examples"])
    for example in meta_data["examples"]:
        assert "description" in example
        assert "model_dimension" in example
        assert "data_dimension" in example
    if "citation" in meta_data:
        assert "doi" in meta_data["citation"]
    if "contacts" in meta_data:
        for contact in meta_data["contacts"]:
            assert "name" in contact
            assert "email" in contact
    if "extra_websites" in meta_data:
        for website in meta_data["extra_websites"]:
            assert "name" in website
            assert "link" in website

    for i in range(n_examples):
        # 5 - functions are defined: set_example_number, suggested_model, data, forward
        contrib_class = getattr(contrib_mod, contrib_name_class)
        contrib_instance = contrib_class(i)
        _model = contrib_instance.suggested_model()
        _data = contrib_instance.data()
        _synthetics = contrib_instance.forward(_model)
        assert _array_like(_model)
        assert _array_like(_data)
        assert _array_like(_synthetics)

        # 6 - optional functions have correct signatures
        try: _synthetics, _jacobian = contrib_instance.forward(_model, with_jacobian=True)
        except NotImplementedError: pass
        else:
            assert _array_like(_synthetics)
            assert _array_like(_jacobian)
        try: _jacobian = contrib_instance.jacobian(_model)
        except NotImplementedError: pass
        else: assert _array_like(_jacobian)
        try: _fig_model = contrib_instance.plot_model(_model)
        except NotImplementedError: pass
        else: assert isinstance(_fig_model, Figure)
        try: _fig_data = contrib_instance.plot_data(_data)
        except NotImplementedError: pass
        else: assert isinstance(_fig_data, Figure)

    # 7 - LICENCE file not empty
    assert os.stat(f"{contrib_sub_folder}/LICENCE").st_size != 0, \
        "LICENCE file shouldn't be empty"

    print(f"✔️ Passed")


def main():
    return pytest.main([Path(__file__)])

if __name__ == "__main__":
    main()
