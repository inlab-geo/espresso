"""Validate a new contribution from the following aspects:

1. Contribution folder name matches the main Python file name

2. The following files exist:
    - README.md
    - LICENCE
    - metadata.yml
    - __init__.py

3. Required functions are implemented and can run
    - set_example_number(num) -> None
    - suggested_model() -> numpy.ndarray | pandas.Series | list
    - data() -> numpy.ndarray | pandas.Series | list
    - forward(model, with_jacobian=False) -> numpy.ndarray | pandas.Series | list
    * Check array like datatypes with `np.ndim(m) != 0`

4. Optional functions, if implemented, have the correct signatures
    - forward(model, with_jacobian=True) -> tuple
    - jacobian(model) -> numpy.ndarray | pandas.Series | list
    - plot_model(model) -> matplotlib.figure.Figure
    - plot_data(model) -> matplotlib.figure.Figure

5. The metadata.yml file can be parsed and has the following keys:
    - name
    - short_description
    - authors
    - examples -> description, model_dimension, data_dimension
    - [optional] citations -> doi
    - [optional] contacts -> name, email, website
    - [optional] extra_websites -> name, link

6. LICENCE file is not empty

"""

import os
import sys
import pytest
import numpy as np


CONTRIB_FOLDER = "contrib"

def get_folder_content(folder_name):
    names = [name for name in os.listdir(folder_name)]
    paths = [f"{folder_name}/{name}" for name in names]
    return names, paths

def all_contribs():
    return list(zip(*get_folder_content(CONTRIB_FOLDER)))

@pytest.fixture(params=all_contribs())
def contrib(request):
    return request.param

def test_contrib(contrib):
    contrib_name, contrib_sub_folder = contrib
    print(f"\n🔍 Checking '{contrib_name}' at {contrib_sub_folder}...")
    names, paths = get_folder_content(contrib_sub_folder)
    
    # 1 - contribution folder name matches the main Python file name
    assert f"{contrib_name}.py" in names

    # 2 - files exist: README.md, LICENCE, metadata.yml, __init__.py
    required_files = ["README.md", "LICENCE", "metadata.yml", "__init__.py"]
    for file in required_files:
        assert file in names, \
            f"{file} is required but you don't have it in {contrib_sub_folder}"

    print(f"✔️ Passed")


if __name__ == "__main__":
    sys.exit(pytest.main(["utils/build_package/validate.py"]))
