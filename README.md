# CoFI Test Suite


CoFi TestSuite is a collection of clearly defined, stand-alone forward and inversion problems that each can be solved using the same code structure. The aim is to create a minimalistic framework that is easily understandable while being flexible enough of hosting a wide range of inversion problems. Information will be conveyed via a sensible, consistent naming convention and an informative output that can consist of text and/or visualisations. 


# Installation

It is recommended to use a clean virtual environment for the install: 

```console
conda create -n cofi_testsuite_env python=3.8 scipy jupyterlab numpy matplotlib
conda activate cofi_testsuite_env
```

CoFi Test Suite is available on PyIP and can be installed using this command:

```console
pip install cofitestsuite
```

# Basic usage

Once installed, each CoFI test problem can be imported using the following command structure:

```console
from cofitestsuite.testproblem import testproblem as tp
```

Replace ``testproblem`` with one of the following currently available problems:

- ``gravityforward``: 
- ``xraytomorgraphy``: 
- ``earthquakebootstrap``: 
- ``earthquakeleastsquares``: 

Once a problem is imported, it's main functions can be called using the same structure for each problem. For more information for each function please use help():

```console

from InversionTestProblems import testproblem as tp

model=tp.init_routine(problem_basics) 

synthetic, gradient = tp.forward(problem_basics, model)

result = tp.solver(problem_basics, model, synthetic, gradient)

tp.plot_model(problem_basics, result)

```

Problem-specific values, for example model resolution or noise level, can be specified after calling ``problem_basics=problem_fcn.basics()``:


```console

from InversionTestProblems import testproblem as tp

problem_basics = problem_fcn.basics()
problem_basics.variable=value
problem_basics.variable2=value2
...

```

# How to contribute to the repository:
 <em>This section should include a text about how to merge branches before it is made available to the wider public.</em>

This section contains the minimum information necessary to contribute to the CoFI Test Suite Github repository. We assume that the potential contributor already has a Github account and established a secure connection using https or SSH. If this is not the case please see the following tutorial: 

https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token

To contribute an existing test problem to the Github repository, first clone the Github repository: 

```console
git clone https://github.com/inlab-geo/inversion-test-problems.git
```

Create a new branch for the new test problem (using a sensible name instead of ``new_branch``):

```console
git checkout -b new_branch
```

For more information about how to use branches, see this article:
https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token

Next, add the new test problem as a new folder to ``scr/cofitestsuite/newproblem`` (replacing ``newproblem`` with the new problem name) and add the Jupyter Notebook with the name ''newproblem.ipynb'' to the ``Jupyter Notebooks`` folder. 

Then use the following commands to upload the new test problem to the repository:

```console
git add .
git commit -m "A sentence that describes the new problem."
git push
```

The new test problem is now uploaded into the repository. As a last step, visit https://github.com/inlab-geo/inversion-test-problems/branches. Locate the new branch and click on "create pull request". If the changes are confined to the new folder and Jupyter Notebook, the new test problem will be quickly added to the repository and released to the pip package with the next update.

# How to create a sensible inversion test problem

The code can include files of any format, but it has to include a file called ``newproblem.py`` that acts as the front end of the new test problem. This file should include the following functions: 
- ``basic()``
- ``init_routine()``
- ``forward()``
- ``solver()``
- ``plot_model()``

These key functions can call any amount of sub-functions from within the file ``newproblem.py``, or from within the same folder.

important points to note on how to convert a locally working problem into one that works within the CoFI Test Suite package:

- We recommend to not use subfolders within ''newproblem''. All data and scripts will be in the same folder after comiling.

- If the code imports functions from the same folder that would usually be called by using ``import auxillaryfunction``, it is necessary to change the line to ``from cofitestsuite.newproblem import auxillaryfunction``.

- If data is included, then the correct path to where the pip package is installed has to be given. When giving the path, replace this: 
  ```console 
  np.load(''data.npz'')
  ```
  with the following, or your own version of it:

  ```console
  np.load(data_path(''data.npz''))
    
  def data_path(filename):
    path_to_current_file = os.path.realpath(__file__)
    current_directory = os.path.split(path_to_current_file)[0]
    data_path = os.path.join(current_directory, filename)
    return data_path
  ```

Additionally, we encourage you to add a Jupyter Notebook with an identical name into the folder ''Jupyter Notebooks`` that contains the following:

- An extensive description of the new inversion test problem, containing information about (but not limited to)...
  - the forward calculation (ie. the underlying physics) and how it was implemented.
  - which inversion method is used (and regularisation) and how it was implemented.
  - the physical unit of relevant variables, but at least of ``model`` and ``data``.
  - all changeable parameters, possibly in a list.

- An example of the new problem being used, with a reasonable output.



<!---  Binder does not work right now.   -->
<!---  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/inlab-geo/inversion-test-problems/HEAD)   -->

