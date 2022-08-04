import importlib
import os
import pkgutil

path_to_current_file = os.path.realpath(__file__)
current_directory = os.path.split(path_to_current_file)[0]
# current_directory=current_directory.replace("/", ".")
# current_directory=current_directory + ".Ex1_dir"
# print(current_directory)


try:
    from . import _version

    __version__ = _version.__version__
except ImportError:
    pass
# gravityforward = getattr(importlib.import_module(".gravityforward", package='inversiontestproblems'), "gravityforward")

for item in os.listdir(current_directory):
    if os.path.isdir(os.path.join(current_directory, item)):
        if not item == ".ipynb_checkpoints" and not item == "__pycache__":
            dirpath = "." + item
            globals()["%s" % item] = getattr(
                importlib.import_module(dirpath, package="inversiontestproblems"), item
            )
