from setuptools import setup, find_packages
import pathlib
# Dummy text
# Dummy text
# Dummy text

#setup(

    #name='testdummy',
    #version='0.0.1',
    #install_requires=[
        #'importlib-metadata; python_version == "3.8"',
    #],

    #packages=find_packages("contrib"),  # include all packages under src
    #package_dir={"": "contrib"},   # tell distutils packages are under src

    #package_data={
        ## If any package contains *.txt files, include them:
        #"": ["*.txt"],
        ## And include any *.dat files found in the "data" subdirectory
        ## of the "mypkg" package, also:
        #"": ["data/*.npz"],
    #}
#)



########################## VERSION ####################################################
_ROOT = pathlib.Path(__file__).parent
with open(str(_ROOT / "contrib" / "_version.py")) as f:
    for line in f:
        if line.startswith("__version__="):
            _, _, version = line.partition("=")
            VERSION = version.strip(" \n'\"")
            break
    else:
        raise RuntimeError("unable to read the version from contib/_version.py")


setup(
    name='inversion-test-problems',
    version=VERSION,
    install_requires=[
        'importlib-metadata; python_version == "3.8"',
    ],
    include_package_data=True,
    #packages=find_packages(exclude=("*.egg_info")),  
    package_dir={"inversiontestproblems": "contrib"},   # tell distutils packages are under src
)

