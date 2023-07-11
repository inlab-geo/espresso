########################## LIBRARY IMPORT #############################################
import io
import sys
import pathlib
import setuptools
import versioningit

try:
    import skbuild
except ImportError:
    print(
        "Please update pip, you need pip 10 or greater,\n"
        " or you need to install the PEP 518 requirements in pyproject.toml yourself",
        file=sys.stderr,
    )
    raise


_ROOT = pathlib.Path(__file__).resolve().parent

########################## VERSION ####################################################
# generate
versioningit_config = {
    "format": {
        "distance": "{base_version}+{distance}.{vcs}{rev}",
        "dirty": "{base_version}+{distance}.{vcs}{rev}.dirty",
        "distance-dirty": "{base_version}+{distance}.{vcs}{rev}.dirty",
    },
    "write": {
        "file": "src/espresso/_version.py"
    },
    "tag2version": {
        "rmprefix": "v",
        "rmsuffix": "-build",
    }
}
try:            # generate _version.py when building the core package
    versioningit.get_version(_ROOT, versioningit_config, True)
except:
    pass        # VCS not found, there should be a version file to read

# read from _version.py
with open(str(_ROOT / "src" / "espresso" / "_version.py")) as f:
    for line in f:
        if line.startswith("__version__ = "):
            _, _, version = line.partition("=")
            VERSION = version.strip(" \n'\"")
            break
    else:
        raise RuntimeError("unable to read the version from src/espresso/_version.py")


########################## LONG DESCRIPTION ###########################################
LONG_DESCRIPTION = (_ROOT / "README.md").read_text()
CONTENT_TYPE = "text/markdown"


########################## CLEAN CACHE ################################################
from shutil import rmtree
_skbuild_dir = _ROOT / "_skbuild"
if _skbuild_dir.exists() and _skbuild_dir.is_dir():
    rmtree(_skbuild_dir)


########################## OTHER METADATA #############################################
PACKAGE_NAME = "geo-espresso"
AUTHOR = f"InLab, Espresso development team"
DESCRIPTION = "Earth Science PRoblems for the Evaluation of Strategies, Solvers and Optimizers"
KEYWORDS = ["inversion", "inference", "python package", "geoscience", "geophysics"]
CLASSIFIERS = [
    "Development Status :: 2 - Pre-Alpha",
    "Intended Audience :: Education",
    "Intended Audience :: Science/Research",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: Implementation :: CPython",
    "Programming Language :: C",
    "Programming Language :: Fortran",
    "Topic :: Scientific/Engineering :: Physics",
    "License :: OSI Approved :: BSD License",
    "License :: Other/Proprietary License",
]
PACKAGE_DIR = {"": "src"}
PACKAGES = setuptools.find_namespace_packages(where='src')
CMAKE_INSTALL_DIR = f"src/espresso"
CMAKE_ARGS = ['-DSKBUILD=ON']
PYTHON_REQUIRES = ">=3.7"
INSTALL_REQUIRES = [
    "numpy>=1.18",
    "scipy>=1.0.0",
    "matplotlib>=3.1",
    "tqdm>=4.0",
    "mpmath>=1.2.0",
    "cython",
    "seislib>=0.6.13"
]


########################## SETUP ######################################################
class NullIO(io.StringIO):
    def write(self, txt):
       pass
sys.stdout = NullIO()
sys.tracebacklimit = 0

try:
    skbuild.setup(
        name=PACKAGE_NAME,
        version=VERSION,
        author=AUTHOR,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        long_description_content_type=CONTENT_TYPE,
        keywords=KEYWORDS,
        classifiers=CLASSIFIERS,
        package_dir=PACKAGE_DIR,
        packages=PACKAGES,
        include_package_data=True,
        cmake_install_dir=CMAKE_INSTALL_DIR,
        cmake_args=CMAKE_ARGS,
        python_requires=PYTHON_REQUIRES,
        install_requires=INSTALL_REQUIRES,
    )
except SystemExit as e:
    skbuild_error : skbuild.exceptions.SKBuildError = e.args[0]
    error_message = skbuild_error.args[0]
    error_message += "\n\nHint: search 'error' in current terminal session to find out the details. "
    # error_message += f"Here are some possible reasons that cause failure in building `{PACKAGE_NAME}`:"
    # error_message += "\n\t1. no Fortran compiler found -> install a Fortran compiler and ensure it's included in the path"
    error_message += "\n\n"
    # error_message += "If the error is not due to above reasons, "
    # error_message += "please feel free to lodge an issue " \
                    #  f"at https://github.com/inlab-geo/{PACKAGE_NAME}/issues for help\n"
    skbuild_error.args = (error_message,)
    sys.exit(skbuild_error)
