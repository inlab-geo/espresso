"""Build the Python package "espresso"

1. clean `_esp_build/`
2. `/<meta-info-files>` => `_esp_build/`
3. `src/` => `_esp_build/src/`
4. generate `_version.py` => `_esp_build/src/espresso/_version.py`
5. `contrib/` => `_esp_build/src/espresso/`;
   `__init__.py` => `_esp_build/src/__init__.py`; 
   `capabilities.py` => `_esp_build/src/capabilities.py`
6. `espresso_machine/` => `_esp_build/src/_machine`
7. remove `_esp_build/src/_machine/versioning/__init__.py` and
   rename `_esp_build/src/_machine/versioning/versioning_for_full_package.py` to `__init__.py`
8. build capability_matrix
9. `pip install .`      (can be disabled by `--no-install`)

Usage: python build.py [--pre] [--post] [--no-install] [-c <example_name>] [--file <file_name>]
"""

import subprocess
import sys
import os
from shutil import copytree, copy, rmtree, ignore_patterns
from pathlib import Path
import versioningit
import json

import _utils
import report
import validate


# ------------------------ constants ------------------------
MODULE_NAME = "espresso"
PKG_NAME = "geo-espresso"
current_directory = Path(__file__).resolve().parent
root = current_directory.parent.parent
ROOT_DIR = str(root)
BUILD_DIR = str(root / "_esp_build")
PKG_SRC = str(root / "src")
CONTRIB_SRC = str(root / "contrib")
VCS_GIT = str(root / ".git")
DOCS_SRC = str(root / "docs")
MACHINE_SRC = str(root / "espresso_machine")
META_FILES = [
    "README.md",
    "pyproject.toml",
    "LICENCE",
    "CMakeLists.txt",
    ".readthedocs.yml",
    ".gitignore",
    "CHANGELOG.md",
]
PROBLEMS_TO_COMPILE_FILE = "problems_to_compile.txt"
validate_script = str(Path(__file__).resolve().parent / "validate.py")


# ------------------------ helpers ------------------------
def is_cache(file_name):
    return (
        file_name.endswith(".pyc")
        or file_name == "__pycache__"
        or file_name == "cmake_install.cmake"
        or file_name.endswith(".mod")
        or file_name.endswith(".out")
        or file_name == "CMakeFiles"
        or file_name == "Makefile"
        or file_name == PROBLEMS_TO_COMPILE_FILE
    )


def move_folder_content(folder_path, dest_path, prefix=None, only_include=None):
    if prefix is None:
        copytree(
            folder_path,
            dest_path,
            # dirs_exist_ok=True,
            ignore=ignore_patterns("*.pyc", "tmp*", "__pycache__"),
        )
    else:  # moving contributions source
        for f in os.listdir(folder_path):
            src = f"{folder_path}/{f}"
            dst = f"{dest_path}/{prefix}{f}"
            if (
                is_cache(f)
                or not os.path.isdir(src)
                or (only_include is not None and f not in only_include)
            ):
                continue
            copytree(
                src,
                dst,
                ignore=ignore_patterns(
                    "*.pyc",
                    "__pycache__",
                    "tmp*",
                    "CMakeFiles",
                    "Makefile",
                    "*.mod",
                    "*.out",
                ),
            )
            # add underscore prefix to file name
            for ff in os.listdir(dst):
                if ff == f"{f}.py":
                    ff_origin = f"{dst}/{ff}"
                    ff_rename = f"{dst}/{prefix}{ff}"
                    os.rename(ff_origin, ff_rename)
                if ff == "__init__.py" or ff == "CMakeLists.txt":
                    with open(f"{dst}/{ff}", "r") as fff:
                        lines = fff.readlines()
                    with open(f"{dst}/{ff}", "w") as fff:
                        for line in lines:
                            fff.write(line.replace(f, f"_{f}"))


# ------------------------ tasks ------------------------
# 1
def clean_build_folder():
    dirpath = Path(BUILD_DIR)
    if dirpath.exists() and dirpath.is_dir():
        rmtree(dirpath)


# 2
def move_pkg_metadata():
    def _fix_gitignore():
        # find and remove line with src/espresso/_version.py
        fname = f"{BUILD_DIR}/.gitignore"
        with open(fname, "r") as file:
            lines = file.readlines()
        with open(fname, "w") as file:
            for line in lines:
                if "src/espresso/_version.py" not in line:
                    file.write(line)

    move_folder_content(DOCS_SRC, f"{BUILD_DIR}/docs")
    for f in META_FILES:
        copy(f"{ROOT_DIR}/{f}", f"{BUILD_DIR}/{f}")
    _fix_gitignore()


# 3
def move_pkg_source():
    move_folder_content(PKG_SRC, f"{BUILD_DIR}/src")


# 4
def write_version():
    versioningit_config = {
        "format": {
            "distance": "{base_version}+{distance}.{vcs}{rev}",
            "dirty": "{base_version}+{distance}.{vcs}{rev}.dirty",
            "distance-dirty": "{base_version}+{distance}.{vcs}{rev}.dirty",
        },
        "write": {"file": "_esp_build/src/espresso/_version.py"},
    }
    versioningit.get_version(root, versioningit_config, True)


# 5
def move_contrib_source():
    # see if any contribution is specified through command line args
    specified_problems = _utils.problems_to_run_names_only()
    # move all contribution subfolders with prefix "_"
    move_folder_content(
        CONTRIB_SRC,
        f"{BUILD_DIR}/src/{MODULE_NAME}",
        prefix="_",
        only_include=specified_problems,
    )
    # collect a list of contributions + related strings to write later
    contribs = []
    init_file_imports = "\n"
    init_file_all_cls = "\n_all_problems = [\n"
    init_file_deletes = "\n"
    for path in Path(CONTRIB_SRC).iterdir():
        contrib = os.path.basename(path)  # name
        if path.is_dir() and (
            specified_problems is None or contrib in specified_problems
        ):
            contrib_class = _utils.problem_name_to_class(contrib)  # class
            contribs.append(contrib)
            init_file_imports += f"from ._{contrib} import {contrib_class}\n"
            init_file_all_cls += f"    {contrib_class},\n"
            init_file_deletes += f"del {contrib_class}\n"
    init_file_all_cls += "]"
    # some constant strings to append to init file later
    init_file_imp_funcs = (
        "\nfrom .capabilities import list_problem_names, list_problems,"
        " list_capabilities\n"
    )
    init_file_add_all_nms = "\n__all__ += list_problem_names()"
    init_file_add_funcs = (
        "\n__all__ += ['list_problem_names', 'list_problems', 'list_capabilities']\n"
    )
    # write all above to files
    # compiled_code_list = set()
    with open(f"{BUILD_DIR}/src/{MODULE_NAME}/CMakeLists.txt", "a") as f:
        f.write(f"install(DIRECTORY _machine DESTINATION espresso)\n")
        f.write(f"install(FILES capabilities.py DESTINATION espresso)\n")
        for contrib in contribs:
            f.write(f"install(DIRECTORY _{contrib} DESTINATION espresso)\n")
            if Path(f"{CONTRIB_SRC}/{contrib}/CMakeLists.txt").exists():
                f.write(f"add_subdirectory(_{contrib})\n")
    with open(f"{BUILD_DIR}/src/{MODULE_NAME}/__init__.py", "a") as f:
        f.write(init_file_imports)
        f.write(init_file_imp_funcs)
        f.write(init_file_add_all_nms)
        f.write(init_file_add_funcs)
    with open(f"{BUILD_DIR}/src/{MODULE_NAME}/capabilities.py", "a") as f:
        f.write(init_file_imports)
        f.write(init_file_all_cls)
        f.write(init_file_deletes)


# 6 move espresso_machine into espresso/_machine
def move_espresso_machine():
    move_folder_content(MACHINE_SRC, f"{BUILD_DIR}/src/espresso/_machine")


# 7 remove versioning/__init__.py and rename versioning_for_full_package.py to __init__.py
def change_versioningit_config():
    os.remove(f"{BUILD_DIR}/src/espresso/_machine/versioning/__init__.py")
    os.rename(
        f"{BUILD_DIR}/src/espresso/_machine/versioning/versioning_for_full_package.py",
        f"{BUILD_DIR}/src/espresso/_machine/versioning/__init__.py",
    )
    # change scikit-build metadata local plugin path
    with open(f"{BUILD_DIR}/pyproject.toml", "r") as f:
        lines = f.readlines()
    # replace `provider-path = "espresso_machine"` with `provider-path = "src/espresso/_machine`
    with open(f"{BUILD_DIR}/pyproject.toml", "w") as f:
        for line in lines:
            f.write(line.replace(
                "provider-path = \"espresso_machine\"", "provider-path = \"src/espresso/_machine\""
            ))


# 8 build capability matrix
def build_problem_capability():
    # see if any contribution is specified through command line args
    specified_problems = _utils.problems_to_run_names_only()
    with _utils.suppress_stdout():
        capability_report = report.capability_report(
            problems_to_check=specified_problems
        )
    report_to_write = json.dumps(capability_report, indent=4)
    with open(f"{BUILD_DIR}/src/{MODULE_NAME}/capabilities.py", "a") as f:
        f.write("\n\n_capability_matrix = ")
        f.write(report_to_write)


# printing helper
def println_with_emoji(content, emoji):
    try:
        print(f"\n{emoji} {content}")
    except:
        print(f"\n{content}")


# ------------------------ main functions ------------------------
build_pipeline = [
    (clean_build_folder, "Cleaning build folder..."),
    (move_pkg_metadata, "Moving package metadata..."),
    (move_pkg_source, "Moving Espresso core packaging files..."),
    (write_version, "Generating version file..."),
    (move_contrib_source, "Moving all contributions..."),
    (move_espresso_machine, "Moving infrastructure code..."),
    (change_versioningit_config, "Changing versioningit configurations..."),
    (
        build_problem_capability,
        "Building capability matrix... (this will take some time)",
    ),
]


def build():
    println_with_emoji("Package building...", "🛠")
    for step, desc in build_pipeline:
        println_with_emoji(desc, "🗂")
        try:
            step()
        except Exception as e:
            println_with_emoji(f"Problem occurred in this step: {desc}\n", "❗️")
            raise e
        print("OK.")
    # println_with_emoji("Espresso installed!", "🍰")


def install_pkg():
    desc = "Building Python package: geo-espresso..."
    println_with_emoji(desc, "🗂")
    subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y", PKG_NAME])
    res = subprocess.call([sys.executable, "-m", "pip", "install", "."], cwd=BUILD_DIR)
    if res != 0:
        sys.exit(res)


def pre_validate():
    validate.main(pre_build=True)


def post_validate():
    validate.main(pre_build=False)


def main():
    _args = _utils.args()
    if _args.pre:
        pre_validate()
    build()
    if not _args.dont_install:
        install_pkg()
    if _args.post:
        post_validate()
    println_with_emoji("All done", "🍰")


if __name__ == "__main__":
    main()
