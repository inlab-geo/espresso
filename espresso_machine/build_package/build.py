"""Build the Python package "espresso"

1. clean "_esp_build/"
2. "/<meta-data-files>" => "_esp_build/"
3. "src/" => "_esp_build/src/"
4. "contrib/" => "_esp_build/src/espresso/" + "__init__.py" + "list_problems.py"
5. "_version.py" => "_esp_build/pyproject.toml"
6. "espresso_machine/" => _esp_build/src/_machine"
6. `pip install .`

"""

import subprocess
import sys
import os
import argparse
import pytest
from shutil import copytree, copy, rmtree, ignore_patterns
from pathlib import Path
import versioningit


# ------------------------ constants ------------------------
PKG_NAME = "espresso"
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
    ".readthedocs.yml",
    ".gitignore",
    "CHANGELOG.md",
]

# ------------------------ argument parser ------------------------
def setup_parser():
    parser = argparse.ArgumentParser(
        description="Script to build Espresso, with/without pre/post-build validation"
    )
    parser.add_argument(
        "--validate", "-v", "--checks", dest="validate", action="store_true", 
        default=False, help="Run tests before and after building the package")
    return parser

args = setup_parser().parse_args()


# ------------------------ helpers ------------------------
def is_cache(file_name):
    return file_name.endswith(".pyc") or \
        file_name == "__pycache__" or \
            file_name == "cmake_install.cmake" or \
                file_name.endswith(".mod") or \
                    file_name.endswith(".out") or \
                        file_name == "CMakeFiles" or \
                            file_name == "Makefile"

def move_folder_content(folder_path, dest_path, prefix=None):
    if prefix is None:
        copytree(
            folder_path, 
            dest_path, 
            # dirs_exist_ok=True, 
            ignore=ignore_patterns('*.pyc', 'tmp*', '__pycache__')
         )
    else:
        for f in os.listdir(folder_path):
            if is_cache(f):
                continue
            src = f"{folder_path}/{f}"
            dst = f"{dest_path}/{prefix}{f}"
            copytree(src, dst, 
                ignore=ignore_patterns(
                    "*.pyc", "__pycache__", "tmp*", "CMakeFiles", "Makefile", "*.mod", "*.out"
                ))
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
    move_folder_content(DOCS_SRC, f"{BUILD_DIR}/docs")
    for f in META_FILES:
        copy(f"{ROOT_DIR}/{f}", f"{BUILD_DIR}/{f}")

# 3
def move_pkg_source():
    move_folder_content(PKG_SRC, f"{BUILD_DIR}/src")

# 4
def move_contrib_source():
    # move all contribution subfolders with prefix "_"
    move_folder_content(CONTRIB_SRC, f"{BUILD_DIR}/src/{PKG_NAME}", prefix="_")
    # collect a list of contributions + related strings to write later
    contribs = []
    init_file_imports = "\n"
    init_file_all_nms = "\n_all_problem_names = [\n"
    init_file_all_cls = "\n_all_problems = [\n"
    for path in Path(CONTRIB_SRC).iterdir():
        if path.is_dir():
            contrib = os.path.basename(path)                    # name
            contrib_class = contrib.title().replace("_", "")    # class
            contribs.append(contrib)
            init_file_imports += f"from ._{contrib} import {contrib_class}\n"
            init_file_all_nms += f"\t'{contrib_class}',\n"
            init_file_all_cls += f"\t{contrib_class},\n"
    init_file_all_nms += "]"
    init_file_all_cls += "]"
    # some constant strings to append to init file later
    init_file_imp_funcs = "\nfrom .list_problems import list_problem_names, list_problems"
    init_file_add_all_nms = "\n__all__ += list_problem_names()"
    init_file_add_funcs = "\n__all__ += ['list_problem_names', 'list_problems']\n"
    # write all above to files
    with open(f"{BUILD_DIR}/src/{PKG_NAME}/__init__.py", "a") as f:
        f.write(init_file_imports)
        f.write(init_file_imp_funcs)
        f.write(init_file_add_all_nms)
        f.write(init_file_add_funcs)
    with open(f"{BUILD_DIR}/src/{PKG_NAME}/list_problems.py", "a") as f:
        f.write(init_file_imports)
        f.write(init_file_all_nms)
        f.write(init_file_all_cls)

# 5
def write_version():
    # get version
    versioningit_config = {
        "format": {
            "distance": "{base_version}+{distance}.{vcs}{rev}",
            "dirty": "{base_version}+{distance}.{vcs}{rev}.dirty",
            "distance-dirty": "{base_version}+{distance}.{vcs}{rev}.dirty",
        },
        "write": {
            "file": "src/espresso/_version.py"
        }
    }
    version = versioningit.get_version(root, versioningit_config, True)
    # --> read pyproject.toml
    with open(f"{BUILD_DIR}/pyproject.toml", "r") as f:
        file_content = f.read()
    # --> process pyproject.toml
    # write version to pyproject.toml (for local build)
    file_content += "\n[tool.versioningit]"
    file_content += f"\ndefault-version = '{version}'\n"
    # change versioningit configs (for build from branch "esp_build")
    file_content = file_content.replace(".core", "")
    file_content += "\n[tool.versioningit.tag2version]"
    file_content += "\nrmprefix = 'v'"
    file_content += "\nrmsuffix = '-build'\n"
    # --> write pyproject.toml
    with open(f"{BUILD_DIR}/pyproject.toml", "w") as f:
        f.write(file_content)

# 6 move espresso_machine into espresso/_machine
def move_espresso_machine():
    move_folder_content(MACHINE_SRC, f"{BUILD_DIR}/src/espresso/_machine")

# 7
def install_pkg():
    subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y", PKG_NAME])
    return subprocess.call([sys.executable, "-m", "pip", "install", "."], cwd=BUILD_DIR)

# printing helper
def println_with_emoji(content, emoji):
    try:
        print(f"\n{emoji}  {content}")
    except:
        print(f"\n{content}")


# ------------------------ main functions ------------------------
def build():
    println_with_emoji("Package building...", "🛠")
    # 1
    println_with_emoji("Cleaning build folder...", "🗂")
    clean_build_folder()
    print("OK.")
    # 2
    println_with_emoji("Moving package metadata...", "🗂")
    move_pkg_metadata()
    print("OK.")
    # 3
    println_with_emoji("Moving Espresso core packaging files...", "🗂")
    move_pkg_source()
    print("OK.")
    # 4
    println_with_emoji("Moving all contributions...", "🗂")
    move_contrib_source()
    print("OK.")
    # 5
    println_with_emoji("Generating version file...", "🗂")
    write_version()
    print("OK.")
    # 6
    println_with_emoji("Moving infrastructure code...", "🗂")
    move_espresso_machine()
    print("OK.")
    # 6
    println_with_emoji("Building Python package: geo-espresso...", "🗂")
    exit_code = install_pkg()
    if exit_code == 0: 
        println_with_emoji("Espresso installed!", "🍰")
    return exit_code

def build_with_validate():
    validate_script = str(Path(__file__).resolve().parent / "validate.py")

    # pre-build validate
    exit_code = subprocess.call([sys.executable, validate_script, "--pre"])
    if exit_code != pytest.ExitCode.OK:
        sys.exit(exit_code)

    # build package
    build()

    # post-build validation
    exit_code = subprocess.call([sys.executable, validate_script, "--post"])
    if exit_code != pytest.ExitCode.OK:
        sys.exit(exit_code)
    else:
        print("\n🍰 All done 🍰")
        sys.exit(exit_code)


def main():
    if args.validate:
        build_with_validate()
    else:
        build()

if __name__ == "__main__":
    sys.exit(main())
