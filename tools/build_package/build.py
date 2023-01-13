"""Build the Python package "cofi_espresso"

1. Create clean folder "_esp_build"
2. Move all files under "src/" into "_esp_build/src/"
3. Move all files under "contrib/" into "_esp_build/src/cofi_espresso/"
4. Move all files under ".git/" into "_esp_build/.git/"
4. Add all contribution's names into "_esp_build/src/cofi_espresso/CMakeLists.txt"
5. Add all contribution's class names into "_esp_build/src/cofi_espresso/__init__.py"
6. Build the package with "pip install ."
7. Test running the workflow again with installed package

"""

import subprocess
import sys
import os
from shutil import copytree, copy, rmtree, ignore_patterns
from pathlib import Path
import versioningit


PKG_NAME = "cofi_espresso"
current_directory = Path(__file__).resolve().parent
root = current_directory.parent.parent
ROOT_DIR = str(root)
BUILD_DIR = str(root / "_esp_build")
PKG_SRC = str(root / "src")
CONTRIB_SRC = str(root / "contrib")
VCS_GIT = str(root / ".git")
DOCS_SRC = str(root / "docs")
META_FILES = [
    "README.md",
    "setup.py",
    "pyproject.toml",
    "CMakeLists.txt",
    "LICENCE",
    ".readthedocs.yml",
    ".gitignore",
    "CHANGELOG.md",
]


def clean_build_folder():
    dirpath = Path(BUILD_DIR)
    if dirpath.exists() and dirpath.is_dir():
        rmtree(dirpath)

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
            dirs_exist_ok=True, 
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

def gen_version_file():
    _ROOT = Path(__file__).resolve().parent
    versioningit_config = {
        "format": {
            "distance": "{base_version}+{distance}.{vcs}{rev}",
            "dirty": "{base_version}+{distance}.{vcs}{rev}.dirty",
            "distance-dirty": "{base_version}+{distance}.{vcs}{rev}.dirty",
        },
        "write": {
            "file": "../../src/cofi_espresso/_version.py"
        }
    }
    versioningit.get_version(_ROOT, versioningit_config, True)

def move_pkg_source():
    move_folder_content(PKG_SRC, f"{BUILD_DIR}/src")

def move_pkg_metadata():
    move_folder_content(DOCS_SRC, f"{BUILD_DIR}/docs")
    for f in META_FILES:
        copy(f"{ROOT_DIR}/{f}", f"{BUILD_DIR}/{f}")

def move_contrib_source():
    move_folder_content(CONTRIB_SRC, f"{BUILD_DIR}/src/{PKG_NAME}", prefix="_")
    contribs = []
    init_file_imports = "\n"
    init_file_all_nms = "\n_all_problem_names = [\n"
    init_file_all_cls = "\n_all_problems = [\n"
    for path in Path(CONTRIB_SRC).iterdir():
        if path.is_dir():
            contrib = os.path.basename(path)
            contrib_class = contrib.title().replace("_", "")
            contribs.append(contrib)
            init_file_imports += f"from ._{contrib} import {contrib_class}\n"
            init_file_all_nms += f"\t'{contrib_class}',\n"
            init_file_all_cls += f"\t{contrib_class},\n"
    init_file_all_nms += "]"
    init_file_all_cls += "]"
    init_file_imp_funcs = "\nfrom .list_problems import list_problem_names, list_problems"
    init_file_add_all_nms = "\n__all__ += list_problem_names()"
    init_file_add_funcs = "\n__all__ += ['list_problem_names', 'list_problems']\n"
    with open(f"{BUILD_DIR}/src/{PKG_NAME}/CMakeLists.txt", "a") as f:
        for contrib in contribs:
            f.write(f"install(DIRECTORY _{contrib} DESTINATION .)\n")
            if Path(f"{CONTRIB_SRC}/{contrib}/CMakeLists.txt").exists():
                f.write(f"add_subdirectory(_{contrib})\n")
    with open(f"{BUILD_DIR}/src/{PKG_NAME}/__init__.py", "a") as f:
        f.write(init_file_imports)
        f.write(init_file_imp_funcs)
        f.write(init_file_add_all_nms)
        f.write(init_file_add_funcs)
    with open(f"{BUILD_DIR}/src/{PKG_NAME}/list_problems.py", "a") as f:
        f.write(init_file_imports)
        f.write(init_file_all_nms)
        f.write(init_file_all_cls)

# def move_vcs_files():
#     move_folder_content(VCS_GIT, f"{BUILD_DIR}/.git")

def install_pkg():
    subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y", PKG_NAME])
    return subprocess.call([sys.executable, "-m", "pip", "install", "."], cwd=BUILD_DIR)

def main():
    println_with_emoji("Package building...", "🛠")
    #
    println_with_emoji("Cleaning build folder...", "🗂")
    clean_build_folder()
    print("OK.")
    #
    println_with_emoji("Generating version file...", "🗂")
    gen_version_file()
    print("OK.")
    #
    println_with_emoji("Moving Espresso core packaging files...", "🗂")
    move_pkg_source()
    print("OK.")
    #
    println_with_emoji("Moving package metadata...", "🗂")
    move_pkg_metadata()
    print("OK.")
    #
    println_with_emoji("Moving all contributions...", "🗂")
    move_contrib_source()
    print("OK.")
    #
    println_with_emoji("Building Python package: cofi-espresso...", "🗂")
    exit_code = install_pkg()
    if exit_code == 0: 
        println_with_emoji("Espresso installed!", "🍰")
    return exit_code

def println_with_emoji(content, emoji):
    try:
        print(f"\n{emoji}  {content}")
    except:
        print(f"\n{content}")

if __name__ == "__main__":
    sys.exit(main())
