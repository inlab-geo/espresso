"""Build the Python package "cofi_espresso"

1. Create clean folder "_esp_build"
2. Move all files under "src/" into "_esp_build/src/"
3. Move all files under "contrib/" into "_esp_build/src/cofi_espresso/"
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


PKG_NAME = "cofi_espresso"
current_directory = Path(__file__).resolve().parent
root = current_directory.parent.parent
ROOT_DIR = str(root)
BUILD_DIR = str(root / "_esp_build")
PKG_SRC = str(root / "src")
CONTRIB_SRC = str(root / "contrib")
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
            if f.endswith(".pyc") or f.startswith("tmp") or f == "__pycache__":
                continue
            src = f"{folder_path}/{f}"
            dst = f"{dest_path}/{prefix}{f}"
            copytree(src, dst, dirs_exist_ok=True)

def move_pkg_source():
    move_folder_content(PKG_SRC, f"{BUILD_DIR}/src")

def move_pkg_metadata():
    move_folder_content(DOCS_SRC, f"{BUILD_DIR}/docs")
    for f in META_FILES:
        copy(f"{ROOT_DIR}/{f}", f"{BUILD_DIR}/{f}")

def move_contrib_source():
    move_folder_content(CONTRIB_SRC, f"{BUILD_DIR}/src/{PKG_NAME}", prefix="_")
    contribs = []
    init_file_imports = "\nimportlib = __import__('importlib')\n"
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
            if Path(f"{CONTRIB_SRC}/_{contrib}/CMakeLists.txt").exists():
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

def install_pkg():
    subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y", PKG_NAME])
    return subprocess.call([sys.executable, "-m", "pip", "install", "."], cwd=BUILD_DIR)

def main():
    print_with_emoji("🛠  Package building...", "\nPackage building...")
    #
    print_with_emoji("\n🗂  Cleaning build folder...", "\nCleaning build folder...")
    clean_build_folder()
    print("OK.")
    #
    print_with_emoji("\n🗂  Moving Espresso core packaging files...", "\nMoving Espresso core packaging files...")
    move_pkg_source()
    print("OK.")
    #
    print_with_emoji("\n🗂  Moving package metadata...", "\nMoving package metadata...")
    move_pkg_metadata()
    print("OK.")
    #
    print_with_emoji("\n🗂  Moving all contributions...", "\nMoving all contributions...")
    move_contrib_source()
    print("OK.")
    #
    print_with_emoji("\n🗂  Building Python package: cofi-espresso...", "\nBuilding Python package: cofi-espresso...")
    exit_code = install_pkg()
    if exit_code == 0: 
        print_with_emoji("🍰 Espresso installed 🍰", "Espresso installed")
    
    return exit_code

def print_with_emoji(content, alt):
    try:
        print(content)
    except:
        print(alt)

if __name__ == "__main__":
    sys.exit(main())
