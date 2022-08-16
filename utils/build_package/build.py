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
current_directory = Path(__file__).parent
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
]


def clean_build_folder():
    dirpath = Path(BUILD_DIR)
    if dirpath.exists() and dirpath.is_dir():
        rmtree(dirpath)

def move_folder_content(folder_path, dest_path):
    copytree(
        folder_path, 
        dest_path, 
        dirs_exist_ok=True, 
        ignore=ignore_patterns('*.pyc', 'tmp*', '__pycache__')
    )

def move_pkg_source():
    move_folder_content(PKG_SRC, f"{BUILD_DIR}/src")

def move_pkg_metadata():
    move_folder_content(DOCS_SRC, f"{BUILD_DIR}/docs")
    for f in META_FILES:
        copy(f"{ROOT_DIR}/{f}", f"{BUILD_DIR}/{f}")

def move_contrib_source():
    move_folder_content(CONTRIB_SRC, f"{BUILD_DIR}/src/{PKG_NAME}")
    contribs = []
    init_file_imports = "\n"
    init_file_all_var = "\n__additional_all__ = [\n"
    for path in Path(CONTRIB_SRC).iterdir():
        if path.is_dir():
            contrib = os.path.basename(path)
            contrib_class = contrib.title().replace("_", "")
            contribs.append(contrib)
            init_file_imports += f"from .{contrib} import {contrib_class}\n"
            init_file_all_var += f"\t'{contrib_class}',\n"
    init_file_all_var += "]\n__all__.append(__additional_all__)"
    with open(f"{BUILD_DIR}/src/{PKG_NAME}/CMakeLists.txt", "a") as f:
        for contrib in contribs:
            f.write(f"install(DIRECTORY {contrib} DESTINATION .)")
    with open(f"{BUILD_DIR}/src/{PKG_NAME}/__init__.py", "a") as f:
        f.write(init_file_imports)
        f.write(init_file_all_var)

def install_pkg():
    return subprocess.call([sys.executable, "-m", "pip", "install", "."], cwd=BUILD_DIR)

def main():
    print("🛠  Package building...")
    #
    print("\n🗂  Cleaning build folder...")
    clean_build_folder()
    print("OK.")
    #
    print("\n🗂  Moving Espresso core packaging files...")
    move_pkg_source()
    print("OK.")
    #
    print("\n🗂  Moving package metadata...")
    move_pkg_metadata()
    print("OK.")
    #
    print("\n🗂  Moving all contributions...")
    move_contrib_source()
    print("OK.")
    #
    print("\n🗂  Building Python package: Espresso..")
    exit_code = install_pkg()
    if exit_code == 0: print("OK.")
    
    return exit_code

if __name__ == "__main__":
    main()
