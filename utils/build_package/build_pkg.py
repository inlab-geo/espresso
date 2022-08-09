"""Build the Python package "espresso"

1. Create clean folder "_esp_build"
2. Move all files under "utils/build_package/package_src/" into "_esp_build/"
3. Move all files under "contrib/" into "_esp_build/src/espresso/"
4. Add all contribution's name into "_esp_build/src/espresso/CMakeLists.txt"
4. Build the package with "pip install ."
5. Test running the workflow again with installed package

"""

import subprocess
import sys
from shutil import copytree, copy, rmtree, ignore_patterns
from pathlib import Path


BUILD_FOLDER = "_esp_build"
PKG_NAME = "espresso"
PKG_SRC = "utils/build_package/_package_src"
CONTRIB_SRC = "contrib"
DOCS_SRC = "docs"


def clean_build_folder():
    dirpath = Path(BUILD_FOLDER)
    if dirpath.exists() and dirpath.is_dir():
        rmtree(dirpath)

def move_folder_content(folder_path, dest_path):
    copytree(
        folder_path, 
        dest_path, 
        dirs_exist_ok=True, 
        ignore=ignore_patterns('*.pyc', 'tmp*')
    )

def move_pkg_source():
    move_folder_content(PKG_SRC, BUILD_FOLDER)

def move_pkg_metadata():
    move_folder_content(DOCS_SRC, f"{BUILD_FOLDER}/{DOCS_SRC}")
    copy("README.md", f"{BUILD_FOLDER}/README.md")
    copy("_version.py", f"{BUILD_FOLDER}/src/{PKG_NAME}/_version.py")

def move_contrib_source():
    move_folder_content(CONTRIB_SRC, f"{BUILD_FOLDER}/src/{PKG_NAME}")


def install_pkg():
    subprocess.call([sys.executable, "-m", "pip", "install", "."], cwd=BUILD_FOLDER)

if __name__ == "__main__":
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
    install_pkg()
    print("OK.")
