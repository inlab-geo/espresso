"""Generate a new contribution package

This file uses the `copier` package to generate a new contribution package based on 
the `gh:scientific-python/cookie` template. Some Espresso-specific adpations are made
after using `copier`. The new contribution package will be created in the `contrib` 
folder of the Espresso repository.

Usage: python create_new_contrib.py <example_name>
"""

import sys
import os
import glob
import pathlib
import shutil
import copier
import tomlkit


SRC_COOKIE = "gh:scientific-python/cookie"
current_dir = pathlib.Path(__file__).resolve().parent
root_dir = current_dir.parent.parent
CONTRIB_FOLDER = str(root_dir / "contrib")
TEMPLATE_FOLDER = str(current_dir / "_template_files")


def validate_contrib_name():
    if len(sys.argv) != 2:
        raise ValueError(
            "No example name detected.\n\n\tUsage:\tpython create_new_contrib.py"
            " <example_name>\n\te.g.\tpython create_new_contrib.py my_example"
        )
    contrib_name = sys.argv[-1]
    existing_examples = glob.glob(CONTRIB_FOLDER + "/*/")
    existing_examples = [e.split("/")[-2] for e in existing_examples]
    if contrib_name in existing_examples:
        raise ValueError(
            "The example name provided already exists, please choose another name"
        )
    return contrib_name

def get_contrib_name_variants(contrib_name):
    return {
        "contrib_name": contrib_name, 
        "contrib-name": contrib_name.replace("_", "-"),
        "Contrib Name": contrib_name.title().replace("_", " "),
        "ContribName": contrib_name.title().replace("_", "")
    }

def generate_new_contrib_folder(contrib_names):
    with copier.Worker(
        src_path=SRC_COOKIE,
        dst_path=f"{CONTRIB_FOLDER}/{contrib_names["contrib_name"]}",
        unsafe=True,
        data={
            "project_name": contrib_names["contrib-name"], 
            "org": "inlab-geo",
            "url": "https://github.com/inlab-geo",
            "project_short_description": f"{contrib_names["Contrib Name"]} plugin for Espresso",
            "vcs": False
        },
        exclude=[
            ".copier-answers.yml",
            ".github",
            "docs",
            "tests",
            ".git_archival.txt",
            ".gitattributes",
            ".pre-commit-config.yaml",
            ".readthedocs.yaml",
            "noxfile.py",
            "README.md",
            f"src/{contrib_names['contrib_name']}/__init__.py",
            f"src/{contrib_names['contrib_name']}/py.typed"
        ]
    ) as worker:
        worker.run_copy()

def move_from_template(src_file_path, dst_file_path, contrib_names):
    with open(src_file_path, "r") as template_file:
        template_content = template_file.read()
    template_content = template_content.replace("example_name", contrib_names["contrib_name"])
    template_content = template_content.replace("Example Name Title", contrib_names["Contrib Name"])
    template_content = template_content.replace("ExampleName", contrib_names["ContribName"])
    with open(dst_file_path, "w") as new_file:
        new_file.write(template_content)

def adapt_from_template(src_rel_path, dst_rel_path, contrib_names):
    src_abs_path = f"{TEMPLATE_FOLDER}/{src_rel_path}"
    dst_abs_path = f"{CONTRIB_FOLDER}/{contrib_names["contrib_name"]}/{dst_rel_path}"
    move_from_template(src_abs_path, dst_abs_path, contrib_names)
    copier.tools.printf("generate", dst_rel_path, style=copier.tools.Style.OK)

def generate_new_contrib_readme(contrib_names):
    adapt_from_template("README.md", "README.md", contrib_names)

def generate_new_contrib_src_pyfiles(contrib_names):
    adapt_from_template("__init__.py", f"src/{contrib_names['contrib_name']}/__init__.py", contrib_names)
    adapt_from_template("example_name.py", f"src/{contrib_names['contrib_name']}/{contrib_names['contrib_name']}.py", contrib_names)

def generate_new_contrib_data(contrib_names):
    data_rel_path = f"src/{contrib_names['contrib_name']}/data"
    data_folder = f"{CONTRIB_FOLDER}/{contrib_names["contrib_name"]}/{data_rel_path}"
    os.makedirs(data_folder)
    shutil.copyfile(f"{TEMPLATE_FOLDER}/my_data.txt", f"{data_folder}/my_data.txt")
    copier.tools.printf("generate", data_rel_path, style=copier.tools.Style.OK)

def update_new_contrib_pyproject(contrib_names):
    pyproject_path = f"{CONTRIB_FOLDER}/{contrib_names["contrib_name"]}/pyproject.toml"
    with open(pyproject_path, "r") as pyproject_file:
        pyproject_content = tomlkit.load(pyproject_file)
    # remove "Typing :: Typed" from classifiers
    pyproject_content["project"]["classifiers"] = [
        c for c in pyproject_content["project"]["classifiers"] if c != "Typing :: Typed"
    ]
    # remove things from project metadata
    pyproject_content["project"].pop("optional-dependencies")
    pyproject_content["project"].pop("urls")
    pyproject_content["tool"].pop("pytest")
    pyproject_content["tool"].pop("coverage")
    pyproject_content["tool"].pop("mypy")
    pyproject_content["tool"].pop("ruff")
    pyproject_content["tool"].pop("pylint")
    try:
        pyproject_content["tool"].pop("cibuildwheel")
    except KeyError:
        pass
    # write to file
    with open(pyproject_path, "w") as pyproject_file:
        tomlkit.dump(pyproject_content, pyproject_file)


def main():
    contrib_name = validate_contrib_name()
    contrib_names = get_contrib_name_variants(contrib_name)
    
    print(
        "🥰 Thanks for contributing! \nAnswer a few quick questions and we will be "
        "generating new contribution folder from the template...\n"
    )
    
    generate_new_contrib_folder(contrib_names)
    print("Adapting from Espresso template")
    generate_new_contrib_readme(contrib_names)
    generate_new_contrib_src_pyfiles(contrib_names)
    generate_new_contrib_data(contrib_names)
    update_new_contrib_pyproject(contrib_names)
    
    print(
        f"\n🎉 OK. Please navigate to {CONTRIB_FOLDER}/{contrib_name}/ to write your own example. "
    )


if __name__ == "__main__":
    main()
