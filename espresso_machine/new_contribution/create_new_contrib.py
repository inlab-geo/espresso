"""Generate a new contribution package

Usage: python create_new_contrib.py <example_name>
"""

import sys
import glob
import pathlib
import copier


SRC_COOKIE = "gh:scientific-python/cookie"
current_dir = pathlib.Path(__file__).resolve().parent
root_dir = current_dir.parent.parent
CONTRIB_FOLDER = str(root_dir / "contrib")


def validate_contrib_name():
    if len(sys.argv) != 2:
        raise RuntimeError(
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
            ".github",
            "docs",
            "tests",
            ".git_archival.txt",
            ".gitattributes",
            "pre-commit-config.yaml",
            ".readthedocs.yaml",
            "noxfile.py",
            "README.md",
            f"src/{contrib_names['contrib_name']}/py.typed"
        ]
    ) as worker:
        worker.run_copy()

def generate_new_contrib_readme(contrib_names):
    pass

def generate_new_contrib_src_pyfile(contrib_names):
    pass

def update_new_contrib_pyproject(contrib_names):
    pass


def main():
    contrib_name = validate_contrib_name()
    contrib_names = get_contrib_name_variants(contrib_name)
    
    print(
        "🥰 Thanks for contributing! \nAnswer a few quick questions and we will be "
        "generating new contribution folder from the template...\n"
    )
    
    generate_new_contrib_folder(contrib_names)
    generate_new_contrib_readme(contrib_names)
    generate_new_contrib_src_pyfile(contrib_names)
    update_new_contrib_pyproject(contrib_names)


if __name__ == "__main__":
    main()
