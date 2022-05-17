from glob import glob
import sys
import os
from shutil import copyfile

NOTEBOOKS_FOLDER = "notebooks"
TEMPLATE_FOLDER = "utils/_template"

if __name__ == '__main__':
    # validate example name
    example_name = sys.argv[-1]
    existing_examples = glob(f"{NOTEBOOKS_FOLDER}/*/")
    existing_examples = [e for e in existing_examples if not e.startswith("notebooks/_")]
    existing_examples = [e.split("/")[1] for e in existing_examples]
    if example_name in existing_examples:
        raise ValueError("The example name provided already exists, please choose another name")

    # create template files under notebooks/example_name
    new_subfolder = f"{NOTEBOOKS_FOLDER}/{example_name}"
    os.mkdir(new_subfolder)
    template_files = glob(f"{TEMPLATE_FOLDER}/*")
    new_files = [f.replace(TEMPLATE_FOLDER, new_subfolder) for f in template_files]
    new_files = [f.replace("example_name", example_name) for f in new_files]
    print("Copying files...")
    for template_file, new_file in zip(template_files, new_files):
        print(f"file: {template_file} -> {new_file}")
        copyfile(template_file, new_file)
    print(f"\nOK. Please navigate to {new_subfolder} to write your own example.")
