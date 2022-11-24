from glob import glob
import sys
import os
from shutil import copyfile
from pathlib import Path

current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent.parent
NOTEBOOKS = "examples"
NOTEBOOKS_DIR = str(root_dir / NOTEBOOKS)
TEMPLATE_DIR = str(root_dir / "tools" / "generate_example" / "_template")

if __name__ == '__main__':
    # validate example name
    example_name = sys.argv[-1]
    if example_name.endswith(".py"):
        raise ValueError("please provide a valid example name as an argument")
    existing_examples = glob(NOTEBOOKS_DIR+"/*/")
    existing_examples = [e for e in existing_examples if not e.startswith("{NOTEBOOKS}/_")]
    existing_examples = [e.split("/")[-2] for e in existing_examples]
    print(existing_examples)
    if example_name in existing_examples:
        raise ValueError("The example name provided already exists, please choose another name")
    
    new_subfolder = NOTEBOOKS_DIR+"/"+example_name
    os.makedirs(new_subfolder)    
    template_files = glob(TEMPLATE_DIR+"/*")
    new_files = [f.replace(TEMPLATE_DIR, new_subfolder) for f in template_files]
    new_files = [f.replace("example_name", example_name) for f in new_files]
    print("Copying files...")
    for template_file, new_file in zip(template_files, new_files):
        print("file: "+template_file+" -> "+new_file)
        copyfile(template_file, new_file)
    print("\nOK. Please navigate to "+ new_subfolder+" to write your own example.")
