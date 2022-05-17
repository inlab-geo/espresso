"""Convert jupyter notebook to sphinx gallery notebook styled examples.
Usage: python ipynb_to_gallery.py <notebook.ipynb>
Dependencies:
pypandoc: install using `pip install pypandoc`

Adapted from source gist link below:
https://gist.github.com/chsasank/7218ca16f8d022e02a9c0deb94a310fe
"""

import sys
from glob import glob
from shutil import copyfile

import pypandoc as pdoc
import json

NOTEBOOKS_FOLDER = "notebooks"
SCRIPTS_FOLDER = "scripts"

def convert_ipynb_to_gallery(file_name):
    python_file = ""

    nb_dict = json.load(open(file_name))
    cells = nb_dict['cells']

    for i, cell in enumerate(cells):
        if i == 0:  
            assert cell['cell_type'] == 'markdown', \
                'First cell has to be markdown'

            md_source = ''.join(cell['source'])
            rst_source = pdoc.convert_text(md_source, 'rst', 'md')
            python_file = '"""\n' + rst_source + '\n"""'
        else:
            if cell['cell_type'] == 'markdown':
                md_source = ''.join(cell['source'])
                rst_source = pdoc.convert_text(md_source, 'rst', 'md')
                commented_source = '\n'.join(['# ' + x for x in
                                              rst_source.split('\n')])
                python_file = python_file + '\n\n\n' + '#' * 70 + '\n' + \
                    commented_source
            elif cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                python_file = python_file + '\n' * 2 + source

    python_file = python_file.replace("\n%", "\n# %")

    file_name_without_path = file_name.split("/")[-1]
    script_file_path = f"{SCRIPTS_FOLDER}/{file_name_without_path}"
    script_file_path = script_file_path.replace(".ipynb", ".py")
    open(script_file_path, 'w').write(python_file)

if __name__ == '__main__':
    # collect notebooks to convert to sphinx gallery scripts
    if sys.argv[-1] == "all":
        all_scripts = glob(f"{NOTEBOOKS_FOLDER}/*/*.ipynb")
        all_scripts = [name for name in all_scripts if "lab" not in name]
    else:
        all_scripts = [sys.argv[-1]]
    # convert
    print("Converting files...")
    for script in all_scripts:
        print(f"file: {script}")
        convert_ipynb_to_gallery(script)
    # collect all data files to move to scripts/
    all_data = glob(f"{NOTEBOOKS_FOLDER}/*/*.npz")
    all_data.extend(glob(f"{NOTEBOOKS_FOLDER}/*/*.dat"))
    all_data.extend(glob(f"{NOTEBOOKS_FOLDER}/*/*.csv"))
    # move
    print("\nMoving dataset files...")
    for data_file in all_data:
        data_filename_without_path = data_file.split("/")[-1]
        dest_file_path = f"{SCRIPTS_FOLDER}/{data_filename_without_path}"
        copyfile(data_file, dest_file_path)
    print("\nOK.")
