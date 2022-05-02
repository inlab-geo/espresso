"""Convert jupyter notebook to sphinx gallery notebook styled examples.
Usage: python ipynb_to_gallery.py <notebook.ipynb>
Dependencies:
pypandoc: install using `pip install pypandoc`

Adapted from source gist link below:
https://gist.github.com/chsasank/7218ca16f8d022e02a9c0deb94a310fe
"""

import pypandoc as pdoc
import json

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
    generated_file_name = file_name.replace(".ipynb", ".py")
    generated_file_name = generated_file_name.replace("notebooks", "scripts")
    open(generated_file_name, 'w').write(python_file)

if __name__ == '__main__':
    import sys
    import glob
    if sys.argv[-1] == "all":
        all_scripts = glob.glob('notebooks/*.ipynb')
        all_scripts = [name for name in all_scripts if "lab" not in name]
    else:
        all_scripts = [sys.argv[-1]]
    for script in all_scripts:
        convert_ipynb_to_gallery(script)
