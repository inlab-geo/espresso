# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import datetime
# import sys
# import subprocess
from pathlib import Path
from shutil import copy
import yaml

import cofi_espresso as esp


# -- Generate documentation for each contrib ---------------------------------
def gen_contrib_docs(_):
    all_contribs = esp.__all__
    base_path = esp.__path__[0]
    dest_path = Path(__file__).parent / "contrib" / "generated"
    os.mkdir(dest_path)
    for contrib in all_contribs:
        contrib_dir = Path(f"{base_path}/{contrib}")
        dest_contrib_dir = Path(f"{dest_path}/{contrib}")
        if contrib_dir.exists() and contrib_dir.is_dir():
            # locate files
            file_metadata = contrib_dir / "metadata.yml"
            file_readme = contrib_dir / "README.md"
            file_licence = contrib_dir / "LICENCE"
            # make new folder docs/source/contrib/<contrib-name>
            os.mkdir(dest_contrib_dir)
            # copy README and LICENCE
            copy(file_readme, f"{dest_contrib_dir}/README.md")
            copy(file_licence, f"{dest_contrib_dir}/LICENCE")
            with open(file_metadata, "r") as f:
                metadata = yaml.safe_load(f)
            lines = []
            # include README.md
            lines.append("```{include} ./README.md\n```")
            # format metadata files
            lines.append(":::{admonition} Contribution Metadata for ")
            lines[-1] += f"{metadata['name']} \n:class: important"
            lines.append(metadata['short_description'])
            lines.append("\n**Author(s)**")
            for author in metadata["authors"]:
                lines.append(f"- {author}")
            lines.append("\n**Contact(s)**")
            for contact in metadata["contacts"]:
                lines.append(f"- {contact['name']} {contact['email']} ")
                lines[-1] += f"[website]({contact['website']})" if "website" in contact else ""
            lines.append("\n**Citation**")
            for citation in metadata["citations"]:
                lines.append(f"- doi: {citation['doi']}")
            lines.append("\n**Extra website**")
            for extra_website in metadata["extra_websites"]:
                lines.append(f"- [{extra_website['name']}]({extra_website['link']})")
            lines.append("\n**Examples**")
            for idx, example in enumerate(metadata["examples"]):
                lines.append(f"{idx+1}. {example['description']}")
                lines.append(f"    - model dimension: {example['model_dimension']}")
                lines.append(f"    - data dimension: {example['data_dimension']}")
            lines.append(":::")
            # include LICENCE
            lines.append("## LICENCE\n")
            lines.append("```{include} ./LICENCE\n```")
            # write to index.md file
            with open(f"{dest_contrib_dir}/index.md", "w") as f:
                f.write("\n".join(lines))
            # add contrib link to contrib/index.rst
            with open(Path(__file__).parent / "contrib" / "_index.rst", "r") as f:
                index_template = f.read()
            with open(Path(__file__).parent / "contrib" / "index.rst", "w") as f:
                f.write(index_template)
                f.write(f"    generated/{contrib}/index.md")

def setup(app):
    app.connect("builder-inited", gen_contrib_docs)           


# -- Project information -----------------------------------------------------
project = 'Espresso'
copyright = f"{datetime.date.today().year}, InLab, {project} development team"
version = "dev" if "dev" in esp.__version__ else f"v{esp.__version__}"
print(version)


# -- General configuration ---------------------------------------------------
extensions = [
    # "sphinx.ext.autodoc",
    # "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.doctest",
    "sphinx.ext.mathjax",
    "sphinx_panels",
    "sphinx_togglebutton",
    "sphinx_copybutton",
    "sphinx.ext.napoleon",
    "myst_nb",
    # "sphinx_gallery.gen_gallery",
    "sphinxcontrib.mermaid",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    ".DS_Store",
    "api/index.rst",
    "contrib/_index.rst",
]

source_suffix = ".rst"
source_encoding = "utf-8"
master_doc = "index"
pygments_style = "trac"        # https://pygments.org/styles/
add_function_parentheses = False

# Configuration to include links to other project docs
intersphinx_mapping = {
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# Disable including boostrap CSS for sphinx_panels since it's already included
# with sphinx-book-theme
panels_add_bootstrap_css = False
panels_css_variables = {
    "tabs-color-label-inactive": "hsla(231, 99%, 66%, 0.5)",
}

# settings for the sphinx-copybutton extension
copybutton_prompt_text = ">>> "


# -- Options for HTML output -------------------------------------------------
html_title = f'{project} <span class="project-version">{version}</span>'
html_short_title = project
# html_logo = "_static/???"
html_favicon = "_static/inlab_logo_60px.png"

html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": "https://github.com/inlab-geo/espresso",
    "repository_branch": "main",
    "path_to_docs": "docs",
    "launch_buttons": {
        "notebook_interface": "classic",
        "inlab_url": "http://www.inlab.edu.au/",
    },
    "extra_footer": "",
    "home_page_in_toc": True,
    "use_repository_button": True,
    "use_edit_page_button": True,
    "use_issues_button": True,
}

html_static_path = ["_static"]
html_css_files = ["style.css"]
html_context = {
    "display_github": True, # Integrate GitHub
    "github_user": "inlab-geo", # Username
    "github_repo": "cofi", # Repo name
    "github_version": "main", # Version
    "conf_py_path": "", # Path in the checkout to the docs root
}


# -- Sphinx Gallery settings --------------------------------------------------
sphinx_gallery_conf = {
    "examples_dirs": "cofi-examples/utils/sphinx_gallery/scripts",
    "gallery_dirs": "cofi-examples/utils/sphinx_gallery/generated",
    "filename_pattern": ".",
    "ignore_pattern": "._lib.py",
    "pypandoc": True,
    "download_all_examples": False,
}

# -- myst-nb settings ---------------------------------------------------------
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]


# -- Cutomised variables ------------------------------------------------------
rst_epilog = """
.. _repository: https://github.com/inlab-geo/espresso
"""
