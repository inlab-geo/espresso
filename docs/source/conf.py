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
import datetime
import os
import sys
import subprocess
from pathlib import Path

import espresso as esp


# -- Generate API references doc ---------------------------------------------
def run_autogen(_):
    cmd_path = "sphinx-autogen"
    if hasattr(sys, "real_prefix"):  # Check to see if we are in a virtualenv
        # If we are, assemble the path manually
        cmd_path = os.path.abspath(os.path.join(sys.prefix, "bin", cmd_path))
    subprocess.check_call(
        [
            cmd_path, "-i", 
            "-t", Path(__file__).parent / "_templates", 
            "-o", Path(__file__).parent / "user_guide" / "api" / "generated", 
            Path(__file__).parent / "user_guide" / "api" / "index.rst"
        ]
    )

def setup(app):
    # This function is automatically called by sphinx-build at the start of the
    # build process.
    app.connect("builder-inited", run_autogen)


# -- Project information -----------------------------------------------------
project = 'Espresso'
copyright = f"{datetime.date.today().year}, InLab, {project} development team"
_version_short = esp.__version__.split("+")[0]
version = "dev" if "dev" in esp.__version__ else _version_short


# -- General configuration ---------------------------------------------------
sys.path.append(os.path.abspath("./_ext"))
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.doctest",
    "sphinx.ext.mathjax",
    "sphinx_design",
    "sphinx_togglebutton",
    "sphinx_copybutton",
    "sphinx.ext.napoleon",
    "myst_nb",
    # "sphinx_gallery.gen_gallery",
    "sphinxcontrib.mermaid",
    "generate_contrib_docs",                # our own extension
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    ".DS_Store",
    "user_guide/contrib/_index.rst",
    "user_guide/contrib/generated/*/examples/*",
]

source_suffix = ".rst"
source_encoding = "utf-8"
master_doc = "index"
pygments_style = "trac"        # https://pygments.org/styles/
add_function_parentheses = False

# Configuration to include links to other project docs
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
}

# settings for the sphinx-copybutton extension
copybutton_prompt_text = ">>> "


# -- Options for HTML output -------------------------------------------------
html_title = f'{project} <span class="project-version">{version}</span>'
html_short_title = project
html_favicon = "_static/inlab_logo_60px.png"

html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": "https://github.com/inlab-geo/espresso",
    "repository_branch": "main",
    "path_to_docs": "docs/source/",
    "launch_buttons": {
        "notebook_interface": "classic",
        "inlab_url": "http://www.inlab.edu.au/",
    },
    "extra_footer": "",
    "home_page_in_toc": True,
    "use_repository_button": True,
    "use_edit_page_button": True,
    "use_source_button": True,
    "use_issues_button": True,
    "use_download_button": True,
    "use_sidenotes": True,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/inlab-geo/geo-espresso",
            "icon": "https://img.shields.io/badge/GitHub-espresso-171515?logo=github&labelColor=f8f9fa&style=flat-square&logoColor=171515",
            "type": "url",
        },
        {
            "name": "Version",
            "url": "https://pypi.org/project/geo-espresso/",
            "icon": "https://img.shields.io/pypi/v/geo-espresso?logo=pypi&style=flat-square&color=83C5BE&labelColor=f8f9fa&label=latest",
            "type": "url",
        },
    ],
}

html_static_path = ["_static"]
html_css_files = ["style.css"]
html_context = {
    "display_github": True, # Integrate GitHub
    "github_user": "inlab-geo", # Username
    "github_repo": "espresso", # Repo name
    "github_version": "main", # Version
    "conf_py_path": "/source/", # Path in the checkout to the docs root
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
.. _newissue: https://github.com/inlab-geo/espresso/issues/new/choose
.. _Slack: https://join.slack.com/t/inlab-community/shared_invite/zt-1ejny069z-v5ZyvP2tDjBR42OAu~TkHg
"""
