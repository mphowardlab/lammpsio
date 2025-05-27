# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import datetime
import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))

# -- Project information -----------------------------------------------------

project = "lammpsio"
year = datetime.date.today().year
copyright = f"2021-{year}, Auburn University"
author = "Michael P. Howard"
version = "0.7.0"
release = "0.7.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "IPython.sphinxext.ipython_console_highlighting",
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_design",
    "sphinx_favicon",
    "myst_parser",
]


templates_path = ["_templates"]

exclude_patterns = []

default_role = "any"

# -- Options for HTML output -------------------------------------------------

html_theme = "furo"
html_static_path = []
html_theme_options = {
    "navigation_with_keys": True,
    "top_of_page_buttons": [],
    "dark_css_variables": {
        "color-brand-primary": "#ef904d",
        "color-brand-content": "#ef904d",
        "color-admonition-background": "#ef904d",
    },
    "light_css_variables": {
        "color-brand-primary": "#e86100",
        "color-brand-content": "#e86100",
        "color-admonition-background": "#e86100",
    },
}

# -- Options for autodoc & autosummary ---------------------------------------

autosummary_generate = True

autodoc_member_order = "bysource"

autodoc_default_options = {"inherited-members": None, "special-members": False}

# -- Options for intersphinx -------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "gsd": ("https://gsd.readthedocs.io/en/stable/", None),
}
