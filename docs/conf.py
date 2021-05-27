# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html
# Written for sphinx 3.2.1

import sys
import os
from agentpy import __version__  # Agentpy must be installed first

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

# -- Project information -----------------------------------------------------

project = 'agentpy'
copyright = '2020-2021, Joël Foramitti'
author = 'Joël Foramitti'

# The full version, including alpha/beta/rc tags
release = __version__

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    "sphinx.ext.intersphinx",
    'sphinx_rtd_theme',  # Read the docs theme
    'nbsphinx'  # Support jupyter notebooks
]

# Remove blank pages
latex_elements = {
    'extraclassoptions': 'openany,oneside'
}

# Define master file
master_doc = 'index'

# Display class attributes as variables
napoleon_use_ivar = True

# Jupyter notebook settings (nbsphinx)
html_sourcelink_suffix = ''
nbsphinx_prolog = """
.. currentmodule:: agentpy
.. note::
    You can download this demonstration as a Jupyter Notebook 
    :download:`here<{{ env.doc2path(env.docname,base=None) }}>`
"""

# Connect to other docs
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'ipywidgets': ('https://ipywidgets.readthedocs.io/en/latest/', None),
    'IPython': ('https://ipython.readthedocs.io/en/stable/', None),
    'matplotlib': ('https://matplotlib.org/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'salib': ('https://salib.readthedocs.io/en/latest/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None)
}

# Remove module name before elements
add_module_names = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'  # 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = ['css/custom.css']
