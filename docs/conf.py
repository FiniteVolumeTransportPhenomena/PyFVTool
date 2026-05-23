# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Make the pyfvtool source importable for autodoc
sys.path.insert(0, os.path.abspath('../src'))

# -- Project information -----------------------------------------------------
project = 'PyFVTool'
copyright = '2024, PyFVTool contributors'
author = 'PyFVTool contributors'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',        # Pull docstrings from source code
    'sphinx.ext.napoleon',       # Support NumPy and Google style docstrings
    'sphinx.ext.viewcode',       # Add links to highlighted source code
    'sphinx.ext.intersphinx',    # Link to other projects' docs (numpy, scipy...)
    'sphinx.ext.mathjax',        # Render math equations
    'myst_parser',               # Write docs in Markdown
    'nbsphinx',                  # Render Jupyter notebooks as pages
    'sphinx_copybutton',         # Add copy button to code blocks
]

# MyST configuration
myst_enable_extensions = [
    'dollarmath',   # $...$ and $$...$$ for inline and display math
    'colon_fence',  # ::: as an alternative to ``` for directives
    'deflist',      # Definition lists
]

# Napoleon settings (adjust to match your docstring style)
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_param = True
napoleon_use_rtype = True

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'show-inheritance': True,
}

# nbsphinx settings
nbsphinx_execute = 'never'   # Don't re-run notebooks on every build
                              # Change to 'auto' once CI is set up

# intersphinx: link to numpy, scipy, matplotlib docs
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
}

# File extensions
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# The root document
root_doc = 'index'

# Exclude patterns
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'furo'   # Clean, modern theme used by many scientific projects
                       # Alternative: 'sphinx_rtd_theme'

html_theme_options = {
    # Furo options — see https://pradyunsg.me/furo/customisation/
    # "light_css_variables": {},
}

# If you have a logo, put it in docs/_static/ and uncomment:
# html_logo = '_static/logo.png'

html_static_path = ['_static']

# -- Options for autodoc -----------------------------------------------------
# Suppress warnings for missing type annotations
nitpicky = False
