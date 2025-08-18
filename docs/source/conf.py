# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'PyFVTool'
copyright = '2025, A. A. Eftekhari, M. H. V. Werts'
author = 'A. A. Eftekhari, M. H. V. Werts'

# The full version, including alpha/beta/rc tags

release = '0.4.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',      # for docstring extraction
    'sphinx.ext.napoleon',     # for numpy/google style docstrings
    'myst_parser',            # for Markdown support
    'nbsphinx',               # for Jupyter notebook support
    # other extensions...
]

templates_path = ['_templates']
exclude_patterns = [
    'Thumbs.db',
    '.DS_Store',
    '**/.virtual_documents',      # Ignore .virtual_documents everywhere
    '**/.ipynb_checkpoints'       # Ignore .ipynb_checkpoints everywhere
]



# Language 

language = "en"



# -- Extension config --------------------------------------------------------



# Optional nbsphinx settings
nbsphinx_execute = 'never'  # or 'always', 'auto'
nbsphinx_allow_errors = True



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']
