# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# import os
# import sys
# sys.path.insert(0, os.path.abspath('../holodeck/'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'holodeck'
copyright = '2024, NANOGrav'
author = 'NANOGrav'
release = '1.2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.duration',
    'sphinx.ext.mathjax',
    'sphinx.ext.autosectionlabel',
]

templates_path = []
exclude_patterns = []

mathjax_path = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js'

# mathjax3_config = {
#     'tex': {
#         # 'inlineMath': [['$', '$'],],
#         'packages': {'[+]': ['ams', 'amsfonts', 'amssymb', 'amsmath']},
#     },
# }



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = []

# -- Setup Sphinx's API-DOC --------------------------------------------------

# Define the configuration for sphinx-apidoc
apidoc_module_dir = 'source/apidoc_modules'
apidoc_excluded_paths = ['../holodeck/detstats.py', '../holodeck/anisotropy.py']
apidoc_command_options = [
    '-M',  # put modules before submodules
    '-e',  # Create separate files for each module
    '-P',  # Include private modules
    '-T',  # Include docstrings in the table of contents
    '-f',   # Overwrite existing files
]

# Run sphinx-apidoc to generate the .rst files
def run_apidoc(_):
    from sphinx.ext.apidoc import main
    argv = [
        # '-M',
        # '-T',
        # '-f',
        *apidoc_command_options,
        '-o', apidoc_module_dir,
        '../../holodeck',
        *apidoc_excluded_paths,
    ]
    main(argv)

# Hook the sphinx-apidoc command into the Sphinx build process
def setup(app):
    app.connect('builder-inited', run_apidoc)