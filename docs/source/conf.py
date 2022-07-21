# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/stable/config

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

import holodeck   # noqa


# -- Project information -----------------------------------------------------

project = 'holodeck'
copyright = holodeck.__copyright__
author = holodeck.__author__
# The short X.Y version
version = holodeck.__version__
# The full version, including alpha/beta/rc tags
release = holodeck.__version__


# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.extlinks',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    # 'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'numpydoc',     # allow numpy/google style docstrings
]

autosummary_generate = True

source_suffix = ['.rst', '.md']

master_doc = 'index'

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# NOTE: `numpy` is actually needed, otherwise things break
autodoc_mock_imports = [
    # 'pytest', 'kalepy', 'astropy', 'h5py', 'kalepy', 'matplotlib', 'scipy', 'tqdm',
]

html_theme = 'sphinx_rtd_theme'

nitpick_ignore = [
    ('py:class', 'numpy.typing._array_like._SupportsArray'),
    ('py:class', 'numpy.typing._nested_sequence._NestedSequence'),
]

# ---- Extensions ------------------------------------------------------------

intersphinx_mapping = {
    'numpy': ('https://numpy.org/doc/stable/', None),
}

numpydoc_show_class_members = False

# Report warnings for all validation checks
# numpydoc_validation_checks = {"all"}

# Report warnings for all checks *except* those listed
numpydoc_validation_checks = {
    "all",
    "GL08",   # The object does not have a docstring
    "ES01",   # No extended summary found
    "PR01",   # Parameters { ... } not documented
    "PR02",   # Unknown parameters { ... }
    "PR07",   # Parameter has no description
    "PR10",   # Parameter "___" requires a space before the colon separating the parameter name and type
    "RT01",   # No Returns section found
    "RT03",   # Return value has no description
    "SS01",   # No summary found (a short summary in a single line should be present at the beginning of the docstring)

    "SA01",   # See Also section not found
    "EX01",   # No examples section found

    "GL01",   # Docstring text (summary) should start in the line immediately after the opening quotes (not in the same line, or leaving a blank line in between)
    "GL02",   # Closing quotes should be placed in the line after the last text in the docstring (do not close the quotes in the same line as the text, or leave a blank line between the last text and the quotes)
    "GL03",   # Double line break found; please use only one blank line to separate sections or paragraphs, and do not leave blank lines at the end of docstrings
    "PR05",   # Parameter "___" type should not finish with "."
    "PR08",   # Parameter "weights" description should start with a capital letter
    "PR09",   # Parameter "___" description should finish with "."
    "RT02",   # The first line of the Returns section should contain only the type, unless multiple values are being returned
    "RT04",   # Return value description should start with a capital letter
    "RT05",   # Return value description should finish with ".
    "SS02",   # Summary does not start with a capital letter
    "SS03",   # Summary does not end with a period
    "SS05",   # Summary must start with infinitive verb, not third person (e.g. use "Generate" instead of "Generates")
}


def run_apidoc(_):
    """

    https://github.com/readthedocs/readthedocs.org/issues/1139#issuecomment-312626491

    """
    from sphinx.ext.apidoc import main
    cur_dir = os.path.abspath(os.path.dirname(__file__))
    sys.path.append(os.path.join(cur_dir, os.path.pardir))
    output_dir = os.path.join(cur_dir, "apidoc_modules")
    # docs/source ==> /docs ==> holodeck/
    input_dir = os.path.join(cur_dir, os.path.pardir, os.path.pardir, "holodeck")
    main(['-e', '-o', output_dir, input_dir, '--force'])
    return


def setup(app):
    app.connect('builder-inited', run_apidoc)
    return
