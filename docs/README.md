# Documents and Documentation

The docs for this project are built with [Sphinx](http://www.sphinx-doc.org/en/master/).

## Notes

* Imported requirements (e.g. python packages) are also required for the docs to build.  They either need to be specified in the `docs/requirements.txt` file (to actually be imported into the build environment), or in some cases can be 'mocked-up' (i.e. faked), in which case they can be added to the `autodoc_mock_imports` list in the `docs/source/conf.py` file.

* `sphinx-apidoc` should be run automatically on readthedocs builds using the code in the bottom of the `conf.py` file.