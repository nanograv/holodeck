# Documents and Documentation

The docs for this project are built with `Sphinx <http://www.sphinx-doc.org/en/master/>`_, and hosted on `readthedocs <https://holodeck-gw.readthedocs.io/en/main/>`_.  The ``sphinx`` configuration is in the ``docs/source/conf.py`` file.


## Building Documentation

1) Make sure the requirements are installed:

    ```bash
    pip install docs/requirements.txt
    ```

    or

    ```bash
    conda install --file docs/requirements.txt
    ```

2) Build the documentation

    ```bash
    ./docs/docgen.sh
    ```

The resulting documentation can be found in the `docs/build` directory, in particular the `docs/build/html/index.html` file.

## Contents

* build/
  * This is the output directory for sphinx builds
* references/
  * PDF files of relevant reference material, mostly published papers
* source/
  * Source files for holodeck documentation.  This is a combination of manually and automatically generated files.  In general, manually created files should live in the `source/` root directory, while automatic files should be stored in subdirectories (e.g. `apidoc_modules`)
* docgen.sh
  * Script to run the sphinx document generation project
* requirements.txt
  * Requirements specific to documentation and the sphinx build.


## Notes

* readthedocs will update the [online documentation](https://holodeck-gw.readthedocs.io/en/main/) on new branches and new tags.
* `sphinx-apidoc` should be run automatically on readthedocs builds using the code in the bottom of the `conf.py` file.
* Citations/References
  * All citations should be defined in the `docs/source/biblio.rst` file.  In individual files that make citations, a smaller references section should be added that link to the `biblio.rst` entries.  If the citations are defined in both places, the links will not behave properly and there will be sphinx build errors.