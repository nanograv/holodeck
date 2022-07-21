# Documents and Documentation

The docs for this project are built with [Sphinx](http://www.sphinx-doc.org/en/master/).

## Contents

* build/
  * This is the output directory for sphinx builds
* holodeck-paper/
  * Manuscript for holodeck methods paper (currently in preparation)
* references/
  * PDF files of relevant reference material, mostly published papers
* source/
  * Source files for holodeck documention.  This is a combination of manually and automatically generated files.  In general, manually created files should live in the `source/` root directory, while automatic files should be stored in subdirectories (e.g. `apidoc_modules`)
* docgen.sh
  * Script to run the sphinx document generation project
* requirements.txt
  * Requirements specific to documention and the sphinx build.


## Notes

* `sphinx-apidoc` should be run automatically on readthedocs builds using the code in the bottom of the `conf.py` file.
* Citations/References
  * All citations should be defined in the `docs/source/biblio.rst` file.  In individual files that make citations, a smaller references section should be added that link to the `biblio.rst` entries.  If the citations are defined in both places, the links will not behave properly and there will be sphinx build errors.