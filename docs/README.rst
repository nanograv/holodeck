Documents and Documentation
===========================

The docs for this project are built with `Sphinx <http://www.sphinx-doc.org/en/master/>`_, and hosted on `readthedocs <https://holodeck-gw.readthedocs.io/en/main/>`_.  The ``sphinx`` configuration is in the ``docs/source/conf.py`` file.

readthedocs will update the online documentation on new tags.


Building Documentation Locally
------------------------------

(1) Make sure the requirements are installed using one of the commands:

.. code-block:: bash

    pip install docs/requirements.txt
    # OR
    conda install --file docs/requirements.txt


(2) Build the documentation

.. code-block:: bash

    make html

The resulting documentation can be found in the ``docs/build`` directory, in particular the ``docs/build/html/index.html`` file.  A symlink is also created in the local directory: ``docs/readthedocs.html``.


Additional Notes
----------------

* **Citations/References**: All citations should be defined in the ``docs/source/biblio.rst`` file.  In individual files that make citations, a smaller references section should be added that link to the ``biblio.rst`` entries.  See the bottom of the ``docs/source/biblio.rst`` file for the formatting of references.  PDF copies of many important reference papers are inclues in the ``docs/references/`` subfolder.