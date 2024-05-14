===========================
Development & Contributions
===========================

**This File:**

.. contents:: :local:
   :depth: 1

This project is being led by the `NANOGrav <http://nanograv.org/>`_ Astrophysics Working Group.  Details on contributions and the mandatory code of conduct can be found in `CONTRIBUTING.md <https://raw.githubusercontent.com/nanograv/holodeck/docs/CONTRIBUTING.md>`_.

Contributions are welcome and encouraged, anywhere from new modules/customizations, to bug-fixes, to improved documentation and usage examples.  The git workflow is based around a ``main`` branch which is intended to be (relatively) stable and operational, and an actively developed ``dev`` branch.  New development should be performed in "feature" branches (made off of the ``dev`` branch), and then incorporated via pull-request (back into the ``dev`` branch).

For active developers, please install the additional development package requirements::

   pip install -r requirements-dev.txt

Formatting
----------

New code should generally abide by `PEP8 formatting <https://peps.python.org/pep-0008/>`_, with `numpy-style docstrings <https://numpydoc.readthedocs.io/en/latest/format.html#>`_.  Exceptions are:

* lines may be broken at either 100 or 120 columns

Notebooks
---------

Please strip all notebook outputs before commiting notebook changes.  The `nbstripout <https://github.com/kynan/nbstripout>`_ package is an excellent option to automatically strip all notebook output only in git commits (i.e. it doesn't change your notebooks in-place).  You can also use ``nbconvert`` to strip output in place::

   jupyter nbconvert --clear-output --inplace <NOTEBOOK-NAME>.ipynb

To install ``nbstripout`` for the ``holodeck`` git repository, make sure you're in the ``holodeck`` root directory and run:

.. code-block:: bash

  pip install --upgrade nbstripout    # install nbstripout
  nbstripout --install                # install git hook in current repo only


Test Suite
----------

**Before submitting a pull request, please run the test suite on your local machine.**

Tests can be run by using ``$ pytest`` in the root holodeck directory.  Tests can also be run against all supported python versions and system configurations by using ``$ tox``.  ``tox`` creates anaconda environments for each supported python version, sets up the package and test suite, and then runs ``pytest`` to execute tests.

Two types of unit-tests are generally used in ``holodeck``.

(1) Simple functions and behaviors are included as normal unit-tests, e.g. in "holodeck/tests" and similar directories.  These are automatically run by ``pytest`` and ``tox``.

(2) More complex functionality should be tested in notebooks (in "notebooks/") where they can also be used as demonstrations/tutorials for that behavior.  Certain notebooks are also converted into unit-test modules to be automatically run by ``pytest`` and ``tox``.  The python script `scripts/convert_notebook_tests.py <https://github.com/nanograv/holodeck/blob/main/scripts/convert_notebook_tests.py>`_ converts target notebooks into python scripts in the ``holodeck/tests/converted_notebooks`` directory, which are then run by ``pytest``.  The script `scripts/holotest.sh <https://github.com/nanograv/holodeck/blob/main/scripts/holotest.sh>`_ will run the conversion script and then run ``pytest``.  For help and usage information, run ``$ scripts/holotest.sh -h``.
