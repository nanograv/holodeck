# `holodeck` Notebooks

The notebooks included with holodeck are used for demonstration and also testing.  `nbconvert` is used to convert notebooks into python modules which are then run during the continuous integration testing suite.  Note that there is a separate `dev-notebooks/` directory in the package root, that contains notebooks intended for development instead of either unit-testing or demonstrations.

## Contents

* `init.ipy`
  * Effectively a header file loaded into each notebook containing standard imports.
* `continuous_observational.ipynb`
  * Observationally-based populations described with smooth density distributions, similar to SAMs.
* `discrete_illustris.ipynb`
  * Discrete binary populations from the cosmological hydrodynamic simulations Illustris.
* `host-relations.ipynb`
  * MBH-Host scaling relationships, mostly corresponding to objects in `holodeck/relations.py`.
* `relations.ipynb`
  * General scaling relationships and phenomenological fits (mostly from `holodeck/relations.py`)
* `sam_discretization.ipynb`
  * Conversion of smooth, semi-analytic models into discretized populations of binaries.
* `semi-analytic-models.ipynb`
  * General notebook for semi-analytic models.
* `utils.ipynb`
  * Utility methods, often logistical or numerical.