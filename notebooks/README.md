# `holodeck` Notebooks

The notebooks included with holodeck are used for demonstration and also testing.  `nbconvert` is used to convert notebooks into python modules which are then run during the continuous integration testing suite.  Note that there is a separate `notebooks/devs/` directory that contains notebooks intended for development instead of either unit-testing or demonstrations.

## Contents

* `discrete_hardening_models.ipynb`
  * Binary evolution models applies to discrete binary populations (Illustris based populations).
* `discrete_illustris.ipynb`
  * Discrete binary populations from the cosmological hydrodynamic simulations Illustris.
* `host-relations.ipynb`
  * MBH-Host scaling relationships, mostly corresponding to objects in `holodeck/relations.py`.
* `relations.ipynb`
  * General scaling relationships and phenomenological fits (mostly from `holodeck/relations.py`)
* `semi-analytic-models.ipynb`
  * General notebook for semi-analytic models.
* `single-cws_demo.ipynb`
  * Quick demonstration of single-source (SS) / continuous-waves (CW) calculations.