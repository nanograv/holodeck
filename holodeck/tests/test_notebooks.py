"""

See:
* https://testbook.readthedocs.io/en/latest/index.html
* https://docs.next.tech/creator/how-tos/testing-techniques/testing-jupyter-notebook-code-with-pytest

"""

import os
import pytest
from testbook import testbook

from holodeck import _PATH_NOTEBOOKS


@pytest.fixture(scope='module')
def tb_discrete_illustris():
    path = os.path.join(_PATH_NOTEBOOKS, "discrete_illustris.ipynb")
    with testbook(path, execute=True) as tb:
        yield tb


@pytest.fixture(scope='module')
def tb_continuous_observational():
    path = os.path.join(_PATH_NOTEBOOKS, "continuous_observational.ipynb")
    with testbook(path, execute=True) as tb:
        yield tb


@pytest.fixture(scope='module')
def tb_semi_analytic_model():
    path = os.path.join(_PATH_NOTEBOOKS, "semi-analytic-model.ipynb")
    with testbook(path, execute=True) as tb:
        yield tb


def test_discrete_illustris(tb_discrete_illustris):
    pass


def test_continuous_observational(tb_continuous_observational):
    pass


def test_semi_analytic_model(tb_semi_analytic_model):
    pass
