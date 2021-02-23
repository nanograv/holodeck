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
def tb_discrete():
    path = os.path.join(_PATH_NOTEBOOKS, "discrete.ipynb")
    with testbook(path, execute=True) as tb:
        yield tb


@pytest.fixture(scope='module')
def tb_continuous():
    path = os.path.join(_PATH_NOTEBOOKS, "continuous.ipynb")
    with testbook(path, execute=True) as tb:
        yield tb


def test_discrete(tb_discrete):
    pass


def test_continuous(tb_continuous):
    pass
