"""Unit and regression test for the holodeck package.

"""

# Import package, test suite, and other packages as needed
import holodeck  # noqa
import pytest  # noqa
import sys


def test_holodeck_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "holodeck" in sys.modules

    assert hasattr(holodeck, 'cosmo')
    assert hasattr(holodeck, 'log')
    assert hasattr(holodeck, '__version__')

    return
