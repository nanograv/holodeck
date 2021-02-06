"""
Unit and regression test for the holodeck package.
"""

# Import package, test suite, and other packages as needed
import holodeck
import pytest
import sys

def test_holodeck_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "holodeck" in sys.modules
