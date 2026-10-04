"""Unit tests for PyTables.

This package contains some modules which provide a ``suite()`` function
(with no arguments) which returns a test suite for some PyTables
functionality.

"""

from tables.tests.test_suite import test, suite

__all__ = [
    "suite",
    "test",
]
