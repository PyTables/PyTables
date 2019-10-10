# -*- coding: utf-8 -*-

"""Run all test cases."""

import sys

import numpy

import tables
from tables.req_versions import min_hdf5_version, min_numpy_version
from tables.tests import common
from tables.tests.common import unittest
from tables.tests.common import print_heavy, print_versions
from tables.tests.test_suite import suite, test


def get_tuple_version(hexversion):
    """Get a tuple from a compact version in hex."""

    h = hexversion
    return(h & 0xff0000) >> 16, (h & 0xff00) >> 8, h & 0xff


if __name__ == '__main__':

    common.parse_argv(sys.argv)

    hdf5_version = get_tuple_version(tables.which_lib_version("hdf5")[0])
    hdf5_version_str = "%s.%s.%s" % hdf5_version
    if hdf5_version_str < min_hdf5_version:
        print("*Warning*: HDF5 version is lower than recommended: %s < %s" %
              (hdf5_version, min_hdf5_version))

    if numpy.__version__ < min_numpy_version:
        print("*Warning*: NumPy version is lower than recommended: %s < %s" %
              (numpy.__version__, min_numpy_version))

    # Handle some global flags (i.e. only useful for test_all.py)
    only_versions = 0
    args = sys.argv[:]
    for arg in args:
        # Remove 'show-versions' for PyTables 2.3 or higher
        if arg in ['--print-versions', '--show-versions']:
            only_versions = True
            sys.argv.remove(arg)
        elif arg == '--show-memory':
            common.show_memory = True
            sys.argv.remove(arg)

    print_versions()
    if not only_versions:
        print_heavy(common.heavy)
        unittest.main(defaultTest='tables.tests.suite')
