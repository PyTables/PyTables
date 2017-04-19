# -*- coding: utf-8 -*-

"""Run all test cases."""

from __future__ import print_function
from __future__ import absolute_import
import sys

import numpy

import tables
from tables.req_versions import min_hdf5_version, min_numpy_version
from tables.tests import common
from tables.tests.common import unittest
from tables.tests.common import print_heavy, print_versions


def get_tuple_version(hexversion):
    """Get a tuple from a compact version in hex."""

    h = hexversion
    return(h & 0xff0000) >> 16, (h & 0xff00) >> 8, h & 0xff


def suite():
    test_modules = [
        'tables.tests.test_attributes',
        'tables.tests.test_basics',
        'tables.tests.test_create',
        'tables.tests.test_backcompat',
        'tables.tests.test_types',
        'tables.tests.test_lists',
        'tables.tests.test_tables',
        'tables.tests.test_tablesMD',
        'tables.tests.test_array',
        'tables.tests.test_earray',
        'tables.tests.test_carray',
        'tables.tests.test_vlarray',
        'tables.tests.test_tree',
        'tables.tests.test_timetype',
        'tables.tests.test_do_undo',
        'tables.tests.test_enum',
        'tables.tests.test_nestedtypes',
        'tables.tests.test_hdf5compat',
        'tables.tests.test_numpy',
        'tables.tests.test_queries',
        'tables.tests.test_expression',
        'tables.tests.test_links',
        'tables.tests.test_indexes',
        'tables.tests.test_indexvalues',
        'tables.tests.test_index_backcompat',
        'tables.tests.test_aux',
        'tables.tests.test_utils',
        # Sub-packages
        'tables.nodes.tests.test_filenode',
    ]

    # print('-=' * 38)

    # The test for garbage must be run *in the last place*.
    # Else, it is not as useful.
    test_modules.append('tables.tests.test_garbage')

    alltests = unittest.TestSuite()
    if common.show_memory:
        # Add a memory report at the beginning
        alltests.addTest(unittest.makeSuite(common.ShowMemTime))
    for name in test_modules:
        # Unexpectedly, the following code doesn't seem to work anymore
        # in python 3
        # exec('from %s import suite as test_suite' % name)
        __import__(name)
        test_suite = sys.modules[name].suite

        alltests.addTest(test_suite())
        if common.show_memory:
            # Add a memory report after each test module
            alltests.addTest(unittest.makeSuite(common.ShowMemTime))
    return alltests


def test(verbose=False, heavy=False):
    """Run all the tests in the test suite.

    If *verbose* is set, the test suite will emit messages with full
    verbosity (not recommended unless you are looking into a certain
    problem).

    If *heavy* is set, the test suite will be run in *heavy* mode (you
    should be careful with this because it can take a lot of time and
    resources from your computer).

    Return 0 (os.EX_OK) if all tests pass, 1 in case of failure

    """

    print_versions()
    print_heavy(heavy)

    # What a context this is!
    #oldverbose, common.verbose = common.verbose, verbose
    oldheavy, common.heavy = common.heavy, heavy
    try:
        result = unittest.TextTestRunner(verbosity=1+int(verbose)).run(suite())
        if result.wasSuccessful():
            return 0
        else:
            return 1
    finally:
        #common.verbose = oldverbose
        common.heavy = oldheavy  # there are pretty young heavies, too ;)


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
