"""Test suite consisting of all testcases."""

import sys

from tests import common
from tables.utils import print_versions


def suite():
    test_modules = [
        "tests.test_attributes",
        "tests.test_pickle",
        "tests.test_basics",
        "tests.test_create",
        "tests.test_backcompat",
        "tests.test_types",
        "tests.test_lists",
        "tests.test_tables",
        "tests.test_tables_md",
        "tests.test_large_tables",
        "tests.test_array",
        "tests.test_earray",
        "tests.test_carray",
        "tests.test_vlarray",
        "tests.test_tree",
        "tests.test_timetype",
        "tests.test_do_undo",
        "tests.test_enum",
        "tests.test_nestedtypes",
        "tests.test_hdf5compat",
        "tests.test_numpy",
        "tests.test_queries",
        "tests.test_expression",
        "tests.test_links",
        "tests.test_indexes",
        "tests.test_indexvalues",
        "tests.test_index_backcompat",
        "tests.test_aux",
        "tests.test_utils",
        "tests.test_direct_chunk",
        "tests.test_lrucache",
        # Sub-packages
        "tests.nodes.test_filenode",
    ]

    # print('-=' * 38)

    # The test for garbage must be run *in the last place*.
    # Else, it is not as useful.
    test_modules.append("tests.test_garbage")

    alltests = common.unittest.TestSuite()
    if common.show_memory:
        # Add a memory report at the beginning
        alltests.addTest(common.make_suite(common.ShowMemTime))
    for name in test_modules:
        # Unexpectedly, the following code doesn't seem to work anymore
        # in python 3
        # exec(f"from {name} import suite as test_suite")
        __import__(name)
        test_suite = sys.modules[name].suite

        alltests.addTest(test_suite())
        if common.show_memory:
            # Add a memory report after each test module
            alltests.addTest(common.make_suite(common.ShowMemTime))
    return alltests


def test(verbose=False, heavy=False, failfast=False):
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
    common.print_heavy(heavy)

    # What a context this is!
    # oldverbose, common.verbose = common.verbose, verbose
    oldheavy, common.heavy = common.heavy, heavy
    try:
        result = common.unittest.TextTestRunner(
            verbosity=1 + int(verbose), failfast=failfast
        ).run(suite())
        if result.wasSuccessful():
            return 0
        return 1
    finally:
        # common.verbose = oldverbose
        common.heavy = oldheavy  # there are pretty young heavies, too ;)
