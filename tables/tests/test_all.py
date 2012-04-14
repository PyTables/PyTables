"""
Run all test cases.
"""

import sys
import os
import re
import unittest

import numpy

import numexpr
import tables
from tables.req_versions import *
from tables.tests import common
from tables.utils import detectNumberOfCores



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
        # Sub-packages
        'tables.nodes.tests.test_filenode',
        #'tables.netcdf3.tests.test_netcdf3',
    ]

    # Numeric is now deprecated
    # Add test_Numeric only if Numeric is installed
    if False:   # if common.numeric_imported:
        import Numeric
        print "Numeric (version %s) is present. Adding the Numeric test suite." % \
              (Numeric.__version__)
        if Numeric.__version__ < min_numeric_version:
            print "*Warning*: Numeric version is lower than recommended: %s < %s" % \
                  (Numeric.__version__, min_numeric_version)
        test_modules.append("tables.tests.test_Numeric")
    else:
        pass
        # Do not print this anymore
        #print "Skipping Numeric test suite."

    # numarray is now deprecated
    # Add test_numarray only if numarray is installed
    if False:   # if common.numarray_imported:
        import numarray
        print \
"""numarray (version %s) is present. Adding the numarray test suite.""" % \
              (numarray.__version__)
        if numarray.__version__ < min_numarray_version:
            print \
"*Warning*: Numarray version is lower than recommended: %s < %s" % \
                  (numarray.__version__, min_numarray_version)
        test_modules.append("tables.tests.test_numarray")
        test_modules.append("tables.nra.tests.test_nestedrecords")
        test_modules.append("tables.nra.tests.test_nriterators")
    else:
        pass
        # Do not print this anymore
        #print "Skipping numarray test suite."
    #print '-=' * 38


    # The test for garbage must be run *in the last place*.
    # Else, it is not as useful.
    test_modules.append('tables.tests.test_garbage')

    alltests = unittest.TestSuite()
    if common.show_memory:
        # Add a memory report at the beginning
        alltests.addTest(unittest.makeSuite(common.ShowMemTime))
    for name in test_modules:
        exec('from %s import suite as test_suite' % name)
        alltests.addTest(test_suite())
        if common.show_memory:
            # Add a memory report after each test module
            alltests.addTest(unittest.makeSuite(common.ShowMemTime))
    return alltests


def print_versions():
    """Print all the versions of software that PyTables relies on."""
    print '-=' * 38
    print "PyTables version:  %s" % tables.__version__
    print "HDF5 version:      %s" % tables.whichLibVersion("hdf5")[1]
    print "NumPy version:     %s" % numpy.__version__
    tinfo = tables.whichLibVersion("zlib")
    if numexpr.use_vml:
        # Get only the main version number and strip out all the rest
        vml_version = numexpr.get_vml_version()
        vml_version = re.findall("[0-9.]+", vml_version)[0]
        vml_avail = "using VML/MKL %s" % vml_version
    else:
        vml_avail = "not using Intel's VML/MKL"
    print "Numexpr version:   %s (%s)" % (numexpr.__version__, vml_avail)
    if tinfo is not None:
        print "Zlib version:      %s (%s)" % (tinfo[1], "in Python interpreter")
    tinfo = tables.whichLibVersion("lzo")
    if tinfo is not None:
        print "LZO version:       %s (%s)" % (tinfo[1], tinfo[2])
    tinfo = tables.whichLibVersion("bzip2")
    if tinfo is not None:
        print "BZIP2 version:     %s (%s)" % (tinfo[1], tinfo[2])
    tinfo = tables.whichLibVersion("blosc")
    blosc_date = tinfo[2].split()[1]
    if tinfo is not None:
        print "Blosc version:     %s (%s)" % (tinfo[1], blosc_date)
    try:
        from Cython.Compiler.Main import Version as Cython_Version
        print 'Cython version:    %s' % Cython_Version.version
    except:
        pass
    print 'Python version:    %s' % sys.version
    if os.name == 'posix':
        (sysname, nodename, release, version, machine) = os.uname()
        print 'Platform:          %s-%s' % (sys.platform, machine)
    print 'Byte-ordering:     %s' % sys.byteorder
    print 'Detected cores:    %s' % detectNumberOfCores()
    print '-=' * 38


def print_heavy(heavy):
    if heavy:
        print """\
Performing the complete test suite!"""
    else:
        print """\
Performing only a light (yet comprehensive) subset of the test suite.
If you want a more complete test, try passing the --heavy flag to this script
(or set the 'heavy' parameter in case you are using tables.test() call).
The whole suite will take more than 4 hours to complete on a relatively
modern CPU and around 512 MB of main memory."""
    print '-=' * 38


def test(verbose=False, heavy=False):
    """
    Run all the tests in the test suite.

    If `verbose` is set, the test suite will emit messages with full
    verbosity (not recommended unless you are looking into a certain
    problem).

    If `heavy` is set, the test suite will be run in *heavy* mode (you
    should be careful with this because it can take a lot of time and
    resources from your computer).
    """
    print_versions()
    print_heavy(heavy)

    # What a context this is!
    oldverbose, common.verbose = common.verbose, verbose
    oldheavy, common.heavy = common.heavy, heavy
    try:
        unittest.TextTestRunner().run(suite())
    finally:
        common.verbose = oldverbose
        common.heavy = oldheavy  # there are pretty young heavies, too ;)


if __name__ == '__main__':

    hdf5_version = get_tuple_version(tables.whichLibVersion("hdf5")[0])
    if hdf5_version < min_hdf5_version:
        print "*Warning*: HDF5 version is lower than recommended: %s < %s" % \
              (hdf5_version, min_hdf5_version)

    if numpy.__version__ < min_numpy_version:
        print "*Warning*: NumPy version is lower than recommended: %s < %s" % \
              (numpy.__version__, min_numpy_version)

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
