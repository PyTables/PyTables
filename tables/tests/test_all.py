"""
Run all test cases.
"""

import sys
import os
import unittest

from numarray import *
from numarray import strings

import common

# Recommended minimum versions for optional packages
minimum_numeric_version = "24.2"
minimum_numpy_version = "0.9.5.2052"

def suite():
    test_modules = [
        'test_attributes',
        'test_basics',
        'test_create',
        'test_backcompat',
        'test_types',
        'test_lists',
        'test_tables',
        'test_tablesMD',
        'test_indexes',
        'test_indexvalues',
        'test_array',
        'test_earray',
        'test_carray',
        'test_vlarray',
        'test_tree',
        'test_timetype',
        'test_do_undo',
        'test_enum',
        'test_nestedrecords',
        'test_nestedtypes',
        'test_hdf5compat',
        # Sub-packages
        'test_filenode',
        'test_NetCDF',
        ]

    # Add test_Numeric only if Numeric is installed
    try:
        import Numeric
        print "Numeric (version %s) is present. Adding the Numeric test suite." % \
              (Numeric.__version__)
        if Numeric.__version__ < minimum_numeric_version:
            print "*Warning*: Numeric version is lower than recommended: %s < %s" % \
                  (Numeric.__version__, minimum_numeric_version)
        test_modules.append("test_Numeric")

        # Warn about conversion between hdf5 <--> NetCDF will only be
        # checked if Numeric *and* Scientific.IO.NetCDF are installed.
        try:
            import Scientific.IO.NetCDF as RealNetCDF
            print \
"Scientific.IO.NetCDF is present. Will check for HDF5 <--> NetCDF conversions."
        except:
            print \
"Scientific.IO.NetCDF not found. Skipping HDF5 <--> NetCDF conversion tests."
    except:
        print "Skipping Numeric test suite"

    # Add test_numpy only if NumPy is installed
    try:
        import numpy
        print "NumPy (version %s) is present. Adding the NumPy test suite." % \
              (numpy.__version__)
        if numpy.__version__ < minimum_numpy_version:
            print "*Warning*: NumPy version is lower than recommended: %s < %s" % \
                  (numpy.__version__, minimum_numpy_version)
        test_modules.append("test_numpy")
    except:
        print "Skipping NumPy test suite"
    print '-=' * 38


    # The test for garbage must be run *in the last place*.
    # Else, it is not as useful.
    test_modules.append('test_garbage')

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


if __name__ == '__main__':
    import numarray
    import tables

    print '-=' * 38
    print "PyTables version:  %s" % tables.__version__
    print "HDF5 version:      %s" % tables.whichLibVersion("hdf5")[1]
    #print "HDF5 version:      %s" % tables.hdf5Version
    print "numarray version:  %s" % numarray.__version__
    #print "Zlib version:      %s" % tables.whichLibVersion("zlib")[1]
    tinfo = tables.whichLibVersion("zlib")
    if tinfo is not None:
        print "Zlib version:      %s" % (tinfo[1])
    tinfo = tables.whichLibVersion("lzo")
    if tinfo is not None:
        print "LZO version:       %s (%s)" % (tinfo[1], tinfo[2])
    tinfo = tables.whichLibVersion("ucl")
    if tinfo is not None:
        print "UCL version:       %s (%s)" % (tinfo[1], tinfo[2])
    tinfo = tables.whichLibVersion("bzip2")
    if tinfo is not None:
        print "BZIP2 version:     %s (%s)" % (tinfo[1], tinfo[2])
    print 'Python version:    %s' % sys.version
    if os.name == 'posix':
        (sysname, nodename, release, version, machine) = os.uname()
        print 'Platform:          %s-%s' % (sys.platform, machine)
    print 'Byte-ordering:     %s' % sys.byteorder
    print '-=' * 38

    # Handle --show-versions-only
    only_versions = 0
    args = sys.argv[:]
    for arg in args:
        if arg == '--show-versions-only':
            only_versions = 1
            sys.argv.remove(arg)
        elif arg == '--show-versions':
            only_versions = 1
            sys.argv.remove(arg)
        elif arg == '--show-memory':
            common.show_memory = True
            sys.argv.remove(arg)

    if not only_versions:
        if common.heavy:
            print \
"""Performing the complete test suite!"""
        else:
            print \
"""Performing only a light (yet comprehensive) subset of the test
suite.  If you have a big system and lots of CPU to waste and want to
do a more complete test, try passing the --heavy flag to this script.
The whole suite will take more than 10 minutes to complete on a
relatively modern CPU and around 100 MB of main memory."""
        print '-=' * 38

        unittest.main( defaultTest='suite' )
