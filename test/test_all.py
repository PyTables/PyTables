"""
Run all test cases.
"""

import sys
import os
import unittest

verbose = 0
if 'verbose' in sys.argv:
    verbose = 1
    sys.argv.remove('verbose')

if 'silent' in sys.argv:  # take care of old flag, just in case
    verbose = 0
    sys.argv.remove('silent')


# This little hack is for when this module is run as main and all the
# other modules import it so they will still be able to get the right
# verbose setting.  It's confusing but it works.
import test_all
test_all.verbose = verbose


def suite():
    test_modules = ['test_basics',
                    'test_create',
                    'test_backcompat',
                    'test_types',
                    'test_numarray',
                    'test_lists',
                    'test_tables',
                    'test_tablesMD',
                    'test_vlarray',
                    'test_earray',
		    'test_tree',
		    ]

    # Add test_Numeric only if Numeric is installed
    try:
        import Numeric
        print "Numeric (version %s) is present. Adding the Numeric test suite." % \
              (Numeric.__version__)
        test_modules.append("test_Numeric")
    except:
        print "Skipping Numeric test suite"

    alltests = unittest.TestSuite()
    for name in test_modules:
        module = __import__(name)
        alltests.addTest(module.suite())
    return alltests


if __name__ == '__main__':
    import tables
    import numarray

    print '-=' * 38
    print "PyTables version:  %s" % tables.__version__
    print "Extension version: %s" % tables.ExtVersion
    print "HDF5 version:      %s" % tables.whichLibVersion("hdf5")[1]
    #print "HDF5 version:      %s" % tables.HDF5Version
    print "numarray version:  %s" % numarray.__version__
    print "Zlib version:      %s" % tables.whichLibVersion("zlib")[1]
    tinfo = tables.whichLibVersion("lzo")
    if tinfo[0]:
        print "LZO version:       %s (%s)" % (tinfo[1], tinfo[2])
    tinfo = tables.whichLibVersion("ucl")
    if tinfo[0]:
        print "UCL version:       %s (%s)" % (tinfo[1], tinfo[2])
    print 'Python version:    %s' % sys.version
    if os.name == 'posix':
        (sysname, nodename, release, version, machine) = os.uname()
        print 'Platform:          %s-%s' % (sys.platform, machine)
    print 'Byte-ordering:     %s' % sys.byteorder
    print '-=' * 38

    # Handle --only-versions
    only_versions = 0
    args = sys.argv[:]
    for arg in args:
        if arg == '--only-versions':
            only_versions = 1
            sys.argv.remove(arg)

    if not only_versions:
        unittest.main( defaultTest='suite' )

