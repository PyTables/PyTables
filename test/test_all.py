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
                    'test_types',
                    'test_numarray',
                    'test_lists',
                    'test_tables',
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

    test_modules2 = [
                    'test_create',
                    'test_types',
                    'test_numarray',
                    'test_tables',
		    'test_tree', 
		    ]
    
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
    print "HDF5 version:      %s" % tables.HDF5Version
    print "numarray version:  %s" % numarray.__version__
    tinfo = tables.isLibAvailable("zlib")
    #print "zlib version:      %s" % tinfo[1]
    tinfo = tables.isLibAvailable("lzo")
    if tinfo[0]:
        print "LZO version:       %s (%s)" % (tinfo[1], tinfo[2])
    tinfo = tables.isLibAvailable("ucl")
    #if type(tinfo) is type(()):
    if tinfo[0]: ### Hi ha que arreglar-ho!
        print "UCL version:       %s (%s)" % (tinfo[1], tinfo[2])
    print 'Python version:    %s' % sys.version
    if os.name == 'posix':
        (sysname, nodename, release, version, machine) = os.uname()
        print 'Platform:          %s-%s' % (sys.platform, machine)
    print 'Byte-ordering:     %s' % sys.byteorder
    print '-=' * 38

    unittest.main( defaultTest='suite' )

