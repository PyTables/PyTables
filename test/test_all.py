"""
Run all test cases.
"""

import sys
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
                    'test_tables',
                    'test_types',
                    'test_array',
		    'test_tree',
		    ]

    alltests = unittest.TestSuite()
    for name in test_modules:
        module = __import__(name)
        alltests.addTest(module.suite())
    return alltests


if __name__ == '__main__':
    import tables
    #from tables.hdf5Extension import getHDF5Version, getExtCVSVersion, cvsid
    from tables.hdf5Extension import getHDF5LibraryVersion, \
                                     getPyTablesVersion, \
                                     getExtCVSVersion
    print '-=' * 38
    print "HDF5 version:            %s" % getHDF5LibraryVersion()
    print "PyTables version:        %s" % getPyTablesVersion()
    #print "Extension CVS version:   %s" % cvsid
    print "Extension CVS version:   %s" % getExtCVSVersion()
    print 'python version:          %s' % sys.version
    print '-=' * 38
    #raise SystemExit

    unittest.main( defaultTest='suite' )

