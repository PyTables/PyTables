"""
Run all test cases.
"""

import sys
import os
import unittest
from numarray import *
from numarray import strings

verbose = 0
heavy = 0  # Default is not doing heavy testing
if 'verbose' in sys.argv:
    verbose = 1
    sys.argv.remove('verbose')

if 'silent' in sys.argv:  # take care of old flag, just in case
    verbose = 0
    sys.argv.remove('silent')

if '--heavy' in sys.argv:
    heavy = 1
    sys.argv.remove('--heavy')


# This little hack is for when this module is run as main and all the
# other modules import it so they will still be able to get the right
# verbose and heavy settings.  It's confusing but it works.
import test_all
test_all.verbose = verbose
test_all.heavy = heavy

def allequal(a,b, flavor="numarray"):
    """Checks if two numarrays are equal"""

#     print "a-->", repr(a)
#     print "b-->", repr(b)
    if not hasattr(b, "shape"):
        # Scalar case
        return a == b

    if flavor == "Numeric":
        # Convert the parameters to numarray objects
        try:
            a = array(buffer(a), type=typeDict[a.typecode()], shape=a.shape)
        except:
            try:
                a = strings.array(a.tolist(), itemsize=1, shape=a.shape)
            except:
                # Fallback case
                a = array(a)
        try:
            b = array(buffer(b), type=typeDict[b.typecode()], shape=b.shape)
        except:
            try:
                b = strings.array(b.tolist(), itemsize=1, shape=b.shape)
            except:
                # Fallback case
                b = array(b)

    if a.shape <> b.shape:
        if verbose:
            print "Shape is not equal:", a.shape, "<>", b.shape
        return 0

    if hasattr(b, "type") and a.type() <> b.type():
        if verbose:
            print "Type is not equal:", a.type(), "<>", b.type()
        return 0

    # Rank-0 case
    if len(a.shape) == 0:
        if str(equal(a,b)) == '1':
            return 1
        else:
            if verbose:
                print "Shape is not equal:", a.shape, "<>", b.shape
            return 0

    # Null arrays
    if len(a._data) == 0:  # len(a) is not correct for generic shapes
        if len(b._data) == 0:
            return 1
        else:
            if verbose:
                print "length is not equal"
                print "len(a._data) ==>", len(a._data)
                print "len(b._data) ==>", len(b._data)
            return 0

    # Multidimensional case
    result = (a == b)
    for i in range(len(a.shape)):
        #print "nonzero(a <> b)", nonzero(a<>b)
        result = logical_and.reduce(result)
    if not result and verbose:
        print "Some of the elements in arrays are not equal"
        
    return result


def suite():
    test_modules = ['test_basics',
                    'test_create',
                    'test_backcompat',
                    'test_types',
                    'test_numarray',
                    'test_lists',
                    'test_tables',
                    'test_indexes',
                    'test_indexvalues',
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
        print '-=' * 38
        test_modules.append("test_Numeric")
    except:
        print "Skipping Numeric test suite"
        print '-=' * 38

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
    #print "Zlib version:      %s" % tables.whichLibVersion("zlib")[1]
    tinfo = tables.whichLibVersion("zlib")
    if tinfo[0]:
        print "Zlib version:      %s" % (tinfo[1])
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

    # Handle --show-versions-only
    only_versions = 0    
    args = sys.argv[:]
    for arg in args:
        if arg == '--show-versions-only':
            only_versions = 1
            sys.argv.remove(arg)

    if not only_versions:
        if heavy:
            print \
"""Performing the complete test suite!"""
        else:
            print \
"""Performing only a light (yet comprehensive) subset of the complete
test suite.  If you have a big system and lots of CPU to waste, try
passing the --heavy flag to this script. The complete suite will take
more than 7 minutes to complete on a relatively modern CPU
(Pentium4@2GHz) and around 80 MB of memory."""
        print '-=' * 38

        unittest.main( defaultTest='suite' )

