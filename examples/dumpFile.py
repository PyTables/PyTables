#!/usr/bin/env python2.3

from tables import *

if __name__=="__main__":
    import sys
    import getopt

    usage = \
"""usage: %s [-v] [-d] file
  -v means dumping Table and Array attributes
  -d means detailed information divided in Groups, Arrays and Tables
            \n""" \
    % sys.argv[0]

    try:
        opts, pargs = getopt.getopt(sys.argv[1:], 'vd')
    except:
        sys.stderr.write(usage)
        sys.exit(0)

    # if we pass too much parameters, abort
    if len(pargs) <> 1: 
        sys.stderr.write(usage)
        sys.exit(0)

    # default options
    verbose = 0
    detailed = 0

    # Get the options
    for option in opts:
        if option[0] == '-v':
            verbose = 1
        if option[0] == '-d':
            detailed = 1

    # Catch the hdf5 file passed as the last argument
    file = pargs[0]

    h5file = openFile(file, 'r')
    # Print all their content
    if verbose:
        print repr(h5file)
    else:
        print h5file

    if verbose:
        # Print detailed info on leaf objects

        print "Attribute info of the objects:"
        for group in h5file(classname="Group"):
            print repr(group._v_attrs)
            for leaf in h5file.listNodes(group, 'Leaf'):
                print repr(leaf.attrs)
        print

    if detailed:
        # Print detailed info on leaf objects

        print "Detailed info on this file object tree follows:"
        print "Groups:"
        for group in h5file.walkGroups(h5file.root):
            print repr(group)

        print
        print "Leaves:"
        for group in h5file.walkGroups(h5file.root):
            for leaf in h5file.listNodes(group, 'Leaf'):
                print leaf

        print
        print "Tables:"
        for group in h5file.walkGroups(h5file.root):
            for table in h5file.listNodes(group, 'Table'):
                print "Info on the object:", repr(table)

        print
        print "Arrays:"
        for group in h5file.walkGroups(h5file.root):
            for array_ in h5file.listNodes(group, 'Array'):
                print "Info on the object:", repr(array_)

        print
        print "EArrays:"
        for group in h5file.walkGroups(h5file.root):
            for array_ in h5file.listNodes(group, 'EArray'):
                print "Info on the object:", repr(array_)
        print
        print "VLArrays:"
        for group in h5file.walkGroups(h5file.root):
            for array_ in h5file.listNodes(group, 'VLArray'):
                print "Info on the object:", repr(array_)
    
    # Close the file
    h5file.close()
    
