from tables import *


if __name__=="__main__":
    import sys
    import getopt

    usage = \
"""usage: %s [-v] file
  -v means dumping detailed Table and Array attributes\n""" \
    % sys.argv[0]

    try:
        opts, pargs = getopt.getopt(sys.argv[1:], 'v')
    except:
        sys.stderr.write(usage)
        sys.exit(0)

    # if we pass too much parameters, abort
    if len(pargs) <> 1: 
        sys.stderr.write(usage)
        sys.exit(0)

    # default options
    verbose = 0

    # Get the options
    for option in opts:
        if option[0] == '-v':
            verbose = 1

    # Catch the hdf5 file passed as the last argument
    file = pargs[0]

    h5file = openFile(file, 'r')
    # Print all their content
    print "Filename:", file
    print "All objects:"
    print h5file

    if verbose:
        # Print detailed info on leaf objects

        print "Detailed info on this file object tree follows:"
        print "Groups:"
        for group in h5file.walkGroups(h5file.root):
            print group

        print
        print "Leaves:"
        for group in h5file.walkGroups(h5file.root):
            for leaf in h5file.listNodes(group, 'Leaf'):
                print leaf

        print
        print "Tables:"
        for group in h5file.walkGroups(h5file.root):
            for table in h5file.listNodes(group, 'Table'):
                print "Info on the object:", table
                print "  name ==>", table.name
                print "  title ==>", table.title
                print "  rows on table ==> %d" % (table.nrows)
                print "  variable names with their type:"
                for i in range(len(table.varnames)):
                    print "    ", table.varnames[i], ':=', table.vartypes[i] 

        print
        print "Arrays:"
        for group in h5file.walkGroups(h5file.root):
            for array_ in h5file.listNodes(group, 'Array'):
                print "Info on the object:", array_
                print "  name ==>", array_.name
                print "  title ==>", array_.title
                print "  shape ==>", array_.shape
                print "  typecode ==>", array_.typecode
                print "  byteorder ==>", array_.byteorder
                #print "  content:\n", array_.read()
    
    # Close the file
    h5file.close()
    
