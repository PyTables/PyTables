from tables import *

if __name__=="__main__":
    import sys
    import getopt

    usage = \
"""usage: %s [-v level] file
  -v level means dumping detailed Table and Array attributes
            1 - medium verbosity
            2 - great verbosity
  -d means detailed information divided in Groups, Arrays and Tables
            \n""" \
    % sys.argv[0]

    try:
        opts, pargs = getopt.getopt(sys.argv[1:], 'v:d')
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
            verbose = int(option[1])
        if option[0] == '-d':
            detailed = 1

    # Catch the hdf5 file passed as the last argument
    file = pargs[0]

    h5file = openFile(file, 'r')
    # Print all their content
    print "Filename:", file
    print "All objects:"
    if verbose >= 1:
        print repr(h5file)
    else:
        print h5file

    if detailed:
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
                print "Info on the object:", repr(table)

        print
        print "Arrays:"
        for group in h5file.walkGroups(h5file.root):
            for array_ in h5file.listNodes(group, 'Array'):
                print "Info on the object:", repr(array_)
    
    # Close the file
    h5file.close()
    
