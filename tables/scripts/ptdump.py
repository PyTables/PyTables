########################################################################
#
#       License: BSD
#       Created: February 10, 2004
#       Author:  Francesc Alted - faltet@pytables.com
#
#       $Id$
#
########################################################################

"""This utility lets you look into the data and metadata of your data files.

Pass the flag -h to this for help on usage.

"""

import sys
import os.path
import getopt

from tables.file import openFile
from tables.group import Group
from tables.leaf import Leaf
from tables.table import Table
from tables.unimplemented import UnImplemented


# default options
class Options(object):
    rng = slice(None)
    showattrs = 0
    verbose = 0
    dump = 0
    colinfo = 0
    idxinfo = 0

options = Options()


def dumpLeaf(leaf):
    if options.verbose:
        print repr(leaf)
    else:
        print str(leaf)
    if options.showattrs:
        print "  "+repr(leaf.attrs)
    if options.dump and not isinstance(leaf, UnImplemented):
        print "  Data dump:"
        # print leaf.read(options.rng.start, options.rng.stop, options.rng.step)
        # This is better for large objects
        if options.rng.start is None:
            start = 0
        else:
            start = options.rng.start
        if options.rng.stop is None:
            if leaf.shape != ():
                stop = leaf.shape[0]
        else:
            stop = options.rng.stop
        if options.rng.step is None:
            step = 1
        else:
            step = options.rng.step
        if leaf.shape == ():
            print "[SCALAR] %s" % (leaf[()])
        else:
            for i in range(start, stop, step):
                print "[%s] %s" % (i, leaf[i])

    if isinstance(leaf, Table) and options.colinfo:
        # Show info of columns
        for colname in leaf.colnames:
            print repr(leaf.cols._f_col(colname))

    if isinstance(leaf, Table) and options.idxinfo:
        # Show info of indexes
        for colname in leaf.colnames:
            if leaf.cols._f_col(colname).index is not None:
                idx = leaf.cols._f_col(colname).index
                print repr(idx)


def dumpGroup(pgroup):
    node_kinds = pgroup._v_file._node_kinds[1:]
    for group in pgroup._f_walkGroups():
        print str(group)
        if options.showattrs:
            print "  "+repr(group._v_attrs)
        for kind in node_kinds:
            for node in group._f_listNodes(kind):
                if options.verbose or options.dump:
                    dumpLeaf(node)
                else:
                    print str(node)



def main():
    usage = \
    """usage: %s [-d] [-v] [-a] [-c] [-i] [-R start,stop,step] [-h] file[:nodepath]
      -d -- Dump data information on leaves
      -v -- Dump more metainformation on nodes
      -a -- Show attributes in nodes (only useful when -v or -d are active)
      -c -- Show info of columns in tables (only useful when -v or -d are active)
      -i -- Show info of indexed columns (only useful when -v or -d are active)
      -R RANGE -- Select a RANGE of rows in the form "start,stop,step"
      -h -- Print help on usage
                \n""" \
    % os.path.basename(sys.argv[0])

    try:
        opts, pargs = getopt.getopt(sys.argv[1:], 'R:ahdvci')
    except:
        sys.stderr.write(usage)
        sys.exit(0)

    # if we pass too much parameters, abort
    if len(pargs) <> 1:
        sys.stderr.write(usage)
        sys.exit(0)

    # Get the options
    for option in opts:
        if option[0] == '-R':
            options.dump = 1
            try:
                options.rng = eval("slice("+option[1]+")")
            except:
                print "Error when getting the range parameter."
                (type, value, traceback) = sys.exc_info()
                print "  The error was:", value
                sys.stderr.write(usage)
                sys.exit(0)

        elif option[0] == '-a':
            options.showattrs = 1
        elif option[0] == '-h':
            sys.stderr.write(usage)
            sys.exit(0)
        elif option[0] == '-v':
            options.verbose = 1
        elif option[0] == '-d':
            options.dump = 1
        elif option[0] == '-c':
            options.colinfo = 1
        elif option[0] == '-i':
            options.idxinfo = 1
        else:
            print option[0], ": Unrecognized option"
            sys.stderr.write(usage)
            sys.exit(0)

    # Catch the files passed as the last arguments
    src = pargs[0].split(':')
    if len(src) == 1:
        filename, nodename = src[0], "/"
    else:
        filename, nodename = src
        if nodename == "":
            # case where filename == "filename:" instead of "filename:/"
            nodename = "/"

    # Check whether the specified node is a group or a leaf
    h5file = openFile(filename, 'r')
    nodeobject = h5file.getNode(nodename)
    if isinstance(nodeobject, Group):
        # Close the file again and reopen using the rootUEP
        dumpGroup(nodeobject)
    elif isinstance(nodeobject, Leaf):
        # If it is not a Group, it must be a Leaf
        dumpLeaf(nodeobject)
    else:
        # This should never happen
        print "Unrecognized object:", nodeobject

    # Close the file
    h5file.close()
