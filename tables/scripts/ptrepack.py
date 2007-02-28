########################################################################
#
#       License: BSD
#       Created: February 10, 2004
#       Author:  Francesc Alted - falted@pytables.org
#
#       $Id$
#
########################################################################

"""This utility lets you repack your data files in a flexible way.

Pass the flag -h to this for help on usage.

"""

import sys
import os.path
import time
import getopt
import warnings

from tables.file import openFile
from tables.group import Group
from tables.leaf import Filters
from tables.exceptions import OldIndexWarning, NoIndexingWarning

# Global variables
verbose = False
regoldindexes = True


def newdstGroup(dstfileh, dstgroup, title, filters):
    group = dstfileh.root
    # Now, create the new group. This works even if dstgroup == '/'
    for nodeName in dstgroup.split('/'):
        if nodeName == '':
            continue
        if not hasattr(dstfileh, nodeName):
            group2 = dstfileh.createGroup(group, nodeName,
                                          title=title,
                                          filters=filters)
        else:
            # We assume that nodeName is a group. If not, an error will
            # be issued later on.
            group2 = dstfileh._f_getChild(nodeName)
        group = group2
    return group


def recreateIndexes(table, dstfileh, dsttable):
    listoldindexes = table._listoldindexes
    if listoldindexes != []:
        if not regoldindexes:
            if verbose:
                print "[I]Not regenerating indexes for table: '%s:%s'" % \
                      (dstfileh.filename, dsttable._v_pathname)
            return
        # Now, recreate the indexed columns
        if verbose:
            print "[I]Regenerating indexes for table: '%s:%s'" % \
                  (dstfileh.filename, dsttable._v_pathname)
        for colname in listoldindexes:
            if verbose:
                print "[I]Indexing column: '%s'. Please wait..." % colname
            colobj = dsttable.cols._f_col(colname)
            # We don't specify the filters for the indexes
            colobj.createIndex(filters = None)


def copyLeaf(srcfile, dstfile, srcnode, dstnode, title,
             filters, copyuserattrs, overwritefile, overwrtnodes, stats,
             start, stop, step):
    # Open the source file
    srcfileh = openFile(srcfile, "r")
    # Get the source node (that should exist)
    srcNode = srcfileh.getNode(srcnode)

    # Get the destination node and its parent
    last_slash = dstnode.rindex('/')
    if last_slash == len(dstnode)-1:
        # print "Detected a trainling slash in destination node. Interpreting it as a destination group."
        dstgroup = dstnode[:-1]
    elif last_slash > 0:
        dstgroup = dstnode[:last_slash]
    else:
        dstgroup = "/"
    dstleaf = dstnode[last_slash+1:]
    if dstleaf == "":
        dstleaf = srcNode.name
    # Check whether the destination group exists or not
    if os.path.isfile(dstfile) and not overwritefile:
        dstfileh = openFile(dstfile, "a")
        try:
            dstGroup = dstfileh.getNode(dstgroup)
        except:
            # The dstgroup does not seem to exist. Try creating it.
            dstGroup = newdstGroup(dstfileh, dstgroup, title, filters)
        else:
            # The node exists, but it is really a group?
            if not isinstance(dstGroup, Group):
                # No. Should we overwrite it?
                if overwrtnodes:
                    parent = dstGroup._v_parent
                    last_slash = dstGroup._v_pathname.rindex('/')
                    dstgroupname = dstGroup._v_pathname[last_slash+1:]
                    dstGroup.remove()
                    dstGroup = dstfileh.createGroup(parent, dstgroupname,
                                                    title=title,
                                                    filters=filters)
                else:
                    raise RuntimeError, "Please check that the node names are not duplicated in destination, and if so, add the --overwrite-nodes flag if desired."
    else:
        # The destination file does not exist or will be overwritten.
        dstfileh = openFile(dstfile, "w", title=title, filters=filters)
        dstGroup = newdstGroup(dstfileh, dstgroup, title="", filters=filters)

    # Finally, copy srcNode to dstNode
    try:
        dstNode = srcNode.copy(
            dstGroup, dstleaf, filters = filters,
            copyuserattrs = copyuserattrs, overwrite = overwrtnodes,
            stats = stats, start = start, stop = stop, step = step)
    except:
        (type, value, traceback) = sys.exc_info()
        print "Problems doing the copy from '%s:%s' to '%s:%s'" % \
              (srcfile, srcnode, dstfile, dstnode)
        print "The error was --> %s: %s" % (type, value)
        print "The destination file looks like:\n", dstfileh
        # Close all the open files:
        srcfileh.close()
        dstfileh.close()
        raise RuntimeError, "Please check that the node names are not duplicated in destination, and if so, add the --overwrite-nodes flag if desired."

    # Recreate possible old indexes in destination node
    if srcNode._c_classId == "TABLE":
        recreateIndexes(srcNode, dstfileh, dstNode)

    # Close all the open files:
    srcfileh.close()
    dstfileh.close()


def copyChildren(srcfile, dstfile, srcgroup, dstgroup, title,
                 recursive, filters, copyuserattrs, overwritefile,
                 overwrtnodes, stats, start, stop, step):
    "Copy the children from source group to destination group"
    # Open the source file with srcgroup as rootUEP
    srcfileh = openFile(srcfile, "r", rootUEP=srcgroup)
    #  Assign the root to srcGroup
    srcGroup = srcfileh.root

    # Check whether the destination group exists or not
    if os.path.isfile(dstfile) and not overwritefile:
        dstfileh = openFile(dstfile, "a")
        try:
            dstGroup = dstfileh.getNode(dstgroup)
        except:
            # The dstgroup does not seem to exist. Try creating it.
            dstGroup = newdstGroup(dstfileh, dstgroup, title, filters)
        else:
            # The node exists, but it is really a group?
            if not isinstance(dstGroup, Group):
                # No. Should we overwrite it?
                if overwrtnodes:
                    parent = dstGroup._v_parent
                    last_slash = dstGroup._v_pathname.rindex('/')
                    dstgroupname = dstGroup._v_pathname[last_slash+1:]
                    dstGroup.remove()
                    dstGroup = dstfileh.createGroup(parent, dstgroupname,
                                                    title=title,
                                                    filters=filters)
                else:
                    raise RuntimeError, "Please check that the node names are not duplicated in destination, and if so, add the --overwrite-nodes flag if desired."
    else:
        # The destination file does not exist or will be overwritten.
        dstfileh = openFile(dstfile, "w", title=title, filters=filters)
        dstGroup = newdstGroup(dstfileh, dstgroup, title="", filters=filters)

    # Finally, copy srcGroup to dstGroup
    try:
        srcGroup._f_copyChildren(
            dstGroup, recursive = recursive, filters = filters,
            copyuserattrs = copyuserattrs, overwrite = overwrtnodes,
            stats = stats, start = start, stop = stop, step = step)
    except:
        (type, value, traceback) = sys.exc_info()
        print "Problems doing the copy from '%s:%s' to '%s:%s'" % \
              (srcfile, srcgroup, dstfile, dstgroup)
        print "The error was --> %s: %s" % (type, value)
        print "The destination file looks like:\n", dstfileh
        # Close all the open files:
        srcfileh.close()
        dstfileh.close()
        raise RuntimeError, "Please check that the node names are not duplicated in destination, and if so, add the --overwrite-nodes flag if desired. In particular, pay attention that rootUEP is not fooling you."

    # Do a second pass and convert the remaining tables with old
    # indexes (if any)
    for table in srcGroup._f_walkNodes("Table"):
        dsttable = dstfileh.getNode(dstGroup, table._v_pathname)
        recreateIndexes(table, dstfileh, dsttable)

    # Close all the open files:
    srcfileh.close()
    dstfileh.close()


def main():
    global verbose
    global regoldindexes

    usage = """usage: %s [-h] [-v] [-o] [-R start,stop,step] [--non-recursive] [--dest-title=title] [--dont-copyuser-attrs] [--overwrite-nodes] [--complevel=(0-9)] [--complib=lib] [--shuffle=(0|1)] [--fletcher32=(0|1)] [--keep-source-filters] [--dont-regenerate-old-indexes] sourcefile:sourcegroup destfile:destgroup
     -h -- Print usage message.
     -v -- Show more information.
     -o -- Overwite destination file.
     -R RANGE -- Select a RANGE of rows (in the form "start,stop,step")
         during the copy of *all* the leaves.
     --non-recursive -- Do not do a recursive copy. Default is to do it.
     --dest-title=title -- Title for the new file (if not specified,
         the source is copied).
     --dont-copy-userattrs -- Do not copy the user attrs (default is to do it)
     --overwrite-nodes -- Overwrite destination nodes if they exist. Default is
         to not overwrite them.
     --complevel=(0-9) -- Set a compression level (0 for no compression, which
         is the default).
     --complib=lib -- Set the compression library to be used during the copy.
         lib can be set to "zlib", "lzo" or "bzip2". Defaults to "zlib".
     --shuffle=(0|1) -- Activate or not the shuffling filter (default is active
         if complevel>0).
     --fletcher32=(0|1) -- Whether to activate or not the fletcher32 filter
        (not active by default).
     --keep-source-filters -- Use the original filters in source files. The
         default is not doing that if any of --complevel, --complib, --shuffle
         or --fletcher32 option is specified.
     --dont-regenerate-old-indexes -- Disable regenerating old indexes. The
         default is to regenerate old indexes as they are found.
    \n""" % os.path.basename(sys.argv[0])


    try:
        opts, pargs = getopt.getopt(sys.argv[1:], 'hvoR:',
                                    ['non-recursive',
                                     'dest-title=',
                                     'dont-copy-userattrs',
                                     'overwrite-nodes',
                                     'complevel=',
                                     'complib=',
                                     'shuffle=',
                                     'fletcher32=',
                                     'keep-source-filters',
                                     'dont-regenerate-old-indexes',
                                     ])
    except:
        (type, value, traceback) = sys.exc_info()
        print "Error parsing the options. The error was:", value
        sys.stderr.write(usage)
        sys.exit(0)

    # default options
    overwritefile = False
    keepfilters = False
    complevel = None
    complib = None
    shuffle = None
    fletcher32 = None
    title = ""
    copyuserattrs = True
    rng = None
    recursive = True
    overwrtnodes = False

    # Get the options
    for option in opts:
        if option[0] == '-h':
            sys.stderr.write(usage)
            sys.exit(0)
        elif option[0] == '-v':
            verbose = True
        elif option[0] == '-o':
            overwritefile = True
        elif option[0] == '-R':
            try:
                rng = eval("slice("+option[1]+")")
            except:
                print "Error when getting the range parameter."
                (type, value, traceback) = sys.exc_info()
                print "  The error was:", value
                sys.stderr.write(usage)
                sys.exit(0)
        elif option[0] == '--dest-title':
            title = option[1]
        elif option[0] == '--dont-copy-userattrs':
            copyuserattrs = False
        elif option[0] == '--non-recursive':
            recursive = False
        elif option[0] == '--overwrite-nodes':
            overwrtnodes = True
        elif option[0] == '--keep-source-filters':
            keepfilters = True
        elif option[0] == '--dont-regenerate-old-indexes':
            regoldindexes = False
        elif option[0] == '--complevel':
            complevel = int(option[1])
        elif option[0] == '--complib':
            complib = option[1]
        elif option[0] == '--shuffle':
            shuffle = int(option[1])
        elif option[0] == '--fletcher32':
            fletcher32 = int(option[1])
        else:
            print option[0], ": Unrecognized option"
            sys.stderr.write(usage)
            sys.exit(0)

    # if we pass a number of files different from 2, abort
    if len(pargs) <> 2:
        print "You need to pass both source and destination!."
        sys.stderr.write(usage)
        sys.exit(0)

    # Catch the files passed as the last arguments
    src = pargs[0].split(':')
    dst = pargs[1].split(':')
    if len(src) == 1:
        srcfile, srcnode = src[0], "/"
    else:
        srcfile, srcnode = src
    if len(dst) == 1:
        dstfile, dstnode = dst[0], "/"
    else:
        dstfile, dstnode = dst

    if srcnode == "":
        # case where filename == "filename:" instead of "filename:/"
        srcnode = "/"

    if dstnode == "":
        # case where filename == "filename:" instead of "filename:/"
        dstnode = "/"

    # Ignore the warnings for tables that contains oldindexes
    # (these will be handled by the copying routines)
    warnings.filterwarnings("ignore", category=OldIndexWarning)
    # Let the user be warned in case he is using ptrepack when copying
    # files with indexes
    #warnings.filterwarnings("ignore", category=NoIndexingWarning)

    # Build the Filters instance
    if ((complevel, complib, shuffle, fletcher32) == (None,)*4 or keepfilters):
        filters = None
    else:
        if complevel is None: complevel = 0
        if complevel > 0 and shuffle is None:
            shuffle = True
        else:
            shuffle = False
        if complib is None: complib = "zlib"
        if fletcher32 is None: fletcher32 = False
        filters = Filters(complevel=complevel, complib=complib,
                          shuffle=shuffle, fletcher32=fletcher32)

    # The start, stop and step params:
    start, stop, step = 0, None, 1  # Defaults
    if rng:
        start, stop, step = rng.start, rng.stop, rng.step

    # Some timing
    t1 = time.time()
    cpu1 = time.clock()
    # Copy the file
    if verbose:
        print "+=+"*20
        print "Recursive copy:", recursive
        print "Applying filters:", filters
        print "Starting copying %s:%s to %s:%s" % (srcfile, srcnode,
                                                   dstfile, dstnode)
        print "+=+"*20

    # Check whether the specified source node is a group or a leaf
    h5srcfile = openFile(srcfile, 'r')
    srcnodeobject = h5srcfile.getNode(srcnode)
    objectclass = srcnodeobject.__class__.__name__
    # Close the file again
    h5srcfile.close()

    stats = {'groups': 0, 'leaves': 0, 'bytes': 0}
    if isinstance(srcnodeobject, Group):
        copyChildren(
            srcfile, dstfile, srcnode, dstnode,
            title = title, recursive = recursive, filters = filters,
            copyuserattrs = copyuserattrs, overwritefile = overwritefile,
            overwrtnodes = overwrtnodes, stats = stats,
            start = start, stop = stop, step = step)
    else:
        # If not a Group, it should be a Leaf
        copyLeaf(
            srcfile, dstfile, srcnode, dstnode,
            title = title, filters = filters, copyuserattrs = copyuserattrs,
            overwritefile = overwritefile, overwrtnodes = overwrtnodes,
            stats = stats, start = start, stop = stop, step = step)

    # Gather some statistics
    t2 = time.time()
    cpu2 = time.clock()
    tcopy = round(t2-t1, 3)
    cpucopy = round(cpu2-cpu1, 3)
    tpercent = int(round(cpucopy/tcopy, 2)*100)

    if verbose:
        ngroups = stats['groups']
        nleafs = stats['leaves']
        nbytescopied = stats['bytes']

        print "Groups copied:", ngroups, " Leaves copied:", nleafs
        if copyuserattrs:
            print "User attrs copied"
        else:
            print "User attrs not copied"
        print "KBytes copied:", round(nbytescopied/1024.,3)
        print "Time copying: %s s (real) %s s (cpu)  %s%%" % \
              (tcopy, cpucopy, tpercent)
        print "Copied nodes/sec: ", round((ngroups+nleafs) / float(tcopy),1)
        print "Copied KB/s :", int(nbytescopied / (tcopy * 1024))
