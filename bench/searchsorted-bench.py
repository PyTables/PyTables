#!/usr/bin/env python

import copy

import time
import numarray as NA
#import bisect
from tables import *
from numarray import random_array

def createFile(filename, chunksize, nchunks, filters, atom, verbose):

    # Open a file in "w"rite mode
    fileh = openFile(filename, mode = "w", title="Searchsorted Benchmark",
                     filters=filters)
    title = "This is the IndexArray title"
    # Create an IndexArray instance
    group = fileh.root
    rowswritten = 0
    if atom == "float":
        atomtype = FloatAtom()
        #arr = random_array.uniform(0, chunksize)
    elif atom == "int":
        atomtype = IntAtom()
        #arr = random_array.randint(0, chunksize)
    elif atom == "bool":
        atomtype = BoolAtom()
    elif atom == "string":
        atomtype = StringAtom(length=4)
    else:
        raise RuntimeError, "This should never happen"
    for j in range(3):
        # Create an entry
        table = fileh.createIndexArray(group, 'iarray'+str(j),
                                       atomtype,
                                       title,
                                       None,  # Filters are inherited
                                       nchunks*chunksize)
        recnchunks = (nchunks * chunksize) / table.nelemslice
        if verbose and j == 0:
            print "elements per slice:", table.nelemslice
            print "chunksize:", table.chunksize
            print "nchunks:", recnchunks

        arr = NA.arange(table.nelemslice, type=atomtype.type,
                        shape=(1, table.nelemslice))
        for i in range(recnchunks):
            #table.append(xrange(table.nelemslice))
            table.append(arr)
#             a = xrange(table.nelemslice)
#             col = NA.array(a, shape=(1, table.nelemslice),
#                            type=atomtype.type)
#             time.sleep(0.2)
            rowswritten += 1
        # Create a new group
        group2 = fileh.createGroup(group, 'group'+str(j))
        # Iterate over this new group (group2)
        group = group2

    rowsize = table.rowsize
    # Close the file (eventually destroy the extended type)
    fileh.close()
    
    return (rowswritten, rowsize)

def readFile(filename, atom, verbose):
    # Open the HDF5 file in read-only mode

    fileh = openFile(filename, mode = "r")
    rowsread = 0
    level = 0
    for groupobj in fileh.walkGroups(fileh.root):
        for table in fileh.listNodes(groupobj, 'IndexArray'):
            rowsize = table.rowsize
            print "reading", table
            if verbose and level == 0:
                print "Max rows in buf:", table._v_maxTuples
                print "Rows in", table._v_pathname, ":", table.nrows
                print "Buffersize:", table.rowsize * table._v_maxTuples
                print "MaxTuples:", table._v_maxTuples
                print "Chunksize:", table.chunksize

            for i in range(table.nrows):
                arr = table[i,:]
            if verbose:
                print "last value read ==>", arr[-1]

	    rowsread += table.nrows
            level += 1
        
    # Close the file (eventually destroy the extended type)
    fileh.close()

    return (rowsread, rowsize)


def searchBin_orig(table, i, item):

    lo = 0
    hi = table.nelemslice   # Number of elements / slice
    bufsize = table.chunksize # Number of elements / chunk
    buffer = table._readSortedSlice(i, 0, bufsize)
    #buffer = table[i,0:bufsize]    # 1.5 / 2.1 sec
    #buffer = NA.arange(0,bufsize)  # test  # 0.44 / 2 sec
    #buffer = range(0,bufsize)  # test   #
    #buffer = xrange(0,bufsize)  # test  # 0.02 / 2 sec
    result = bisect_left(buffer, item)
    niter = 1
    if 0 <= result < bufsize:
        return result, niter
    # The end
    buffer = table._readSortedSlice(i, hi-bufsize, hi)
    #buffer = table[i,-bufsize:]
    #buffer = NA.arange(hi-bufsize,hi)  # test
    #buffer = range(hi-bufsize,hi) # test
    #buffer = xrange(hi-bufsize,hi) # test
    niter = 2
    result = bisect_left(buffer, item)
    if 0 < result < bufsize:
        return hi - bufsize + result, niter
    elif result == bufsize:
        return hi, niter
    while lo < hi:
        mid = (lo+hi)//2
        start = (mid/bufsize)*bufsize
        buffer = table._readSortedSlice(i, start, start+bufsize)
        #buffer = table[i,start:start+bufsize]
        #buffer = NA.arange(start,start+bufsize)  # test
        #buffer = range(start,start+bufsize) # test
        #buffer = xrange(start,start+bufsize) # test
        niter += 1
        result = bisect_left(buffer, item)
        if result == 0:
            if buffer[result] == item:
                lo = start
                break
            # The item is at left
            hi = mid
        elif result == bufsize:
            # The item is at the right
            lo = mid+1
        else:
            # Item has been found. Exit the loop and return
            lo = result+start
            break
    return (lo, niter)

# This has been copied from the standard module bisect.
# Checks for the values out of limits has been added at the beginning
# because I forsee that this should be a very common case.
# 2004-05-20
def bisect_left(a, x, lo=0, hi=None):
    """Return the index where to insert item x in list a, assuming a is sorted.

    The return value i is such that all e in a[:i] have e < x, and all e in
    a[i:] have e >= x.  So if x already appears in the list, i points just
    before the leftmost x already there.

    """

    if hi == None:
        hi = len(a)
    if x <= a[0]: return 0
    if a[-1] < x: return hi
    while lo < hi:
        mid = (lo+hi)//2
        if a[mid] < x: lo = mid+1
        else: hi = mid
    return lo

def interSearch(table, nrow, bufsize, item, lo, hi):
    niter = 0
    while lo < hi:
        mid = (lo+hi)//2
        start = (mid/bufsize)*bufsize
        buffer = table._readSortedSlice(nrow, start, start+bufsize)
        niter += 1
        result = bisect_left(buffer, item, hi=bufsize)
        if result == 0:
            if buffer[result] == item:
                lo = start
                break
            # The item is at left
            hi = mid
        elif result == bufsize:
            # The item is at the right
            lo = mid+1
        else:
            # Item has been found. Exit the loop and return
            lo = result+start
            break
    return (lo, niter)

def searchBin(table, nrow, item):
    hi = table.nelemslice   # Number of elements / chunk
    item1, item2 = item
    item1done = 0; item2done = 0
    # Look for items at the beginning
    bufsize = table.chunksize # Number of elements/chunksize
    buffer = table._readSortedSlice(nrow, 0, bufsize)
    niter = 1
    result1 = bisect_left(buffer, item1, hi=bufsize)
    if 0 <= result1 < bufsize:
        item1done = 1
    result2 = bisect_left(buffer, item2, hi=bufsize)
    if 0 <= result2 < bufsize:
        item2done = 1
    if item1done and item2done:
        return (result1, result2, niter)
    # The end
    buffer = table._readSortedSlice(nrow, hi-bufsize, hi)
    niter = 2
    if not item1done:
        result1 = bisect_left(buffer, item1, hi=bufsize)
        if 0 < result1 < bufsize:
            item1done = 1
            result1 = hi - bufsize + result
        elif result1 == bufsize:
            item1done = 1
            result1 = hi
    if not item2done:
        result2 = bisect_left(buffer, item2, hi=bufsize)
        if 0 < result2 < bufsize:
            item2done = 1
            result2 = hi - bufsize + result
        elif result2 == bufsize:
            item2done = 1
            result2 = hi
    if item1done and item2done:
        return (result1, result2, niter)

    lo = 0
    # Intermediate look for item1
    if not item1done:
        (result1, iter) = interSearch(table, nrow, bufsize, item1, lo, hi)
        niter += iter
    # Intermediate look for item1
    if not item2done:
        (result2, iter) = interSearch(table, nrow, bufsize, item2, lo, hi)
        niter += iter
    return (result1, result2, niter)


def searchFile(filename, atom, verbose, item):
    # Open the HDF5 file in read-only mode

    fileh = openFile(filename, mode = "r")
    rowsread = 0
    level = 0
    uncomprBytes = 0
    ntotaliter = 0
    for groupobj in fileh.walkGroups(fileh.root):
        for table in fileh.listNodes(groupobj, 'IndexArray'):
            rowsize = table.rowsize
            print "reading", table
            if verbose and level == 0:
                print "Chunk size:", table.chunksize
                print "Chunk number in", table._v_pathname, ":", table.nrows
                print "Buffersize:", table.rowsize * table._v_maxTuples
                print "MaxTuples:", table._v_maxTuples

            #buffer = table[0]  # Test
#             niter = 0  # for tests
#             #buffer = xrange(table.shape[1])  # Test
#             bufsize = table.chunksize # # of elements/chunksize
#             table._initSortedSlice(bufsize, table.type)
#             for i in range(table.nrows):
#                 #(result1, result2, niter) = searchBin(table, i, item)
#                 (result1, result2, niter) = table._searchBin(i, item)
#                 #result = searchBin(buffer, 0, item)
#                 #result = bisect_left(buffer, item)
#             table._destroySortedSlice()
            (result1, result2, niter) = table.searchBin(item)
            if verbose and level == 0:
                print "Position for item ==>", (item, (result1[0],
                                                       result2[0]),
                                                niter/table.nrows)

	    rowsread += table.nrows
            uncomprBytes += table.chunksize * niter * table.itemsize
            ntotaliter += niter
            level += 1
        
    # Close the file (eventually destroy the extended type)
    fileh.close()

    return (rowsread, table.nelemslice, uncomprBytes, ntotaliter)


if __name__=="__main__":
    import sys
    import getopt
    try:
        import psyco
        psyco_imported = 1
    except:
        psyco_imported = 0
    
    import time
    
    usage = """usage: %s [-v] [-p] [-R range] [-r] [-w] [-s item ] [-a
    atom] [-c level] [-l complib] [-S] [-F] [-i chunksize] [-n nchunks] file
            -v verbose
	    -p use "psyco" if available
            -R select a range in a field in the form "start,stop,step"
	    -r only read test
	    -w only write test
            -s item to search
            -a use [float], [int], [bool] or [string] atom
            -c sets a compression level (do not set it or 0 for no compression)
            -S activate shuffling filter
            -F activate fletcher32 filter
            -l sets the compression library to be used ("zlib", "lzo", "ucl")
            -i sets the chunksize (number of records on each chunk)
            -k sets the number of chunks on each index\n""" % sys.argv[0]

    try:
        opts, pargs = getopt.getopt(sys.argv[1:], 'vpSFR:rws:a:c:l:i:n:')
    except:
        sys.stderr.write(usage)
        sys.exit(0)

    # if we pass too much parameters, abort
    if len(pargs) <> 1: 
        sys.stderr.write(usage)
        sys.exit(0)

    # default options
    verbose = 0
    rng = None
    item = None
    atom = "float"
    fieldName = None
    testread = 1
    testwrite = 1
    usepsyco = 0
    complevel = 0
    shuffle = 0
    fletcher32 = 0
    complib = "zlib"
    chunksize = 1000000
    nchunks = 100

    # Get the options
    for option in opts:
        if option[0] == '-v':
            verbose = 1
        if option[0] == '-p':
            usepsyco = 1
        if option[0] == '-S':
            shuffle = 1
        if option[0] == '-F':
            fletcher32 = 1
        elif option[0] == '-R':
            rng = [int(i) for i in option[1].split(",")]
        elif option[0] == '-r':
            testwrite = 0
        elif option[0] == '-w':
            testread = 0
        elif option[0] == '-s':
            item = eval(option[1])
        elif option[0] == '-a':
            atom = option[1]
            if atom not in ["float", "int", "bool", "string"]:
                sys.stderr.write(usage)
                sys.exit(0)
        elif option[0] == '-c':
            complevel = int(option[1])
        elif option[0] == '-l':
            complib = option[1]
        elif option[0] == '-i':
            chunksize = int(option[1])
        elif option[0] == '-n':
            nchunks = int(option[1])
            
#     if atom == "float":
#         item = float(eval(item))
#     elif atom == "int":
#         item = int(eval(item))
#     elif atom == "bool":
#         item = int(eval(item))
#     else:  # string atom case
#         item = int(eval(item))
    # Build the Filters instance
    filters = Filters(complevel=complevel, complib=complib,
                      shuffle=shuffle, fletcher32=fletcher32)

    # Catch the hdf5 file passed as the last argument
    file = pargs[0]

    if testwrite:
        print "Compression level:", complevel
        if complevel > 0:
            print "Compression library:", complib
            if shuffle:
                print "Suffling..."
	t1 = time.time()
	cpu1 = time.clock()
        if psyco_imported and usepsyco:
            psyco.bind(createFile)
	(rowsw, rowsz) = createFile(file, chunksize, nchunks, filters,
                                    atom, verbose)
	t2 = time.time()
        cpu2 = time.clock()
	tapprows = round(t2-t1, 3)
	cpuapprows = round(cpu2-cpu1, 3)
        tpercent = int(round(cpuapprows/tapprows, 2)*100)
	print "Rows written:", rowsw, " Row size:", rowsz
	print "Time writing rows: %s s (real) %s s (cpu)  %s%%" % \
              (tapprows, cpuapprows, tpercent)
	print "Write rows/sec: ", int(rowsw / float(tapprows))
	print "Write KB/s :", int(rowsw * rowsz / (tapprows * 1024))
	
    if testread:
        if psyco_imported and usepsyco:
            psyco.bind(readFile)
            psyco.bind(searchBin)
	t1 = time.time()
        cpu1 = time.clock()
        if rng or item:
            (rowsr, rowsz, uncomprB, niter) = searchFile(file, atom, verbose, item)
        else:
            for i in range(1):
                (rowsr, rowsz) = readFile(file, atom, verbose)
	t2 = time.time()
        cpu2 = time.clock()
	treadrows = round(t2-t1, 3)
        cpureadrows = round(cpu2-cpu1, 3)
        tpercent = int(round(cpureadrows/treadrows, 2)*100)
        tMrows = rowsr*rowsz/(1000*1000)
	print "Rows read:", rowsr, " Row size:", rowsz, \
              "Total:", tMrows, "Mrows"
	print "Time reading rows: %s s (real) %s s (cpu)  %s%%" % \
              (treadrows, cpureadrows, tpercent)
	print "Read Mrows/sec: ", int(tMrows / float(treadrows))
	#print "Read KB/s :", int(rowsr * rowsz / (treadrows * 1024))
	print "Uncompr MB :", int(uncomprB / (1024 * 1024))
	print "Uncompr MB/s :", int(uncomprB / (treadrows * 1024 * 1024))
	print "Total chunks uncompr :", int(niter)
    
