#!/usr/bin/env python

import copy
                
import numarray as NA
from tables import *
import random

def createFile(filename, totalrows, filters, atom):

    # Open a file in "w"rite mode
    fileh = openFile(filename, mode = "w", title="Searchsorted Benchmark",
                     filters=filters)
    title = "This is the EArray title"
    # Create an EArray instance
    group = fileh.root
    rowswritten = 0
    for j in range(3):
        # Create a table
        if atom == "float":
            atomtype = Atom(dtype="Float64", shape=(0,))
            arr = NA.arange(totalrows, type=NA.Float64)
        elif atom == "int":
            atomtype = Atom(dtype="Int32", shape=(0,))
            arr = NA.arange(totalrows, type=NA.Int32)
        elif atom == "string":
            atomtype = StringAtom(shape=(0,), length=4, flavor="CharArray")
            arr = strings.num2char(NA.arange(totalrows), '%4d')
        else:
            raise RuntimeError, "This should never happen"

        table = fileh.createEArray(group, 'earray'+str(j),
                                   atomtype,
                                   title,
                                   None,
                                   totalrows)
        table.append(arr)
        rowswritten += totalrows
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
        for table in fileh.listNodes(groupobj, 'EArray'):
            rowsize = table.rowsize
            print "reading", table
            if verbose and level == 0:
                print "Max rows in buf:", table._v_maxTuples
                print "Rows in", table._v_pathname, ":", table.nrows
                print "Buffersize:", table.rowsize * table._v_maxTuples
                print "MaxTuples:", table._v_maxTuples
                print "Chunksize:", table._v_chunksize

            if atom == "float":
                arr = table[:]
            elif atom == "int":
                arr = table[:]
            else:  # string atom case
                arr = table[:]
            if verbose:
                print "last value read ==>", arr[-1]

	    rowsread += table.nrows
            level += 1
        
    # Close the file (eventually destroy the extended type)
    fileh.close()

    return (rowsread, rowsize)

def searchBin(table, item):

#     table = NA.arange(1000*1000, type=table.type)
#     tablelen = len(table)
    tablelen = table.nrows
    bufsize = 5000
    tsections = tablelen / bufsize
    remainder = tablelen % bufsize
    correct = 0
    npoint = 0
    niter = 0
    npoint = 0
    additer = 0
    addsections = 0
    itemfound = 0 # Sentinel
    while itemfound == 0:
#         print "tsections-->", tsections
#         print "npoint-->", npoint
#         print "correct-->", correct
        buffer = table[npoint:npoint+bufsize+correct]
        correct = 0
        result = NA.searchsorted(buffer, item)
#         print "result-->", result
        if result == 0:
            if buffer[result] == item:
                itemfound = 1
                print "Item found!"
                break
            # The item is at left
            npoint = int(npoint - tsections * bufsize)
#             print "going left"
            if npoint < 0:
                # We are before the begining. Exit with result == 0
                print "Attempt to go before the beginning"
                break
        elif result == len(buffer):
                # The item is at the right
                npoint = int(npoint + (tsections-1) * bufsize)
#                 print "going right"
                if niter == 0:
                    # Add a correction when in the end of the table
                    correct = remainder
                if npoint > tablelen:
                    # We are after the end. Exit with result == tablelen
                    print "Attempt to go past the end"
                    break
        else:
            # Item has been found. Exit the loop and return
            itemfound = 1
            break
        # Reduce the number of tsections by half
        if tsections % 2 and not tsections == 1:
            # If we skip some iterations due to roundings, add some at the end
            additer += 1
            addsections = 1
        tsections /= 2
        if addsections:
            tsections += 1
            addsections = 0
        niter += 1
        if tsections == 0:
#             print "additer-->", additer
            if additer == 0:
                print "That should never happen"
                break
            tsections = 1
            additer -= 1

    return (result+npoint, itemfound, niter)

def searchFile(filename, atom, verbose, item):
    # Open the HDF5 file in read-only mode

    fileh = openFile(filename, mode = "r")
    rowsread = 0
    level = 0
    for groupobj in fileh.walkGroups(fileh.root):
        for table in fileh.listNodes(groupobj, 'EArray'):
            rowsize = table.rowsize
            print "reading", table
            if verbose and level == 0:
                print "Max rows in buf:", table._v_maxTuples
                print "Rows in", table._v_pathname, ":", table.nrows
                print "Buffersize:", table.rowsize * table._v_maxTuples
                print "MaxTuples:", table._v_maxTuples
                print "Chunksize:", table._v_chunksize

            if atom == "float":
                result = searchBin(table, float(eval(item)))
            elif atom == "int":
                result = searchBin(table, int(eval(item)))
            else:  # string atom case
                result = searchBin(table, str(eval(item)))
            if verbose:
                print "Position for item ==>", item, result

	    rowsread += table.nrows
            level += 1
        
    # Close the file (eventually destroy the extended type)
    fileh.close()

    return (rowsread, rowsize)


if __name__=="__main__":
    import sys
    import getopt
    try:
        import psyco
        psyco_imported = 1
    except:
        psyco_imported = 0
    
    import time
    
    usage = """usage: %s [-v] [-p] [-R range] [-r] [-w] [-s item ] [-a atom] [-f field] [-c level] [-l complib] [-i iterations] [-S] [-F] file
            -v verbose
	    -p use "psyco" if available
            -R select a range in a field in the form "start,stop,step"
	    -r only read test
	    -w only write test
            -s item to search
            -a use [float], [int] or [string] atom
            -c sets a compression level (do not set it or 0 for no compression)
            -S activate shuffling filter
            -F activate fletcher32 filter
            -l sets the compression library to be used ("zlib", "lzo", "ucl")
            -i sets the number of elements on each index\n""" % sys.argv[0]

    try:
        opts, pargs = getopt.getopt(sys.argv[1:], 'vpSFR:rws:a:c:l:i:')
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
    iterations = 100

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
            item = option[1]
        elif option[0] == '-a':
            atom = option[1]
            if atom not in ["float", "int", "string"]:
                sys.stderr.write(usage)
                sys.exit(0)
        elif option[0] == '-c':
            complevel = int(option[1])
        elif option[0] == '-l':
            complib = option[1]
        elif option[0] == '-i':
            iterations = int(option[1])
            
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
	(rowsw, rowsz) = createFile(file, iterations, filters, atom)
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
	t1 = time.time()
        cpu1 = time.clock()
        if psyco_imported and usepsyco:
            psyco.bind(readFile)
            psyco.bind(searchFile)
        if rng or item:
            (rowsr, rowsz) = searchFile(file, atom, verbose, item)
        else:
            for i in range(1):
                (rowsr, rowsz) = readFile(file, atom, verbose)
	t2 = time.time()
        cpu2 = time.clock()
	treadrows = round(t2-t1, 3)
        cpureadrows = round(cpu2-cpu1, 3)
        tpercent = int(round(cpureadrows/treadrows, 2)*100)
	print "Rows read:", rowsr, " Row size:", rowsz
	print "Time reading rows: %s s (real) %s s (cpu)  %s%%" % \
              (treadrows, cpureadrows, tpercent)
	print "Read rows/sec: ", int(rowsr / float(treadrows))
	print "Read KB/s :", int(rowsr * rowsz / (treadrows * 1024))
    
