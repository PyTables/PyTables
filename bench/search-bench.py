#!/usr/bin/env python

import copy

import time
import numarray as NA
from tables import *
import random
import warnings

# class Small(IsDescription):
#     _v_indexprops = IndexProps(auto=0, filters=Filters(complevel=1, complib="zlib", shuffle=1))
#     var1 = StringCol(length=4, dflt="", indexed=1)
#     var2 = IntCol(0, indexed=1)
#     var3 = FloatCol(0, indexed=1)
#     var4 = BoolCol(0, indexed=1)

Small = {
    "_v_indexprops" : IndexProps(auto=0, filters=Filters(complevel=1, complib="zlib", shuffle=1)),
    # var1 column will be indexed if not heavy test
    "var1" : StringCol(length=4, dflt="", indexed=0, pos=2),
    "var2" : IntCol(0, indexed=1, pos=1),
    "var3" : FloatCol(0, indexed=1, pos=0),
    #"var4" : BoolCol(0, indexed=1),
    }

def createNewBenchFile(bfile, verbose):

    class Create(IsDescription):
        nrows   = IntCol(pos=0)
        irows   = IntCol(pos=1)
        tfill   = FloatCol(pos=2)
        tidx    = FloatCol(pos=3)
        tcfill  = FloatCol(pos=4)
        tcidx   = FloatCol(pos=5)
        rowsecf = FloatCol(pos=6)
        rowseci = FloatCol(pos=7)
        psyco   = BoolCol(pos=8)

    class Search(IsDescription):
        nrows   = IntCol(pos=0)
        rowsel  = IntCol(pos=1)
        time1   = FloatCol(pos=2)
        time2   = FloatCol(pos=3)
        tcpu1   = FloatCol(pos=4)
        tcpu2   = FloatCol(pos=5)
        rowsec1 = FloatCol(pos=6)
        rowsec2 = FloatCol(pos=7)
        psyco   = BoolCol(pos=8)

    if verbose:
        print "Creating a new benchfile:", bfile
    # Open the benchmarking file
    bf = openFile(bfile, "w")
    # Create groups
    for recsize in ["small"]:
        group = bf.createGroup("/", recsize, recsize+" Group")
        # Attach the row size of table as attribute
        if recsize == "small":
            group._v_attrs.rowsize = 16
        # Create a Table for writing bench
        bf.createTable(group, "create", Create, "create bench")
        # create a group for searching bench
        groupS = bf.createGroup(group, "search", "Search Group")
        # Create Tables for searching
        for mode in ["indexed", "inkernel", "standard"]:
            group = bf.createGroup(groupS, mode, mode+" Group")
            # for searching bench
            #for atom in ["string", "int", "float", "bool"]:
            for atom in ["string", "int", "float"]:
                bf.createTable(group, atom, Search, atom+" bench")
    bf.close()
    
def createFile(filename, nrows, filters, index, heavy, verbose):

    # Open a file in "w"rite mode
    fileh = openFile(filename, mode = "w", title="Searchsorted Benchmark",
                     filters=filters)
    rowswritten = 0
    # set the properties of the index (the same of table)
    Small["_v_indexprops"] = IndexProps(auto=0, filters=filters)
    if not heavy:
        # make the index entry indexed as well
        Small["var1"] = StringCol(length=4, dflt="", indexed=1)
        
    # Create the test table
    table = fileh.createTable(fileh.root, 'table', Small, "test table",
                              None, nrows)
    print "Filling...",
    #print "Filling...nrows:", nrows, "Mrows:", nrows / 10**6
    t1 = time.time()
    cpu1 = time.clock()
    for i in xrange(nrows):
        # Assigning a string takes lots of time!
        if not heavy:
            table.row['var1'] = str(i)
        #table.row['var2'] = random.randrange(nrows)
        table.row['var2'] = i
        table.row['var3'] = float(nrows-i)
        #table.row['var4'] = i % 2
        #table.row['var4'] = i > 2
        table.row.append()
    rowswritten += nrows
    table.flush()
    time1 = time.time()-t1
    tcpu1 = time.clock()-cpu1
    print "Time for filling:", round(time1,3), "rows/s:", round(nrows/time1,3)
    rowsize = table.rowsize
    indexrows = 0
    if index:
        print "Indexing...",
        #print "Indexing... nrows:", nrows, "Mrows:", nrows / 10**6
        t1 = time.time()
        cpu1 = time.clock()
        # Index all entries
        indexrows = table.flushRowsToIndex()
        time2 = time.time()-t1
        print "Time for indexing:", round(time2,3), \
              "irows/s:", round(indexrows/time2,3)
	tcpu2 = time.clock()-cpu1
    if verbose:
        if index:
            idx = table.cols.var1.index
            print "Index parameters:", repr(idx)
        else:
            print "NOT indexing rows"
    # Close the file (eventually destroy the extended type)
    fileh.close()
    return (rowswritten, indexrows, rowsize, time1, time2, tcpu1, tcpu2)

def benchCreate(file, nrows, filters, index, bfile, heavy, psyco, verbose):

    # Open the benchfile in append mode
    bf = openFile(bfile,"a")
    recsize = "small"
    table = bf.getNode("/"+recsize+"/create")
    (rowsw, irows, rowsz, time1, time2, tcpu1, tcpu2) = \
            createFile(file, nrows, filters, index, heavy, verbose)
    # Collect data
    table.row["nrows"] = rowsw
    table.row["irows"] = irows
    table.row["tfill"] = time1
    table.row["tidx"]  = time2
    table.row["tcfill"] = tcpu1
    table.row["tcidx"] = tcpu2
    table.row["psyco"] = psyco
    tapprows = round(time1, 3)
    cpuapprows = round(tcpu1, 3)
    tpercent = int(round(cpuapprows/tapprows, 2)*100)
    print "Rows written:", rowsw, " Row size:", rowsz
    print "Time writing rows: %s s (real) %s s (cpu)  %s%%" % \
          (tapprows, cpuapprows, tpercent)
    rowsecf = int(rowsw / float(tapprows))
    table.row["rowsecf"] = rowsecf
    #print "Write rows/sec: ", rowsecf
    print "Write KB/s :", int(rowsw * rowsz / (tapprows * 1024))
    tidxrows = round(time2, 3)
    cpuidxrows = round(tcpu2, 3)
    tpercent = int(round(cpuidxrows/tidxrows, 2)*100)
    print "Rows indexed:", irows, " (IMRows):", irows / float(10**6)
    print "Time indexing rows: %s s (real) %s s (cpu)  %s%%" % \
          (tidxrows, cpuidxrows, tpercent)
    rowseci = int(irows / float(tidxrows))
    table.row["rowseci"] = rowseci
    #print "Index rows/sec: ", rowseci
    table.row.append()
    bf.close()
    
def readFile(filename, atom, riter, indexmode, verbose):
    # Open the HDF5 file in read-only mode

    fileh = openFile(filename, mode = "r")
    table = fileh.root.table
    var1 = table.cols.var1
    var2 = table.cols.var2
    var3 = table.cols.var3
    #var4 = table.cols.var4
    if indexmode == "indexed":
        if var2.index.nelements > 0:
            where = table.whereIndexed
        else:
            warnings.warn("Not indexed table or empty index. Defaulting to in-kernel selections")
            indexmode = "inkernel"
            where = table.whereInRange
    elif indexmode == "inkernel":
        where = table.whereInRange
    if verbose:
        print "Max rows in buf:", table._v_maxTuples
        print "Rows in", table._v_pathname, ":", table.nrows
        print "Buffersize:", table.rowsize * table._v_maxTuples
        print "MaxTuples:", table._v_maxTuples
        if indexmode == "indexed":
            print "Chunk size:", var2.index.sorted.chunksize
            print "Number of elements per slice:", var2.index.nelemslice
            print "Slice number in", table._v_pathname, ":", var2.index.nrows

    rowselected = 0
    time2 = 0.
    tcpu2 = 0.
    results = []
    print "Select mode:", indexmode, ". Selecting for type:", atom
    for i in xrange(riter):
        rnd = random.randrange(table.nrows)
        cpu1 = time.clock()
        t1 = time.time()
        if atom == "string":
            if indexmode in ["indexed", "inkernel"]:
                results = [p.nrow()
                           # for p in where("1000" <= var1 <= "1010")]
                           #for p in where(var1 == "1111")]
                           for p in where(var1 == str(rnd)[-4:])]
            else:
                results = [p.nrow() for p in table
                           # if "1000" <= p["var1"] <= "1010"]
                           #if p["var1"] == "1111"]
                           if p["var1"] == str(rnd)[-4:]]
        elif atom == "int":
            if indexmode in ["indexed", "inkernel"]:
                results = [p.nrow()
                           # for p in where(2+i<= var2 < 10+i)]
                           # for p in where(2<= var2 < 10)]
                           # for p in where(110*i <= var2 < 110*(i+1))]
                           # for p in where(1000-30 < var2 < 1000+60)]
                           # for p in where(3 <= var2 < 5)]
                           for p in where(rnd <= var2 < rnd+3)]
            else:
                results = [p.nrow() for p in table
                           # if p["var2"] < 10+i]
                           # if 2 <= p["var2"] < 10)]
                           # if 110*i <= p["var2"] < 110*(i+1)]
                           # if 1000-30 < p["var2"] < 1000+60]
                           #if 3 <= p["var2"] < 5]
                           if rnd <= p["var2"] < rnd+3]
        elif atom == "float":
            if indexmode in ["indexed", "inkernel"]:
                results = [p.nrow()
                           # for p in where(var3 < 5.)]
                           #for p in where(3. <= var3 < 5.)]
                           #for p in where(float(rnd) <= var3 < float(rnd+3))]
                           for p in where(rnd <= var3 < rnd+3)]
                           # for p in where(1000.-i <= var3 < 1000.+i)]
                           # for p in where(100*i <= var3 < 100*(i+1))]
            else:
                results = [p.nrow() for p in table
                           # if p["var3"] < 5.]
                           #if 3. <= p["var3"] < 5.]
                           if float(rnd) <= p["var3"] < float(rnd+3)]
                           # if 1000.-i <= p["var3"] < 1000.+i]
                           # if 100*i <= p["var3"] < 100*(i+1)]
#         elif atom == "bool":
#             if indexmode in ["indexed", "inkernel"]:
#                 results = [p.nrow() for p in where(var4 == 0)]
#             else:
#                 results = [p.nrow() for p in table if p["var4"] == 0]
        else:
            raise ValueError, "Value for atom '%s' not supported." % atom
        rowselected += len(results)
        if i == 0:
            # First iteration
            time1 = time.time() - t1
            tcpu1 = time.clock() - cpu1
        else:
            if indexmode == "indexed":
                # if indexed, wait until the 5th iteration (in order to
                # insure that the index is effectively cached) to take times
                if i >= 5:
                    time2 += time.time() - t1
                    tcpu2 += time.clock() - cpu1
            else:
                time2 += time.time() - t1
                tcpu2 += time.clock() - cpu1
                        
    if riter > 1:
        if indexmode == "indexed" and riter >= 5:
            correction = 5
        else:
            correction = 1
        time2 = time2 / (riter - correction)
        tcpu2 = tcpu2 / (riter - correction)
    if verbose and 1:
        print "Values that fullfill the conditions:"
        print results

    #rowsread = table.nrows * riter
    rowsread = table.nrows
    rowsize = table.rowsize 
        
    # Close the file
    fileh.close()

    return (rowsread, rowselected, rowsize, time1, time2, tcpu1, tcpu2)

def benchSearch(file, riter, indexmode, bfile, heavy, psyco, verbose):

    # Open the benchfile in append mode
    bf = openFile(bfile,"a")
    recsize = "small"
    tableparent = "/"+recsize+"/search/"+indexmode+"/"
    # Do the benchmarks
    if not heavy:
        #atomlist = ["string", "int", "float", "bool"]
        atomlist = ["string", "int", "float"]
    else:
        #atomlist = ["int", "float", "bool"]
        atomlist = ["int", "float"]
    for atom in atomlist:
        tablepath = tableparent + atom
        table = bf.getNode(tablepath)
        (rowsr, rowsel, rowssz, time1, time2, tcpu1, tcpu2) = \
                readFile(file, atom, riter, indexmode, verbose)
        table.row["nrows"] = rowsr
        table.row["rowsel"] = rowsel
        treadrows = round(time1, 4)
        table.row["time1"] = time1
        treadrows2 = round(time2, 4)
        table.row["time2"] = time2
        cpureadrows = round(tcpu1, 4)
        table.row["tcpu1"] = tcpu1
        cpureadrows2 = round(tcpu1, 4)
        table.row["tcpu2"] = tcpu2
        table.row["psyco"] = psyco
        tpercent = int(round(cpureadrows/treadrows, 2)*100)
        if riter > 1:
            tpercent2 = int(round(cpureadrows2/treadrows2, 2)*100)
        else:
            tpercent2 = 0.
        tMrows = rowsr/(1000*1000.)
        sKrows = rowsel/1000.
        if atom == "string": # just to print once
            print "Rows read:", rowsr, "Mread:", round(tMrows, 4), "Mrows"
        print "Rows selected:", rowsel, "Ksel:", round(sKrows,4), "Krows"
        print "Time selecting (1st time): %s s (real) %s s (cpu)  %s%%" % \
              (treadrows, cpureadrows, tpercent)
        if riter > 1:
            print "Time selecting (cached): %s s (real) %s s (cpu)  %s%%" % \
                  (treadrows2, cpureadrows2, tpercent2)
        rowsec1 = round(rowsr / float(treadrows), 4)/10**6
        table.row["rowsec1"] = rowsec1
        print "Read Mrows/sec: ",
        print rowsec1, "(first time)",
        if riter > 1:
            rowsec2 = round(rowsr / float(treadrows2), 4)/10**6
            table.row["rowsec2"] = rowsec2
            print rowsec2, "(cached time)"
        else:
            print
        # Append the info to the table
        table.row.append()
    # Close the benchmark file
    bf.close()

if __name__=="__main__":
    import sys
    import os.path
    import getopt
    try:
        import psyco
        psyco_imported = 1
    except:
        psyco_imported = 0
    
    import time
    
    usage = """usage: %s [-v] [-p] [-R range] [-r] [-w] [-c level] [-l complib] [-S] [-F] [-n nrows] [-x] [-b file] [-h] [-k riter] [-m indexmode] datafile
            -v verbose
	    -p use "psyco" if available
            -R select a range in a field in the form "start,stop,step"
	    -r only read test
	    -w only write test
            -c sets a compression level (do not set it or 0 for no compression)
            -l sets the compression library to be used ("zlib", "lzo", "ucl")
            -S activate shuffling filter
            -F activate fletcher32 filter
            -n set the number of rows in tables (in krows)
            -x don't make indexes
            -b bench filename
            -h heavy benchmark (operations with strings)
            -m index mode for reading ("indexed" | "inkernel" | "standard")
            -k number of iterations for reading\n""" % sys.argv[0]

    try:
        opts, pargs = getopt.getopt(sys.argv[1:], 'vpSFR:rowxhk:b:c:l:n:m:')
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
    fieldName = None
    testread = 1
    testwrite = 1
    usepsyco = 0
    complevel = 0
    shuffle = 0
    fletcher32 = 0
    complib = "zlib"
    nrows = 100
    index = 1
    heavy = 0
    bfile = "bench.h5"
    supported_imodes = ["indexed","inkernel","standard"]
    indexmode = "indexed"
    riter = 1

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
        elif option[0] == '-x':
            index = 0
        elif option[0] == '-h':
            heavy = 1
        elif option[0] == '-b':
            bfile = option[1]
        elif option[0] == '-c':
            complevel = int(option[1])
        elif option[0] == '-l':
            complib = option[1]
        elif option[0] == '-m':
            indexmode = option[1]
            if indexmode not in supported_imodes:
                raise ValueError, "Indexmode should be any of '%s' and you passed '%s'" % (supported_imodes, indexmode)
        elif option[0] == '-n':
            nrows = int(float(option[1])*1000)
        elif option[0] == '-k':
            riter = int(option[1])
            
    # Catch the hdf5 file passed as the last argument
    file = pargs[0]

    # Build the Filters instance
    filters = Filters(complevel=complevel, complib=complib,
                      shuffle=shuffle, fletcher32=fletcher32)

    # Create the benchfile (if needed)
    if not os.path.exists(bfile):
        createNewBenchFile(bfile, verbose)

    if testwrite:
        if verbose:
            print "Compression level:", complevel
            if complevel > 0:
                print "Compression library:", complib
                if shuffle:
                    print "Suffling..."
        if psyco_imported and usepsyco:
            psyco.bind(createFile)
        benchCreate(file, nrows, filters, index, bfile, heavy,
                    usepsyco, verbose)
    if testread:
        if psyco_imported and usepsyco:
            psyco.bind(readFile)
        benchSearch(file, riter, indexmode, bfile, heavy, usepsyco, verbose)
