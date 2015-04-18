#!/usr/bin/env python

from __future__ import print_function
import sys
import math
import time
import random
import warnings
import os

import numpy

from tables import *

# Initialize the random generator always with the same integer
# in order to have reproductible results
random.seed(19)
numpy.random.seed(19)

randomvalues = 0
worst = 0

Small = {
    "var1": StringCol(itemsize=4, dflt="Hi!", pos=2),
    "var2": Int32Col(pos=1),
    "var3": Float64Col(pos=0),
    #"var4" : BoolCol(),
}


def createNewBenchFile(bfile, verbose):

    class Create(IsDescription):
        nrows = Int32Col(pos=0)
        irows = Int32Col(pos=1)
        tfill = Float64Col(pos=2)
        tidx = Float64Col(pos=3)
        tcfill = Float64Col(pos=4)
        tcidx = Float64Col(pos=5)
        rowsecf = Float64Col(pos=6)
        rowseci = Float64Col(pos=7)
        fsize = Float64Col(pos=8)
        isize = Float64Col(pos=9)
        psyco = BoolCol(pos=10)

    class Search(IsDescription):
        nrows = Int32Col(pos=0)
        rowsel = Int32Col(pos=1)
        time1 = Float64Col(pos=2)
        time2 = Float64Col(pos=3)
        tcpu1 = Float64Col(pos=4)
        tcpu2 = Float64Col(pos=5)
        rowsec1 = Float64Col(pos=6)
        rowsec2 = Float64Col(pos=7)
        psyco = BoolCol(pos=8)

    if verbose:
        print("Creating a new benchfile:", bfile)
    # Open the benchmarking file
    bf = open_file(bfile, "w")
    # Create groups
    for recsize in ["small"]:
        group = bf.create_group("/", recsize, recsize + " Group")
        # Attach the row size of table as attribute
        if recsize == "small":
            group._v_attrs.rowsize = 16
        # Create a Table for writing bench
        bf.create_table(group, "create_best", Create, "best case")
        bf.create_table(group, "create_worst", Create, "worst case")
        for case in ["best", "worst"]:
            # create a group for searching bench (best case)
            groupS = bf.create_group(group, "search_" + case, "Search Group")
            # Create Tables for searching
            for mode in ["indexed", "inkernel", "standard"]:
                groupM = bf.create_group(groupS, mode, mode + " Group")
                # for searching bench
                # for atom in ["string", "int", "float", "bool"]:
                for atom in ["string", "int", "float"]:
                    bf.create_table(groupM, atom, Search, atom + " bench")
    bf.close()


def createFile(filename, nrows, filters, index, heavy, noise, verbose):

    # Open a file in "w"rite mode
    fileh = open_file(filename, mode="w", title="Searchsorted Benchmark",
                      filters=filters)
    rowswritten = 0

    # Create the test table
    table = fileh.create_table(fileh.root, 'table', Small, "test table",
                               None, nrows)

    t1 = time.time()
    cpu1 = time.clock()
    nrowsbuf = table.nrowsinbuf
    minimum = 0
    maximum = nrows
    for i in range(0, nrows, nrowsbuf):
        if i + nrowsbuf > nrows:
            j = nrows
        else:
            j = i + nrowsbuf
        if randomvalues:
            var3 = numpy.random.uniform(minimum, maximum, size=j - i)
        else:
            var3 = numpy.arange(i, j, dtype=numpy.float64)
            if noise > 0:
                var3 += numpy.random.uniform(-noise, noise, size=j - i)
        var2 = numpy.array(var3, dtype=numpy.int32)
        var1 = numpy.empty(shape=[j - i], dtype="S4")
        if not heavy:
            var1[:] = var2
        table.append([var3, var2, var1])
    table.flush()
    rowswritten += nrows
    time1 = time.time() - t1
    tcpu1 = time.clock() - cpu1
    print("Time for filling:", round(time1, 3),
          "Krows/s:", round(nrows / 1000. / time1, 3), end=' ')
    fileh.close()
    size1 = os.stat(filename)[6]
    print(", File size:", round(size1 / (1024. * 1024.), 3), "MB")
    fileh = open_file(filename, mode="a", title="Searchsorted Benchmark",
                      filters=filters)
    table = fileh.root.table
    rowsize = table.rowsize
    if index:
        t1 = time.time()
        cpu1 = time.clock()
        # Index all entries
        if not heavy:
            indexrows = table.cols.var1.create_index(filters=filters)
        for colname in ['var2', 'var3']:
            table.colinstances[colname].create_index(filters=filters)
        time2 = time.time() - t1
        tcpu2 = time.clock() - cpu1
        print("Time for indexing:", round(time2, 3),
              "iKrows/s:", round(indexrows / 1000. / time2, 3), end=' ')
    else:
        indexrows = 0
        time2 = 0.0000000001  # an ugly hack
        tcpu2 = 0.

    if verbose:
        if index:
            idx = table.cols.var1.index
            print("Index parameters:", repr(idx))
        else:
            print("NOT indexing rows")
    # Close the file
    fileh.close()

    size2 = os.stat(filename)[6] - size1
    if index:
        print(", Index size:", round(size2 / (1024. * 1024.), 3), "MB")
    return (rowswritten, indexrows, rowsize, time1, time2,
            tcpu1, tcpu2, size1, size2)


def benchCreate(file, nrows, filters, index, bfile, heavy,
                psyco, noise, verbose):

    # Open the benchfile in append mode
    bf = open_file(bfile, "a")
    recsize = "small"
    if worst:
        table = bf.get_node("/" + recsize + "/create_worst")
    else:
        table = bf.get_node("/" + recsize + "/create_best")

    (rowsw, irows, rowsz, time1, time2, tcpu1, tcpu2, size1, size2) = \
        createFile(file, nrows, filters, index, heavy, noise, verbose)
    # Collect data
    table.row["nrows"] = rowsw
    table.row["irows"] = irows
    table.row["tfill"] = time1
    table.row["tidx"] = time2
    table.row["tcfill"] = tcpu1
    table.row["tcidx"] = tcpu2
    table.row["fsize"] = size1
    table.row["isize"] = size2
    table.row["psyco"] = psyco
    tapprows = round(time1, 3)
    cpuapprows = round(tcpu1, 3)
    tpercent = int(round(cpuapprows / tapprows, 2) * 100)
    print("Rows written:", rowsw, " Row size:", rowsz)
    print("Time writing rows: %s s (real) %s s (cpu)  %s%%" %
          (tapprows, cpuapprows, tpercent))
    rowsecf = rowsw / tapprows
    table.row["rowsecf"] = rowsecf
    # print "Write rows/sec: ", rowsecf
    print("Total file size:",
          round((size1 + size2) / (1024. * 1024.), 3), "MB", end=' ')
    print(", Write KB/s (pure data):", int(rowsw * rowsz / (tapprows * 1024)))
    # print "Write KB/s :", int((size1+size2) / ((time1+time2) * 1024))
    tidxrows = time2
    cpuidxrows = round(tcpu2, 3)
    tpercent = int(round(cpuidxrows / tidxrows, 2) * 100)
    print("Rows indexed:", irows, " (IMRows):", irows / float(10 ** 6))
    print("Time indexing rows: %s s (real) %s s (cpu)  %s%%" %
          (round(tidxrows, 3), cpuidxrows, tpercent))
    rowseci = irows / tidxrows
    table.row["rowseci"] = rowseci
    table.row.append()
    bf.close()


def readFile(filename, atom, riter, indexmode, dselect, verbose):
    # Open the HDF5 file in read-only mode

    fileh = open_file(filename, mode="r")
    table = fileh.root.table
    var1 = table.cols.var1
    var2 = table.cols.var2
    var3 = table.cols.var3
    if indexmode == "indexed":
        if var2.index.nelements > 0:
            where = table._whereIndexed
        else:
            warnings.warn(
                "Not indexed table or empty index. Defaulting to in-kernel "
                "selection")
            indexmode = "inkernel"
            where = table._whereInRange
    elif indexmode == "inkernel":
        where = table.where
    if verbose:
        print("Max rows in buf:", table.nrowsinbuf)
        print("Rows in", table._v_pathname, ":", table.nrows)
        print("Buffersize:", table.rowsize * table.nrowsinbuf)
        print("MaxTuples:", table.nrowsinbuf)
        if indexmode == "indexed":
            print("Chunk size:", var2.index.sorted.chunksize)
            print("Number of elements per slice:", var2.index.nelemslice)
            print("Slice number in", table._v_pathname, ":", var2.index.nrows)

    #table.nrowsinbuf = 10
    # print "nrowsinbuf-->", table.nrowsinbuf
    rowselected = 0
    time2 = 0.
    tcpu2 = 0.
    results = []
    print("Select mode:", indexmode, ". Selecting for type:", atom)
    # Initialize the random generator always with the same integer
    # in order to have reproductible results on each read iteration
    random.seed(19)
    numpy.random.seed(19)
    for i in range(riter):
        # The interval for look values at. This is aproximately equivalent to
        # the number of elements to select
        rnd = numpy.random.randint(table.nrows)
        cpu1 = time.clock()
        t1 = time.time()
        if atom == "string":
            val = str(rnd)[-4:]
            if indexmode in ["indexed", "inkernel"]:
                results = [p.nrow
                           for p in where('var1 == val')]
            else:
                results = [p.nrow for p in table
                           if p["var1"] == val]
        elif atom == "int":
            val = rnd + dselect
            if indexmode in ["indexed", "inkernel"]:
                results = [p.nrow
                           for p in where('(rnd <= var3) & (var3 < val)')]
            else:
                results = [p.nrow for p in table
                           if rnd <= p["var2"] < val]
        elif atom == "float":
            val = rnd + dselect
            if indexmode in ["indexed", "inkernel"]:
                t1 = time.time()
                results = [p.nrow
                           for p in where('(rnd <= var3) & (var3 < val)')]
            else:
                results = [p.nrow for p in table
                           if float(rnd) <= p["var3"] < float(val)]
        else:
            raise ValueError("Value for atom '%s' not supported." % atom)
        rowselected += len(results)
        # print "selected values-->", results
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
        print("Values that fullfill the conditions:")
        print(results)

    #rowsread = table.nrows * riter
    rowsread = table.nrows
    rowsize = table.rowsize

    # Close the file
    fileh.close()

    return (rowsread, rowselected, rowsize, time1, time2, tcpu1, tcpu2)


def benchSearch(file, riter, indexmode, bfile, heavy, psyco, dselect, verbose):

    # Open the benchfile in append mode
    bf = open_file(bfile, "a")
    recsize = "small"
    if worst:
        tableparent = "/" + recsize + "/search_worst/" + indexmode + "/"
    else:
        tableparent = "/" + recsize + "/search_best/" + indexmode + "/"

    # Do the benchmarks
    if not heavy:
        #atomlist = ["string", "int", "float", "bool"]
        atomlist = ["string", "int", "float"]
    else:
        #atomlist = ["int", "float", "bool"]
        atomlist = ["int", "float"]
    for atom in atomlist:
        tablepath = tableparent + atom
        table = bf.get_node(tablepath)
        (rowsr, rowsel, rowssz, time1, time2, tcpu1, tcpu2) = \
            readFile(file, atom, riter, indexmode, dselect, verbose)
        row = table.row
        row["nrows"] = rowsr
        row["rowsel"] = rowsel
        treadrows = round(time1, 6)
        row["time1"] = time1
        treadrows2 = round(time2, 6)
        row["time2"] = time2
        cpureadrows = round(tcpu1, 6)
        row["tcpu1"] = tcpu1
        cpureadrows2 = round(tcpu2, 6)
        row["tcpu2"] = tcpu2
        row["psyco"] = psyco
        tpercent = int(round(cpureadrows / treadrows, 2) * 100)
        if riter > 1:
            tpercent2 = int(round(cpureadrows2 / treadrows2, 2) * 100)
        else:
            tpercent2 = 0.
        tMrows = rowsr / (1000 * 1000.)
        sKrows = rowsel / 1000.
        if atom == "string":  # just to print once
            print("Rows read:", rowsr, "Mread:", round(tMrows, 6), "Mrows")
        print("Rows selected:", rowsel, "Ksel:", round(sKrows, 6), "Krows")
        print("Time selecting (1st time): %s s (real) %s s (cpu)  %s%%" %
              (treadrows, cpureadrows, tpercent))
        if riter > 1:
            print("Time selecting (cached): %s s (real) %s s (cpu)  %s%%" %
                  (treadrows2, cpureadrows2, tpercent2))
        #rowsec1 = round(rowsr / float(treadrows), 6)/10**6
        rowsec1 = rowsr / treadrows
        row["rowsec1"] = rowsec1
        print("Read Mrows/sec: ", end=' ')
        print(round(rowsec1 / 10. ** 6, 6), "(first time)", end=' ')
        if riter > 1:
            rowsec2 = rowsr / treadrows2
            row["rowsec2"] = rowsec2
            print(round(rowsec2 / 10. ** 6, 6), "(cache time)")
        else:
            print()
        # Append the info to the table
        row.append()
        table.flush()
    # Close the benchmark file
    bf.close()


if __name__ == "__main__":
    import getopt
    try:
        import psyco
        psyco_imported = 1
    except:
        psyco_imported = 0

    usage = """usage: %s [-v] [-p] [-R] [-r] [-w] [-c level] [-l complib] [-S] [-F] [-n nrows] [-x] [-b file] [-t] [-h] [-k riter] [-m indexmode] [-N range] [-d range] datafile
            -v verbose
            -p use "psyco" if available
            -R use Random values for filling
            -r only read test
            -w only write test
            -c sets a compression level (do not set it or 0 for no compression)
            -l sets the compression library ("zlib", "lzo", "ucl", "bzip2" or "none")
            -S activate shuffling filter
            -F activate fletcher32 filter
            -n set the number of rows in tables (in krows)
            -x don't make indexes
            -b bench filename
            -t worsT searching case
            -h heavy benchmark (operations without strings)
            -m index mode for reading ("indexed" | "inkernel" | "standard")
            -N introduce (uniform) noise within range into the values
            -d the interval for look values (int, float) at. Default is 3.
            -k number of iterations for reading\n""" % sys.argv[0]

    try:
        opts, pargs = getopt.getopt(
            sys.argv[1:], 'vpSFRrowxthk:b:c:l:n:m:N:d:')
    except:
        sys.stderr.write(usage)
        sys.exit(0)

    # if we pass too much parameters, abort
    if len(pargs) != 1:
        sys.stderr.write(usage)
        sys.exit(0)

    # default options
    dselect = 3.
    noise = 0.
    verbose = 0
    fieldName = None
    testread = 1
    testwrite = 1
    usepsyco = 0
    complevel = 0
    shuffle = 0
    fletcher32 = 0
    complib = "zlib"
    nrows = 1000
    index = 1
    heavy = 0
    bfile = "bench.h5"
    supported_imodes = ["indexed", "inkernel", "standard"]
    indexmode = "inkernel"
    riter = 1

    # Get the options
    for option in opts:
        if option[0] == '-v':
            verbose = 1
        if option[0] == '-p':
            usepsyco = 1
        if option[0] == '-R':
            randomvalues = 1
        if option[0] == '-S':
            shuffle = 1
        if option[0] == '-F':
            fletcher32 = 1
        elif option[0] == '-r':
            testwrite = 0
        elif option[0] == '-w':
            testread = 0
        elif option[0] == '-x':
            index = 0
        elif option[0] == '-h':
            heavy = 1
        elif option[0] == '-t':
            worst = 1
        elif option[0] == '-b':
            bfile = option[1]
        elif option[0] == '-c':
            complevel = int(option[1])
        elif option[0] == '-l':
            complib = option[1]
        elif option[0] == '-m':
            indexmode = option[1]
            if indexmode not in supported_imodes:
                raise ValueError(
                    "Indexmode should be any of '%s' and you passed '%s'" %
                    (supported_imodes, indexmode))
        elif option[0] == '-n':
            nrows = int(float(option[1]) * 1000)
        elif option[0] == '-N':
            noise = float(option[1])
        elif option[0] == '-d':
            dselect = float(option[1])
        elif option[0] == '-k':
            riter = int(option[1])

    if worst:
        nrows -= 1  # the worst case

    if complib == "none":
        # This means no compression at all
        complib = "zlib"  # just to make PyTables not complaining
        complevel = 0

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
            print("Compression level:", complevel)
            if complevel > 0:
                print("Compression library:", complib)
                if shuffle:
                    print("Suffling...")
        if psyco_imported and usepsyco:
            psyco.bind(createFile)
        benchCreate(file, nrows, filters, index, bfile, heavy,
                    usepsyco, noise, verbose)
    if testread:
        if psyco_imported and usepsyco:
            psyco.bind(readFile)
        benchSearch(file, riter, indexmode, bfile, heavy, usepsyco,
                    dselect, verbose)
