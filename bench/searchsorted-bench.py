#!/usr/bin/env python

from time import perf_counter as clock
from time import process_time as cpuclock

import tables as tb


class Small(tb.IsDescription):
    var1 = tb.StringCol(itemsize=4)
    var2 = tb.Int32Col()
    var3 = tb.Float64Col()
    var4 = tb.BoolCol()

# Define a user record to characterize some kind of particles


class Medium(tb.IsDescription):
    var1 = tb.StringCol(itemsize=16)   # 16-character String
    #float1 = Float64Col(dflt=2.3)
    #float2 = Float64Col(dflt=2.3)
    # zADCcount    = Int16Col()      # signed short integer
    var2 = tb.Int32Col()               # signed short integer
    var3 = tb.Float64Col()
    grid_i = tb.Int32Col()             # integer
    grid_j = tb.Int32Col()             # integer
    pressure = tb.Float32Col()         # float  (single-precision)
    energy = tb.Float64Col(shape=2)    # double (double-precision)


def createFile(filename, nrows, filters, atom, recsize, index, verbose):

    # Open a file in "w"rite mode
    fileh = tb.open_file(filename, mode="w", title="Searchsorted Benchmark",
                      filters=filters)
    title = "This is the IndexArray title"
    # Create an IndexArray instance
    rowswritten = 0
    # Create an entry
    klass = {"small": Small, "medium": Medium}
    table = fileh.create_table(fileh.root, 'table', klass[recsize], title,
                               None, nrows)
    for i in range(nrows):
        #table.row['var1'] = str(i)
        #table.row['var2'] = random.randrange(nrows)
        table.row['var2'] = i
        table.row['var3'] = i
        #table.row['var4'] = i % 2
        #table.row['var4'] = i > 2
        table.row.append()
    rowswritten += nrows
    table.flush()
    rowsize = table.rowsize
    indexrows = 0

    # Index one entry:
    if index:
        if atom == "string":
            indexrows = table.cols.var1.create_index()
        elif atom == "bool":
            indexrows = table.cols.var4.create_index()
        elif atom == "int":
            indexrows = table.cols.var2.create_index()
        elif atom == "float":
            indexrows = table.cols.var3.create_index()
        else:
            raise ValueError("Index type not supported yet")
        if verbose:
            print("Number of indexed rows:", indexrows)
    # Close the file (eventually destroy the extended type)
    fileh.close()

    return (rowswritten, rowsize)


def readFile(filename, atom, niter, verbose):
    # Open the HDF5 file in read-only mode

    fileh = tb.open_file(filename, mode="r")
    table = fileh.root.table
    print("reading", table)
    if atom == "string":
        idxcol = table.cols.var1.index
    elif atom == "bool":
        idxcol = table.cols.var4.index
    elif atom == "int":
        idxcol = table.cols.var2.index
    else:
        idxcol = table.cols.var3.index
    if verbose:
        print("Max rows in buf:", table.nrowsinbuf)
        print("Rows in", table._v_pathname, ":", table.nrows)
        print("Buffersize:", table.rowsize * table.nrowsinbuf)
        print("MaxTuples:", table.nrowsinbuf)
        print("Chunk size:", idxcol.sorted.chunksize)
        print("Number of elements per slice:", idxcol.nelemslice)
        print("Slice number in", table._v_pathname, ":", idxcol.nrows)

    rowselected = 0
    if atom == "string":
        for i in range(niter):
            #results = [table.row["var3"] for i in table.where(2+i<=table.cols.var2 < 10+i)]
            #results = [table.row.nrow() for i in table.where(2<=table.cols.var2 < 10)]
            results = [p["var1"]  # p.nrow()
                       for p in table.where(table.cols.var1 == "1111")]
#                      for p in table.where("1000"<=table.cols.var1<="1010")]
            rowselected += len(results)
    elif atom == "bool":
        for i in range(niter):
            results = [p["var2"]  # p.nrow()
                       for p in table.where(table.cols.var4 == 0)]
            rowselected += len(results)
    elif atom == "int":
        for i in range(niter):
            #results = [table.row["var3"] for i in table.where(2+i<=table.cols.var2 < 10+i)]
            #results = [table.row.nrow() for i in table.where(2<=table.cols.var2 < 10)]
            results = [p["var2"]  # p.nrow()
                       # for p in table.where(110*i<=table.cols.var2<110*(i+1))]
                       # for p in table.where(1000-30<table.cols.var2<1000+60)]
                       for p in table.where(table.cols.var2 <= 400)]
            rowselected += len(results)
    elif atom == "float":
        for i in range(niter):
#         results = [(table.row.nrow(), table.row["var3"])
#                    for i in table.where(3<=table.cols.var3 < 5.)]
#             results = [(p.nrow(), p["var3"])
# for p in table.where(1000.-i<=table.cols.var3<1000.+i)]
            results = [
                p["var3"]  # (p.nrow(), p["var3"])
                for p in table.where(
                    100 * i <= table.cols.var3 < 100 * (i + 1))
            ]
#                        for p in table
#                        if 100*i<=p["var3"]<100*(i+1)]
#             results = [ (p.nrow(), p["var3"]) for p in table
#                         if (1000.-i <= p["var3"] < 1000.+i) ]
            rowselected += len(results)
        else:
            raise ValueError("Unsuported atom value")
    if verbose and 1:
        print("Values that fullfill the conditions:")
        print(results)

    rowsread = table.nrows * niter
    rowsize = table.rowsize

    # Close the file (eventually destroy the extended type)
    fileh.close()

    return (rowsread, rowselected, rowsize)


def searchFile(filename, atom, verbose, item):
    # Open the HDF5 file in read-only mode

    fileh = tb.open_file(filename, mode="r")
    rowsread = 0
    uncomprBytes = 0
    table = fileh.root.table
    if atom == "int":
        idxcol = table.cols.var2.index
    elif atom == "float":
        idxcol = table.cols.var3.index
    else:
        raise ValueError("Unsuported atom value")
    print("Searching", table, "...")
    if verbose:
        print("Chunk size:", idxcol.sorted.chunksize)
        print("Number of elements per slice:", idxcol.sorted.nelemslice)
        print("Slice number in", table._v_pathname, ":", idxcol.sorted.nrows)

    (positions, niter) = idxcol.search(item)
    if verbose:
        print("Positions for item", item, "==>", positions)
        print("Total iterations in search:", niter)

    rowsread += table.nrows
    uncomprBytes += idxcol.sorted.chunksize * niter * idxcol.sorted.itemsize

    results = table.read(coords=positions)
    print("results length:", len(results))
    if verbose:
        print("Values that fullfill the conditions:")
        print(results)

    # Close the file (eventually destroy the extended type)
    fileh.close()

    return (rowsread, uncomprBytes, niter)


if __name__ == "__main__":
    import sys
    import getopt
    try:
        import psyco
        psyco_imported = 1
    except:
        psyco_imported = 0

    usage = """usage: %s [-v] [-p] [-R range] [-r] [-w] [-s recsize ] [-a
    atom] [-c level] [-l complib] [-S] [-F] [-i item] [-n nrows] [-x]
    [-k niter] file
            -v verbose
            -p use "psyco" if available
            -R select a range in a field in the form "start,stop,step"
            -r only read test
            -w only write test
            -s record size
            -a use [float], [int], [bool] or [string] atom
            -c sets a compression level (do not set it or 0 for no compression)
            -S activate shuffling filter
            -F activate fletcher32 filter
            -l sets the compression library to be used ("zlib", "lzo", "ucl", "bzip2")
            -i item to search
            -n set the number of rows in tables
            -x don't make indexes
            -k number of iterations for reading\n""" % sys.argv[0]

    try:
        opts, pargs = getopt.getopt(sys.argv[1:], 'vpSFR:rwxk:s:a:c:l:i:n:')
    except:
        sys.stderr.write(usage)
        sys.exit(0)

    # if we pass too much parameters, abort
    if len(pargs) != 1:
        sys.stderr.write(usage)
        sys.exit(0)

    # default options
    verbose = 0
    rng = None
    item = None
    atom = "int"
    fieldName = None
    testread = 1
    testwrite = 1
    usepsyco = 0
    complevel = 0
    shuffle = 0
    fletcher32 = 0
    complib = "zlib"
    nrows = 100
    recsize = "small"
    index = 1
    niter = 1

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
        elif option[0] == '-s':
            recsize = option[1]
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
            item = eval(option[1])
        elif option[0] == '-n':
            nrows = int(option[1])
        elif option[0] == '-k':
            niter = int(option[1])

    # Build the Filters instance
    filters = tb.Filters(complevel=complevel, complib=complib,
                         shuffle=shuffle, fletcher32=fletcher32)

    # Catch the hdf5 file passed as the last argument
    file = pargs[0]

    if testwrite:
        print("Compression level:", complevel)
        if complevel > 0:
            print("Compression library:", complib)
            if shuffle:
                print("Suffling...")
        t1 = clock()
        cpu1 = cpuclock()
        if psyco_imported and usepsyco:
            psyco.bind(createFile)
        (rowsw, rowsz) = createFile(file, nrows, filters,
                                    atom, recsize, index, verbose)
        t2 = clock()
        cpu2 = cpuclock()
        tapprows = t2 - t1
        cpuapprows = cpu2 - cpu1
        print(f"Rows written:", rowsw, " Row size:", rowsz)
        print(
            f"Time writing rows: {tapprows:.3f} s (real) "
            f"{cpuapprows:.3f} s (cpu)  {cpuapprows / tapprows:.0%}")
        print(f"Write rows/sec: {rowsw / tapprows:.0f}")
        print(f"Write KB/s : {rowsw * rowsz / tapprows / 1024:.0f}")

    if testread:
        if psyco_imported and usepsyco:
            psyco.bind(readFile)
            psyco.bind(searchFile)
        t1 = clock()
        cpu1 = cpuclock()
        if rng or item:
            (rowsr, uncomprB, niter) = searchFile(file, atom, verbose, item)
        else:
            for i in range(1):
                (rowsr, rowsel, rowsz) = readFile(file, atom, niter, verbose)
        t2 = clock()
        cpu2 = cpuclock()
        treadrows = t2 - t1
        cpureadrows = cpu2 - cpu1
        tMrows = rowsr / 1000 / 1000
        sKrows = rowsel / 1000
        print(f"Rows read: {rowsr} Mread: {tMrows:.3f} Mrows")
        print(f"Rows selected: {rowsel} Ksel: {sKrows:.3f} Krows")
        print(
            f"Time reading rows: {treadrows:.3f} s (real) "
            f"{cpureadrows:.3f} s (cpu)  {cpureadrows / treadrows:.0%}")
        print(f"Read Mrows/sec: {tMrows / treadrows:.3f}")
        # print "Read KB/s :", int(rowsr * rowsz / (treadrows * 1024))
#       print "Uncompr MB :", int(uncomprB / (1024 * 1024))
#       print "Uncompr MB/s :", int(uncomprB / (treadrows * 1024 * 1024))
#       print "Total chunks uncompr :", int(niter)
