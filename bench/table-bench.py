#!/usr/bin/env python

from __future__ import print_function
import numpy as NP
from tables import *

# This class is accessible only for the examples


class Small(IsDescription):
    var1 = StringCol(itemsize=4, pos=2)
    var2 = Int32Col(pos=1)
    var3 = Float64Col(pos=0)

# Define a user record to characterize some kind of particles


class Medium(IsDescription):
    name = StringCol(itemsize=16, pos=0)    # 16-character String
    float1 = Float64Col(shape=2, dflt=NP.arange(2), pos=1)
    #float1 = Float64Col(dflt=2.3)
    #float2 = Float64Col(dflt=2.3)
    # zADCcount    = Int16Col()               # signed short integer
    ADCcount = Int32Col(pos=6)              # signed short integer
    grid_i = Int32Col(pos=7)                # integer
    grid_j = Int32Col(pos=8)                # integer
    pressure = Float32Col(pos=9)            # float  (single-precision)
    energy = Float64Col(pos=2)              # double (double-precision)
    # unalig      = Int8Col()                 # just to unalign data

# Define a user record to characterize some kind of particles


class Big(IsDescription):
    name = StringCol(itemsize=16)           # 16-character String
    float1 = Float64Col(shape=32, dflt=NP.arange(32))
    float2 = Float64Col(shape=32, dflt=2.2)
    TDCcount = Int8Col()                    # signed short integer
    #ADCcount    = Int32Col()
    # ADCcount = Int16Col()                   # signed short integer
    grid_i = Int32Col()                       # integer
    grid_j = Int32Col()                       # integer
    pressure = Float32Col()                   # float  (single-precision)
    energy = Float64Col()                     # double (double-precision)


def createFile(filename, totalrows, filters, recsize):

    # Open a file in "w"rite mode
    fileh = open_file(filename, mode="w", title="Table Benchmark",
                      filters=filters)

    # Table title
    title = "This is the table title"

    # Create a Table instance
    group = fileh.root
    rowswritten = 0
    for j in range(3):
        # Create a table
        if recsize == "big":
            table = fileh.create_table(group, 'tuple' + str(j), Big, title,
                                       None,
                                       totalrows)
        elif recsize == "medium":
            table = fileh.create_table(group, 'tuple' + str(j), Medium, title,
                                       None,
                                       totalrows)
        elif recsize == "small":
            table = fileh.create_table(group, 'tuple' + str(j), Small, title,
                                       None,
                                       totalrows)
        else:
            raise RuntimeError("This should never happen")

        table.attrs.test = 2
        rowsize = table.rowsize
        # Get the row object associated with the new table
        d = table.row
        # Fill the table
        if recsize == "big":
            for i in range(totalrows):
                # d['name']  = 'Part: %6d' % (i)
                d['TDCcount'] = i % 256
                #d['float1'] = NP.array([i]*32, NP.float64)
                #d['float2'] = NP.array([i**2]*32, NP.float64)
                #d['float1'][0] = float(i)
                #d['float2'][0] = float(i*2)
                # Common part with medium
                d['grid_i'] = i
                d['grid_j'] = 10 - i
                d['pressure'] = float(i * i)
                # d['energy'] = float(d['pressure'] ** 4)
                d['energy'] = d['pressure']
                # d['idnumber'] = i * (2 ** 34)
                d.append()
        elif recsize == "medium":
            for i in range(totalrows):
                #d['name']  = 'Part: %6d' % (i)
                #d['float1'] = NP.array([i]*2, NP.float64)
                #d['float1'] = arr
                #d['float1'] = i
                #d['float2'] = float(i)
                # Common part with big:
                d['grid_i'] = i
                d['grid_j'] = 10 - i
                d['pressure'] = i * 2
                # d['energy'] = float(d['pressure'] ** 4)
                d['energy'] = d['pressure']
                d.append()
        else:  # Small record
            for i in range(totalrows):
                #d['var1'] = str(random.randrange(1000000))
                #d['var3'] = random.randrange(10000000)
                d['var1'] = str(i)
                #d['var2'] = random.randrange(totalrows)
                d['var2'] = i
                #d['var3'] = 12.1e10
                d['var3'] = totalrows - i
                d.append()  # This is a 10% faster than table.append()
        rowswritten += totalrows

        if recsize == "small":
            # Testing with indexing
            pass
#            table._createIndex("var3", Filters(1,"zlib",shuffle=1))

        # table.flush()
        group._v_attrs.test2 = "just a test"
        # Create a new group
        group2 = fileh.create_group(group, 'group' + str(j))
        # Iterate over this new group (group2)
        group = group2
        table.flush()

    # Close the file (eventually destroy the extended type)
    fileh.close()
    return (rowswritten, rowsize)


def readFile(filename, recsize, verbose):
    # Open the HDF5 file in read-only mode

    fileh = open_file(filename, mode="r")
    rowsread = 0
    for groupobj in fileh.walk_groups(fileh.root):
        # print "Group pathname:", groupobj._v_pathname
        row = 0
        for table in fileh.list_nodes(groupobj, 'Table'):
            rowsize = table.rowsize
            print("reading", table)
            if verbose:
                print("Max rows in buf:", table.nrowsinbuf)
                print("Rows in", table._v_pathname, ":", table.nrows)
                print("Buffersize:", table.rowsize * table.nrowsinbuf)
                print("MaxTuples:", table.nrowsinbuf)

            if recsize == "big" or recsize == "medium":
                # e = [ p.float1 for p in table.iterrows()
                #      if p.grid_i < 2 ]
                #e = [ str(p) for p in table.iterrows() ]
                #      if p.grid_i < 2 ]
#                 e = [ p['grid_i'] for p in table.iterrows()
#                       if p['grid_j'] == 20 and p['grid_i'] < 20 ]
#                 e = [ p['grid_i'] for p in table
#                       if p['grid_i'] <= 2 ]
#                e = [ p['grid_i'] for p in table.where("grid_i<=20")]
#                 e = [ p['grid_i'] for p in
#                       table.where('grid_i <= 20')]
                e = [p['grid_i'] for p in
                     table.where('(grid_i <= 20) & (grid_j == 20)')]
#                 e = [ p['grid_i'] for p in table.iterrows()
#                       if p.nrow() == 20 ]
#                 e = [ table.delrow(p.nrow()) for p in table.iterrows()
#                       if p.nrow() == 20 ]
                # The version with a for loop is only 1% better than
                # comprenhension list
                #e = []
                # for p in table.iterrows():
                #    if p.grid_i < 20:
                #        e.append(p.grid_j)
            else:  # small record case
#                 e = [ p['var3'] for p in table.iterrows()
#                       if p['var2'] < 20 and p['var3'] < 20 ]
#                e = [ p['var3'] for p in table.where("var3 <= 20")
#                      if p['var2'] < 20 ]
#               e = [ p['var3'] for p in table.where("var3 <= 20")]
# Cuts 1) and 2) issues the same results but 2) is about 10 times faster
# Cut 1)
#                e = [ p.nrow() for p in
#                      table.where(table.cols.var2 > 5)
#                      if p["var2"] < 10]
# Cut 2)
#                 e = [ p.nrow() for p in
#                       table.where(table.cols.var2 < 10)
#                       if p["var2"] > 5]
#                e = [ (p._nrow,p["var3"]) for p in
#                e = [ p["var3"] for p in
#                      table.where(table.cols.var3 < 10)]
#                      table.where(table.cols.var3 < 10)]
#                      table if p["var3"] <= 10]
#               e = [ p['var3'] for p in table.where("var3 <= 20")]
#                e = [ p['var3'] for p in
# table.where(table.cols.var1 == "10")]  # More
                     # than ten times faster than the next one
#                e = [ p['var3'] for p in table
#                      if p['var1'] == "10"]
#                e = [ p['var3'] for p in table.where('var2 <= 20')]
                e = [p['var3']
                     for p in table.where('(var2 <= 20) & (var2 >= 3)')]
                # e = [ p[0] for p in table.where('var2 <= 20')]
                #e = [ p['var3'] for p in table if p['var2'] <= 20 ]
                # e = [ p[:] for p in table if p[1] <= 20 ]
#                  e = [ p['var3'] for p in table._whereInRange(table.cols.var2 <=20)]
                #e = [ p['var3'] for p in table.iterrows(0,21) ]
#                  e = [ p['var3'] for p in table.iterrows()
#                       if p.nrow() <= 20 ]
                #e = [ p['var3'] for p in table.iterrows(1,0,1000)]
                #e = [ p['var3'] for p in table.iterrows(1,100)]
                # e = [ p['var3'] for p in table.iterrows(step=2)
                #      if p.nrow() < 20 ]
                # e = [ p['var2'] for p in table.iterrows()
                #      if p['var2'] < 20 ]
                # for p in table.iterrows():
                #      pass
            if verbose:
                # print "Last record read:", p
                print("resulting selection list ==>", e)

            rowsread += table.nrows
            row += 1
            if verbose:
                print("Total selected records ==> ", len(e))

    # Close the file (eventually destroy the extended type)
    fileh.close()

    return (rowsread, rowsize)


def readField(filename, field, rng, verbose):
    fileh = open_file(filename, mode="r")
    rowsread = 0
    if rng is None:
        rng = [0, -1, 1]
    if field == "all":
        field = None
    for groupobj in fileh.walk_groups(fileh.root):
        for table in fileh.list_nodes(groupobj, 'Table'):
            rowsize = table.rowsize
            # table.nrowsinbuf = 3 # For testing purposes
            if verbose:
                print("Max rows in buf:", table.nrowsinbuf)
                print("Rows in", table._v_pathname, ":", table.nrows)
                print("Buffersize:", table.rowsize * table.nrowsinbuf)
                print("MaxTuples:", table.nrowsinbuf)
                print("(field, start, stop, step) ==>", (field, rng[0], rng[1],
                                                         rng[2]))

            e = table.read(rng[0], rng[1], rng[2], field)

            rowsread += table.nrows
            if verbose:
                print("Selected rows ==> ", e)
                print("Total selected rows ==> ", len(e))

    # Close the file (eventually destroy the extended type)
    fileh.close()
    return (rowsread, rowsize)

if __name__ == "__main__":
    import sys
    import getopt

    try:
        import psyco
        psyco_imported = 1
    except:
        psyco_imported = 0

    import time

    usage = """usage: %s [-v] [-p] [-P] [-R range] [-r] [-w] [-s recsize] [-f field] [-c level] [-l complib] [-i iterations] [-S] [-F] file
            -v verbose
            -p use "psyco" if available
            -P do profile
            -R select a range in a field in the form "start,stop,step"
            -r only read test
            -w only write test
            -s use [big] record, [medium] or [small]
            -f only read stated field name in tables ("all" means all fields)
            -c sets a compression level (do not set it or 0 for no compression)
            -S activate shuffling filter
            -F activate fletcher32 filter
            -l sets the compression library to be used ("zlib", "lzo", "blosc", "bzip2")
            -i sets the number of rows in each table\n""" % sys.argv[0]

    try:
        opts, pargs = getopt.getopt(sys.argv[1:], 'vpPSFR:rwf:s:c:l:i:')
    except:
        sys.stderr.write(usage)
        sys.exit(0)

    # if we pass too much parameters, abort
    if len(pargs) != 1:
        sys.stderr.write(usage)
        sys.exit(0)

    # default options
    verbose = 0
    profile = 0
    rng = None
    recsize = "medium"
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
        if option[0] == '-P':
            profile = 1
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
        elif option[0] == '-f':
            fieldName = option[1]
        elif option[0] == '-s':
            recsize = option[1]
            if recsize not in ["big", "medium", "small"]:
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

    if verbose:
        print("numpy version:", NP.__version__)
        if psyco_imported and usepsyco:
            print("Using psyco version:", psyco.version_info)

    if testwrite:
        print("Compression level:", complevel)
        if complevel > 0:
            print("Compression library:", complib)
            if shuffle:
                print("Suffling...")
        t1 = time.time()
        cpu1 = time.clock()
        if psyco_imported and usepsyco:
            psyco.bind(createFile)
        if profile:
            import profile as prof
            import pstats
            prof.run(
                '(rowsw, rowsz) = createFile(file, iterations, filters, '
                'recsize)',
                'table-bench.prof')
            stats = pstats.Stats('table-bench.prof')
            stats.strip_dirs()
            stats.sort_stats('time', 'calls')
            stats.print_stats(20)
        else:
            (rowsw, rowsz) = createFile(file, iterations, filters, recsize)
        t2 = time.time()
        cpu2 = time.clock()
        tapprows = round(t2 - t1, 3)
        cpuapprows = round(cpu2 - cpu1, 3)
        tpercent = int(round(cpuapprows / tapprows, 2) * 100)
        print("Rows written:", rowsw, " Row size:", rowsz)
        print("Time writing rows: %s s (real) %s s (cpu)  %s%%" %
              (tapprows, cpuapprows, tpercent))
        print("Write rows/sec: ", int(rowsw / float(tapprows)))
        print("Write KB/s :", int(rowsw * rowsz / (tapprows * 1024)))

    if testread:
        t1 = time.time()
        cpu1 = time.clock()
        if psyco_imported and usepsyco:
            psyco.bind(readFile)
            # psyco.bind(readField)
            pass
        if rng or fieldName:
            (rowsr, rowsz) = readField(file, fieldName, rng, verbose)
            pass
        else:
            for i in range(1):
                (rowsr, rowsz) = readFile(file, recsize, verbose)
        t2 = time.time()
        cpu2 = time.clock()
        treadrows = round(t2 - t1, 3)
        cpureadrows = round(cpu2 - cpu1, 3)
        tpercent = int(round(cpureadrows / treadrows, 2) * 100)
        print("Rows read:", rowsr, " Row size:", rowsz)
        print("Time reading rows: %s s (real) %s s (cpu)  %s%%" %
              (treadrows, cpureadrows, tpercent))
        print("Read rows/sec: ", int(rowsr / float(treadrows)))
        print("Read KB/s :", int(rowsr * rowsz / (treadrows * 1024)))
