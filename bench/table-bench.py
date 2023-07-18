#!/usr/bin/env python

import numpy as np
import tables as tb


class Small(tb.IsDescription):
    var1 = tb.StringCol(itemsize=4, pos=2)
    var2 = tb.Int32Col(pos=1)
    var3 = tb.Float64Col(pos=0)

# Define a user record to characterize some kind of particles


class Medium(tb.IsDescription):
    name = tb.StringCol(itemsize=16, pos=0)    # 16-character String
    float1 = tb.Float64Col(shape=2, dflt=np.arange(2), pos=1)
    #float1 = Float64Col(dflt=2.3)
    #float2 = Float64Col(dflt=2.3)
    # zADCcount    = Int16Col()               # signed short integer
    ADCcount = tb.Int32Col(pos=6)              # signed short integer
    grid_i = tb.Int32Col(pos=7)                # integer
    grid_j = tb.Int32Col(pos=8)                # integer
    pressure = tb.Float32Col(pos=9)            # float  (single-precision)
    energy = tb.Float64Col(pos=2)              # double (double-precision)
    # unalig      = Int8Col()                 # just to unalign data

# Define a user record to characterize some kind of particles


class Big(tb.IsDescription):
    name = tb.StringCol(itemsize=16)           # 16-character String
    float1 = tb.Float64Col(shape=32, dflt=np.arange(32))
    float2 = tb.Float64Col(shape=32, dflt=2.2)
    TDCcount = tb.Int8Col()                    # signed short integer
    #ADCcount    = Int32Col()
    # ADCcount = Int16Col()                   # signed short integer
    grid_i = tb.Int32Col()                       # integer
    grid_j = tb.Int32Col()                       # integer
    pressure = tb.Float32Col()                   # float  (single-precision)
    energy = tb.Float64Col()                     # double (double-precision)


def createFile(filename, totalrows, filters, recsize):

    # Open a file in "w"rite mode
    fileh = tb.open_file(filename, mode="w", title="Table Benchmark",
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

    fileh = tb.open_file(filename, mode="r")
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
    fileh = tb.open_file(filename, mode="r")
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

    from time import perf_counter as clock
    from time import process_time as cpuclock

    usage = """usage: %s [-v] [-P] [-R range] [-r] [-w] [-s recsize] [-f field] [-c level] [-l complib] [-n nrows] [-S] [-B] [-F] file
            -v verbose
            -P do profile
            -R select a range in a field in the form "start,stop,step"
            -r only read test
            -w only write test
            -s use [big] record, [medium] or [small]
            -f only read stated field name in tables ("all" means all fields)
            -c sets a compression level (do not set it or 0 for no compression)
            -S activate shuffle filter
            -B activate bitshuffle filter
            -F activate fletcher32 filter
            -l sets the compression library to be used ("zlib", "lzo", "blosc", "bzip2")
            -n sets the number of rows in each table\n""" % sys.argv[0]

    try:
        opts, pargs = getopt.getopt(sys.argv[1:], 'vPSBFR:rwf:s:c:l:n:')
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
    complevel = 9
    shuffle = 0
    fletcher32 = 0
    complib = "blosc2:blosclz"
    nrows = 1_000_000

    # Get the options
    for option in opts:
        if option[0] == '-v':
            verbose = 1
        if option[0] == '-P':
            profile = 1
        if option[0] == '-S':
            shuffle = 1
        if option[0] == '-B':
            shuffle = 2  # bitshuffle
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
        elif option[0] == '-n':
            nrows = int(option[1])

    # Build the Filters instance
    filters = tb.Filters(complevel=complevel, complib=complib,
                         shuffle=(True if shuffle == 1 else False),
                         bitshuffle=(True if shuffle == 2 else False),
                         fletcher32=fletcher32)

    # Catch the hdf5 file passed as the last argument
    file = pargs[0]

    if verbose:
        print("numpy version:", np.__version__)

    if testwrite:
        print("Compression level:", complevel)
        if complevel > 0:
            print("Compression library:", complib)
            if shuffle == 1:
                print("Shuffling...")
            elif shuffle == 2:
                print("Bitshuffling...")
        t1 = clock()
        cpu1 = cpuclock()
        if profile:
            import profile as prof
            import pstats
            prof.run(
                '(rowsw, rowsz) = createFile(file, nrows, filters, '
                'recsize)',
                'table-bench.prof')
            stats = pstats.Stats('table-bench.prof')
            stats.strip_dirs()
            stats.sort_stats('time', 'calls')
            stats.print_stats(20)
        else:
            (rowsw, rowsz) = createFile(file, nrows, filters, recsize)
        t2 = clock()
        cpu2 = cpuclock()
        tapprows = t2 - t1
        cpuapprows = cpu2 - cpu1
        print(f"Rows written: {rowsw}  Row size: {rowsz}")
        print(
            f"Time writing rows: {tapprows:.3f} s (real) "
            f"{cpuapprows:.3f} s (cpu)  {cpuapprows / tapprows:.0%}")
        print(f"Write Mrows/sec:  {rowsw / (tapprows * 1e6):.3f}")
        print(f"Write MB/s : {rowsw * rowsz / (tapprows * 1024 * 1024):.3f}")

    if testread:
        t1 = clock()
        cpu1 = cpuclock()
        if rng or fieldName:
            (rowsr, rowsz) = readField(file, fieldName, rng, verbose)
            pass
        else:
            for i in range(1):
                (rowsr, rowsz) = readFile(file, recsize, verbose)
        t2 = clock()
        cpu2 = cpuclock()
        treadrows = t2 - t1
        cpureadrows = cpu2 - cpu1
        print(f"Rows read: {rowsr}  Row size: {rowsz}")
        print(
            f"Time reading rows: {treadrows:.3f} s (real) "
            f"{cpureadrows:.3f} s (cpu)  {cpureadrows / treadrows:.0%}")
        print(f"Read Mrows/sec:  {rowsr / (treadrows * 1e6):.3f}")
        print(f"Read MB/s : {rowsr * rowsz / (treadrows * 1024 * 1024):.3f}")
