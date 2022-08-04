#!/usr/bin/env python

"""This script allows to create arbitrarily large files with the desired
combination of groups, tables per group and rows per table.

Issue "python stress-test3.py" without parameters for a help on usage.

"""

import gc
import sys
from time import perf_counter as clock
from time import process_time as cpuclock
import tables as tb


class Test(tb.IsDescription):
    ngroup = tb.Int32Col(pos=1)
    ntable = tb.Int32Col(pos=2)
    nrow = tb.Int32Col(pos=3)
    float = tb.Float32Col(pos=3)
    #string = tb.StringCol(500, pos=4)


def createFile(filename, ngroups, ntables, nrows, complevel, complib, recsize):

    # First, create the groups

    # Open a file in "w"rite mode
    fileh = tb.open_file(filename, mode="w", title="PyTables Stress Test")

    for k in range(ngroups):
        # Create the group
        group = fileh.create_group("/", 'group%04d' % k, "Group %d" % k)

    fileh.close()

    # Now, create the tables
    rowswritten = 0
    rowsize = 0
    for k in range(ngroups):
        fileh = tb.open_file(filename, mode="a", root_uep='group%04d' % k)
        # Get the group
        group = fileh.root
        for j in range(ntables):
            # Create a table
            table = fileh.create_table(group, 'table%04d' % j, Test,
                                       'Table%04d' % j,
                                       tb.Filters(complevel, complib), nrows)
            rowsize = table.rowsize
            # Get the row object associated with the new table
            row = table.row
            # Fill the table
            for i in range(nrows):
                row['ngroup'] = k
                row['ntable'] = j
                row['nrow'] = i
                row.append()

            rowswritten += nrows
            table.flush()

        # Close the file
        fileh.close()

    return (rowswritten, rowsize)


def readFile(filename, ngroups, recsize, verbose):
    # Open the HDF5 file in read-only mode

    rowsread = 0
    rowsize = 0
    buffersize = 0
    for ngroup in range(ngroups):
        fileh = tb.open_file(filename, mode="r")
        # Get the group
        group = fileh.get_node(fileh.root, 'group%04d' % ngroup)
        ntable = 0
        if verbose:
            print("Group ==>", group)
        for table in fileh.list_nodes(group, 'Leaf'):
            rowsize = table.rowsize
            buffersize = table.rowsize * table.nrowsinbuf
            if verbose > 1:
                print("Table ==>", table)
                print("Max rows in buf:", table.nrowsinbuf)
                print("Rows in", table._v_pathname, ":", table.nrows)
                print("Buffersize:", table.rowsize * table.nrowsinbuf)
                print("MaxTuples:", table.nrowsinbuf)

            nrow = 0
            for row in table:
                try:
                    assert row["ngroup"] == ngroup
                    assert row["ntable"] == ntable
                    assert row["nrow"] == nrow
                except:
                    print("Error in group: %d, table: %d, row: %d" %
                          (ngroup, ntable, nrow))
                    print("Record ==>", row)
                nrow += 1

            assert nrow == table.nrows
            rowsread += table.nrows
            ntable += 1

        # Close the file (eventually destroy the extended type)
        fileh.close()

    return (rowsread, rowsize, buffersize)


def dump_garbage():
    """show us what the garbage is about."""
    # Force collection
    print("\nGARBAGE:")
    gc.collect()

    print("\nGARBAGE OBJECTS:")
    for x in gc.garbage:
        s = str(x)
        #if len(s) > 80: s = s[:77] + "..."
        print(type(x), "\n   ", s)

if __name__ == "__main__":
    import getopt

    usage = """usage: %s [-d debug] [-v level] [-p] [-r] [-w] [-l complib] [-c complevel] [-g ngroups] [-t ntables] [-i nrows] file
    -d debugging level
    -v verbosity level
    -r only read test
    -w only write test
    -l sets the compression library to be used ("zlib", "bzip2", "blosc", "blosc:codec")
    -c sets a compression level (do not set it or 0 for no compression)
    -g number of groups hanging from "/"
    -t number of tables per group
    -i number of rows per table
"""

    try:
        opts, pargs = getopt.getopt(sys.argv[1:], 'd:v:parwl:c:g:t:i:')
    except:
        sys.stderr.write(usage)
        sys.exit(0)

    # if we pass too much parameters, abort
    if len(pargs) != 1:
        sys.stderr.write(usage)
        sys.exit(0)

    # default options
    ngroups = 5
    ntables = 5
    nrows = 100
    verbose = 0
    debug = 0
    recsize = "medium"
    testread = 1
    testwrite = 1
    complevel = 0
    complib = "zlib"

    # Get the options
    for option in opts:
        if option[0] == '-d':
            debug = int(option[1])
        if option[0] == '-v':
            verbose = int(option[1])
        elif option[0] == '-r':
            testwrite = 0
        elif option[0] == '-w':
            testread = 0
        elif option[0] == '-l':
            complib = option[1]
        elif option[0] == '-c':
            complevel = int(option[1])
        elif option[0] == '-g':
            ngroups = int(option[1])
        elif option[0] == '-t':
            ntables = int(option[1])
        elif option[0] == '-i':
            nrows = int(option[1])

    if debug:
        gc.enable()
        gc.set_debug(gc.DEBUG_LEAK)

    # Catch the hdf5 file passed as the last argument
    file = pargs[0]

    print("Compression level:", complevel)
    if complevel > 0:
        print("Compression library:", complib)
    rowsw = 0
    if testwrite:
        t1 = clock()
        cpu1 = cpuclock()
        (rowsw, rowsz) = createFile(file, ngroups, ntables, nrows,
                                    complevel, complib, recsize)
        t2 = clock()
        cpu2 = cpuclock()
        tapprows = t2 - t1
        cpuapprows = cpu2 - cpu1
        print(f"Rows written: {rowsw}  Row size: {rowsz}")
        print(
            f"Time writing rows: {tapprows:.3f} s (real) "
            f"{cpuapprows:.3f} s (cpu)  {cpuapprows / tapprows:.0%}")
        print(f"Write Krows/sec:  {rowsw / (tapprows * 1000):.3f}")
        print(f"Write MB/s : {rowsw * rowsz / (tapprows * 2**20):.3f}")

    if testread:
        t1 = clock()
        cpu1 = cpuclock()
        (rowsr, rowsz, bufsz) = readFile(file, ngroups, recsize, verbose)
        t2 = clock()
        cpu2 = cpuclock()
        treadrows = t2 - t1
        cpureadrows = cpu2 - cpu1
        print(f"Rows read: {rowsw}  Row size: {rowsz}, Buf size: {bufsz}")
        print(
            f"Time reading rows: {treadrows:.3f} s (real) "
            f"{cpureadrows:.3f} s (cpu)  {cpureadrows / treadrows:.0%}")
        print(f"Read Krows/sec:  {rowsr / (treadrows * 1000):.3f}")
        print(f"Read MB/s : {rowsr * rowsz / (treadrows * 2**20):.3f}")

    # Show the dirt
    if debug > 1:
        dump_garbage()
