#!/usr/bin/env python

import struct
import sys
import shelve
import numpy as np
import tables as tb
import psyco

# This class is accessible only for the examples


class Small(tb.IsDescription):

    """Record descriptor.

    A record has several columns. They are represented here as class
    attributes, whose names are the column names and their values will
    become their types. The IsDescription class will take care the user
    will not add any new variables and that its type is correct.

    """

    var1 = tb.StringCol(itemsize=4)
    var2 = tb.Int32Col()
    var3 = tb.Float64Col()

# Define a user record to characterize some kind of particles


class Medium(tb.IsDescription):
    name = tb.StringCol(itemsize=16)   # 16-character String
    float1 = tb.Float64Col(shape=2, dflt=2.3)
    #float1 = Float64Col(dflt=1.3)
    #float2 = Float64Col(dflt=2.3)
    ADCcount = tb.Int16Col()           # signed short integer
    grid_i = tb.Int32Col()             # integer
    grid_j = tb.Int32Col()             # integer
    pressure = tb.Float32Col()         # float  (single-precision)
    energy = tb.Float64Col()           # double (double-precision)

# Define a user record to characterize some kind of particles


class Big(tb.IsDescription):
    name = tb.StringCol(itemsize=16)   # 16-character String
    #float1 = Float64Col(shape=32, dflt=np.arange(32))
    #float2 = Float64Col(shape=32, dflt=np.arange(32))
    float1 = tb.Float64Col(shape=32, dflt=range(32))
    float2 = tb.Float64Col(shape=32, dflt=[2.2] * 32)
    ADCcount = tb.Int16Col()           # signed short integer
    grid_i = tb.Int32Col()             # integer
    grid_j = tb.Int32Col()             # integer
    pressure = tb.Float32Col()         # float  (single-precision)
    energy = tb.Float64Col()           # double (double-precision)


def createFile(filename, totalrows, recsize):

    # Open a 'n'ew file
    fileh = shelve.open(filename, flag="n")

    rowswritten = 0
    # Get the record object associated with the new table
    if recsize == "big":
        d = Big()
        arr = np.arange(32, dtype=np.float64)
        arr2 = np.arange(32, dtype=np.float64)
    elif recsize == "medium":
        d = Medium()
    else:
        d = Small()
    # print d
    # sys.exit(0)
    for j in range(3):
        # Create a table
        # table = fileh.create_table(group, 'tuple'+str(j), Record(), title,
        #                          compress = 6, expectedrows = totalrows)
        # Create a Table instance
        tablename = 'tuple' + str(j)
        table = []
        # Fill the table
        if recsize == "big" or recsize == "medium":
            for i in range(totalrows):
                d.name = 'Particle: %6d' % (i)
                #d.TDCcount = i % 256
                d.ADCcount = (i * 256) % (1 << 16)
                if recsize == "big":
                    #d.float1 = np.array([i]*32, np.float64)
                    #d.float2 = np.array([i**2]*32, np.float64)
                    arr[0] = 1.1
                    d.float1 = arr
                    arr2[0] = 2.2
                    d.float2 = arr2
                    pass
                else:
                    d.float1 = np.array([i ** 2] * 2, np.float64)
                    #d.float1 = float(i)
                    #d.float2 = float(i)
                d.grid_i = i
                d.grid_j = 10 - i
                d.pressure = float(i * i)
                d.energy = float(d.pressure ** 4)
                table.append((d.ADCcount, d.energy, d.float1, d.float2,
                              d.grid_i, d.grid_j, d.name, d.pressure))
                # Only on float case
                # table.append((d.ADCcount, d.energy, d.float1,
                #              d.grid_i, d.grid_j, d.name, d.pressure))
        else:
            for i in range(totalrows):
                d.var1 = str(i)
                d.var2 = i
                d.var3 = 12.1e10
                table.append((d.var1, d.var2, d.var3))

        # Save this table on disk
        fileh[tablename] = table
        rowswritten += totalrows

    # Close the file
    fileh.close()
    return (rowswritten, struct.calcsize(d._v_fmt))


def readFile(filename, recsize):
    # Open the HDF5 file in read-only mode
    fileh = shelve.open(filename, "r")
    for table in ['tuple0', 'tuple1', 'tuple2']:
        if recsize == "big" or recsize == "medium":
            e = [t[2] for t in fileh[table] if t[4] < 20]
            # if there is only one float (array)
            #e = [ t[1] for t in fileh[table] if t[3] < 20 ]
        else:
            e = [t[1] for t in fileh[table] if t[1] < 20]

        print("resulting selection list ==>", e)
        print("Total selected records ==> ", len(e))

    # Close the file (eventually destroy the extended type)
    fileh.close()


# Add code to test here
if __name__ == "__main__":
    import getopt
    from time import perf_counter as clock

    usage = """usage: %s [-f] [-s recsize] [-i iterations] file
            -s use [big] record, [medium] or [small]
            -i sets the number of rows in each table\n""" % sys.argv[0]

    try:
        opts, pargs = getopt.getopt(sys.argv[1:], 's:fi:')
    except:
        sys.stderr.write(usage)
        sys.exit(0)

    # if we pass too much parameters, abort
    if len(pargs) != 1:
        sys.stderr.write(usage)
        sys.exit(0)

    # default options
    recsize = "medium"
    iterations = 100

    # Get the options
    for option in opts:
        if option[0] == '-s':
            recsize = option[1]
            if recsize not in ["big", "medium", "small"]:
                sys.stderr.write(usage)
                sys.exit(0)
        elif option[0] == '-i':
            iterations = int(option[1])

    # Catch the hdf5 file passed as the last argument
    file = pargs[0]

    t1 = clock()
    psyco.bind(createFile)
    (rowsw, rowsz) = createFile(file, iterations, recsize)
    t2 = clock()
    tapprows = t2 - t1

    t1 = clock()
    psyco.bind(readFile)
    readFile(file, recsize)
    t2 = clock()
    treadrows = t2 - t1

    print(f"Rows written: {rowsw}  Row size: {rowsz}")
    print(f"Time appending rows: {tapprows:.3f}")
    print(f"Write rows/sec: {iterations * 3 / tapprows:.0f}")
    print(f"Write KB/s : {rowsw * rowsz / (tapprows * 1024):.0f}")
    print(f"Time reading rows: {treadrows:.3f}")
    print(f"Read rows/sec:  {iterations * 3 / treadrows:.0f}")
    print(f"Read KB/s : {rowsw * rowsz / (treadrows * 1024):.0f}")
