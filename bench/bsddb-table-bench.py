#!/usr/bin/env python
###### WARNING #######
### This script is obsoleted ###
# If you get it working again, please drop me a line
# F. Alted 2004-01-27

import sys
import struct
import cPickle

import numpy as np
import tables as tb

from bsddb import db
import psyco


# This class is accessible only for the examples
class Small(tb.IsDescription):
    """Record descriptor.

    A record has several columns. They are represented here as class
    attributes, whose names are the column names and their values will
    become their types. The IsColDescr class will take care the user
    will not add any new variables and that its type is correct.

    """

    var1 = tb.StringCol(itemsize=16)
    var2 = tb.Int32Col()
    var3 = tb.Float64Col()

# Define a user record to characterize some kind of particles


class Medium(tb.IsDescription):
    name = tb.StringCol(itemsize=16, pos=0)  # 16-character String
    #float1      = Float64Col(shape=2, dflt=2.3)
    float1 = tb.Float64Col(dflt=1.3, pos=1)
    float2 = tb.Float64Col(dflt=2.3, pos=2)
    ADCcount = tb.Int16Col(pos=3)     # signed short integer
    grid_i = tb.Int32Col(pos=4)        # integer
    grid_j = tb.Int32Col(pos=5)        # integer
    pressure = tb.Float32Col(pos=6)    # float  (single-precision)
    energy = tb.Float64Col(pos=7)      # double (double-precision)

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


def createFile(filename, totalrows, recsize, verbose):

    # Open a 'n'ew file
    dd = db.DB()
    if recsize == "big":
        isrec = tb.Description(Big)
    elif recsize == "medium":
        isrec = Medium()
    else:
        isrec = tb.Description(Small)
    # dd.set_re_len(struct.calcsize(isrec._v_fmt))  # fixed length records
    dd.open(filename, db.DB_RECNO, db.DB_CREATE | db.DB_TRUNCATE)

    rowswritten = 0
    # Get the record object associated with the new table
    if recsize == "big":
        isrec = Big()
        arr = np.array(np.arange(32), type=np.float64)
        arr2 = np.array(np.arange(32), type=np.float64)
    elif recsize == "medium":
        isrec = Medium()
        arr = np.array(np.arange(2), type=np.float64)
    else:
        isrec = Small()
    # print d
    # Fill the table
    if recsize == "big" or recsize == "medium":
        d = {"name": " ",
             "float1": 1.0,
             "float2": 2.0,
             "ADCcount": 12,
             "grid_i": 1,
             "grid_j": 1,
             "pressure": 1.9,
             "energy": 1.8,
             }
        for i in range(totalrows):
            #d['name']  = 'Particle: %6d' % (i)
            #d['TDCcount'] = i % 256
            d['ADCcount'] = (i * 256) % (1 << 16)
            if recsize == "big":
                #d.float1 = np.array([i]*32, np.float64)
                #d.float2 = np.array([i**2]*32, np.float64)
                arr[0] = 1.1
                d['float1'] = arr
                arr2[0] = 2.2
                d['float2'] = arr2
                pass
            else:
                d['float1'] = float(i)
                d['float2'] = float(i)
            d['grid_i'] = i
            d['grid_j'] = 10 - i
            d['pressure'] = float(i * i)
            d['energy'] = d['pressure']
            dd.append(cPickle.dumps(d))
#             dd.append(struct.pack(isrec._v_fmt,
#                                   d['name'], d['float1'], d['float2'],
#                                   d['ADCcount'],
#                                   d['grid_i'], d['grid_j'],
#                                   d['pressure'],  d['energy']))
    else:
        d = {"var1": " ", "var2": 1, "var3": 12.1e10}
        for i in range(totalrows):
            d['var1'] = str(i)
            d['var2'] = i
            d['var3'] = 12.1e10
            dd.append(cPickle.dumps(d))
            #dd.append(
            #    struct.pack(isrec._v_fmt, d['var1'], d['var2'], d['var3']))

    rowswritten += totalrows

    # Close the file
    dd.close()
    return (rowswritten, struct.calcsize(isrec._v_fmt))


def readFile(filename, recsize, verbose):
    # Open the HDF5 file in read-only mode
    #fileh = shelve.open(filename, "r")
    dd = db.DB()
    if recsize == "big":
        isrec = Big()
    elif recsize == "medium":
        isrec = Medium()
    else:
        isrec = Small()
    # dd.set_re_len(struct.calcsize(isrec._v_fmt))  # fixed length records
    # dd.set_re_pad('-') # sets the pad character...
    # dd.set_re_pad(45)  # ...test both int and char
    dd.open(filename, db.DB_RECNO)
    if recsize == "big" or recsize == "medium":
        print(isrec._v_fmt)
        c = dd.cursor()
        rec = c.first()
        e = []
        while rec:
            record = cPickle.loads(rec[1])
            #record = struct.unpack(isrec._v_fmt, rec[1])
            # if verbose:
            #    print record
            if record['grid_i'] < 20:
                e.append(record['grid_j'])
            # if record[4] < 20:
            #    e.append(record[5])
            rec = next(c)
    else:
        print(isrec._v_fmt)
        #e = [ t[1] for t in fileh[table] if t[1] < 20 ]
        c = dd.cursor()
        rec = c.first()
        e = []
        while rec:
            record = cPickle.loads(rec[1])
            #record = struct.unpack(isrec._v_fmt, rec[1])
            # if verbose:
            #    print record
            if record['var2'] < 20:
                e.append(record['var1'])
            # if record[1] < 20:
            #    e.append(record[2])
            rec = next(c)

    print("resulting selection list ==>", e)
    print("last record read ==>", record)
    print("Total selected records ==> ", len(e))

    # Close the file (eventually destroy the extended type)
    dd.close()


# Add code to test here
if __name__ == "__main__":
    import getopt
    from time import perf_counter as clock

    usage = """usage: %s [-v] [-s recsize] [-i iterations] file
            -v verbose
            -s use [big] record, [medium] or [small]
            -i sets the number of rows in each table\n""" % sys.argv[0]

    try:
        opts, pargs = getopt.getopt(sys.argv[1:], 's:vi:')
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
    verbose = 0

    # Get the options
    for option in opts:
        if option[0] == '-s':
            recsize = option[1]
            if recsize not in ["big", "medium", "small"]:
                sys.stderr.write(usage)
                sys.exit(0)
        elif option[0] == '-i':
            iterations = int(option[1])
        elif option[0] == '-v':
            verbose = 1

    # Catch the hdf5 file passed as the last argument
    file = pargs[0]

    t1 = clock()
    psyco.bind(createFile)
    (rowsw, rowsz) = createFile(file, iterations, recsize, verbose)
    t2 = clock()
    tapprows = t2 - t1

    t1 = clock()
    psyco.bind(readFile)
    readFile(file, recsize, verbose)
    t2 = clock()
    treadrows = t2 - t1

    print(f"Rows written: {rowsw}, Row size: {rowsz}")
    print(f"Time appending rows: {tapprows:.3f}")
    if tapprows > 0.001:
        print(f"Write rows/sec: {iterations / tapprows:.0f}")
        print(f"Write KB/s: {rowsw * rowsz / (tapprows * 1024):.0f}")
    print(f"Time reading rows: {treadrows:.3f}")
    if treadrows > 0.001:
        print(f"Read rows/sec: {iterations / treadrows:.0f}")
        print(f"Read KB/s: {rowsw * rowsz / (treadrows * 1024):.0f}")
