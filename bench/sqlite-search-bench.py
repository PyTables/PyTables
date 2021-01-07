#!/usr/bin/python

import os
import random
import sqlite3
import sys
from pathlib import Path
from time import perf_counter as clock
from time import process_time as cpuclock

import numpy as np
import tables as tb

randomvalues = 0
standarddeviation = 10_000
# Initialize the random generator always with the same integer
# in order to have reproductible results
random.seed(19)
np.random.seed((19, 20))

# defaults
psycon = 0
worst = 0


def createNewBenchFile(bfile, verbose):

    class Create(tb.IsDescription):
        nrows = tb.Int32Col(pos=0)
        irows = tb.Int32Col(pos=1)
        tfill = tb.Float64Col(pos=2)
        tidx = tb.Float64Col(pos=3)
        tcfill = tb.Float64Col(pos=4)
        tcidx = tb.Float64Col(pos=5)
        rowsecf = tb.Float64Col(pos=6)
        rowseci = tb.Float64Col(pos=7)
        fsize = tb.Float64Col(pos=8)
        isize = tb.Float64Col(pos=9)
        psyco = tb.BoolCol(pos=10)

    class Search(tb.IsDescription):
        nrows = tb.Int32Col(pos=0)
        rowsel = tb.Int32Col(pos=1)
        time1 = tb.Float64Col(pos=2)
        time2 = tb.Float64Col(pos=3)
        tcpu1 = tb.Float64Col(pos=4)
        tcpu2 = tb.Float64Col(pos=5)
        rowsec1 = tb.Float64Col(pos=6)
        rowsec2 = tb.Float64Col(pos=7)
        psyco = tb.BoolCol(pos=8)

    if verbose:
        print("Creating a new benchfile:", bfile)
    # Open the benchmarking file
    bf = tb.open_file(bfile, "w")
    # Create groups
    for recsize in ["sqlite_small"]:
        group = bf.create_group("/", recsize, recsize + " Group")
        # Attach the row size of table as attribute
        if recsize == "small":
            group._v_attrs.rowsize = 16
        # Create a Table for writing bench
        bf.create_table(group, "create_indexed", Create, "indexed values")
        bf.create_table(group, "create_standard", Create, "standard values")
        # create a group for searching bench
        groupS = bf.create_group(group, "search", "Search Group")
        # Create Tables for searching
        for mode in ["indexed", "standard"]:
            group = bf.create_group(groupS, mode, mode + " Group")
            # for searching bench
            # for atom in ["string", "int", "float", "bool"]:
            for atom in ["string", "int", "float"]:
                bf.create_table(group, atom, Search, atom + " bench")
    bf.close()


def createFile(filename, nrows, filters, indexmode, heavy, noise, bfile,
               verbose):

    # Initialize some variables
    t1 = 0
    t2 = 0
    tcpu1 = 0
    tcpu2 = 0
    rowsecf = 0
    rowseci = 0
    size1 = 0
    size2 = 0

    if indexmode == "standard":
        print("Creating a new database:", dbfile)
        instd = os.popen("/usr/local/bin/sqlite " + dbfile, "w")
        CREATESTD = """
CREATE TABLE small (
-- Name         Type            -- Example
---------------------------------------
recnum  INTEGER PRIMARY KEY,  -- 345
var1            char(4),        -- Abronia villosa
var2            INTEGER,        -- 111
var3            FLOAT        --  12.32
);
"""
        CREATEIDX = """
CREATE TABLE small (
-- Name         Type            -- Example
---------------------------------------
recnum  INTEGER PRIMARY KEY,  -- 345
var1            char(4),        -- Abronia villosa
var2            INTEGER,        -- 111
var3            FLOAT        --  12.32
);
CREATE INDEX ivar1 ON small(var1);
CREATE INDEX ivar2 ON small(var2);
CREATE INDEX ivar3 ON small(var3);
"""
        # Creating the table first and indexing afterwards is a bit faster
        instd.write(CREATESTD)
        instd.close()

    conn = sqlite3.connect(dbfile)
    cursor = conn.cursor()
    if indexmode == "standard":
        place_holders = ",".join(['%s'] * 3)
        # Insert rows
        SQL = "insert into small values(NULL, %s)" % place_holders
        time1 = clock()
        cpu1 = cpuclock()
        # This way of filling is to copy the PyTables benchmark
        nrowsbuf = 1000
        minimum = 0
        maximum = nrows
        for i in range(0, nrows, nrowsbuf):
            if i + nrowsbuf > nrows:
                j = nrows
            else:
                j = i + nrowsbuf
            if randomvalues:
                var3 = np.random.uniform(minimum, maximum, shape=[j - i])
            else:
                var3 = np.arange(i, j, type=np.float64)
                if noise:
                    var3 += np.random.uniform(-3, 3, shape=[j - i])
            var2 = np.array(var3, type=np.int32)
            var1 = np.array(None, shape=[j - i], dtype='s4')
            if not heavy:
                for n in range(j - i):
                    var1[n] = str("%.4s" % var2[n])
            for n in range(j - i):
                fields = (var1[n], var2[n], var3[n])
                cursor.execute(SQL, fields)
            conn.commit()
        t1 = clock() - time1
        tcpu1 = cpuclock() - cpu1
        rowsecf = nrows / t1
        size1 = Path(dbfile).stat().st_size
        print(f"******** Results for writing nrows = {nrows} *********")
        print(f"Insert time: {t1:.5f}, KRows/s: {nrows / 1000 / t1:.3f}")
        print(f", File size: {size1 / 1024 / 1024:.3f} MB")

    # Indexem
    if indexmode == "indexed":
        time1 = clock()
        cpu1 = cpuclock()
        if not heavy:
            cursor.execute("CREATE INDEX ivar1 ON small(var1)")
            conn.commit()
        cursor.execute("CREATE INDEX ivar2 ON small(var2)")
        conn.commit()
        cursor.execute("CREATE INDEX ivar3 ON small(var3)")
        conn.commit()
        t2 = clock() - time1
        tcpu2 = cpuclock() - cpu1
        rowseci = nrows / t2
        print(f"Index time: {t2:.5f}, IKRows/s: {nrows / 1000 / t2:.3f}")
        size2 = Path(dbfile).stat().st_size - size1
        print(f", Final size with index: {size2 / 1024 / 1024:.3f} MB")

    conn.close()

    # Collect benchmark data
    bf = tb.open_file(bfile, "a")
    recsize = "sqlite_small"
    if indexmode == "indexed":
        table = bf.get_node("/" + recsize + "/create_indexed")
    else:
        table = bf.get_node("/" + recsize + "/create_standard")
    table.row["nrows"] = nrows
    table.row["irows"] = nrows
    table.row["tfill"] = t1
    table.row["tidx"] = t2
    table.row["tcfill"] = tcpu1
    table.row["tcidx"] = tcpu2
    table.row["psyco"] = psycon
    table.row["rowsecf"] = rowsecf
    table.row["rowseci"] = rowseci
    table.row["fsize"] = size1
    table.row["isize"] = size2
    table.row.append()
    bf.close()

    return


def readFile(dbfile, nrows, indexmode, heavy, dselect, bfile, riter):
    # Connect to the database.
    conn = sqlite3.connect(db=dbfile, mode=755)
    # Obtain a cursor
    cursor = conn.cursor()

    #      select count(*), avg(var2)
    SQL1 = """
    select recnum
    from small where var1 = %s
    """
    SQL2 = """
    select recnum
    from small where var2 >= %s and var2 < %s
    """
    SQL3 = """
    select recnum
    from small where var3 >= %s and var3 < %s
    """

    # Open the benchmark database
    bf = tb.open_file(bfile, "a")
    # default values for the case that columns are not indexed
    t2 = 0
    tcpu2 = 0
    # Some previous computations for the case of random values
    if randomvalues:
        # algorithm to choose a value separated from mean
# If want to select fewer values, select this
#         if nrows/2 > standarddeviation*3:
# Choose five standard deviations away from mean value
#             dev = standarddeviation*5
# dev = standarddeviation*math.log10(nrows/1000.)

        # This algorithm give place to too asymmetric result values
#         if standarddeviation*10 < nrows/2:
# Choose four standard deviations away from mean value
#             dev = standarddeviation*4
#         else:
#             dev = 100
        # Yet Another Algorithm
        if nrows / 2 > standarddeviation * 10:
            dev = standarddeviation * 4
        elif nrows / 2 > standarddeviation:
            dev = standarddeviation * 2
        elif nrows / 2 > standarddeviation / 10:
            dev = standarddeviation / 10
        else:
            dev = standarddeviation / 100

        valmax = round(nrows / 2 - dev)
        # split the selection range in regular chunks
        if riter > valmax * 2:
            riter = valmax * 2
        chunksize = (valmax * 2 / riter) * 10
        # Get a list of integers for the intervals
        randlist = range(0, valmax, chunksize)
        randlist.extend(range(nrows - valmax, nrows, chunksize))
        # expand the list ten times so as to use the cache
        randlist = randlist * 10
        # shuffle the list
        random.shuffle(randlist)
        # reset the value of chunksize
        chunksize = chunksize / 10
        # print "chunksize-->", chunksize
        # randlist.sort();print "randlist-->", randlist
    else:
        chunksize = 3
    if heavy:
        searchmodelist = ["int", "float"]
    else:
        searchmodelist = ["string", "int", "float"]

    # Execute queries
    for atom in searchmodelist:
        time2 = 0
        cpu2 = 0
        rowsel = 0
        for i in range(riter):
            rnd = random.randrange(nrows)
            time1 = clock()
            cpu1 = cpuclock()
            if atom == "string":
                #cursor.execute(SQL1, "1111")
                cursor.execute(SQL1, str(rnd)[-4:])
            elif atom == "int":
                #cursor.execute(SQL2 % (rnd, rnd+3))
                cursor.execute(SQL2 % (rnd, rnd + dselect))
            elif atom == "float":
                #cursor.execute(SQL3 % (float(rnd), float(rnd+3)))
                cursor.execute(SQL3 % (float(rnd), float(rnd + dselect)))
            else:
                raise ValueError(
                    "atom must take a value in ['string','int','float']")
            if i == 0:
                t1 = clock() - time1
                tcpu1 = cpuclock() - cpu1
            else:
                if indexmode == "indexed":
                    # if indexed, wait until the 5th iteration to take
                    # times (so as to insure that the index is
                    # effectively cached)
                    if i >= 5:
                        time2 += clock() - time1
                        cpu2 += cpuclock() - cpu1
                else:
                    time2 += clock() - time1
                    time2 += cpuclock() - cpu1
        if riter > 1:
            if indexmode == "indexed" and riter >= 5:
                correction = 5
            else:
                correction = 1
            t2 = time2 / (riter - correction)
            tcpu2 = cpu2 / (riter - correction)

        print(
            f"*** Query results for atom = {atom}, "
            f"nrows = {nrows}, indexmode = {indexmode} ***")
        print(f"Query time: {t1:.5f}, cached time: {t2:.5f}")
        print(f"MRows/s: {nrows / 1_000_000 / t1:.3f}", end=' ')
        if t2 > 0:
            print(f", cached MRows/s: {nrows / 10 ** 6 / t2:.3f}")
        else:
            print()

        # Collect benchmark data
        recsize = "sqlite_small"
        tablepath = "/" + recsize + "/search/" + indexmode + "/" + atom
        table = bf.get_node(tablepath)
        table.row["nrows"] = nrows
        table.row["rowsel"] = rowsel
        table.row["time1"] = t1
        table.row["time2"] = t2
        table.row["tcpu1"] = tcpu1
        table.row["tcpu2"] = tcpu2
        table.row["psyco"] = psycon
        table.row["rowsec1"] = nrows / t1
        if t2 > 0:
            table.row["rowsec2"] = nrows / t2
        table.row.append()
        table.flush()  # Flush the data

    # Close the database
    conn.close()
    bf.close()  # the bench database

    return

if __name__ == "__main__":
    import getopt
    try:
        import psyco
        psyco_imported = 1
    except:
        psyco_imported = 0

    usage = """usage: %s [-v] [-p] [-R] [-h] [-t] [-r] [-w] [-n nrows] [-b file] [-k riter] [-m indexmode] [-N range] datafile
            -v verbose
            -p use "psyco" if available
            -R use Random values for filling
            -h heavy mode (exclude strings from timings)
            -t worsT searching case (to emulate PyTables worst cases)
            -r only read test
            -w only write test
            -n the number of rows (in krows)
            -b bench filename
            -N introduce (uniform) noise within range into the values
            -d the interval for look values (int, float) at. Default is 3.
            -k number of iterations for reading\n""" % sys.argv[0]

    try:
        opts, pargs = getopt.getopt(sys.argv[1:], 'vpRhtrwn:b:k:m:N:d:')
    except:
        sys.stderr.write(usage)
        sys.exit(0)

    # if we pass too much parameters, abort
    if len(pargs) != 1:
        sys.stderr.write(usage)
        sys.exit(0)

    # default options
    dselect = 3
    noise = 0
    verbose = 0
    heavy = 0
    testread = 1
    testwrite = 1
    usepsyco = 0
    nrows = 1000
    bfile = "sqlite-bench.h5"
    supported_imodes = ["indexed", "standard"]
    indexmode = "indexed"
    riter = 2

    # Get the options
    for option in opts:
        if option[0] == '-v':
            verbose = 1
        if option[0] == '-p':
            usepsyco = 1
        elif option[0] == '-R':
            randomvalues = 1
        elif option[0] == '-h':
            heavy = 1
        elif option[0] == '-t':
            worst = 1
        elif option[0] == '-r':
            testwrite = 0
        elif option[0] == '-w':
            testread = 0
        elif option[0] == '-b':
            bfile = option[1]
        elif option[0] == '-N':
            noise = float(option[1])
        elif option[0] == '-m':
            indexmode = option[1]
            if indexmode not in supported_imodes:
                raise ValueError(
                    "Indexmode should be any of '%s' and you passed '%s'" %
                    (supported_imodes, indexmode))
        elif option[0] == '-n':
            nrows = int(float(option[1]) * 1000)
        elif option[0] == '-d':
            dselect = float(option[1])
        elif option[0] == '-k':
            riter = int(option[1])

    # remaining parameters
    dbfile = pargs[0]

    if worst:
        nrows -= 1  # the worst case

    # Create the benchfile (if needed)
    if not Path(bfile).exists():
        createNewBenchFile(bfile, verbose)

    if testwrite:
        if psyco_imported and usepsyco:
            psyco.bind(createFile)
            psycon = 1
        createFile(dbfile, nrows, None, indexmode, heavy, noise, bfile,
                   verbose)

    if testread:
        if psyco_imported and usepsyco:
            psyco.bind(readFile)
            psycon = 1
        readFile(dbfile, nrows, indexmode, heavy, dselect, bfile, riter)
