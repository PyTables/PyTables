import random
from pathlib import Path
from time import perf_counter as clock

import numpy as np
import tables as tb

# in order to always generate the same random sequence
random.seed(19)
np.random.seed((19, 20))


def open_db(filename, remove=0):
    if remove and Path(filename).is_file():
        Path(filename).unlink()
    con = tb.open_file(filename, 'a')
    return con


def create_db(filename, nrows):

    class Record(tb.IsDescription):
        col1 = tb.Int32Col()
        col2 = tb.Int32Col()
        col3 = tb.Float64Col()
        col4 = tb.Float64Col()

    con = open_db(filename, remove=1)
    table = con.create_table(con.root, 'table', Record,
                             filters=filters, expectedrows=nrows)
    table.indexFilters = filters
    step = 1000 * 100
    scale = 0.1
    t1 = clock()
    j = 0
    for i in range(0, nrows, step):
        stop = (j + 1) * step
        if stop > nrows:
            stop = nrows
        arr_f8 = np.arange(i, stop, type=np.float64)
        arr_i4 = np.arange(i, stop, type=np.int32)
        if userandom:
            arr_f8 += np.random.normal(0, stop * scale, shape=[stop - i])
            arr_i4 = np.array(arr_f8, type=np.int32)
        recarr = np.rec.fromarrays([arr_i4, arr_i4, arr_f8, arr_f8])
        table.append(recarr)
        j += 1
    table.flush()
    ctime = clock() - t1
    if verbose:
        print(f"insert time: {ctime:.5f}")
        print(f"Krows/s: {nrows / 1000 / ctime:.5f}")
    index_db(table)
    close_db(con)


def index_db(table):
    t1 = clock()
    table.cols.col2.create_index()
    itime = clock() - t1
    if verbose:
        print(f"index time (int): {itime:.5f}")
        print(f"Krows/s: {nrows / 1000 / itime:.5f}")
    t1 = clock()
    table.cols.col4.create_index()
    itime = clock() - t1
    if verbose:
        print(f"index time (float): {itime:.5f}")
        print(f"Krows/s: {nrows / 1000 / itime:.5f}")


def query_db(filename, rng):
    con = open_db(filename)
    table = con.root.table
    # Query for integer columns
    # Query for non-indexed column
    if not doqueryidx:
        t1 = clock()
        ntimes = 10
        for i in range(ntimes):
            results = [
                r['col1'] for r in table.where(
                    rng[0] + i <= table.cols.col1 <= rng[1] + i)
            ]
        qtime = (clock() - t1) / ntimes
        if verbose:
            print(f"query time (int, not indexed): {qtime:.5f}")
            print(f"Krows/s: {nrows / 1000 / qtime:.5f}")
            print(results)
    # Query for indexed column
    t1 = clock()
    ntimes = 10
    for i in range(ntimes):
        results = [
            r['col1'] for r in table.where(
                rng[0] + i <= table.cols.col2 <= rng[1] + i)
        ]
    qtime = (clock() - t1) / ntimes
    if verbose:
        print(f"query time (int, indexed): {qtime:.5f}")
        print(f"Krows/s: {nrows / 1000 / qtime:.5f}")
        print(results)
    # Query for floating columns
    # Query for non-indexed column
    if not doqueryidx:
        t1 = clock()
        ntimes = 10
        for i in range(ntimes):
            results = [
                r['col3'] for r in table.where(
                    rng[0] + i <= table.cols.col3 <= rng[1] + i)
            ]
        qtime = (clock() - t1) / ntimes
        if verbose:
            print(f"query time (float, not indexed): {qtime:.5f}")
            print(f"Krows/s: {nrows / 1000 / qtime:.5f}")
            print(results)
    # Query for indexed column
    t1 = clock()
    ntimes = 10
    for i in range(ntimes):
        results = [r['col3'] for r in
                   table.where(rng[0] + i <= table.cols.col4 <= rng[1] + i)]
    qtime = (clock() - t1) / ntimes
    if verbose:
        print(f"query time (float, indexed): {qtime:.5f}")
        print(f"Krows/s: {nrows / 1000 / qtime:.5f}")
        print(results)
    close_db(con)


def close_db(con):
    con.close()

if __name__ == "__main__":
    import sys
    import getopt
    try:
        import psyco
        psyco_imported = 1
    except:
        psyco_imported = 0

    usage = """usage: %s [-v] [-p] [-m] [-c] [-q] [-i] [-z complevel] [-l complib] [-R range] [-n nrows] file
            -v verbose
            -p use "psyco" if available
            -m use random values to fill the table
            -q do a query (both indexed and non-indexed version)
            -i do a query (exclude non-indexed version)
            -c create the database
            -z compress with zlib (no compression by default)
            -l use complib for compression (zlib used by default)
            -R select a range in a field in the form "start,stop" (def "0,10")
            -n sets the number of rows (in krows) in each table
            \n""" % sys.argv[0]

    try:
        opts, pargs = getopt.getopt(sys.argv[1:], 'vpmcqiz:l:R:n:')
    except:
        sys.stderr.write(usage)
        sys.exit(0)

    # default options
    verbose = 0
    usepsyco = 0
    userandom = 0
    docreate = 0
    docompress = 0
    complib = "zlib"
    doquery = 0
    doqueryidx = 0
    rng = [0, 10]
    nrows = 1

    # Get the options
    for option in opts:
        if option[0] == '-v':
            verbose = 1
        elif option[0] == '-p':
            usepsyco = 1
        elif option[0] == '-m':
            userandom = 1
        elif option[0] == '-c':
            docreate = 1
            createindex = 1
        elif option[0] == '-q':
            doquery = 1
        elif option[0] == '-i':
            doqueryidx = 1
        elif option[0] == '-z':
            docompress = int(option[1])
        elif option[0] == '-l':
            complib = option[1]
        elif option[0] == '-R':
            rng = [int(i) for i in option[1].split(",")]
        elif option[0] == '-n':
            nrows = int(option[1])

    # Catch the hdf5 file passed as the last argument
    filename = pargs[0]

    # The filters chosen
    filters = tb.Filters(complevel=docompress, complib=complib)

    if verbose:
        print("pytables version:", tb.__version__)
        if userandom:
            print("using random values")
        if doqueryidx:
            print("doing indexed queries only")

    if docreate:
        if verbose:
            print("writing %s krows" % nrows)
        if psyco_imported and usepsyco:
            psyco.bind(create_db)
        nrows *= 1000
        create_db(filename, nrows)

    if doquery:
        query_db(filename, rng)
