from time import perf_counter as clock
import numpy as np
import random

DSN = "dbname=test port = 5435"

# in order to always generate the same random sequence
random.seed(19)


def flatten(l):
    """Flattens list of tuples l."""
    return [x[0] for x in l]


def fill_arrays(start, stop):
    col_i = np.arange(start, stop, type=np.int32)
    if userandom:
        col_j = np.random.uniform(0, nrows, size=[stop - start])
    else:
        col_j = np.array(col_i, type=np.float64)
    return col_i, col_j

# Generator for ensure pytables benchmark compatibility


def int_generator(nrows):
    step = 1000 * 100
    j = 0
    for i in range(nrows):
        if i >= step * j:
            stop = (j + 1) * step
            if stop > nrows:  # Seems unnecessary
                stop = nrows
            col_i, col_j = fill_arrays(i, stop)
            j += 1
            k = 0
        yield (col_i[k], col_j[k])
        k += 1


def int_generator_slow(nrows):
    for i in range(nrows):
        if userandom:
            yield (i, float(random.randint(0, nrows)))
        else:
            yield (i, float(i))


class Stream32:

    "Object simulating a file for reading"

    def __init__(self):
        self.n = None
        self.read_it = self.read_iter()

    # No va! Hi ha que convertir a un de normal!
    def readline(self, n=None):
        for tup in int_generator(nrows):
            sout = "%s\t%s\n" % tup
            if n is not None and len(sout) > n:
                for i in range(0, len(sout), n):
                    yield sout[i:i + n]
            else:
                yield sout

    def read_iter(self):
        sout = ""
        n = self.n
        for tup in int_generator(nrows):
            sout += "%s\t%s\n" % tup
            if n is not None and len(sout) > n:
                for i in range(n, len(sout), n):
                    rout = sout[:n]
                    sout = sout[n:]
                    yield rout
        yield sout

    def read(self, n=None):
        self.n = n
        try:
            str = next(self.read_it)
        except StopIteration:
            str = ""
        return str


def open_db(filename, remove=0):
    if not filename:
        con = sqlite.connect(DSN)
    else:
        con = sqlite.connect(filename)
    cur = con.cursor()
    return con, cur


def create_db(filename, nrows):
    con, cur = open_db(filename, remove=1)
    try:
        cur.execute("create table ints(i integer, j double precision)")
    except:
        con.rollback()
        cur.execute("DROP TABLE ints")
        cur.execute("create table ints(i integer, j double precision)")
    con.commit()
    con.set_isolation_level(2)
    t1 = clock()
    st = Stream32()
    cur.copy_from(st, "ints")
    # In case of postgres, the speeds of generator and loop are similar
    #cur.executemany("insert into ints values (%s,%s)", int_generator(nrows))
#     for i in xrange(nrows):
#         cur.execute("insert into ints values (%s,%s)", (i, float(i)))
    con.commit()
    ctime = clock() - t1
    if verbose:
        print(f"insert time: {ctime:.5f}")
        print(f"Krows/s: {nrows / 1000 / ctime:.5f}")
    close_db(con, cur)


def index_db(filename):
    con, cur = open_db(filename)
    t1 = clock()
    cur.execute("create index ij on ints(j)")
    con.commit()
    itime = clock() - t1
    if verbose:
        print(f"index time: {itime:.5f}")
        print(f"Krows/s: {nrows / itime:.5f}")
    # Close the DB
    close_db(con, cur)


def query_db(filename, rng):
    con, cur = open_db(filename)
    t1 = clock()
    ntimes = 10
    for i in range(ntimes):
        # between clause does not seem to take advantage of indexes
        # cur.execute("select j from ints where j between %s and %s" % \
        cur.execute("select i from ints where j >= %s and j <= %s" %
                    # cur.execute("select i from ints where i >= %s and i <=
                    # %s" %
                    (rng[0] + i, rng[1] + i))
        results = cur.fetchall()
    con.commit()
    qtime = (clock() - t1) / ntimes
    if verbose:
        print(f"query time: {qtime:.5f}")
        print(f"Mrows/s: {nrows / 1000 / qtime:.5f}")
        results = sorted(flatten(results))
        print(results)
    close_db(con, cur)


def close_db(con, cur):
    cur.close()
    con.close()

if __name__ == "__main__":
    import sys
    import getopt
    try:
        import psyco
        psyco_imported = 1
    except:
        psyco_imported = 0

    usage = """usage: %s [-v] [-p] [-m] [-i] [-q] [-c] [-R range] [-n nrows] file
            -v verbose
            -p use "psyco" if available
            -m use random values to fill the table
            -q do query
            -c create the database
            -i index the table
            -2 use sqlite2 (default is use sqlite3)
            -R select a range in a field in the form "start,stop" (def "0,10")
            -n sets the number of rows (in krows) in each table
            \n""" % sys.argv[0]

    try:
        opts, pargs = getopt.getopt(sys.argv[1:], 'vpmiqc2R:n:')
    except:
        sys.stderr.write(usage)
        sys.exit(0)

    # default options
    verbose = 0
    usepsyco = 0
    userandom = 0
    docreate = 0
    createindex = 0
    doquery = 0
    sqlite_version = "3"
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
        elif option[0] == '-i':
            createindex = 1
        elif option[0] == '-q':
            doquery = 1
        elif option[0] == '-c':
            docreate = 1
        elif option[0] == "-2":
            sqlite_version = "2"
        elif option[0] == '-R':
            rng = [int(i) for i in option[1].split(",")]
        elif option[0] == '-n':
            nrows = int(option[1])

    # Catch the hdf5 file passed as the last argument
    filename = pargs[0]

#     if sqlite_version == "2":
#         import sqlite
#     else:
#         from pysqlite2 import dbapi2 as sqlite
    import psycopg2 as sqlite

    if verbose:
        # print "pysqlite version:", sqlite.version
        if userandom:
            print("using random values")

    if docreate:
        if verbose:
            print("writing %s krows" % nrows)
        if psyco_imported and usepsyco:
            psyco.bind(create_db)
        nrows *= 1000
        create_db(filename, nrows)

    if createindex:
        index_db(filename)

    if doquery:
        query_db(filename, rng)
