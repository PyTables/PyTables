from time import time
#import subprocess  # requires Python 2.4
import popen2
import random
import numarray
from numarray import random_array

# in order to always generate the same random sequence
random_array.seed(19, 20)

# Constants
STEP = 1000*100  # the size of the buffer to fill the table, in rows
SCALE = 0.1      # standard deviation of the noise compared with actual values
NI_NTIMES = 2      # The number of queries for doing a mean (non-idx cols)
I_NTIMES = 10      # The number of queries for doing a mean (idx cols)
READ_TIMES = 100    # The number of complete calls to DB.query_db()
MROW = 1000*1000.

# global variables
reg_cols = ['col1','col3']
idx_cols = ['col2','col4']
rdm_cod = ['lin', 'rnd']


def get_nrows(nrows_str):
    if nrows_str.endswith("k"):
        return int(float(nrows_str[:-1])*1000)
    elif nrows_str.endswith("m"):
        return int(float(nrows_str[:-1])*1000*1000)
    elif nrows_str.endswith("g"):
        return int(float(nrows_str[:-1])*1000*1000*1000)
    else:
        raise ValueError, "value of nrows must end with either 'k', 'm' or 'g' suffixes."

class DB(object):

    def __init__(self, nrows, rng, userandom):
        global step, scale
        self.step = STEP
        self.scale = SCALE
        self.rng = rng
        self.userandom = userandom
        self.filename = '-'.join([rdm_cod[userandom], nrows])
        self.nrows = get_nrows(nrows)

    def get_db_size(self):
#         sout = subprocess.Popen("du -s %s" % self.filename, shell=True,
#                                 stdout=subprocess.PIPE).stdout
        (sout, sin) = popen2.popen2("sync;du -s %s" % self.filename)
        line = [l for l in sout][0]
        return int(line.split()[0])

    def print_mtime(self, t1, explain):
        mtime = time()-t1
        print "%s:" % explain, round(mtime, 5)
        print "Krows/s:", round((self.nrows/1000.)/mtime, 5)

    def print_qtime(self, colname, ltimes):
        ntimes = len(ltimes)
        qtime1 = ltimes[0] # First measured time
        if colname in idx_cols and ntimes > 5:
            # if indexed, wait until the 5th iteration (in order to
            # insure that the index is effectively cached) to take times
            qtime2 = sum(ltimes[5:])/(ntimes-5)
        else:
            qtime2 = ltimes[-1]  # Last measured time
        print "Query time for %s:" % colname, round(qtime1, 5)
        print "Mrows/s:", round((self.nrows/(MROW))/qtime1, 5)
        if colname in idx_cols:
            if ntimes > 5:
                print "Query time for %s (cached):" % colname, round(qtime2, 5)
                print "Mrows/s (cached):", round((self.nrows/(MROW))/qtime2, 5)
            else:
                print "Not enough iterations to compute cache times."
        else:
            print "Query time for %s (cached):" % colname, round(qtime2, 5)
            print "Mrows/s (cached):", round((self.nrows/(MROW))/qtime2, 5)

    def print_db_sizes(self, init, filled, indexed):
        table_size = (filled-init)/1024.
        indexes_size = (indexed-filled)/1024.
        print "Table size (MB):", round(table_size, 3)
        print "Indexes size (MB):", round(indexes_size, 3)
        print "Full size (MB):", round(table_size+indexes_size, 3)

    def fill_arrays(self, start, stop):
        arr_f8 = numarray.arange(start, stop, type=numarray.Float64)
        arr_i4 = numarray.arange(start, stop, type=numarray.Int32)
        if self.userandom:
            arr_f8 += random_array.normal(0, stop*self.scale,
                                          shape=[stop-start])
            arr_i4 = numarray.array(arr_f8, type=numarray.Int32)
        return arr_i4, arr_f8

    def create_db(self, dtype, optlevel, verbose):
        con = self.open_db(remove=1)
        self.create_table(con)
        init_size = self.get_db_size()
        t1=time()
        self.fill_table(con)
        table_size = self.get_db_size()
        self.print_mtime(t1, 'Insert time')
        self.index_db(con, dtype, optlevel, verbose)
        indexes_size = self.get_db_size()
        self.print_db_sizes(init_size, table_size, indexes_size)
#         if optlevel > 0:
#             self.optimize_index(con, optlevel, verbose)
        self.close_db(con)

    def index_db(self, con, dtype, optlevel, verbose):
        if dtype == "int":
            idx_cols = ['col2']
        elif dtype == "float":
            idx_cols = ['col4']
        else:
            idx_cols = ['col2', 'col4']
        for colname in idx_cols:
            t1=time()
            self.index_col(con, colname, optlevel, verbose)
            self.print_mtime(t1, 'Index time (%s)' % colname)

#     def optimize_index(self, con, level, verbose):   # Only for PyTables Pro
#         for colname in idx_cols:
#             t1=time()
#             self.optimizeIndex(con, colname, level=level, verbose=verbose)
#             self.print_mtime(t1, 'Optimize time (%s)' % colname)

    def query_db(self, dtype, onlyidxquery, onlynonidxquery, avoidfscache, verbose):
        if dtype == "int":
            reg_cols = ['col1']
            idx_cols = ['col2']
        elif dtype == "float":
            reg_cols = ['col3']
            idx_cols = ['col4']
        else:
            reg_cols = ['col1', 'col3']
            idx_cols = ['col2', 'col4']
        con = self.open_db()
        if avoidfscache:
            rseed = random.random()
        else:
            rseed = 19
        # Query for non-indexed columns
        if not onlyidxquery:
            for colname in reg_cols:
                ltimes = []
                random.seed(rseed)
                t1=time()
                for i in range(NI_NTIMES):
                    results = self.do_query(con, colname,
                                            #base)
                                            random.randrange(self.nrows))
                ltimes.append((time()-t1)/NI_NTIMES)
                #results.sort()
                if verbose:
                    print results
                self.print_qtime(colname, ltimes)
        # Query for indexed columns
        if not onlynonidxquery:
            for colname in idx_cols:
                ltimes = []
                for j in range(READ_TIMES):
                    random.seed(rseed)
                    t1=time()
                    for i in range(I_NTIMES):
                        results = self.do_query(con, colname,
                                                #base)
                                                random.randrange(self.nrows))
                    ltimes.append((time()-t1)/I_NTIMES)
                #results.sort()
                if verbose:
                    print results
                self.print_qtime(colname, ltimes)
        self.close_db(con)

    def close_db(self, con):
        con.close()

if __name__=="__main__":
    import sys
    import getopt

    try:
        import psyco
        psyco_imported = 1
    except:
        psyco_imported = 0

    usage = """usage: %s [-T] [-S] [-P] [-v] [-f] [-p] [-m] [-c] [-q] [-i] [-I] [-x] [-z complevel] [-l complib] [-R range] [-n nrows] [-d datadir] [-O level] [-s] col
            -T use Pytables
            -S use Sqlite3
            -P use Postgres
            -v verbose
            -f do a profile of the run (only query functionality & Python 2.4)
            -p use "psyco" if available
            -m use random values to fill the table
            -q do a query (both indexed and non-indexed versions)
            -i do a query (just indexed versions)
            -I do a query (just non-indexed versions)
            -x choose a different seed for random numbers (i.e. avoid FS cache)
            -c create the database
            -z compress with zlib (no compression by default)
            -l use complib for compression (zlib used by default)
            -R select a range in a field in the form "start,stop" (def "0,10")
            -n sets the number of rows (in krows) in each table
            -d directory to save data (default: data.nobackup)
            -O set the optimization level for PyTables Pro indexes
            -s select a type column for operations ('int' or 'float'. def all)
            \n""" % sys.argv[0]

    try:
        opts, pargs = getopt.getopt(sys.argv[1:], 'TSPvfpmcqiIxz:l:R:n:d:O:s:')
    except:
        sys.stderr.write(usage)
        sys.exit(0)

    # default options
    usepytables = 0
    usesqlite3 = 0
    usepostgres = 0
    verbose = 0
    doprofile = 0
    usepsyco = 0
    userandom = 0
    docreate = 0
    optlevel = 0
    docompress = 0
    complib = "zlib"
    doquery = 0
    onlyidxquery = 0
    onlynonidxquery = 0
    avoidfscache = 0
    rng = [0,10]
    krows = '1k'
    dtype = "all"
    datadir = "data.nobackup"

    # Get the options
    for option in opts:
        if option[0] == '-T':
            usepytables = 1
        elif option[0] == '-S':
            usesqlite3 = 1
        elif option[0] == '-P':
            usepostgres = 1
        elif option[0] == '-v':
            verbose = 1
        elif option[0] == '-f':
            doprofile = 1
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
            doquery = 1
            onlyidxquery = 1
        elif option[0] == '-I':
            doquery = 1
            onlynonidxquery = 1
        elif option[0] == '-x':
            avoidfscache = 1
        elif option[0] == '-z':
            docompress = int(option[1])
        elif option[0] == '-l':
            complib = option[1]
        elif option[0] == '-R':
            rng = [int(i) for i in option[1].split(",")]
        elif option[0] == '-n':
            krows = option[1]
        elif option[0] == '-d':
            datadir = option[1]
        elif option[0] == '-O':
            optlevel = int(option[1])
        elif option[0] == '-s':
            if option[1] in ('int', 'float'):
                dtype = option[1]
            else:
                print "column should be either 'int' or 'float'"
                sys.exit(0)

    # If not database backend selected, abort
    if not usepytables and not usesqlite3 and not usepostgres:
        print "Please, select a backend:"
        print "PyTables: -T"
        print "Sqlite3:  -S"
        print "Postgres: -P"
        sys.exit(0)

    # Create the class for the database
    if usepytables:
        from pytables_backend import PyTables_DB
        db = PyTables_DB(krows, rng, userandom, datadir,
                         docompress, complib, optlevel)
    elif usesqlite3:
        from sqlite3_backend import Sqlite3_DB
        db = Sqlite3_DB(krows, rng, userandom, datadir)
    elif usepostgres:
        from postgres_backend import Postgres_DB
        db = Postgres_DB(krows, rng, userandom)

    if verbose:
        if userandom:
            print "using random values"
        if onlyidxquery:
            print "doing indexed queries only"

    if psyco_imported and usepsyco:
        psyco.bind(db.create_db)
        psyco.bind(db.query_db)

    if docreate:
        if verbose:
            print "writing %s rows" % krows
        db.create_db(dtype, optlevel, verbose)

    if doquery:
        print "Calling query_db() %s times" % READ_TIMES
        if doprofile:
            import pstats
            import profile as prof
            prof.run('db.query_db(dtype, onlyidxquery, avoidfscache, verbose)',
                     'query_db.prof')
            stats = pstats.Stats('query_db.prof')
            stats.strip_dirs()
            stats.sort_stats('time', 'calls')
            if verbose:
                stats.print_stats()
            else:
                stats.print_stats(20)
        else:
            db.query_db(dtype, onlyidxquery, onlynonidxquery, avoidfscache, verbose)
