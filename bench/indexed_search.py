from time import time
import subprocess  # requires Python 2.4
import popen2
import random
import numarray
from numarray import random_array
import numpy

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
        sout = subprocess.Popen("sync;du -s %s" % self.filename, shell=True,
                                stdout=subprocess.PIPE).stdout
#         (sout, sin) = popen2.popen2("sync;du -s %s" % self.filename)
        line = [l for l in sout][0]
        return int(line.split()[0])

    def print_mtime(self, t1, explain):
        mtime = time()-t1
        print "%s:" % explain, round(mtime, 6)
        print "Krows/s:", round((self.nrows/1000.)/mtime, 6)

    def print_qtime(self, colname, ltimes):
        ntimes = len(ltimes)
        qtime1 = ltimes[0] # First measured time
        if colname in idx_cols and ntimes > 5:
            # if indexed, wait until the 5th iteration (in order to
            # insure that the index is effectively cached) to take times
            qtime2 = sum(ltimes[5:])/(ntimes-5)
        else:
            qtime2 = ltimes[-1]  # Last measured time
        print "Query time for %s:" % colname, round(qtime1, 6)
        print "Mrows/s:", round((self.nrows/(MROW))/qtime1, 6)
        if colname in idx_cols:
            if ntimes > 5:
                print "Query time for %s (cached):" % colname, round(qtime2, 6)
                print "Mrows/s (cached):", round((self.nrows/(MROW))/qtime2, 6)
            else:
                print "Not enough iterations to compute cache times."
        else:
            print "Query time for %s (cached):" % colname, round(qtime2, 6)
            print "Mrows/s (cached):", round((self.nrows/(MROW))/qtime2, 6)

    def print_db_sizes(self, init, filled, indexed):
        table_size = (filled-init)/1024.
        indexes_size = (indexed-filled)/1024.
        print "Table size (MB):", round(table_size, 3)
        print "Indexes size (MB):", round(indexes_size, 3)
        print "Full size (MB):", round(table_size+indexes_size, 3)

    def fill_arrays_na(self, start, stop):
        arr_f8 = numarray.arange(start, stop, type=numarray.Float64)
        arr_i4 = numarray.arange(start, stop, type=numarray.Int32)
        if self.userandom:
            arr_f8 += random_array.normal(0, stop*self.scale,
                                          shape=[stop-start])
            arr_i4 = numarray.array(arr_f8, type=numarray.Int32)
        return arr_i4, arr_f8

    def fill_arrays(self, start, stop):
        arr_f8 = numpy.arange(start, stop, dtype='float64')
        arr_i4 = numpy.arange(start, stop, dtype='int32')
        if self.userandom:
            arr_f8 += numpy.random.normal(0, stop*self.scale,
                                          size=stop-start)
            arr_i4 = numpy.array(arr_f8, dtype='int32')
        return arr_i4, arr_f8

    def create_db(self, dtype, optlevel, verbose):
        self.con = self.open_db(remove=1)
        self.create_table(self.con)
        init_size = self.get_db_size()
        t1=time()
        self.fill_table(self.con)
        table_size = self.get_db_size()
        self.print_mtime(t1, 'Insert time')
        self.index_db(dtype, optlevel, verbose)
        indexes_size = self.get_db_size()
        self.print_db_sizes(init_size, table_size, indexes_size)
        self.close_db(self.con)

    def index_db(self, dtype, optlevel, verbose):
        if dtype == "int":
            idx_cols = ['col2']
        elif dtype == "float":
            idx_cols = ['col4']
        else:
            idx_cols = ['col2', 'col4']
        for colname in idx_cols:
            t1=time()
            self.index_col(self.con, colname, optlevel, verbose)
            self.print_mtime(t1, 'Index time (%s)' % colname)

    def query_db(self, niter, dtype, onlyidxquery, onlynonidxquery,
                 avoidfscache, verbose):
        self.con = self.open_db()
        if dtype == "int":
            reg_cols = ['col1']
            idx_cols = ['col2']
        elif dtype == "float":
            reg_cols = ['col3']
            idx_cols = ['col4']
        else:
            reg_cols = ['col1', 'col3']
            idx_cols = ['col2', 'col4']
        if avoidfscache:
            rseed = random.random()
        else:
            rseed = 19
        # Query for non-indexed columns
        random.seed(rseed)
        base = random.randrange(self.nrows)
        if not onlyidxquery:
            for colname in reg_cols:
                ltimes = []
                random.seed(rseed)
                t1=time()
                for i in range(NI_NTIMES):
                    results = self.do_query(self.con, colname,
                                            #base)
                                            random.randrange(self.nrows))
                ltimes.append((time()-t1)/NI_NTIMES)
                #results.sort()
                if verbose:
                    print "Results len:", results
                self.print_qtime(colname, ltimes)
        # Query for indexed columns
        if not onlynonidxquery:
            for colname in idx_cols:
                ltimes = []
                for j in xrange(niter):
                    random.seed(rseed)
                    t1=time()
                    for i in range(I_NTIMES):
                        results = self.do_query(self.con, colname,
                                                base)
                                                #random.randrange(self.nrows))
                    ltimes.append((time()-t1)/I_NTIMES)
                #results.sort()
                if verbose:
                    print "Results len:", results
                self.print_qtime(colname, ltimes)
        if hasattr(self, "table_cache"):
            del self.table_cache
        self.close_db(self.con)

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

    usage = """usage: %s [-T] [-S] [-P] [-v] [-f] [-k] [-p] [-m] [-c] [-q] [-i] [-I] [-x] [-z complevel] [-l complib] [-R range] [-N niter] [-n nrows] [-d datadir] [-O level] [-s] col -Q [suplim]
            -T use Pytables
            -S use Sqlite3
            -P use Postgres
            -v verbose
            -f do a profile of the run (only query functionality & Python 2.5)
            -k do a profile for kcachegrind use (out file is 'indexed_search.kcg')
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
            -N number of iterations for reading
            -n sets the number of rows (in krows) in each table
            -d directory to save data (default: data.nobackup)
            -O set the optimization level for PyTables Pro indexes
            -s select a type column for operations ('int' or 'float'. def all)
            -Q do a repeteated query up to 10**value
            \n""" % sys.argv[0]

    try:
        opts, pargs = getopt.getopt(sys.argv[1:], 'TSPvfkpmcqiIxz:l:R:N:n:d:O:s:Q:')
    except:
        sys.stderr.write(usage)
        sys.exit(0)

    # default options
    usepytables = 0
    usesqlite3 = 0
    usepostgres = 0
    verbose = 0
    doprofile = 0
    dokprofile = 0
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
    rng = [-10, 10]
    repeatquery = 0
    repeatvalue = 0
    krows = '1k'
    niter = READ_TIMES
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
        elif option[0] == '-k':
            dokprofile = 1
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
            onlyidxquery = 1
        elif option[0] == '-I':
            onlynonidxquery = 1
        elif option[0] == '-x':
            avoidfscache = 1
        elif option[0] == '-z':
            docompress = int(option[1])
        elif option[0] == '-l':
            complib = option[1]
        elif option[0] == '-R':
            rng = [int(i) for i in option[1].split(",")]
        elif option[0] == '-N':
            niter = int(option[1])
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
        elif option[0] == '-Q':
            repeatquery = 1
            repeatvalue = int(option[1])

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
        print "Calling query_db() %s times" % niter
        if doprofile:
            import pstats
            import cProfile as prof
            prof.run('db.query_db(niter, dtype, onlyidxquery, onlynonidxquery, avoidfscache, verbose)', 'indexed_search.prof')
            stats = pstats.Stats('indexed_search.prof')
            stats.strip_dirs()
            stats.sort_stats('time', 'calls')
            if verbose:
                stats.print_stats()
            else:
                stats.print_stats(20)
        elif dokprofile:
            from cProfile import Profile
            import lsprofcalltree
            prof = Profile()
            prof.run('db.query_db(niter, dtype, onlyidxquery, onlynonidxquery, avoidfscache, verbose)')
            kcg = lsprofcalltree.KCacheGrind(prof)
            ofile = open('indexed_search.kcg','w')
            kcg.output(ofile)
            ofile.close()
        elif doprofile:
            import hotshot, hotshot.stats
            prof = hotshot.Profile("indexed_search.prof")
            benchtime, stones = prof.run('db.query_db(niter, dtype, onlyidxquery, onlynonidxquery, avoidfscache, verbose)')
            prof.close()
            stats = hotshot.stats.load("indexed_search.prof")
            stats.strip_dirs()
            stats.sort_stats('time', 'calls')
            stats.print_stats(20)
        else:
            db.query_db(niter, dtype, onlyidxquery, onlynonidxquery,
                        avoidfscache, verbose)

    if repeatquery:
        # Start by a range which is almost None
        db.rng = [1, 1]
        if verbose:
            print "range:", db.rng
        db.query_db(niter, dtype, onlyidxquery, onlynonidxquery,
                    avoidfscache, verbose)
        for i in xrange(repeatvalue):
            rng = 10**i
            db.rng = [-rng/2, rng/2]
            if verbose:
                print "range:", db.rng
            db.query_db(niter, dtype, onlyidxquery, onlynonidxquery,
                        avoidfscache, verbose)

