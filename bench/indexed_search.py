import random
import subprocess
from pathlib import Path
from time import perf_counter as clock

import numpy as np

# Constants

STEP = 1000 * 100   # the size of the buffer to fill the table, in rows
SCALE = 0.1         # standard deviation of the noise compared with actual
                    # values
NI_NTIMES = 1       # The number of queries for doing a mean (non-idx cols)
# COLDCACHE = 10   # The number of reads where the cache is considered 'cold'
# WARMCACHE = 50   # The number of reads until the cache is considered 'warmed'
# READ_TIMES = WARMCACHE+50    # The number of complete calls to DB.query_db()
# COLDCACHE = 50   # The number of reads where the cache is considered 'cold'
# WARMCACHE = 50   # The number of reads until the cache is considered 'warmed'
# READ_TIMES = WARMCACHE+50    # The number of complete calls to DB.query_db()
MROW = 1000 * 1000

# Test values
COLDCACHE = 5   # The number of reads where the cache is considered 'cold'
WARMCACHE = 5   # The number of reads until the cache is considered 'warmed'
READ_TIMES = 10    # The number of complete calls to DB.query_db()

# global variables
rdm_cod = ['lin', 'rnd']
prec = 6  # precision for printing floats purposes


def get_nrows(nrows_str):
    powers = {'k': 3, 'm': 6, 'g': 9}
    try:
        return int(float(nrows_str[:-1]) * 10 ** powers[nrows_str[-1]])
    except KeyError:
        raise ValueError(
            "value of nrows must end with either 'k', 'm' or 'g' suffixes.")


class DB:

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
        line = [l for l in sout][0]
        return int(line.split()[0])

    def print_mtime(self, t1, explain):
        mtime = clock() - t1
        print(f"{explain}: {mtime:.6f}")
        print(f"Krows/s: {self.nrows / 1000 / mtime:.6f}")

    def print_qtime(self, colname, ltimes):
        qtime1 = ltimes[0]  # First measured time
        qtime2 = ltimes[-1]  # Last measured time
        print(f"Query time for {colname}: {qtime1:.6f}")
        print(f"Mrows/s: {self.nrows / MROW / qtime1:.6f}")
        print(f"Query time for {colname} (cached): {qtime2:.6f}")
        print(f"Mrows/s (cached): {self.nrows / MROW / qtime2:.6f}")

    def norm_times(self, ltimes):
        "Get the mean and stddev of ltimes, avoiding the extreme values."
        lmean = ltimes.mean()
        lstd = ltimes.std()
        ntimes = ltimes[ltimes < lmean + lstd]
        nmean = ntimes.mean()
        nstd = ntimes.std()
        return nmean, nstd

    def print_qtime_idx(self, colname, ltimes, repeated, verbose):
        if repeated:
            r = "[REP] "
        else:
            r = "[NOREP] "
        ltimes = np.array(ltimes)
        ntimes = len(ltimes)
        qtime1 = ltimes[0]  # First measured time
        ctimes = ltimes[1:COLDCACHE]
        cmean, cstd = self.norm_times(ctimes)
        wtimes = ltimes[WARMCACHE:]
        wmean, wstd = self.norm_times(wtimes)
        if verbose:
            print("Times for cold cache:\n", ctimes)
            # print "Times for warm cache:\n", wtimes
            hist1, hist2 = np.histogram(wtimes)
            print(f"Histogram for warm cache: {hist1}\n{hist2}")
        print(f"{r}1st query time for {colname}: {qtime1:.{prec}f}")
        print(f"{r}Query time for {colname} (cold cache): "
              f"{cmean:.{prec}f} +- {cstd:.{prec}f}")
        print(f"{r}Query time for {colname} (warm cache): "
              f"{wmean:.{prec}f} +- {wstd:.{prec}f}")

    def print_db_sizes(self, init, filled, indexed):
        table_size = (filled - init) / 1024
        indexes_size = (indexed - filled) / 1024
        print(f"Table size (MB): {table_size:.3f}")
        print(f"Indexes size (MB): {indexes_size:.3f}")
        print(f"Full size (MB): {table_size + indexes_size:.3f}")

    def fill_arrays(self, start, stop):
        arr_f8 = np.arange(start, stop, dtype='float64')
        arr_i4 = np.arange(start, stop, dtype='int32')
        if self.userandom:
            arr_f8 += np.random.normal(0, stop * self.scale, size=stop - start)
            arr_i4 = np.array(arr_f8, dtype='int32')
        return arr_i4, arr_f8

    def create_db(self, dtype, kind, optlevel, verbose):
        self.con = self.open_db(remove=1)
        self.create_table(self.con)
        init_size = self.get_db_size()
        t1 = clock()
        self.fill_table(self.con)
        table_size = self.get_db_size()
        self.print_mtime(t1, 'Insert time')
        self.index_db(dtype, kind, optlevel, verbose)
        indexes_size = self.get_db_size()
        self.print_db_sizes(init_size, table_size, indexes_size)
        self.close_db(self.con)

    def index_db(self, dtype, kind, optlevel, verbose):
        if dtype == "int":
            idx_cols = ['col2']
        elif dtype == "float":
            idx_cols = ['col4']
        else:
            idx_cols = ['col2', 'col4']
        for colname in idx_cols:
            t1 = clock()
            self.index_col(self.con, colname, kind, optlevel, verbose)
            self.print_mtime(t1, 'Index time (%s)' % colname)

    def query_db(self, niter, dtype, onlyidxquery, onlynonidxquery,
                 avoidfscache, verbose, inkernel):
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
            rseed = int(np.random.randint(self.nrows))
        else:
            rseed = 19
        # Query for non-indexed columns
        np.random.seed(rseed)
        base = np.random.randint(self.nrows)
        if not onlyidxquery:
            for colname in reg_cols:
                ltimes = []
                random.seed(rseed)
                for i in range(NI_NTIMES):
                    t1 = clock()
                    results = self.do_query(self.con, colname, base, inkernel)
                    ltimes.append(clock() - t1)
                if verbose:
                    print("Results len:", results)
                self.print_qtime(colname, ltimes)
            # Always reopen the file after *every* query loop.
            # Necessary to make the benchmark to run correctly.
            self.close_db(self.con)
            self.con = self.open_db()
        # Query for indexed columns
        if not onlynonidxquery:
            for colname in idx_cols:
                ltimes = []
                np.random.seed(rseed)
                rndbase = np.random.randint(self.nrows, size=niter)
                # First, non-repeated queries
                for i in range(niter):
                    base = rndbase[i]
                    t1 = clock()
                    results = self.do_query(self.con, colname, base, inkernel)
                    #results, tprof = self.do_query(
                    #    self.con, colname, base, inkernel)
                    ltimes.append(clock() - t1)
                if verbose:
                    print("Results len:", results)
                self.print_qtime_idx(colname, ltimes, False, verbose)
                # Always reopen the file after *every* query loop.
                # Necessary to make the benchmark to run correctly.
                self.close_db(self.con)
                self.con = self.open_db()
                ltimes = []
# Second, repeated queries
#                 for i in range(niter):
#                     t1=time()
#                     results = self.do_query(
#                         self.con, colname, base, inkernel)
# results, tprof = self.do_query(self.con, colname, base, inkernel)
#                     ltimes.append(time()-t1)
#                 if verbose:
#                     print "Results len:", results
#                 self.print_qtime_idx(colname, ltimes, True, verbose)
                # Print internal PyTables index tprof statistics
                #tprof = numpy.array(tprof)
                #tmean, tstd = self.norm_times(tprof)
                # print "tprof-->", round(tmean, prec), "+-", round(tstd, prec)
                # print "tprof hist-->", \
                #    numpy.histogram(tprof)
                # print "tprof raw-->", tprof
                # Always reopen the file after *every* query loop.
                # Necessary to make the benchmark to run correctly.
                self.close_db(self.con)
                self.con = self.open_db()
        # Finally, close the file.
        self.close_db(self.con)

    def close_db(self, con):
        con.close()


if __name__ == "__main__":
    import sys
    import getopt

    try:
        import psyco
        psyco_imported = 1
    except:
        psyco_imported = 0

    usage = """usage: %s [-T] [-P] [-v] [-f] [-k] [-p] [-m] [-c] [-q] [-i] [-I] [-S] [-x] [-z complevel] [-l complib] [-R range] [-N niter] [-n nrows] [-d datadir] [-O level] [-t kind] [-s] col -Q [suplim]
            -T use Pytables
            -P use Postgres
            -v verbose
            -f do a profile of the run (only query functionality)
            -k do a profile for kcachegrind use (out file is 'indexed_search.kcg')
            -p use "psyco" if available
            -m use random values to fill the table
            -q do a query (both indexed and non-indexed versions)
            -i do a query (just indexed one)
            -I do a query (just in-kernel one)
            -S do a query (just standard one)
            -x choose a different seed for random numbers (i.e. avoid FS cache)
            -c create the database
            -z compress with zlib (no compression by default)
            -l use complib for compression (zlib used by default)
            -R select a range in a field in the form "start,stop" (def "0,10")
            -N number of iterations for reading
            -n sets the number of rows (in krows) in each table
            -d directory to save data (default: data.nobackup)
            -O set the optimization level for PyTables indexes
            -t select the index type: "medium" (default) or "full", "light", "ultralight"
            -s select a type column for operations ('int' or 'float'. def all)
            -Q do a repeteated query up to 10**value
            \n""" % sys.argv[0]

    try:
        opts, pargs = getopt.getopt(
            sys.argv[1:], 'TPvfkpmcqiISxz:l:R:N:n:d:O:t:s:Q:')
    except:
        sys.stderr.write(usage)
        sys.exit(1)

    # default options
    usepytables = 0
    usepostgres = 0
    verbose = 0
    doprofile = 0
    dokprofile = 0
    usepsyco = 0
    userandom = 0
    docreate = 0
    optlevel = 0
    kind = "medium"
    docompress = 0
    complib = "zlib"
    doquery = False
    onlyidxquery = False
    onlynonidxquery = False
    inkernel = True
    avoidfscache = 0
    #rng = [-10, 10]
    rng = [-1000, -1000]
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
        elif option[0] == '-q':
            doquery = True
        elif option[0] == '-i':
            doquery = True
            onlyidxquery = True
        elif option[0] == '-I':
            doquery = True
            onlynonidxquery = True
        elif option[0] == '-S':
            doquery = True
            onlynonidxquery = True
            inkernel = False
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
        elif option[0] == '-t':
            if option[1] in ('full', 'medium', 'light', 'ultralight'):
                kind = option[1]
            else:
                print("kind should be either 'full', 'medium', 'light' or "
                      "'ultralight'")
                sys.exit(1)
        elif option[0] == '-s':
            if option[1] in ('int', 'float'):
                dtype = option[1]
            else:
                print("column should be either 'int' or 'float'")
                sys.exit(1)
        elif option[0] == '-Q':
            repeatquery = 1
            repeatvalue = int(option[1])

    # If not database backend selected, abort
    if not usepytables and not usepostgres:
        print("Please select a backend:")
        print("PyTables: -T")
        print("Postgres: -P")
        sys.exit(1)

    # Create the class for the database
    if usepytables:
        from pytables_backend import PyTables_DB
        db = PyTables_DB(krows, rng, userandom, datadir,
                         docompress, complib, kind, optlevel)
    elif usepostgres:
        from postgres_backend import Postgres_DB
        db = Postgres_DB(krows, rng, userandom)

    if not avoidfscache:
        # in order to always generate the same random sequence
        np.random.seed(20)

    if verbose:
        if userandom:
            print("using random values")
        if onlyidxquery:
            print("doing indexed queries only")

    if psyco_imported and usepsyco:
        psyco.bind(db.create_db)
        psyco.bind(db.query_db)

    if docreate:
        if verbose:
            print("writing %s rows" % krows)
        db.create_db(dtype, kind, optlevel, verbose)

    if doquery:
        print("Calling query_db() %s times" % niter)
        if doprofile:
            import pstats
            import cProfile as prof
            prof.run(
                'db.query_db(niter, dtype, onlyidxquery, onlynonidxquery, '
                'avoidfscache, verbose, inkernel)',
                'indexed_search.prof')
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
            prof.run(
                'db.query_db(niter, dtype, onlyidxquery, onlynonidxquery, '
                'avoidfscache, verbose, inkernel)')
            kcg = lsprofcalltree.KCacheGrind(prof)
            with Path('indexed_search.kcg').open('w') as ofile:
                kcg.output(ofile)
        elif doprofile:
            import hotshot
            import hotshot.stats
            prof = hotshot.Profile("indexed_search.prof")
            benchtime, stones = prof.run(
                'db.query_db(niter, dtype, onlyidxquery, onlynonidxquery, '
                'avoidfscache, verbose, inkernel)')
            prof.close()
            stats = hotshot.stats.load("indexed_search.prof")
            stats.strip_dirs()
            stats.sort_stats('time', 'calls')
            stats.print_stats(20)
        else:
            db.query_db(niter, dtype, onlyidxquery, onlynonidxquery,
                        avoidfscache, verbose, inkernel)

    if repeatquery:
        # Start by a range which is almost None
        db.rng = [1, 1]
        if verbose:
            print("range:", db.rng)
        db.query_db(niter, dtype, onlyidxquery, onlynonidxquery,
                    avoidfscache, verbose, inkernel)
        for i in range(repeatvalue):
            for j in (1, 2, 5):
                rng = j * 10 ** i
                db.rng = [-rng / 2, rng / 2]
                if verbose:
                    print("range:", db.rng)
#                 if usepostgres:
#                     os.system(
#                         "echo 1 > /proc/sys/vm/drop_caches;"
#                         " /etc/init.d/postgresql restart")
#                 else:
#                     os.system("echo 1 > /proc/sys/vm/drop_caches")
                db.query_db(niter, dtype, onlyidxquery, onlynonidxquery,
                            avoidfscache, verbose, inkernel)
