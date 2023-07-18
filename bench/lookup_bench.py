"""Benchmark to help choosing the best chunksize so as to optimize the access
time in random lookups."""

import subprocess
from pathlib import Path
from time import perf_counter as clock

import numpy as np
import tables as tb

# Constants
NOISE = 1e-15    # standard deviation of the noise compared with actual values

rdm_cod = ['lin', 'rnd']


def get_nrows(nrows_str):
    powers = {'k': 3, 'm': 6, 'g': 9}
    try:
        return int(float(nrows_str[:-1]) * 10 ** powers[nrows_str[-1]])
    except KeyError:
        raise ValueError(
            "value of nrows must end with either 'k', 'm' or 'g' suffixes.")


class DB:

    def __init__(self, nrows, dtype, chunksize, userandom, datadir,
                 docompress=0, complib='zlib'):
        self.dtype = dtype
        self.docompress = docompress
        self.complib = complib
        self.filename = '-'.join([rdm_cod[userandom],
                                  "n" + nrows, "s" + chunksize, dtype])
        # Complete the filename
        self.filename = "lookup-" + self.filename
        if docompress:
            self.filename += '-' + complib + str(docompress)
        self.filename = datadir + '/' + self.filename + '.h5'
        print("Processing database:", self.filename)
        self.userandom = userandom
        self.nrows = get_nrows(nrows)
        self.chunksize = get_nrows(chunksize)
        self.step = self.chunksize
        self.scale = NOISE

    def get_db_size(self):
        sout = subprocess.Popen("sync;du -s %s" % self.filename, shell=True,
                                stdout=subprocess.PIPE).stdout
        line = [l for l in sout][0]
        return int(line.split()[0])

    def print_mtime(self, t1, explain):
        mtime = clock() - t1
        print(f"{explain}: {mtime:.6f}")
        print(f"Krows/s: {self.nrows / 1000 / mtime:.6f}")

    def print_db_sizes(self, init, filled):
        array_size = (filled - init) / 1024
        print(f"Array size (MB): {array_size:.3f}")

    def open_db(self, remove=0):
        if remove and Path(self.filename).is_file():
            Path(self.filename).unlink()
        con = tb.open_file(self.filename, 'a')
        return con

    def create_db(self, verbose):
        self.con = self.open_db(remove=1)
        self.create_array()
        init_size = self.get_db_size()
        t1 = clock()
        self.fill_array()
        array_size = self.get_db_size()
        self.print_mtime(t1, 'Insert time')
        self.print_db_sizes(init_size, array_size)
        self.close_db()

    def create_array(self):
        # The filters chosen
        filters = tb.Filters(complevel=self.docompress,
                             complib=self.complib)
        atom = tb.Atom.from_kind(self.dtype)
        self.con.create_earray(self.con.root, 'earray', atom, (0,),
                               filters=filters,
                               expectedrows=self.nrows,
                               chunkshape=(self.chunksize,))

    def fill_array(self):
        "Fills the array"
        earray = self.con.root.earray
        j = 0
        arr = self.get_array(0, self.step)
        for i in range(0, self.nrows, self.step):
            stop = (j + 1) * self.step
            if stop > self.nrows:
                stop = self.nrows
            ###arr = self.get_array(i, stop, dtype)
            earray.append(arr)
            j += 1
        earray.flush()

    def get_array(self, start, stop):
        arr = np.arange(start, stop, dtype='float')
        if self.userandom:
            arr += np.random.normal(0, stop * self.scale, size=stop - start)
        arr = arr.astype(self.dtype)
        return arr

    def print_qtime(self, ltimes):
        ltimes = np.array(ltimes)
        print("Raw query times:\n", ltimes)
        print("Histogram times:\n", np.histogram(ltimes[1:]))
        ntimes = len(ltimes)
        qtime1 = ltimes[0]  # First measured time
        if ntimes > 5:
            # Wait until the 5th iteration (in order to
            # ensure that the index is effectively cached) to take times
            qtime2 = sum(ltimes[5:]) / (ntimes - 5)
        else:
            qtime2 = ltimes[-1]  # Last measured time
        print(f"1st query time: {qtime1:.3f}")
        print(f"Mean (skipping the first 5 meas.): {qtime2:.3f}")

    def query_db(self, niter, avoidfscache, verbose):
        self.con = self.open_db()
        earray = self.con.root.earray
        if avoidfscache:
            rseed = int(np.random.randint(self.nrows))
        else:
            rseed = 19
        np.random.seed(rseed)
        np.random.randint(self.nrows)
        ltimes = []
        for i in range(niter):
            t1 = clock()
            self.do_query(earray, np.random.randint(self.nrows))
            ltimes.append(clock() - t1)
        self.print_qtime(ltimes)
        self.close_db()

    def do_query(self, earray, idx):
        return earray[idx]

    def close_db(self):
        self.con.close()


if __name__ == "__main__":
    import sys
    import getopt

    usage = """usage: %s [-v] [-m] [-c] [-q] [-x] [-z complevel] [-l complib] [-N niter] [-n nrows] [-d datadir] [-t] type [-s] chunksize
            -v verbose
            -m use random values to fill the array
            -q do a (random) lookup
            -x choose a different seed for random numbers (i.e. avoid FS cache)
            -c create the file
            -z compress with zlib (no compression by default)
            -l use complib for compression (zlib used by default)
            -N number of iterations for reading
            -n sets the number of rows in the array
            -d directory to save data (default: data.nobackup)
            -t select the type for array ('int' or 'float'. def 'float')
            -s select the chunksize for array
            \n""" % sys.argv[0]

    try:
        opts, pargs = getopt.getopt(sys.argv[1:], 'vmcqxz:l:N:n:d:t:s:')
    except:
        sys.stderr.write(usage)
        sys.exit(0)

    # default options
    verbose = 0
    userandom = 0
    docreate = 0
    optlevel = 0
    docompress = 0
    complib = "zlib"
    doquery = False
    avoidfscache = 0
    krows = '1k'
    chunksize = '32k'
    niter = 50
    datadir = "data.nobackup"
    dtype = "float"

    # Get the options
    for option in opts:
        if option[0] == '-v':
            verbose = 1
        elif option[0] == '-m':
            userandom = 1
        elif option[0] == '-c':
            docreate = 1
            createindex = 1
        elif option[0] == '-q':
            doquery = True
        elif option[0] == '-x':
            avoidfscache = 1
        elif option[0] == '-z':
            docompress = int(option[1])
        elif option[0] == '-l':
            complib = option[1]
        elif option[0] == '-N':
            niter = int(option[1])
        elif option[0] == '-n':
            krows = option[1]
        elif option[0] == '-d':
            datadir = option[1]
        elif option[0] == '-t':
            if option[1] in ('int', 'float'):
                dtype = option[1]
            else:
                print("type should be either 'int' or 'float'")
                sys.exit(0)
        elif option[0] == '-s':
            chunksize = option[1]

    if not avoidfscache:
        # in order to always generate the same random sequence
        np.random.seed(20)

    if verbose:
        if userandom:
            print("using random values")

    db = DB(krows, dtype, chunksize, userandom, datadir, docompress, complib)

    if docreate:
        if verbose:
            print("writing %s rows" % krows)
        db.create_db(verbose)

    if doquery:
        print("Calling query_db() %s times" % niter)
        db.query_db(niter, avoidfscache, verbose)
