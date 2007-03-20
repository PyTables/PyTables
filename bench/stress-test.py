import sys, time, random, gc, types
import numarray
from tables import Group, metaIsDescription
from tables import *

class Test(IsDescription):
    ngroup = Int32Col(pos=1)
    ntable = Int32Col(pos=2)
    nrow = Int32Col(pos=3)
    #string = StringCol(itemsize=500, pos=4)

TestDict = {
    "ngroup": Int32Col(pos=1),
    "ntable": Int32Col(pos=2),
    "nrow": Int32Col(pos=3),
    }

def createFileArr(filename, ngroups, ntables, nrows):

    # First, create the groups

    # Open a file in "w"rite mode
    fileh = openFile(filename, mode="w", title="PyTables Stress Test")

    for k in range(ngroups):
        # Create the group
        group = fileh.createGroup("/", 'group%04d'% k, "Group %d" % k)

    fileh.close()

    # Now, create the arrays
    rowswritten = 0
    arr = numarray.arange(nrows)
    for k in range(ngroups):
        fileh = openFile(filename, mode="a", rootUEP='group%04d'% k)
        # Get the group
        group = fileh.root
        for j in range(ntables):
            # Create the array
            group = fileh.createArray("/", 'array%04d'% j,
                                      arr, "Array %d" % j)
        fileh.close()

    return (ngroups*ntables*nrows, 4)

def readFileArr(filename, ngroups, recsize, verbose):

    rowsread = 0
    for ngroup in range(ngroups):
        fileh = openFile(filename, mode="r", rootUEP='group%04d'% ngroup)
        # Get the group
        group = fileh.root
        narrai = 0
        if verbose:
            print "Group ==>", group
        for arrai in fileh.listNodes(group, 'Array'):
            if verbose > 1:
                print "Array ==>", arrai
                print "Rows in", arrai._v_pathname, ":", arrai.shape

            nrow = 0
            arr = arrai.read()

            rowsread += len(arr)
            narrai += 1

        # Close the file (eventually destroy the extended type)
        fileh.close()

    return (rowsread, 4, rowsread*4)

def createFile(filename, ngroups, ntables, nrows, complevel, complib, recsize):

    # First, create the groups

    # Open a file in "w"rite mode
    fileh = openFile(filename, mode="w", title="PyTables Stress Test")

    for k in range(ngroups):
        # Create the group
        group = fileh.createGroup("/", 'group%04d'% k, "Group %d" % k)

    fileh.close()

    # Now, create the tables
    rowswritten = 0
    if not ntables:
        rowsize = 0

    for k in range(ngroups):
        print "Filling tables in group:", k
        fileh = openFile(filename, mode="a", rootUEP='group%04d'% k)
        # Get the group
        group = fileh.root
        for j in range(ntables):
            # Create a table
            #table = fileh.createTable(group, 'table%04d'% j, Test,
            table = fileh.createTable(group, 'table%04d'% j, TestDict,
                                      'Table%04d'%j,
                                      complevel, complib, nrows)
            rowsize = table.rowsize
            # Get the row object associated with the new table
            row = table.row
            # Fill the table
            for i in xrange(nrows):
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

    rowsize = 0
    buffersize = 0
    rowsread = 0
    for ngroup in range(ngroups):
        fileh = openFile(filename, mode="r", rootUEP='group%04d'% ngroup)
        # Get the group
        group = fileh.root
        ntable = 0
        if verbose:
            print "Group ==>", group
        for table in fileh.listNodes(group, 'Table'):
            rowsize = table.rowsize
            buffersize=table.rowsize * table.nrowsinbuf
            if verbose > 1:
                print "Table ==>", table
                print "Max rows in buf:", table.nrowsinbuf
                print "Rows in", table._v_pathname, ":", table.nrows
                print "Buffersize:", table.rowsize * table.nrowsinbuf
                print "MaxTuples:", table.nrowsinbuf

            nrow = 0
            if table.nrows > 0:  # only read if we have rows in tables
                for row in table:
                    try:
                        assert row["ngroup"] == ngroup
                        assert row["ntable"] == ntable
                        assert row["nrow"] == nrow
                    except:
                        print "Error in group: %d, table: %d, row: %d" % \
                              (ngroup, ntable, nrow)
                        print "Record ==>", row
                    nrow += 1

            assert nrow == table.nrows
            rowsread += table.nrows
            ntable += 1

        # Close the file (eventually destroy the extended type)
        fileh.close()

    return (rowsread, rowsize, buffersize)

class TrackRefs:
    """Object to track reference counts across test runs."""

    def __init__(self, verbose=0):
        self.type2count = {}
        self.type2all = {}
        self.verbose = verbose

    def update(self, verbose=0):
        obs = sys.getobjects(0)
        type2count = {}
        type2all = {}
        for o in obs:
            all = sys.getrefcount(o)
            t = type(o)
            if verbose:
                #if t == types.TupleType:
                if isinstance(o, Group):
                #if isinstance(o, metaIsDescription):
                    print "-->", o, "refs:", all
                    refrs = gc.get_referrers(o)
                    trefrs = []
                    for refr in refrs:
                        trefrs.append(type(refr))
                    print "Referrers -->", refrs
                    print "Referrers types -->", trefrs
            #if t == types.StringType: print "-->",o
            if t in type2count:
                type2count[t] += 1
                type2all[t] += all
            else:
                type2count[t] = 1
                type2all[t] = all

        ct = [(type2count[t] - self.type2count.get(t, 0),
               type2all[t] - self.type2all.get(t, 0),
               t)
              for t in type2count.iterkeys()]
        ct.sort()
        ct.reverse()
        for delta1, delta2, t in ct:
            if delta1 or delta2:
                print "%-55s %8d %8d" % (t, delta1, delta2)

        self.type2count = type2count
        self.type2all = type2all

def dump_refs(preheat=10, iter1=10, iter2=10, *testargs):

    rc1 = rc2 = None
    #testMethod()
    for i in xrange(preheat):
        testMethod(*testargs)
    gc.collect()
    rc1 = sys.gettotalrefcount()
    track = TrackRefs()
    for i in xrange(iter1):
        testMethod(*testargs)
    print "First output of TrackRefs:"
    gc.collect()
    rc2 = sys.gettotalrefcount()
    track.update()
    print >>sys.stderr, "Inc refs in function testMethod --> %5d" % (rc2-rc1)
    for i in xrange(iter2):
        testMethod(*testargs)
        track.update(verbose=1)
    print "Second output of TrackRefs:"
    gc.collect()
    rc3 = sys.gettotalrefcount()

    print >>sys.stderr, "Inc refs in function testMethod --> %5d" % (rc3-rc2)
    ok = 1

def dump_garbage():
    """
    show us waht the garbage is about
    """
    # Force collection
    print "\nGARBAGE:"
    gc.collect()

    print "\nGARBAGE OBJECTS:"
    for x in gc.garbage:
        s = str(x)
        #if len(s) > 80: s = s[:77] + "..."
        print type(x),"\n   ", s

    #print "\nTRACKED OBJECTS:"
    #reportLoggedInstances("*")

def testMethod(file, usearray, testwrite, testread, complib, complevel,
               ngroups, ntables, nrows):

    if complevel > 0:
        print "Compression library:", complib
    if testwrite:
        t1 = time.time()
        cpu1 = time.clock()
        if usearray:
            (rowsw, rowsz) = createFileArr(file, ngroups, ntables, nrows)
        else:
            (rowsw, rowsz) = createFile(file, ngroups, ntables, nrows,
                                        complevel, complib, recsize)
        t2 = time.time()
        cpu2 = time.clock()
        tapprows = round(t2-t1, 3)
        cpuapprows = round(cpu2-cpu1, 3)
        tpercent = int(round(cpuapprows/tapprows, 2)*100)
        print "Rows written:", rowsw, " Row size:", rowsz
        print "Time writing rows: %s s (real) %s s (cpu)  %s%%" % \
              (tapprows, cpuapprows, tpercent)
        print "Write rows/sec: ", int(rowsw / float(tapprows))
        print "Write KB/s :", int(rowsw * rowsz / (tapprows * 1024))

    if testread:
        t1 = time.time()
        cpu1 = time.clock()
        if usearray:
            (rowsr, rowsz, bufsz)=readFileArr(file, ngroups, recsize, verbose)
        else:
            (rowsr, rowsz, bufsz) = readFile(file, ngroups, recsize, verbose)
        t2 = time.time()
        cpu2 = time.clock()
        treadrows = round(t2-t1, 3)
        cpureadrows = round(cpu2-cpu1, 3)
        tpercent = int(round(cpureadrows/treadrows, 2)*100)
        print "Rows read:", rowsr, " Row size:", rowsz, "Buf size:", bufsz
        print "Time reading rows: %s s (real) %s s (cpu)  %s%%" % \
              (treadrows, cpureadrows, tpercent)
        print "Read rows/sec: ", int(rowsr / float(treadrows))
        print "Read KB/s :", int(rowsr * rowsz / (treadrows * 1024))

if __name__=="__main__":
    import getopt
    import profile
    try:
        import psyco
        psyco_imported = 1
    except:
        psyco_imported = 0


    usage = """usage: %s [-d debug] [-v level] [-p] [-r] [-w] [-l complib] [-c complevel] [-g ngroups] [-t ntables] [-i nrows] file
    -d debugging level
    -v verbosity level
    -p use "psyco" if available
    -a use Array objects instead of Table
    -r only read test
    -w only write test
    -l sets the compression library to be used ("zlib", "lzo", "ucl", "bzip2")
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
    if len(pargs) <> 1:
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
    usepsyco = 0
    usearray = 0
    complevel = 0
    complib = "zlib"

    # Get the options
    for option in opts:
        if option[0] == '-d':
            debug = int(option[1])
        if option[0] == '-v':
            verbose = int(option[1])
        if option[0] == '-p':
            usepsyco = 1
        if option[0] == '-a':
            usearray = 1
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

    if debug == 1:
        gc.set_debug(gc.DEBUG_LEAK)

    # Catch the hdf5 file passed as the last argument
    file = pargs[0]

    if psyco_imported and usepsyco:
        psyco.bind(createFile)
        psyco.bind(readFile)

    if debug == 2:
        dump_refs(10, 10, 15, file, usearray, testwrite, testread, complib,
                  complevel, ngroups, ntables, nrows)
    else:
#         testMethod(file, usearray, testwrite, testread, complib, complevel,
#                    ngroups, ntables, nrows)
        profile.run("testMethod(file, usearray, testwrite, testread, " + \
                    "complib, complevel, ngroups, ntables, nrows)")

    # Show the dirt
    if debug == 1:
        dump_garbage()
