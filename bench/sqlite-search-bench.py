#!/usr/bin/python
import sqlite
import random
import time
import sys
import os
import os.path
from tables import *

# pysco is off by default
psycon = 0

def createNewBenchFile(bfile, verbose):

    class Create(IsDescription):
        nrows   = IntCol(pos=0)
        irows   = IntCol(pos=1)
        tfill   = FloatCol(pos=2)
        tidx    = FloatCol(pos=3)
        tcfill  = FloatCol(pos=4)
        tcidx   = FloatCol(pos=5)
        rowsecf = FloatCol(pos=6)
        rowseci = FloatCol(pos=7)
        fsizek  = FloatCol(pos=8)
        fsizeik = FloatCol(pos=9)
        psyco   = BoolCol(pos=10)

    class Search(IsDescription):
        nrows   = IntCol(pos=0)
        rowsel  = IntCol(pos=1)
        time1   = FloatCol(pos=2)
        time2   = FloatCol(pos=3)
        tcpu1   = FloatCol(pos=4)
        tcpu2   = FloatCol(pos=5)
        rowsec1 = FloatCol(pos=6)
        rowsec2 = FloatCol(pos=7)
        psyco   = BoolCol(pos=8)

    if verbose:
        print "Creating a new benchfile:", bfile
    # Open the benchmarking file
    bf = openFile(bfile, "w")
    # Create groups
    for recsize in ["sqlite_small"]:
        group = bf.createGroup("/", recsize, recsize+" Group")
        # Attach the row size of table as attribute
        if recsize == "small":
            group._v_attrs.rowsize = 16
        # Create a Table for writing bench
        bf.createTable(group, "create", Create, "create bench")
        # create a group for searching bench
        groupS = bf.createGroup(group, "search", "Search Group")
        # Create Tables for searching
        for mode in ["indexed", "standard"]:
            group = bf.createGroup(groupS, mode, mode+" Group")
            # for searching bench
            #for atom in ["string", "int", "float", "bool"]:
            for atom in ["string", "int", "float"]:
                bf.createTable(group, atom, Search, atom+" bench")
    bf.close()
    
def createFile(dbfile, nrows, indexmode, bfile):

    if os.path.exists(dbfile):
        print "removing:", dbfile
        os.remove(dbfile)
    print "Creating a new database:", dbfile
    instd=os.popen("/usr/local/bin/sqlite "+dbfile, "w")
    CREATESTD="""
CREATE TABLE small (	
-- Name		Type	        -- Example 
---------------------------------------
recnum	INTEGER PRIMARY KEY,  -- 345
var1		char(4),	-- Abronia villosa
var2		INTEGER,	-- 111
var3            FLOAT        --  12.32
);
"""
    CREATEIDX="""
CREATE TABLE small (	
-- Name		Type	        -- Example 
---------------------------------------
recnum	INTEGER PRIMARY KEY,  -- 345
var1		char(4),	-- Abronia villosa
var2		INTEGER,	-- 111
var3            FLOAT        --  12.32
);
CREATE INDEX ivar1 ON small(var1);
CREATE INDEX ivar2 ON small(var2);
CREATE INDEX ivar3 ON small(var3);
"""
#     if indexmode == "indexed":
#         instd.write(CREATEIDX)
#     else:
#         instd.write(CREATESTD)
    # Creating the table first and indexing afterwards is a little bit faster
    instd.write(CREATESTD)
    instd.close()

    conn = sqlite.connect(dbfile)
    cursor = conn.cursor()
    place_holders = ",".join(['%s']*3)
    # Insert rows
    SQL = "insert into small values(NULL, %s)" % place_holders
    time1 = time.time()
    cpu1 = time.clock()
    for rowcount in xrange(nrows):
        fields = ("%.4s" % rowcount, rowcount, float(nrows-rowcount))
        cursor.execute(SQL, fields)
    conn.commit()
    t1 = round(time.time()-time1, 5)
    tcpu1 = round(time.clock()-cpu1, 5)
    rowsecf = round((nrows/10.**6)/t1, 3)
    size1 = round(os.stat(dbfile)[6] / (1024.*1024), 3)
    print "******** Results for writing nrows = %s" % (nrows), "*********"
    print "Insert time:", t1, ", KRows/s:", round((nrows/10.**3)/t1, 3),
    print ", File size:", size1, "MB"
    # Indexem
    if indexmode == "indexed":
        time1 = time.time()
        cpu1 = time.clock()
        cursor.execute("CREATE INDEX ivar1 ON small(var1)")
        cursor.execute("CREATE INDEX ivar2 ON small(var2)")
        cursor.execute("CREATE INDEX ivar3 ON small(var3)")
        conn.commit()
        t2 = round(time.time()-time1, 5)
        tcpu2 = round(time.clock()-cpu1, 5)
        rowseci = round((nrows/10.**6)/t2, 3)
        print "Index time:", t2, ", IKRows/s:", round((nrows/10.**3)/t2, 3),
        size2 = round(os.stat(dbfile)[6] / (1024.*1024), 3)
        print ", Final size with index:", size2, "MB"
    else:
        t2 = 0.
        tcpu2 = 0.
        rowseci = 0.
        size2 = size1

    conn.close()

    # Collect benchmark data
    bf = openFile(bfile,"a")
    recsize = "sqlite_small"
    table = bf.getNode("/"+recsize+"/create")
    table.row["nrows"] = nrows
    table.row["irows"] = nrows
    table.row["tfill"] = t1
    table.row["tidx"]  = t2
    table.row["tcfill"] = tcpu1
    table.row["tcidx"] = tcpu2
    table.row["psyco"] = psycon
    table.row["rowsecf"] = rowsecf
    table.row["rowseci"] = rowseci
    table.row["fsizek"] = size1
    table.row["fsizeik"] = size2
    table.row.append()
    bf.close()

    return

def readFile(dbfile, nrows, indexmode, bfile, riter):
    # Connect to the database.
    conn = sqlite.connect(db=dbfile, mode=755)
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
    bf = openFile(bfile,"a")
    # Execute queries
    rowselected = 0
    t2 = 0.
    tcpu2 = 0.
    for atom in ["string", "int", "float"]:
        time2 = 0
        cpu2 = 0
        rowsel = 0
        for i in xrange(riter):
            rnd = random.randrange(nrows)
            time1 = time.time()
            cpu1 = time.clock()
            if atom == "string":
                #cursor.execute(SQL1, "1111")
                cursor.execute(SQL1, str(rnd)[-4:])
            elif atom == "int":
                #cursor.execute(SQL2 % (3, 5))
                cursor.execute(SQL2 % (rnd, rnd+3))
            elif atom == "float":
                #cursor.execute(SQL3 % (3, 5))
                cursor.execute(SQL3 % (float(rnd), float(rnd+3)))
            else:
                raise ValueError, "atom must take a value in ['string','int','float']"
            if i == 0:
                t1 = time.time() - time1
                tcpu1 = time.clock() - cpu1
            else:
                if indexmode == "indexed":
                    # if indexed, wait until the 5th iteration to take
                    # times (so as to insure that the index is
                    # effectively cached)
                    if i >= 5:
                        time2 += time.time() - time1
                        cpu2 += time.clock() - cpu1
                else:
                    time2 += time.time() - time1
                    time2 += time.clock() - cpu1
        if riter > 1:
            if indexmode == "indexed" and riter >= 5:
                correction = 5
            else:
                correction = 1
            t2 = time2/(riter-correction)
            tcpu2 = cpu2/(riter-correction)

        print "*** Query results for atom = %s, nrows = %s, indexmode = %s ***" % (atom, nrows, indexmode)
        print "Query time:", round(t1,5), ", cached time:", round(t2, 5)
        print "MRows/s:", round((nrows/10.**6)/t1, 3),
        if t2 > 0:
             print ", cached MRows/s:", round((nrows/10.**6)/t2, 3)
        else:
            print

        # Collect benchmark data
        recsize = "sqlite_small"
        tablepath = "/"+recsize+"/search/"+indexmode+"/"+atom
        table = bf.getNode(tablepath)
        table.row["nrows"] = nrows
        table.row["rowsel"] = rowsel
        table.row["time1"] = t1
        table.row["time2"] = t2
        table.row["tcpu1"] = tcpu1
        table.row["tcpu2"] = tcpu2
        table.row["psyco"] = psycon
        table.row["rowsec1"] = round(nrows / float(t1), 4)/10**6
        if t2 > 0:
            table.row["rowsec2"] = round(nrows / float(t2), 4)/10**6
        table.row.append()
        table.flush()  # Flush the data

    # Close the database
    conn.close()
    bf.close()  # the bench database

    return

if __name__=="__main__":
    import sys
    import os.path
    import getopt
    try:
        import psyco
        psyco_imported = 1
    except:
        psyco_imported = 0
    
    import time

    usage = """usage: %s [-v] [-p] [-r] [-w] [-n nrows] [-b file] [-k riter] [-m indexmode] datafile
            -v verbose
	    -p use "psyco" if available
	    -r only read test
	    -w only write test
            -n the number of rows (in krows)
            -b bench filename
            -k number of iterations for reading\n""" % sys.argv[0]

    try:
        opts, pargs = getopt.getopt(sys.argv[1:], 'vprwn:b:k:m:')
    except:
        sys.stderr.write(usage)
        sys.exit(0)

    # if we pass too much parameters, abort
    if len(pargs) <> 1: 
        sys.stderr.write(usage)
        sys.exit(0)

    # default options
    verbose = 0
    testread = 1
    testwrite = 1
    usepsyco = 0
    nrows = 1000
    bfile = "sqlite-bench.h5"
    supported_imodes = ["indexed","standard"]
    indexmode = "indexed"
    riter = 2

    # Get the options
    for option in opts:
        if option[0] == '-v':
            verbose = 1
        if option[0] == '-p':
            usepsyco = 1
        elif option[0] == '-r':
            testwrite = 0
        elif option[0] == '-w':
            testread = 0
        elif option[0] == '-b':
            bfile = option[1]
        elif option[0] == '-m':
            indexmode = option[1]
            if indexmode not in supported_imodes:
                raise ValueError, "Indexmode should be any of '%s' and you passed '%s'" % (supported_imodes, indexmode)
        elif option[0] == '-n':
            nrows = int(float(option[1])*1000)
        elif option[0] == '-k':
            riter = int(option[1])
            
    # remaining parameters
    dbfile=pargs[0]

    # Create the benchfile (if needed)
    if not os.path.exists(bfile):
        createNewBenchFile(bfile, verbose)

    if testwrite:
        if psyco_imported and usepsyco:
            psyco.bind(createFile)
            psycon = 1
        createFile(dbfile, nrows, indexmode, bfile)

    if testread:
        if psyco_imported and usepsyco:
            psyco.bind(readFile)
            psycon = 1
        readFile(dbfile, nrows, indexmode, bfile, riter)
