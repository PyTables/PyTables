#!/usr/bin/env python2.2

from tables import *

# Verbosity level
verbose = 0

# This class is accessible only for the examples
class Record(IsRecord):
    """ A record has several columns. They are represented here as
    class attributes, whose names are the column names and their
    values will become their types. The IsRecord class will take care
    the user won't add any new variables and that its type is
    correct."""
    
    var1 = '4s'
    var2 = 'i'
    var3 = 'd'

# Define a user record to characterize some kind of particles
class Particle(IsRecord):
    name        = '16s'  # 16-character String
    idnumber    = 'Q'    # unsigned long long (i.e. 64-bit integer)
    TDCcount    = 'B'    # unsigned byte
    ADCcount    = 'H'    # unsigned short integer
    grid_i      = 'i'    # integer
    grid_j      = 'i'    # integer
    pressure    = 'f'    # float  (single-precision)
    energy      = 'd'    # double (double-precision)

def createFile(filename, totalrows, fast, complevel, bigrec):

    # Open a file in "w"rite mode
    fileh = openFile(filename, mode = "w")

    # Table title
    title = "This is the table title"
    
    # Create a Table instance
    group = fileh.root
    for j in range(3):
        # Create a table
        if bigrec:
            table = fileh.createTable(group, 'tuple'+str(j), Particle(), title,
                                      complevel, totalrows)
        else:
            table = fileh.createTable(group, 'tuple'+str(j), Record(), title,
                                      complevel, totalrows)
            
        # Get the record object associated with the new table
        d = table.record 
        # Fill the table
        for i in xrange(totalrows):
            if fast:
                if bigrec:
                    table.appendAsValues((i * 256) % (1 << 16),
                                         i % 256,
                                         float((i*i) ** 4),
                                         i,
                                         10 - i,
                                         i * (2 ** 34),
                                         'Particle: %6d' % (i),
                                         float(i * i),
                                         )
                else:
                    table.appendAsValues(str(i), i * j, 12.1e10)
            else:
                if bigrec:
                    d.name  = 'Particle: %6d' % (i)
                    d.TDCcount = i % 256    
                    d.ADCcount = (i * 256) % (1 << 16)
                    d.grid_i = i 
                    d.grid_j = 10 - i
                    d.pressure = float(i*i)
                    d.energy = float(d.pressure ** 4)
                    d.idnumber = i * (2 ** 34) 
                    table.appendAsRecord(d)
                else:
                    d.var1 = str(i)
                    d.var2 = i * j
                    d.var3 = 12.1e10
                    table.appendAsRecord(d)

        # Create a new group
        group2 = fileh.createGroup(group, 'group'+str(j))
        # Iterate over this new group (group2)
        group = group2
    
    # Close the file (eventually destroy the extended type)
    fileh.close()

def readFile(filename, fast, bigrec):
    # Open the HDF5 file in read-only mode

    fileh = openFile(filename, mode = "r")
    for groupobj in fileh.walkGroups(fileh.root):
        #print "Group pathname:", groupobj._v_pathname
        for table in fileh.listNodes(groupobj, 'Table'):
            #print "Table title for", table._v_pathname, ":", table.tableTitle
            if verbose:
                print "Rows in", table._v_pathname, ":", table.nrows

            if fast:
                if bigrec:
                    e = [ t[3] for t in table.readAsTuples() if t[3] < 20 ]
                else:
                    e = [ t[1] for t in table.readAsTuples() if t[1] < 20 ]
            else:
                if bigrec:
                    e = [ p.grid_i for p in table.readAsRecords() 
                          if p.grid_i < 20 ]
                else:
                    e = [ p.var2 for p in table.readAsRecords()
                          if p.var2 < 20 ]
                if verbose:
                    print "Last record read:", p

            if verbose:
                print "Total selected records ==> ", len(e)
        
    # Close the file (eventually destroy the extended type)
    fileh.close()


if __name__=="__main__":
    import sys
    import getopt
    import time
    
    usage = """usage: %s [-v] [-b] [-f] [-c level] [-i iterations] file
            -v verbose
            -b use big record (Particle); else use small record (Record)
            -f means use fast methods (unsafer)
            -c sets a compression level (don't set it or 0 for no compression)
            -i sets the number of rows in each table\n""" % sys.argv[0]

    try:
        opts, pargs = getopt.getopt(sys.argv[1:], 'vbfc:i:')
    except:
        sys.stderr.write(usage)
        sys.exit(0)

    # if we pass too much parameters, abort
    if len(pargs) <> 1: 
        sys.stderr.write(usage)
        sys.exit(0)

    # default options
    #verbose = 0
    bigrec = 0
    fast = 0
    complevel = 0
    iterations = 100

    # Get the options
    for option in opts:
        if option[0] == '-v':
            global verbose
            verbose = 1
        elif option[0] == '-b':
            bigrec = 1
        elif option[0] == '-f':
            fast = 1
        elif option[0] == '-c':
            complevel = int(option[1])
        elif option[0] == '-i':
            iterations = int(option[1])

    # Catch the hdf5 file passed as the last argument
    file = pargs[0]

    t1 = time.clock()
    createFile(file, iterations, fast, complevel, bigrec)
    t2 = time.clock()
    tapprows = round(t2-t1, 3)
    
    t1 = time.clock()    
    readFile(file, fast, bigrec)
    t2 = time.clock()
    treadrows = round(t2-t1, 3)
    
    #addRecords(file, iterations * 2, fast)

    if fast:
        print "-*-"*8, " FAST mode ", "-*-"*8
    else:
        print "-*-"*8, " NORMAL mode ", "-*-"*8
    print "Compression level:", complevel
    print "Time appending rows:", tapprows
    print "Write rows/sec: ", int(iterations * 3/ float(tapprows))
    print "Time reading rows:", treadrows
    print "Read rows/sec: ", int(iterations * 3 / float(treadrows))
