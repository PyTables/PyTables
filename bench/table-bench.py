#!/usr/bin/env python

import copy
                
import numarray as NA
from tables import *
import random

# This class is accessible only for the examples
class Small(IsDescription):
    """ A record has several columns. They are represented here as
    class attributes, whose names are the column names and their
    values will become their types. The IsDescription class will take care
    the user will not add any new variables and that its type is
    correct."""
    
    var1 = StringCol(length=4, dflt="")
    var2 = IntCol(0)
    var3 = FloatCol(0)

# Define a user record to characterize some kind of particles
class Medium(IsDescription):
    name        = StringCol(length=16, dflt="")  # 16-character String
    float1      = Col("Float64", 2, NA.arange(2))
    #float1      = Col("Float64", 1, 2.3)
    #float2      = Col("Float64", 1, 2.3)
    #zADCcount    = Col("Int16", 1, 0)    # signed short integer
    ADCcount    = Col("Int32", 1, 0)    # signed short integer
    grid_i      = Col("Int32", 1, 0)    # integer
    grid_j      = Col("Int32", 1, 0)    # integer
    pressure    = Col("Float32", 1, 0)    # float  (single-precision)
    energy      = Col("Float64", 1, 0)    # double (double-precision)

# Define a user record to characterize some kind of particles
class Big(IsDescription):
    name        = StringCol(length=16, dflt="")  # 16-character String
    float1      = Col("Float64", 32, NA.arange(32))
    float2      = Col("Float64", 32, 2.2)
    TDCcount    = Col("Int8", 1, 0)    # signed short integer
    #ADCcount    = Col("Int32", 1, 0)
    #ADCcount    = Col("Int16", 1, 0)    # signed short integer
    grid_i      = Col("Int32", 1, 0)    # integer
    grid_j      = Col("Int32", 1, 0)    # integer
    pressure    = Col("Float32", 1, 0)    # float  (single-precision)
    energy      = Col("Float64", 1, 0)    # double (double-precision)

def createFile(filename, totalrows, filters, recsize):

    # Open a file in "w"rite mode
    fileh = openFile(filename, mode = "w", title="Table Benchmark",
                     filters=filters)

    # Table title
    title = "This is the table title"
    
    # Create a Table instance
    group = fileh.root
    rowswritten = 0
    for j in range(3):
        # Create a table
        if recsize == "big":
            table = fileh.createTable(group, 'tuple'+str(j), Big, title,
                                      None,
                                      totalrows)
            arr = NA.array(NA.arange(32), type=NA.Float64)
            arr2 = NA.array(NA.arange(32), type=NA.Float64)
        elif recsize == "medium":
            table = fileh.createTable(group, 'tuple'+str(j), Medium, title,
                                      None,
                                      totalrows)
            arr = NA.array(NA.arange(2), type=NA.Float64)
        elif recsize == "small":
            table = fileh.createTable(group, 'tuple'+str(j), Small, title,
                                      None,
                                      totalrows)
        else:
            raise RuntimeError, "This should never happen"

        table.attrs.test = 2
        rowsize = table.rowsize
        # Get the row object associated with the new table
        d = table.row
        # Fill the table
        if recsize == "big":
            for i in xrange(totalrows):
                # d['name']  = 'Part: %6d' % (i)
                d['TDCcount'] = i % 256
                #d['float1'] = NA.array([i]*32, NA.Float64)
                #d['float2'] = NA.array([i**2]*32, NA.Float64)
                #d['float1'][0] = float(i)
                #d['float2'][0] = float(i*2)
                # Common part with medium
                d['grid_i'] = i 
                d['grid_j'] = 10 - i
                d['pressure'] = float(i*i)
                # d['energy'] = float(d['pressure'] ** 4)
                d['energy'] = d['pressure']
                # d['idnumber'] = i * (2 ** 34) 
                d.append()
        elif recsize == "medium":
            for i in xrange(totalrows):
                #d['name']  = 'Part: %6d' % (i)
                #d['float1'] = NA.array([i]*2, NA.Float64)
                #d['float1'] = arr
                #d['float1'] = i
                #d['float2'] = float(i)
                # Common part with big:
                d['grid_i'] = i 
                d['grid_j'] = 10 - i
                d['pressure'] = i*2
                # d['energy'] = float(d['pressure'] ** 4)
                d['energy'] = d['pressure']
                d.append()
        else: # Small record
            for i in xrange(totalrows):
                #d['var1'] = str(random.randrange(1000000))
                #d['var3'] = random.randrange(10000000)
                #d['var1'] = str(i)
                d['var2'] = random.randrange(totalrows)
                #d['var3'] = 12.1e10
                d['var3'] = totalrows-i
                d.append()  # This is a 10% faster than table.append()
		    
        rowswritten += totalrows

        group._v_attrs.test2 = "just a test"
        # Create a new group
        group2 = fileh.createGroup(group, 'group'+str(j))
        # Iterate over this new group (group2)
        group = group2
    
    # Close the file (eventually destroy the extended type)
    fileh.close()
    
    return (rowswritten, rowsize)

def readFile(filename, recsize, verbose):
    # Open the HDF5 file in read-only mode

    fileh = openFile(filename, mode = "r")
    rowsread = 0
    for groupobj in fileh.walkGroups(fileh.root):
        #print "Group pathname:", groupobj._v_pathname
        row = 0
        for table in fileh.listNodes(groupobj, 'Table'):
            rowsize = table.rowsize
            print "reading", table
            if verbose:
                print "Max rows in buf:", table._v_maxTuples
                print "Rows in", table._v_pathname, ":", table.nrows
                print "Buffersize:", table.rowsize * table._v_maxTuples
                print "MaxTuples:", table._v_maxTuples

            if recsize == "big" or recsize == "medium":
                #e = [ p.float1 for p in table.iterrows() 
                #      if p.grid_i < 2 ]
                #e = [ str(p) for p in table.iterrows() ]
                #      if p.grid_i < 2 ]
#                 e = [ p['grid_i'] for p in table.iterrows() 
#                       if p['grid_j'] == 20 and p['grid_i'] < 20 ]
#                 e = [ p['grid_i'] for p in table(step=1) 
#                       if p['grid_j'] <= 2 ]
                e = [ p['grid_i'] for p in table(step=1, where=("grid_i<=20"))]
#                 e = [ p['grid_i'] for p in table.iterrows() 
#                       if p.nrow() == 20 ]
#                 e = [ table.delrow(p.nrow()) for p in table.iterrows() 
#                       if p.nrow() == 20 ]
                # The version with a for loop is only 1% better than
                # comprenhension list
                #e = []
                #for p in table.iterrows(): 
                #    if p.grid_i < 20:
                #        e.append(p.grid_j)
            else:  # small record case
#                 e = [ p['var3'] for p in table.iterrows()
#                       if p['var2'] < 20 and p['var3'] < 20 ]
               e = [ p['var3'] for p in table(where="var3 <= 20")
                     if p['var2'] < 20 ]
#                 e = [ p['var3'] for p in table.iterrows()
#                       if p['var2'] <= 20 ]
                #e = [ p['var3'] for p in table.iterrows(0,21) ]
#                  e = [ p['var3'] for p in table.iterrows()
#                       if p.nrow() <= 20 ]
                #e = [ p['var3'] for p in table.iterrows(1,0,1000)]
                #e = [ p['var3'] for p in table.iterrows(1,100)]
                #e = [ p['var3'] for p in table.iterrows(step=2)
                #      if p.nrow() < 20 ]
                #e = [ p['var2'] for p in table.iterrows()
                #      if p['var2'] < 20 ]
                #for p in table.iterrows():
                #      pass
            if verbose:
                #print "Last record read:", p
                print "resulting selection list ==>", e

	    rowsread += table.nrows
            row += 1
            if verbose:
                print "Total selected records ==> ", len(e)
        
    # Close the file (eventually destroy the extended type)
    fileh.close()

    return (rowsread, rowsize)

def readField(filename, field, rng, verbose):
    fileh = openFile(filename, mode = "r")
    rowsread = 0
    for groupobj in fileh.walkGroups(fileh.root):
        row = 0
        for table in fileh.listNodes(groupobj, 'Table'):
            rowsize = table.rowsize
            #table._v_maxTuples = 3 # For testing purposes
            if verbose:
                print "Max rows in buf:", table._v_maxTuples
                print "Rows in", table._v_pathname, ":", table.nrows
                print "Buffersize:", table.rowsize * table._v_maxTuples
                print "MaxTuples:", table._v_maxTuples
                print "(field, start, stop, step) ==>", (field, rng[0], rng[1], rng[2])

            e = table.read(rng[0], rng[1], rng[2], field)

	    rowsread += table.nrows
            if verbose:
                print "Selected rows ==> ", e
                print "Total selected rows ==> ", len(e)
        
    # Close the file (eventually destroy the extended type)
    fileh.close()
    return (rowsread, rowsize)

if __name__=="__main__":
    import sys
    import getopt
    try:
        import psyco
        psyco_imported = 1
    except:
        psyco_imported = 0
    
    import time
    
    usage = """usage: %s [-v] [-p] [-R range] [-r] [-w] [-s recsize] [-f field] [-c level] [-l complib] [-i iterations] [-S] [-F] file
            -v verbose
	    -p use "psyco" if available
            -R select a range in a field in the form "start,stop,step"
	    -r only read test
	    -w only write test
            -s use [big] record, [medium] or [small]
            -f only read stated field name in tables
            -c sets a compression level (do not set it or 0 for no compression)
            -S activate shuffling filter
            -F activate fletcher32 filter
            -l sets the compression library to be used ("zlib", "lzo", "ucl")
            -i sets the number of rows in each table\n""" % sys.argv[0]

    try:
        opts, pargs = getopt.getopt(sys.argv[1:], 'vpSFR:rwf:s:c:l:i:')
    except:
        sys.stderr.write(usage)
        sys.exit(0)

    # if we pass too much parameters, abort
    if len(pargs) <> 1: 
        sys.stderr.write(usage)
        sys.exit(0)

    # default options
    verbose = 0
    rng = None
    recsize = "medium"
    fieldName = None
    testread = 1
    testwrite = 1
    usepsyco = 0
    complevel = 0
    shuffle = 0
    fletcher32 = 0
    complib = "zlib"
    iterations = 100

    # Get the options
    for option in opts:
        if option[0] == '-v':
            verbose = 1
        if option[0] == '-p':
            usepsyco = 1
        if option[0] == '-S':
            shuffle = 1
        if option[0] == '-F':
            fletcher32 = 1
        elif option[0] == '-R':
            rng = [int(i) for i in option[1].split(",")]
        elif option[0] == '-r':
            testwrite = 0
        elif option[0] == '-w':
            testread = 0
        elif option[0] == '-f':
            fieldName = option[1]
        elif option[0] == '-s':
            recsize = option[1]
            if recsize not in ["big", "medium", "small"]:
                sys.stderr.write(usage)
                sys.exit(0)
        elif option[0] == '-c':
            complevel = int(option[1])
        elif option[0] == '-l':
            complib = option[1]
        elif option[0] == '-i':
            iterations = int(option[1])
            
    # Build the Filters instance
    filters = Filters(complevel=complevel, complib=complib,
                      shuffle=shuffle, fletcher32=fletcher32)

    # Catch the hdf5 file passed as the last argument
    file = pargs[0]

    if testwrite:
        print "Compression level:", complevel
        if complevel > 0:
            print "Compression library:", complib
            if shuffle:
                print "Suffling..."
	t1 = time.time()
	cpu1 = time.clock()
        if psyco_imported and usepsyco:
            psyco.bind(createFile)
	(rowsw, rowsz) = createFile(file, iterations, filters, recsize)
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
        if psyco_imported and usepsyco:
            psyco.bind(readFile)
            psyco.bind(readField)
            pass
        if rng:
            (rowsr, rowsz) = readField(file, fieldName, rng, verbose)
            pass
        else:
            for i in range(1):
                (rowsr, rowsz) = readFile(file, recsize, verbose)
	t2 = time.time()
        cpu2 = time.clock()
	treadrows = round(t2-t1, 3)
        cpureadrows = round(cpu2-cpu1, 3)
        tpercent = int(round(cpureadrows/treadrows, 2)*100)
	print "Rows read:", rowsr, " Row size:", rowsz
	print "Time reading rows: %s s (real) %s s (cpu)  %s%%" % \
              (treadrows, cpureadrows, tpercent)
	print "Read rows/sec: ", int(rowsr / float(treadrows))
	print "Read KB/s :", int(rowsr * rowsz / (treadrows * 1024))
    
