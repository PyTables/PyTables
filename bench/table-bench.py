#!/usr/bin/env python2.2

import copy
                
import numarray as NA
from tables import *

# Verbosity level
verbose = 0

# This class is accessible only for the examples
class Small(IsRecord):
    """ A record has several columns. They are represented here as
    class attributes, whose names are the column names and their
    values will become their types. The IsRecord class will take care
    the user will not add any new variables and that its type is
    correct."""
    
    var1 = Col("CharType", 16, "")
    var2 = Col("Int32", 1, 0)
    var3 = Col("Float64", 1, 0)

# Define a user record to characterize some kind of particles
class Medium(IsRecord):
    name        = Col('CharType', 16, "")  # 16-character String
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
class Big(IsRecord):
    name        = Col('CharType', 16, "")  # 16-character String
    float1      = Col("Float64", 32, NA.arange(32))
    float2      = Col("Float64", 32, 2.2)
    TDCcount    = Col("Int8", 1, 0)    # signed short integer
    #ADCcount    = Col("Int32", 1, 0)
    #ADCcount    = Col("Int16", 1, 0)    # signed short integer
    grid_i      = Col("Int32", 1, 0)    # integer
    grid_j      = Col("Int32", 1, 0)    # integer
    pressure    = Col("Float32", 1, 0)    # float  (single-precision)
    energy      = Col("Float64", 1, 0)    # double (double-precision)

def createFile(filename, totalrows, complevel, recsize):

    # Open a file in "w"rite mode
    fileh = openFile(filename, mode = "w")

    # Table title
    title = "This is the table title"
    
    # Create a Table instance
    group = fileh.root
    rowswritten = 0
    for j in range(3):
        # Create a table
        if recsize == "big":
            table = fileh.createTable(group, 'tuple'+str(j), Big(), title,
                                      complevel, totalrows)
            arr = NA.array(NA.arange(32), type=NA.Float64)
            arr2 = NA.array(NA.arange(32), type=NA.Float64)
        elif recsize == "medium":
            table = fileh.createTable(group, 'tuple'+str(j), Medium(), title,
                                      complevel, totalrows)
        elif recsize == "small":
            table = fileh.createTable(group, 'tuple'+str(j), Small(), title,
                                      complevel, totalrows)
        else:
            raise RuntimeError, "This should never happen"
            
        # Get the record object associated with the new table
        #d = table.record
        # In PyTables 0.3 this is changed to a row object
        d = table.row
        # Fill the table
        if recsize == "big" or recsize == "medium":
            for i in xrange(totalrows):
                d.name  = 'Part: %6d' % (i)
                #d.float1 = [1.2, 3.6]
                #d.float1 = arr
                #d.float2 = 2.4
                #d.zADCcount = (i * 256) % (1 << 16)
                #d.ADCcount = (i * 256) % (1 << 16)
                if recsize == "big":
                    d.TDCcount = i % 256
                    d.float1 = NA.array([i]*32, NA.Float64)
                    d.float2 = NA.array([i**2]*32, NA.Float64)
                    #d.float1[0] = float(i)
                    #d.float2[0] = float(i*2)
                    pass
                else:
                    #d.float1 = NA.array([i]*2, NA.Float64)
                    #d.float1 = float(i)
                    #d.float2 = float(i)
                    pass
                d.grid_i = i 
                d.grid_j = 10 - i
                d.pressure = float(i*i)
                #d.energy = float(d.pressure ** 4)
                d.energy = d.pressure
                #d.idnumber = i * (2 ** 34) 
                table.append(d)
        else: # Small record
            for i in xrange(totalrows):
                # __setattr__ is faster than setField!
                #d.var1 = str(i)
                #d['var1'] = str(i)
                #d.var2 = i
                d['var2'] = i
                #d.var3 = 12.1e10
                d['var3'] = 12.1e10
                d.add()  # This is a 10% faster than table.append()
                #table.append(d)
		    
            #rowswritten += 1
        rowswritten += totalrows

        # Create a new group
        group2 = fileh.createGroup(group, 'group'+str(j))
        # Iterate over this new group (group2)
        group = group2
    
    # Close the file (eventually destroy the extended type)
    fileh.close()
    
    return (rowswritten, table._v_rowsize)

def readFile(filename, recsize):
    # Open the HDF5 file in read-only mode

    fileh = openFile(filename, mode = "r")
    rowsread = 0
    for groupobj in fileh.walkGroups(fileh.root):
        #print "Group pathname:", groupobj._v_pathname
        row = 0
        for table in fileh.listNodes(groupobj, 'Table'):
            #print "Table title for", table._v_pathname, ":", table.tableTitle
            if verbose:
                print "Rows in", table._v_pathname, ":", table.nrows

            if recsize == "big" or recsize == "medium":
                # There are two possibilities in doing selects
                # 1.- Modify recarray2.Record2 to check if the field is a
                #     NumArray, and if yes, do a deepcopy before to deliver it
                # Pros: the array object is a real one to the user,
                #       not a reference
                # Cons: This makes all the selections a 25% to 35% slower
                # 2.- Inform to the user that if he wants to keep the array
                #     as a separate object, he have to deeply copy it.
                # Pros: The selection speed do not degrade
                # Cons: The user has to be concious to copy the array if he
                #       want to use it outside the loop

                # For the moment we work under the assumption that the user is
                # responsible to do it (case 2).
                #e = [ p._row for p in table.fetchall()
                #      if p.grid_i < 2 ]
                #e = [ copy.deepcopy(p.float1) for p in table.fetchall()
                #      if p.grid_i < 2 ]
                # Next line can be used in case 1. If used in case 2 you will
                # get corrupted data!.
                #e = [ p.float1 for p in table.fetchall() 
                #      if p.grid_i < 2 ]
                #e = [ str(p) for p in table.fetchall() ]
                #      if p.grid_i < 2 ]
                e = [ p.grid_j for p in table.fetchall() 
                      if p.grid_i < 20 ]
                # The version with a for loop is only 1% better than
                # comprenhension list
                #e = []
                #for p in table.fetchall(): 
                #    if p.grid_i < 20:
                #        e.append(p.grid_j)
            else:
                e = [ p['var3'] for p in table.fetchall()
                      if p['var2'] == 20 ]
                #e = [ p.var3 for p in table.fetchall()
                #      if p.nrow() == 20 ]
                #e = [ p.var3 for p in table.fetchall()
                #      if p.var2 == 2 ]
                #for p in table.fetchall():
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

    return (rowsread, table._v_rowsize)

if __name__=="__main__":
    import sys
    import getopt
    try:
        import psyco
        psyco_imported = 1
    except:
        psyco_imported = 0
    
    import time
    
    usage = """usage: %s [-v] [-r] [-w] [-s recsize] [-f] [-c level] [-i iterations] file
            -v verbose
	    -r only read test
	    -w only write test
            -s use [big] record, [medium] or [small]
            -c sets a compression level (do not set it or 0 for no compression)
            -i sets the number of rows in each table\n""" % sys.argv[0]

    try:
        opts, pargs = getopt.getopt(sys.argv[1:], 'vrws:c:i:')
    except:
        sys.stderr.write(usage)
        sys.exit(0)

    # if we pass too much parameters, abort
    if len(pargs) <> 1: 
        sys.stderr.write(usage)
        sys.exit(0)

    # default options
    #verbose = 0
    recsize = "medium"
    testread = 1
    testwrite = 1
    complevel = 0
    iterations = 100

    # Get the options
    for option in opts:
        if option[0] == '-v':
            global verbose
            verbose = 1
        elif option[0] == '-r':
            testwrite = 0
        elif option[0] == '-w':
            testread = 0
        elif option[0] == '-s':
            recsize = option[1]
            if recsize not in ["big", "medium", "small"]:
                sys.stderr.write(usage)
                sys.exit(0)
        elif option[0] == '-c':
            complevel = int(option[1])
        elif option[0] == '-i':
            iterations = int(option[1])

    # Catch the hdf5 file passed as the last argument
    file = pargs[0]

    print "Compression level:", complevel
    if testwrite:
	t1 = time.clock()
        if psyco_imported:
            psyco.bind(createFile)
            pass
	(rowsw, rowsz) = createFile(file, iterations, complevel, recsize)
	t2 = time.clock()
	tapprows = round(t2-t1, 3)
	print "Rows written:", rowsw, " Row size:", rowsz
	print "Time appending rows:", tapprows
	print "Write rows/sec: ", int(rowsw / float(tapprows))
	print "Write KB/s :", int(rowsw * rowsz / (tapprows * 1024))
	
    if testread:
	t1 = time.clock()
        if psyco_imported:
            psyco.bind(readFile)
            pass
	(rowsr, rowsz) = readFile(file, recsize)
	t2 = time.clock()
	treadrows = round(t2-t1, 3)
	print "Rows read:", rowsr, " Row size:", rowsz
	print "Time reading rows:", treadrows
	print "Read rows/sec: ", int(rowsr / float(treadrows))
	print "Read KB/s :", int(rowsr * rowsz / (treadrows * 1024))
    

