#!/usr/bin/env python2.2


from tables import IsRecord
import shelve


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

def createFile(filename, totalrows, fast, bigrec):
    
    # Open a 'n'ew file
    fileh = shelve.open(filename, flag = "n")

    for j in range(3):
        # Create a table
        #table = fileh.createTable(group, 'tuple'+str(j), Record(), title,
	#                          compress = 6, expectedrows = totalrows)
	# Create a Table instance
	tablename = 'tuple'+str(j)
	table = []
        # Get the record object associated with the new table
        if bigrec:
            d = Particle()
        else:
            d = Record() 
        # Fill the table
        for i in xrange(totalrows):
            if fast:
                if bigrec:
                    table.append(((i * 256) % (1 << 16),
                                 i % 256,
                                 float((i*i) ** 4),
                                 i,
                                 10 - i,
                                 i * (2 ** 34),
                                 'Particle: %6d' % (i),
                                 float(i * i),
                                 ))
                else:
                    table.append((str(i), i * j, 12.1e10))
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
                    table.append((d.ADCcount, d.TDCcount, d.energy, d.grid_i,
                                  d.grid_j, d.idnumber, d.name, d.pressure))
                else:
                    d.var1 = str(i)
                    d.var2 = i * j
                    d.var3 = 12.1e10
                    table.append((d.var1, d.var2, d.var3))

	# Save this table on disk
	fileh[tablename] = table
                
    # Close the file
    fileh.close()

def readFile(filename, fast, bigrec):
    # Open the HDF5 file in read-only mode
    fileh = shelve.open(filename, "r")
    for table in ['tuple0', 'tuple1', 'tuple2']:
	if fast:
            if bigrec:
                e = [ t[3] for t in fileh[table] if t[3] < 20 ]
            else:
                e = [ t[1] for t in fileh[table] if t[1] < 20 ]
	    #e = [ t[1] for t in fileh[table] if t[1] < 20 ]
		
	else:
	    # Record method (how to simulate it in shelve?)
            if bigrec:
                e = [ t[3] for t in fileh[table] if t[3] < 20 ]
            else:
                e = [ t[1] for t in fileh[table] if t[1] < 20 ]

	    #e = [ t[1] for t in fileh[table] if t[1] < 20 ]
	    #e = [ p.var2 for p in table.readAsRecords() if p.var2 < 20 ]
	    # print "Last record ==>", p
    
	print "Total selected records ==> ", len(e)
        
    # Close the file (eventually destroy the extended type)
    fileh.close()


# Add code to test here
if __name__=="__main__":
    import sys
    import getopt
    import time

    usage = """usage: %s [-f] [-b] [-i iterations] file
            -b use big record (Particle); else use small record (Record)
            -f means use fast methods (unsafer)
            -i sets the number of rows in each table\n""" % sys.argv[0]

    try:
        opts, pargs = getopt.getopt(sys.argv[1:], 'bfi:')
    except:
        sys.stderr.write(usage)
        sys.exit(0)

    # if we pass too much parameters, abort
    if len(pargs) <> 1: 
        sys.stderr.write(usage)
        sys.exit(0)

    # default options
    bigrec = 0
    fast = 0
    iterations = 100

    # Get the options
    for option in opts:
        if option[0] == '-b':
            bigrec = 1
        if option[0] == '-f':
            fast = 1
        if option[0] == '-i':
            iterations = int(option[1])

    # Catch the hdf5 file passed as the last argument
    file = pargs[0]

    t1 = time.clock()
    createFile(file, iterations, fast, bigrec)
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

    print "Time appending rows:", tapprows
    print "Write rows/sec: ", int(iterations * 3/ float(tapprows))
    print "Time reading rows:", treadrows
    print "Read rows/sec: ", int(iterations * 3/ float(treadrows))
