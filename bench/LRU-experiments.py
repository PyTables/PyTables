# Testbed to perform experiments in order to determine best values for
# the node numbers in LRU cache. Tables version.

from time import time
from tables import *
import tables

print "PyTables version-->", tables.__version__

filename = "/tmp/junk-tables-100.h5"
NLEAVES = 2000
NROWS = 1000

class Particle(IsDescription):
    name        = StringCol(16, pos=1)   # 16-character String
    lati        = Int32Col(pos=2)        # integer
    longi       = Int32Col(pos=3)        # integer
    pressure    = Float32Col(pos=4)      # float  (single-precision)
    temperature = Float64Col(pos=5)      # double (double-precision)

def create_junk():
    # Open a file in "w"rite mode
    fileh = openFile(filename, mode = "w")
    # Create a new group
    group = fileh.createGroup(fileh.root, "newgroup")

    for i in xrange(NLEAVES):
        # Create a new table in newgroup group
        table = fileh.createTable(group, 'table'+str(i), Particle,
                                  "A table", Filters(1))
        particle = table.row
        print "Creating table-->", table._v_name

        # Fill the table with particles
        for i in xrange(NROWS):
            # This injects the row values.
            particle.append()
        table.flush()

    # Finally, close the file
    fileh.close()

def modify_junk_LRU():
    fileh = openFile(filename,'a')
    group = fileh.root.newgroup
    for j in range(5):
        print "iter -->", j
        for tt in fileh.walkNodes(group):
            if isinstance(tt,Table):
                pass
#                 for row in tt:
#                     pass
    fileh.close()

def modify_junk_LRU2():
    fileh = openFile(filename,'a')
    group = fileh.root.newgroup
    for j in range(20):
        t1 = time()
        for i in range(100):
#              print "table-->", tt._v_name
            tt = getattr(group,"table"+str(i))
#             for row in tt:
#                 pass
        print "iter and time -->", j+1, round(time()-t1,3)
    fileh.close()

def modify_junk_LRU3():
    fileh = openFile(filename,'a')
    group = fileh.root.newgroup
    for j in range(3):
        t1 = time()
        for tt in fileh.walkNodes(group, "Table"):
            title = tt.attrs.TITLE
            for row in tt:
                pass
        print "iter and time -->", j+1, round(time()-t1,3)
    fileh.close()

if 1:
    #create_junk()
    #modify_junk_LRU()    # uses the iterator version (walkNodes)
    #modify_junk_LRU2()   # uses a regular loop (getattr)
    modify_junk_LRU3()   # uses a regular loop (getattr)
else:
    import profile, pstats
    profile.run('modify_junk_LRU2()', 'modify.prof')
    stats = pstats.Stats('modify.prof')
    stats.strip_dirs()
    stats.sort_stats('time', 'calls')
    stats.print_stats()
