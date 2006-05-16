# Testbed to perform experiments in order to determine best values for
# the node numbers in LRU cache. Arrays version.

from time import time
import tables
print "PyTables version-->", tables.__version__

filename = "/tmp/junk-array.h5"
NOBJS = 1000

def create_junk():
    fileh = tables.openFile(filename, mode = "w")
    for i in xrange(NOBJS):
        table = fileh.createArray(fileh.root, 'array'+str(i), [1])
    fileh.close()

def modify_junk_LRU():
    fileh = tables.openFile(filename,'a')
    group = fileh.root
    for j in range(5):
        print "iter -->", j
        for tt in fileh.walkNodes(group):
            if isinstance(tt,tables.Array):
#                 d = tt.read()
                pass

    fileh.close()

def modify_junk_LRU2():
    fileh = tables.openFile(filename,'a')
    group = fileh.root
    for j in range(5):
        t1 = time()
        for i in range(100):  # The number
#              print "table-->", tt._v_name
            tt = getattr(group,"array"+str(i))
#                 d = tt.read()
        print "iter and time -->", j+1, round(time()-t1,3)
    fileh.close()

if 1:
    #create_junk()
    #modify_junk_LRU()    # uses the iterador version (walkNodes)
    modify_junk_LRU2()   # uses a regular loop (getattr)
else:
    import profile, pstats
    profile.run('modify_junk_LRU2()', 'modify.prof')
    stats = pstats.Stats('modify.prof')
    stats.strip_dirs()
    stats.sort_stats('time', 'calls')
    stats.print_stats()
