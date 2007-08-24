import numpy
import tables
from time import time
import psyco

filename = "/tmp/LRU-bench.h5"
nodespergroup = 250
niter = 100

f = tables.openFile(filename, "w")
g = f.createGroup("/", "NodeContainer")
print "Creating nodes"
for i in range(nodespergroup):
    f.createArray(g, "arr%d"%i, [i])
f.close()

f = tables.openFile(filename)

def iternodes():
#     for a in f.root.NodeContainer:
#         pass
    indices = numpy.random.randn(nodespergroup*niter)*30+nodespergroup/2.
    indices = indices.astype('i4').clip(0, nodespergroup-1)
    g = f.getNode("/", "NodeContainer")
    for i in indices:
        a = f.getNode(g, "arr%d"%i)
        #print "a-->", a

print "reading nodes..."
# First iteration (put in LRU cache)
t1 = time()
for a in f.root.NodeContainer:
    pass
print "time (init cache)-->", round(time()-t1, 3)

def timeLRU():
    # Next iterations
    t1 = time()
#     for i in range(niter):
#         iternodes()
    iternodes()
    print "time (from cache)-->", round((time()-t1)/niter, 3)

def profile(verbose=False):
    import pstats
    import cProfile as prof
    prof.run('timeLRU()', 'out.prof')
    stats = pstats.Stats('out.prof')
    stats.strip_dirs()
    stats.sort_stats('time', 'calls')
    if verbose:
        stats.print_stats()
    else:
        stats.print_stats(20)

#profile()
#psyco.bind(timeLRU)
timeLRU()

f.close()
