"""Small benchmark on the effect of chunksizes and compression on HDF5 files.

This script is meant to be used on a Linux kernel > 2.6.16 because of
the trick used to empty the OS pagecache.  Because of this trick, it
does need to be run as root!

Francesc Altet
2007-11-25
"""

import os, math, subprocess
from time import time
import numpy
import tables

#N, M = 512, 2**16
N, M = 512, 2**19
#N, M = 2000, 1000000
datom = tables.Float64Atom()

def quantize(data, least_significant_digit):
    """quantize data to improve compression.

    data is quantized using around(scale*data)/scale, where scale is
    2**bits, and bits is determined from the least_significant_digit.
    For example, if least_significant_digit=1, bits will be 4."""

    precision = 10.**-least_significant_digit
    exp = math.log(precision,10)
    if exp < 0:
        exp = int(math.floor(exp))
    else:
        exp = int(math.ceil(exp))
    bits = math.ceil(math.log(10.**-exp,2))
    scale = 2.**bits
    return numpy.around(scale*data)/scale


def get_db_size(filename):
    sout = subprocess.Popen("ls -sh %s" % filename, shell=True,
                            stdout=subprocess.PIPE).stdout
    line = [l for l in sout][0]
    return line.split()[0]


def bench(chunkshape, filters):
    numpy.random.seed(1)   # to have reproductible results
    #filename = '/oldScr/ivilata/pytables/data.nobackup/test.h5'
    #filename = '/scratch2/faltet/data.nobackup/test.h5'
    filename = '/scratch3/faltet/test.h5'
    f = tables.openFile(filename, 'w')
    e = f.createEArray(f.root, 'earray', datom, shape=(0, M),
                       filters = filters,
                       chunkshape = chunkshape)
    # Fill the array
    t1 = time()
    for i in xrange(N):
        #e.append([numpy.random.rand(M)])  # use this for less compressibility
        e.append([quantize(numpy.random.rand(M), 6)])
    os.system("sync")
    print "Creation time:", round(time()-t1, 3),
    filesize = get_db_size(filename)
    filesize_bytes = os.stat(filename)[6]
    print "\t\tFile size: %d -- (%s)" % (filesize_bytes, filesize)

    # Read in sequential mode:
    t1 = time()
    # Flush everything to disk and flush caches
    os.system("sync; echo 1 > /proc/sys/vm/drop_caches")
    for row in e:
        t = row
    print "Sequential read time:", round(time()-t1, 3),

    # Read in random mode:
    i_index = numpy.random.randint(0, N, 128)
    j_index = numpy.random.randint(0, M, 256)
    # Flush everything to disk and flush caches
    os.system("sync; echo 1 > /proc/sys/vm/drop_caches")
    t1 = time()
    for i in i_index:
        for j in j_index:
            t = e[i,j]
    print "\tRandom read time:", round(time()-t1, 3)

    f.close()

# Benchmark with different chunksizes and filters
#for complevel in (0, 1, 3, 6, 9):
for complib in (None, 'zlib', 'lzo'):
    if complib:
        filters = tables.Filters(complevel=1, complib=complib)
    else:
        filters = tables.Filters(complevel=0)
    print "8<--"*20, "\nFilters:", filters, "\n"+"-"*80
    #for ecs in (11, 14, 17, 20):
    for ecs in range(10, 24):
        chunksize = 2**ecs
        chunk1 = 1
        chunk2 = chunksize/datom.itemsize
        if chunk2 > M:
            chunk1 = chunk2 / M
            chunk2 = M
        chunkshape = (chunk1, chunk2)
        cs_str = str(chunksize / 1024) + " KB"
        print "***** Chunksize:", cs_str, "/ Chunkshape:", chunkshape, "*****"
        bench(chunkshape, filters)
