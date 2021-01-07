"""Small benchmark on the effect of chunksizes and compression on HDF5 files.

Francesc Alted
2007-11-25

"""

import math
import subprocess
import tempfile
from pathlib import Path
from time import perf_counter as clock
import numpy as np
import tables as tb

# Size of dataset
# N, M = 512, 2**16     # 256 MB
# N, M = 512, 2**18     # 1 GB
# N, M = 512, 2**19     # 2 GB
N, M = 2000, 1_000_000  # 15 GB
# N, M = 4000, 1000000  # 30 GB
datom = tb.Float64Atom()   # elements are double precision


def quantize(data, least_significant_digit):
    """Quantize data to improve compression.

    data is quantized using around(scale*data)/scale, where scale is
    2**bits, and bits is determined from the least_significant_digit.
    For example, if least_significant_digit=1, bits will be 4.

    """

    precision = 10 ** -least_significant_digit
    exp = math.log(precision, 10)
    if exp < 0:
        exp = math.floor(exp)
    else:
        exp = math.ceil(exp)
    bits = math.ceil(math.log(10 ** -exp, 2))
    scale = 2 ** bits
    return np.around(scale * data) / scale


def get_db_size(filename):
    sout = subprocess.Popen("ls -sh %s" % filename, shell=True,
                            stdout=subprocess.PIPE).stdout
    line = [l for l in sout][0]
    return line.split()[0]


def bench(chunkshape, filters):
    np.random.seed(1)   # to have reproductible results
    filename = tempfile.mktemp(suffix='.h5')
    print("Doing test on the file system represented by:", filename)

    f = tb.open_file(filename, 'w')
    e = f.create_earray(f.root, 'earray', datom, shape=(0, M),
                        filters = filters,
                        chunkshape = chunkshape)
    # Fill the array
    t1 = clock()
    for i in range(N):
        # e.append([numpy.random.rand(M)])  # use this for less compressibility
        e.append([quantize(np.random.rand(M), 6)])
    # os.system("sync")
    print(f"Creation time: {clock() - t1:.3f}", end=' ')
    filesize = get_db_size(filename)
    filesize_bytes = Path(filename).stat().st_size
    print("\t\tFile size: %d -- (%s)" % (filesize_bytes, filesize))

    # Read in sequential mode:
    e = f.root.earray
    t1 = clock()
    # Flush everything to disk and flush caches
    #os.system("sync; echo 1 > /proc/sys/vm/drop_caches")
    for row in e:
        t = row
    print(f"Sequential read time: {clock() - t1:.3f}", end=' ')

    # f.close()
    # return

    # Read in random mode:
    i_index = np.random.randint(0, N, 128)
    j_index = np.random.randint(0, M, 256)
    # Flush everything to disk and flush caches
    #os.system("sync; echo 1 > /proc/sys/vm/drop_caches")

    # Protection against too large chunksizes
    # 4 MB
    if 0 and filters.complevel and chunkshape[0] * chunkshape[1] * 8 > 2 ** 22:
        f.close()
        return

    t1 = clock()
    for i in i_index:
        for j in j_index:
            t = e[i, j]
    print(f"\tRandom read time: {clock() - t1:.3f}")

    f.close()

# Benchmark with different chunksizes and filters
# for complevel in (0, 1, 3, 6, 9):
for complib in (None, 'zlib', 'lzo', 'blosc'):
# for complib in ('blosc',):
    if complib:
        filters = tb.Filters(complevel=5, complib=complib)
    else:
        filters = tb.Filters(complevel=0)
    print("8<--" * 20, "\nFilters:", filters, "\n" + "-" * 80)
    # for ecs in (11, 14, 17, 20, 21, 22):
    for ecs in range(10, 24):
    # for ecs in (19,):
        chunksize = 2 ** ecs
        chunk1 = 1
        chunk2 = chunksize / datom.itemsize
        if chunk2 > M:
            chunk1 = chunk2 / M
            chunk2 = M
        chunkshape = (chunk1, chunk2)
        cs_str = str(chunksize / 1024) + " KB"
        print("***** Chunksize:", cs_str, "/ Chunkshape:", chunkshape, "*****")
        bench(chunkshape, filters)
