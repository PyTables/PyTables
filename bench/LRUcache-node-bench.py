import sys
from time import perf_counter as clock

import numpy as np

import tables as tb

# import psyco

filename = "/tmp/LRU-bench.h5"
nodespergroup = 250
niter = 100

print("nodespergroup:", nodespergroup)
print("niter:", niter)

if len(sys.argv) > 1:
    NODE_CACHE_SLOTS = int(sys.argv[1])
    print("NODE_CACHE_SLOTS:", NODE_CACHE_SLOTS)
else:
    NODE_CACHE_SLOTS = tb.parameters.NODE_CACHE_SLOTS
f = tb.open_file(filename, "w", node_cache_slots=NODE_CACHE_SLOTS)
g = f.create_group("/", "NodeContainer")
print("Creating nodes")
for i in range(nodespergroup):
    f.create_array(g, "arr%d" % i, [i])
f.close()

f = tb.open_file(filename)


def iternodes():
    #     for a in f.root.NodeContainer:
    #         pass
    indices = np.random.randn(nodespergroup * niter) * 30 + nodespergroup / 2
    indices = indices.astype("i4").clip(0, nodespergroup - 1)
    g = f.get_node("/", "NodeContainer")
    for i in indices:
        _ = f.get_node(g, "arr%d" % i)
        # print("a-->", _)


print("reading nodes...")
# First iteration (put in LRU cache)
t1 = clock()
for a in f.root.NodeContainer:
    pass
print(f"time (init cache)--> {clock() - t1:.3f}")


def time_lru():
    # Next iterations
    t1 = clock()
    #     for i in range(niter):
    #         iternodes()
    iternodes()
    print(f"time (from cache)--> {(clock() - t1) / niter:.3f}")


def profile(verbose=False):
    import pstats
    import cProfile

    cProfile.run("timeLRU()", "out.prof")
    stats = pstats.Stats("out.prof")
    stats.strip_dirs()
    stats.sort_stats("time", "calls")
    if verbose:
        stats.print_stats()
    else:
        stats.print_stats(20)


# profile()
# psyco.bind(timeLRU)
time_lru()

f.close()

# for N in 0 4 8 16 32 64 128 256 512 1024 2048 4096; do
#     env PYTHONPATH=../build/lib.linux-x86_64-2.7 \
#     python LRUcache-node-bench.py $N;
# done
