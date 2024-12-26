# Testbed to perform experiments in order to determine best values for
# the node numbers in LRU cache. Arrays version.

from time import perf_counter as clock

import tables as tb

print("PyTables version-->", tb.__version__)

filename = "/tmp/junk-array.h5"
NOBJS = 1000


def create_junk():
    fileh = tb.open_file(filename, mode="w")
    for i in range(NOBJS):
        fileh.create_array(fileh.root, "array" + str(i), [1])
    fileh.close()


def modify_junk_lru():
    fileh = tb.open_file(filename, "a")
    group = fileh.root
    for j in range(5):
        print("iter -->", j)
        for tt in fileh.walk_nodes(group):
            if isinstance(tt, tb.Array):
                #                 d = tt.read()
                pass

    fileh.close()


def modify_junk_lru2():
    fileh = tb.open_file(filename, "a")
    group = fileh.root
    for j in range(5):
        t1 = clock()
        for i in range(100):  # The number
            # print("table-->", tt._v_name)
            _ = getattr(group, "array" + str(i))
            # d = _.read()
        print(f"iter and time --> {j + 1} {clock() - t1:.3f}")
    fileh.close()


if 1:
    # create_junk()
    # modify_junk_LRU()    # uses the iterador version (walk_nodes)
    modify_junk_lru2()  # uses a regular loop (getattr)
else:
    import pstats
    import profile

    profile.run("modify_junk_LRU2()", "modify.prof")
    stats = pstats.Stats("modify.prof")
    stats.strip_dirs()
    stats.sort_stats("time", "calls")
    stats.print_stats()
