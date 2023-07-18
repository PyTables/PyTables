# Testbed to perform experiments in order to determine best values for
# the node numbers in LRU cache. Tables version.

from time import perf_counter as clock
import tables as tb

print("PyTables version-->", tb.__version__)

filename = "/tmp/junk-tables-100.h5"
NLEAVES = 2000
NROWS = 1000


class Particle(tb.IsDescription):
    name = tb.StringCol(16, pos=1)         # 16-character String
    lati = tb.Int32Col(pos=2)              # integer
    longi = tb.Int32Col(pos=3)             # integer
    pressure = tb.Float32Col(pos=4)        # float  (single-precision)
    temperature = tb.Float64Col(pos=5)     # double (double-precision)


def create_junk():
    # Open a file in "w"rite mode
    fileh = tb.open_file(filename, mode="w")
    # Create a new group
    group = fileh.create_group(fileh.root, "newgroup")

    for i in range(NLEAVES):
        # Create a new table in newgroup group
        table = fileh.create_table(group, 'table' + str(i), Particle,
                                   "A table", tb.Filters(1))
        particle = table.row
        print("Creating table-->", table._v_name)

        # Fill the table with particles
        for i in range(NROWS):
            # This injects the row values.
            particle.append()
        table.flush()

    # Finally, close the file
    fileh.close()


def modify_junk_LRU():
    fileh = tb.open_file(filename, 'a')
    group = fileh.root.newgroup
    for j in range(5):
        print("iter -->", j)
        for tt in fileh.walk_nodes(group):
            if isinstance(tt, tb.Table):
                pass
#                 for row in tt:
#                     pass
    fileh.close()


def modify_junk_LRU2():
    fileh = tb.open_file(filename, 'a')
    group = fileh.root.newgroup
    for j in range(20):
        t1 = clock()
        for i in range(100):
            #print("table-->", tt._v_name)
            tt = getattr(group, "table" + str(i))
            #for row in tt:
            #    pass
        print(f"iter and time --> {j + 1} {clock() - t1:.3f}")
    fileh.close()


def modify_junk_LRU3():
    fileh = tb.open_file(filename, 'a')
    group = fileh.root.newgroup
    for j in range(3):
        t1 = clock()
        for tt in fileh.walk_nodes(group, "Table"):
            tt.attrs.TITLE
            for row in tt:
                pass
        print(f"iter and time --> {j + 1} {clock() - t1:.3f}")
    fileh.close()

if 1:
    # create_junk()
    # modify_junk_LRU()    # uses the iterator version (walk_nodes)
    # modify_junk_LRU2()   # uses a regular loop (getattr)
    modify_junk_LRU3()   # uses a regular loop (getattr)
else:
    import profile
    import pstats
    profile.run('modify_junk_LRU2()', 'modify.prof')
    stats = pstats.Stats('modify.prof')
    stats.strip_dirs()
    stats.sort_stats('time', 'calls')
    stats.print_stats()
