from __future__ import print_function
import numpy as np
import tables
from time import time

N = 1000 * 1000
NCOLL = 200  # 200 collections maximum

# In order to have reproducible results
np.random.seed(19)


class Energies(tables.IsDescription):
    collection = tables.UInt8Col()
    energy = tables.Float64Col()


def fill_bucket(lbucket):
    #c = np.random.normal(NCOLL/2, NCOLL/10, lbucket)
    c = np.random.normal(NCOLL / 2, NCOLL / 100, lbucket)
    e = np.arange(lbucket, dtype='f8')
    return c, e

# Fill the table
t1 = time()
f = tables.open_file("data.nobackup/collations.h5", "w")
table = f.create_table("/", "Energies", Energies, expectedrows=N)
# Fill the table with values
lbucket = 1000   # Fill in buckets of 1000 rows, for speed
for i in range(0, N, lbucket):
    bucket = fill_bucket(lbucket)
    table.append(bucket)
# Fill the remaining rows
bucket = fill_bucket(N % lbucket)
table.append(bucket)
f.close()
print("Time to create the table with %d entries: %.3f" % (N, time() - t1))

# Now, read the table and group it by collection
f = tables.open_file("data.nobackup/collations.h5", "a")
table = f.root.Energies

#########################################################
# First solution: load the table completely in memory
#########################################################
t1 = time()
t = table[:]  # convert to structured array
coll1 = []
collections = np.unique(t['collection'])
for c in collections:
    cond = t['collection'] == c
    energy_this_collection = t['energy'][cond]
    sener = energy_this_collection.sum()
    coll1.append(sener)
    print(c, ' : ', sener)
del collections, energy_this_collection
print("Time for first solution: %.3f" % (time() - t1))

#########################################################
# Second solution: load all the collections in memory
#########################################################
t1 = time()
collections = {}
for row in table:
    c = row['collection']
    e = row['energy']
    if c in collections:
        collections[c].append(e)
    else:
        collections[c] = [e]
# Convert the lists in numpy arrays
coll2 = []
for c in sorted(collections):
    energy_this_collection = np.array(collections[c])
    sener = energy_this_collection.sum()
    coll2.append(sener)
    print(c, ' : ', sener)
del collections, energy_this_collection
print("Time for second solution: %.3f" % (time() - t1))

t1 = time()
table.cols.collection.create_csindex()
# table.cols.collection.reindex()
print("Time for indexing: %.3f" % (time() - t1))

#########################################################
# Third solution: load each collection separately
#########################################################
t1 = time()
coll3 = []
for c in np.unique(table.col('collection')):
    energy_this_collection = table.read_where(
        'collection == c', field='energy')
    sener = energy_this_collection.sum()
    coll3.append(sener)
    print(c, ' : ', sener)
del energy_this_collection
print("Time for third solution: %.3f" % (time() - t1))


t1 = time()
table2 = table.copy('/', 'EnergySortedByCollation', overwrite=True,
                    sortby="collection", propindexes=True)
print("Time for sorting: %.3f" % (time() - t1))

#####################################################################
# Fourth solution: load each collection separately.  Sorted table.
#####################################################################
t1 = time()
coll4 = []
for c in np.unique(table2.col('collection')):
    energy_this_collection = table2.read_where(
        'collection == c', field='energy')
    sener = energy_this_collection.sum()
    coll4.append(sener)
    print(c, ' : ', sener)
    del energy_this_collection
print("Time for fourth solution: %.3f" % (time() - t1))


# Finally, check that all solutions do match
assert coll1 == coll2 == coll3 == coll4

f.close()
