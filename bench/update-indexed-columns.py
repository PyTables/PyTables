# This is a small benchmark in order to detect whether the index code
# would unnecessarily reindex updated rows that don't affect indexed
# columns.  This is to cope with bug #139.
# Francesc Alted
# 2007-12-17

import sys
import numpy
from tables import *
from time import time

class Record(IsDescription):
    col1 = BoolCol()
    col2 = IntCol()
    col3 = FloatCol()
    col4 = StringCol(itemsize=4)

# Create a table and fill it with some values
f = openFile("update-indexed-columns.h5", "w")
t = f.createTable(f.root, "table", Record)
row = t.row
print "Filling table",
t1 = time()
for i in xrange(10**5):
    row['col2'] = i
    row.append()
t.flush()
print round(time()-t1, 3)

print "indexing col2",
t1 = time()
t.cols.col2.createIndex()
tidx = time()-t1
print round(tidx, 3)

print "indexing col4",
t1 = time()
t.cols.col4.createIndex()
tidx2 = time()-t1
print round(tidx2, 3)

# Modify some indexed columns (update method)
t1 = time()
for row in t.iterrows(1,10):
    row['col2'] = 1
    row['col4'] = "Hola"
    row.update()
t.flush()
t2 = time()-t1
print "Time for modifying indexed columns (update):", round(t2, 3)
if t2 < 0.1*tidx:
    print "WARNING! Necessary re-indexing in update method not done."

# Modify some non-indexed columns (update method)
t1 = time()
for row in t.iterrows(1,10):
    row['col1'] = True
    row['col3'] = 1.0
    row.update()
t.flush()
t2 = time()-t1
print "Time for modifying non-indexed columns (update):", round(t2, 3)
if t2 > 0.1*tidx:
    print "WARNING! Possible unnecessary re-indexing in update method."

# Modify some indexed columns (modifyColumn method)
t1 = time()
t.modifyColumn(0,1,10,[0],'col2')
t2 = time()-t1
print "Time for modifying indexed columns (modifyColumn):", round(t2, 3)
if t2 < 0.1*tidx:
    print "WARNING! Necessary re-indexing in modifyColumn method no done."

# Modify some un-indexed columns (modifyColumn method)
t1 = time()
t.modifyColumn(0,1,10,[0],'col3')
t2 = time()-t1
print "Time for modifying non-indexed columns (modifyColumn):", round(t2, 3)
if t2 > 0.1*tidx:
    print "WARNING! Possible unnecessary re-indexing in modifyColumn method."

# Modify some indexed columns (modifyColumns method)
t1 = time()
t.modifyColumns(0,1,1, numpy.array([(0,"Hol")], dtype="i4,S4"), ['col2','col4'])
t2 = time()-t1
print "Time for modifying indexed columns (modifyColumns):", round(t2, 3)
if t2 < 0.1*tidx:
    print "WARNING! Necessary re-indexing in modifyColumns method no done."

# Modify some un-indexed columns (modifyColumns method)
t1 = time()
t.modifyColumns(0,1,1, numpy.array([(0,1)], dtype="b1,f8"), ['col1','col3'])
t2 = time()-t1
print "Time for modifying non-indexed columns (modifyColumns):", round(t2, 3)
if t2 > 0.1*tidx:
    print "WARNING! Possible unnecessary re-indexing in modifyColumns method."

f.close()
