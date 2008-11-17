# Script for creating different kind of indexes in a small space as possible.
# This is intended for testing purposes.

from tables import *

class Descr(IsDescription):
    var1 = StringCol(itemsize=4, shape=(), dflt='', pos=0)
    var2 = BoolCol(shape=(), dflt=False, pos=1)
    var3 = Int32Col(shape=(), dflt=0, pos=2)
    var4 = Float64Col(shape=(), dflt=0.0, pos=3)

# Parameters for the table and index creation
small_chunkshape = (2,)
small_blocksizes = (64, 32, 16, 8)
nrows = 43

# Create the new file
f = openFile('indexes_2_1.h5', 'w')
t1 = f.createTable(f.root, 'table1', Descr)
row = t1.row
for i in range(nrows):
    row['var1'] = i
    row['var2'] = i
    row['var3'] = i
    row['var4'] = i
    row.append()
t1.flush()

# Do a copy of table1
t1.copy(f.root, 'table2')

# Create indexes of all kinds
t1.cols.var1.createIndex(0,'ultralight',_blocksizes=small_blocksizes)
t1.cols.var2.createIndex(3,'light',_blocksizes=small_blocksizes)
t1.cols.var3.createIndex(6,'medium',_blocksizes=small_blocksizes)
t1.cols.var4.createIndex(9,'full',_blocksizes=small_blocksizes)

f.close()
