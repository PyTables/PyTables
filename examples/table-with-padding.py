# This is an example on how to use complex columns
from __future__ import print_function
import numpy as np
import tables

N = 1000

padded_dtype = np.dtype([('string', 'S3'), ('int', 'i4'), ('double', 'f8')], align=True)
#assert padded_dtype.itemsize == 16
padded_struct = np.zeros(N, padded_dtype)

padded_struct['string'] = np.arange(N).astype('S3')
padded_struct['int'] = np.arange(N, dtype='i4')
padded_struct['double'] = np.arange(N, dtype='f8')

# Open a file in "w"rite mode
fileh = tables.open_file("table-with-padding.h5", mode="w", pytables_sys_attrs=False)
table = fileh.create_table(fileh.root, 'table', padded_struct, "A table with padding")
print("str(Cols)-->", table.cols)
print("repr(Cols)-->", repr(table.cols))
print("Column handlers:")
for name in table.colnames:
    print(table.cols._f_col(name))

fileh.close()

print("   ***After closing***")

fileh = tables.open_file("table-with-padding.h5", mode="r")
table = fileh.root.table
print("str(Cols)-->", table.cols)
print("repr(Cols)-->", repr(table.cols))
print("Column handlers:")
for name in table.colnames:
    print(table.cols._f_col(name))
print("col 'string' ->", table.cols.string[0:10])

fileh.close()
