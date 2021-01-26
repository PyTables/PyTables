# This is an example on how to use complex columns
import numpy as np
import tables as tb

N = 1000

padded_dtype = np.dtype([('string', 'S3'), ('int', 'i4'), ('double', 'f8')], align=True)
#assert padded_dtype.itemsize == 16
padded_struct = np.zeros(N, padded_dtype)

padded_struct['string'] = np.arange(N).astype('S3')
padded_struct['int'] = np.arange(N, dtype='i4')
padded_struct['double'] = np.arange(N, dtype='f8')

# Create a file with padding (the default)
fileh = tb.open_file("tables-with-padding.h5", mode="w", pytables_sys_attrs=False)
table = fileh.create_table(fileh.root, 'table', padded_struct, "A table with padding")
print("table *with* padding -->", table)
print("table.description --> ", table.description)
print("table.descrition._v_offsets-->", table.description._v_offsets)
print("table.descrition._v_itemsize-->", table.description._v_itemsize)

fileh.close()

# Create another file without padding
fileh = tb.open_file("tables-without-padding.h5", mode="w", pytables_sys_attrs=False, allow_padding=False)
table = fileh.create_table(fileh.root, 'table', padded_struct, "A table without padding")
print("\ntable *without* padding -->", table)
print("table.description --> ", table.description)
print("table.descrition._v_offsets-->", table.description._v_offsets)
print("table.descrition._v_itemsize-->", table.description._v_itemsize)

fileh.close()

print("\n   ***After closing***\n")

fileh = tb.open_file("tables-with-padding.h5", mode="r")
table = fileh.root.table
print("table *with* padding -->", table)
print("table.description --> ", table.description)
print("table.descrition._v_offsets-->", table.description._v_offsets)
print("table.descrition._v_itemsize-->", table.description._v_itemsize)

fileh.close()

fileh = tb.open_file("tables-without-padding.h5", mode="r")
table = fileh.root.table
print("\ntable *without* padding -->", table)
print("table.description --> ", table.description)
print("table.descrition._v_offsets-->", table.description._v_offsets)
print("table.descrition._v_itemsize-->", table.description._v_itemsize)

fileh.close()
