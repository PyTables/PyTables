# This is an example on how to use complex columns
from __future__ import print_function
import numpy as np
import tables

dt = np.dtype('i4,f8', align=True)

# Create a file with regular padding
fileh = tables.open_file("attrs-with-padding.h5", mode="w", pytables_sys_attrs=False)
attrs = fileh.root._v_attrs
# Set some attrs
attrs.pq = np.zeros(2, dt)
attrs.qr = np.ones((2, 2), dt)
attrs.rs = np.array([(1, 2.)], dt)
print("attrs *with* padding:")
print("repr(attrs)-->", repr(attrs))

fileh.close()

# Create a file with no padding
fileh = tables.open_file("attrs-without-padding.h5", mode="w", pytables_sys_attrs=False, allow_padding=False)
attrs = fileh.root._v_attrs
# Set some attrs
attrs.pq = np.zeros(2, dt)
attrs.qr = np.ones((2, 2), dt)
attrs.rs = np.array([(1, 2.)], dt)
print("\nattrs *without* padding:")
print("repr(attrs)-->", repr(attrs))

fileh.close()

print("\n   ***After closing***\n")

fileh = tables.open_file("attrs-with-padding.h5", mode="r")
attrs = fileh.root._v_attrs
print("attrs *with* padding:")
print("repr(attrs)-->", repr(attrs))

fileh.close()

fileh = tables.open_file("attrs-without-padding.h5", mode="r")
attrs = fileh.root._v_attrs
print("\nattrs *without* padding:")
print("repr(attrs)-->", repr(attrs))

fileh.close()
