# This is an example on how to use complex columns
from __future__ import print_function
import numpy as np
import tables

dt = np.dtype('i4,f8', align=True)

# Open a file in "w"rite mode
fileh = tables.open_file("attrs-with-padding.h5", mode="w", pytables_sys_attrs=False)
attrs = fileh.root._v_attrs
# Set some attrs
attrs.pq = np.zeros(2, dt)
attrs.qr = np.ones((2, 2), dt)
attrs.rs = np.array([(1, 2.)], dt)

print("str(attrs)-->", attrs)
print("repr(attrs)-->", repr(attrs))
print("attributes:")
# for attr in attrs:
#     print(attr[:])

fileh.close()

print("   ***After closing***")

fileh = tables.open_file("attrs-with-padding.h5", mode="r")
attrs = fileh.root._v_attrs

print("str(attrs)-->", attrs)
print("repr(attrs)-->", repr(attrs))
print("attributes:")
# for attr in attrs:
#     print(attr[:])

fileh.close()
