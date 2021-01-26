# This is an example on how to use complex columns
import numpy as np
import tables as tb

dt = np.dtype('i4,f8', align=True)

# Create a file with regular padding
print("attrs *with* padding:")
fileh = tb.open_file("attrs-with-padding.h5", mode="w", pytables_sys_attrs=False)
attrs = fileh.root._v_attrs
# Set some attrs
attrs.pq = np.zeros(2, dt)
attrs.qr = np.ones((2, 2), dt)
attrs.rs = np.array([(1, 2)], dt)
print("repr(attrs)-->", repr(attrs))

fileh.close()

# Create a file with no padding
print("\nattrs *without* padding:")
fileh = tb.open_file("attrs-without-padding.h5", mode="w", pytables_sys_attrs=False, allow_padding=False)
attrs = fileh.root._v_attrs
# Set some attrs
attrs.pq = np.zeros(2, dt)
attrs.qr = np.ones((2, 2), dt)
attrs.rs = np.array([(1, 2)], dt)
print("repr(attrs)-->", repr(attrs))

fileh.close()

print("\n   ***After closing***\n")

print("attrs *with* padding:")
fileh = tb.open_file("attrs-with-padding.h5", mode="r")
attrs = fileh.root._v_attrs
print("repr(attrs)-->", repr(attrs))

fileh.close()

print("\nattrs *without* padding:")
fileh = tb.open_file("attrs-without-padding.h5", mode="r")
attrs = fileh.root._v_attrs
print("repr(attrs)-->", repr(attrs))

fileh.close()
