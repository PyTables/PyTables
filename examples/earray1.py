import tables as tb
import numpy as np

fileh = tb.open_file('earray1.h5', mode='w')
a = tb.StringAtom(itemsize=8)
# Use ``a`` as the object type for the enlargeable array.
array_c = fileh.create_earray(fileh.root, 'array_c', a, (0,), "Chars")
array_c.append(np.array(['a' * 2, 'b' * 4], dtype='S8'))
array_c.append(np.array(['a' * 6, 'b' * 8, 'c' * 10], dtype='S8'))

# Read the string ``EArray`` we have created on disk.
for s in array_c:
    print(f'array_c[{array_c.nrow}] => {s!r}')
# Close the file.
fileh.close()
