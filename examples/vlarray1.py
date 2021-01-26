import tables as tb
import numpy as np

# Create a VLArray:
fileh = tb.open_file('vlarray1.h5', mode='w')
vlarray = fileh.create_vlarray(fileh.root, 'vlarray1',
                               tb.Int32Atom(shape=()),
                               "ragged array of ints",
                               filters=tb.Filters(1))
# Append some (variable length) rows:
vlarray.append(np.array([5, 6]))
vlarray.append(np.array([5, 6, 7]))
vlarray.append([5, 6, 9, 8])

# Now, read it through an iterator:
print('-->', vlarray.title)
for x in vlarray:
    print('%s[%d]--> %s' % (vlarray.name, vlarray.nrow, x))

# Now, do the same with native Python strings.
vlarray2 = fileh.create_vlarray(fileh.root, 'vlarray2',
                                tb.StringAtom(itemsize=2),
                                "ragged array of strings",
                                filters=tb.Filters(1))
vlarray2.flavor = 'python'
# Append some (variable length) rows:
print('-->', vlarray2.title)
vlarray2.append(['5', '66'])
vlarray2.append(['5', '6', '77'])
vlarray2.append(['5', '6', '9', '88'])

# Now, read it through an iterator:
for x in vlarray2:
    print('%s[%d]--> %s' % (vlarray2.name, vlarray2.nrow, x))

# Close the file.
fileh.close()
