from __future__ import print_function
import tables


class MyClass(object):
    foo = 'bar'

# An object of my custom class.
myObject = MyClass()

h5f = tables.open_file('test.h5', 'w')
h5f.root._v_attrs.obj = myObject  # store the object
print(h5f.root._v_attrs.obj.foo)  # retrieve it
h5f.close()

# Delete class of stored object and reopen the file.
del MyClass, myObject

h5f = tables.open_file('test.h5', 'r')
print(h5f.root._v_attrs.obj.foo)
# Let us inspect the object to see what is happening.
print(repr(h5f.root._v_attrs.obj))
# Maybe unpickling the string will yield more information:
import pickle
pickle.loads(h5f.root._v_attrs.obj)
# So the problem was not in the stored object,
# but in the *environment* where it was restored.
h5f.close()
