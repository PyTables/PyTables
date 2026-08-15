import tables as tb


class MyClass:
    foo = "bar"


# An object of my custom class.
my_object = MyClass()

with tb.open_file("test.h5", "w") as h5f:
    h5f.root._v_attrs.obj = my_object  # store the object
    print(h5f.root._v_attrs.obj.foo)  # retrieve it

# Automatic unpickling is disabled, so the raw payload is returned.
with tb.open_file("test.h5", "r") as h5f:
    print(repr(h5f.root._v_attrs.obj))

# Only enable automatic unpickling for files from trusted sources.
with tb.open_file("test.h5", "r", allow_pickle=True) as h5f:
    print(h5f.root._v_attrs.obj.foo)
