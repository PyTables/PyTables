import tables as tb


class MyClass:
    foo = "bar"


# An object of my custom class.
my_object = MyClass()

with tb.open_file("test.h5", "w") as h5f:
    h5f.root._v_attrs.obj = my_object  # store the object
    print(h5f.root._v_attrs.obj.foo)  # retrieve it

# Automatic unpickling is still enabled by default.
with tb.open_file("test.h5", "r") as h5f:
    print(h5f.root._v_attrs.obj.foo)

# Disable it for files that are not trusted.
with tb.open_file("test.h5", "r", allow_pickle=False) as h5f:
    print(repr(h5f.root._v_attrs.obj))
