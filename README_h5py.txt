**FOR DEVELOPERS EYES ONLY**

I have started experimenting with using h5py to replace low-level access to
HDF5. To see things clearly I have spent some amount of time splitting
hdf5extension and indexesextension into smaller parts.

h5py at the moment lacks some features we require (like fully supporting the
file image API) so that should be done too.

-- Andrea 24/3/16

If you are experimenting with this take notice of this **important warning**

The HDF5 uses global variables to maintain its state. In case PyTables and
h5py get linked to two different copies of HDF5, all the handles returned by
library call will mismatch and everything will crash and burn.

This happens regurarly on OS X since h5py's wheel ships its own version of
HDF5 while PyTables requires compiling from source (picking up the version
installed in the system).

You can assess the situation with otool on OSX (and similarly with ldd on
linux):

otool -L path/to/tables/*.so path/to/h5py/*.so

-- Andrea 5/7/16


Running tests
=============

There is a very simple test suite in tables/tests_h5py_backend that you can
run via::

  $ py.test tables/tests_h5py_backend

Our first goal should be to pass these simple tests using the concrete implementation
of the h5py backend.  Then we can concentrate in getting the regular test suite pass.

-- Francesc 2016-08-08

