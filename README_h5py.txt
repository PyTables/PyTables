**FOR DEVELOPERS EYES ONLY**

I have started experimenting with using h5py to replace low-level access to
HDF5. To see things clearly I have spent some amount of time splitting
hdf5extension and indexesextension into smaller parts.

h5py at the moment lacks some features we require (like fully supporting the
file image API) so that should be done too.

-- Andrea 24/3/16

