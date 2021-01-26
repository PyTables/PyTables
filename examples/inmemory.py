#!/usr/bin/env python3

"""inmemory.py.

Example usage of creating in-memory HDF5 file with a specified chunksize
using PyTables 3.0.0+

See also Cookbook page
http://pytables.github.io/cookbook/inmemory_hdf5_files.html and available
drivers
http://pytables.github.io/usersguide/parameter_files.html#hdf5-driver-management

"""

import numpy as np
import tables as tb

CHUNKY = 30
CHUNKX = 4320

if __name__ == '__main__':

    # create dataset and add global attrs
    file_path = 'demofile_chunk%sx%d.h5' % (CHUNKY, CHUNKX)

    with tb.open_file(file_path, 'w',
                      title='PyTables HDF5 In-memory example',
                      driver='H5FD_CORE') as h5f:

        # dummy some data
        lats = np.empty([2160])
        lons = np.empty([4320])

        # create some simple arrays
        lat_node = h5f.create_array('/', 'lat', lats, title='latitude')
        lon_node = h5f.create_array('/', 'lon', lons, title='longitude')

        # create a 365 x 4320 x 8640 CArray of 32bit float
        shape = (5, 2160, 4320)
        atom = tb.Float32Atom(dflt=np.nan)

        # chunk into daily slices and then further chunk days
        sst_node = h5f.create_carray(
            h5f.root, 'sst', atom, shape, chunkshape=(1, CHUNKY, CHUNKX))

        # dummy up an ndarray
        sst = np.empty([2160, 4320], dtype=np.float32)
        sst.fill(30.0)

        # write ndarray to a 2D plane in the HDF5
        sst_node[0] = sst
