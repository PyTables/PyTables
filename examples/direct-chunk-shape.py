# This script uses h5py to write two versions of the following 6x6 array into
# two datasets with 4x4 chunks::
#
#                chunk boundary
#                |
#   [[ 0  1  2  3  0  1]
#    [ 4  5  6  7  4  5]
#    [ 8  9 10 11  8  9]
#    [12 13 14 15 12 13] __ chunk boundary
#    [ 0  1  2  3  0  1]
#    [ 4  5  6  7  4  5]]
#
# So its chunks contain, left-to-right, top-to-bottom, 4x4, 4x2, 2x4, and 2x2
# items worth of data.
#
# - ``/data_full`` has all its chunks with shape 4x4 in storage, even if the
#   actual data does not fill the whole chunk.
# - ``/data_part`` has chunks with the shape of actual data, as listed above.
#
# Direct chunking is used in h5py to control the exact storage format for each
# chunk, using a Blosc2 ND container.  Then PyTables is used to read both
# datasets and compare them against the original array, so as to test whether
# there are problems in loading partial chunks.
#
# Since PyTables has an optimized path for uncompressing Blosc2 ND data that
# does not use the HDF5 filter pipeline, you may force it to only use the
# latter by setting ``BLOSC2_FILTER=1`` in the environment.

import os
import tempfile

import h5py
import numpy
import blosc2
import hdf5plugin

# Import after h5py part is done so that it uses its own `hdf5-blosc2`.
import tables

# Both parameter sets below are equivalent.
fparams = hdf5plugin.Blosc2(
    cname="zstd", clevel=1, filters=hdf5plugin.Blosc2.SHUFFLE
)
cparams = {
    "codec": blosc2.Codec.ZSTD,
    "clevel": 1,
    "filters": [blosc2.Filter.SHUFFLE],
}


def np2b2(a):
    return blosc2.asarray(
        numpy.ascontiguousarray(a),
        chunks=a.shape,
        blocks=a.shape,
        cparams=cparams,
    )


# Assemble the array.
achunk = numpy.arange(4 * 4).reshape((4, 4))
adata = numpy.zeros((6, 6), dtype=achunk.dtype)
adata[0:4, 0:4] = achunk[:, :]
adata[0:4, 4:6] = achunk[:, 0:2]
adata[4:6, 0:4] = achunk[0:2, :]
adata[4:6, 4:6] = achunk[0:2, 0:2]

h5fdesc, h5fpath = tempfile.mkstemp()
os.close(h5fdesc)

h5f = h5py.File(h5fpath, "w")
print(f"Writing with full chunks to {h5fpath}:/data_full...")
dataset = h5f.create_dataset(
    "data_full", adata.shape, dtype=adata.dtype, chunks=achunk.shape, **fparams
)
b2chunk = np2b2(achunk)  # need to keep the ref or cframe becomes bogus
b2frame = b2chunk._schunk.to_cframe()
dataset.id.write_direct_chunk((0, 0), b2frame)
dataset.id.write_direct_chunk((0, 4), b2frame)
dataset.id.write_direct_chunk((4, 0), b2frame)
dataset.id.write_direct_chunk((4, 4), b2frame)
print(f"Writing with partial chunks to {h5fpath}:/data_part...")
dataset = h5f.create_dataset(
    "data_part", adata.shape, dtype=adata.dtype, chunks=achunk.shape, **fparams
)
b2chunk = np2b2(achunk[:, :])  # need to keep the ref or cframe becomes bogus
b2frame = b2chunk._schunk.to_cframe()
dataset.id.write_direct_chunk((0, 0), b2frame)
b2chunk = np2b2(achunk[:, 0:2])
# Uncomment to introduce a bogus partial chunk in the midle of data,
# too small even for a margin chunk.
# b2chunk = np2b2(achunk[0:2, 0:2])
b2frame = b2chunk._schunk.to_cframe()
dataset.id.write_direct_chunk((0, 4), b2frame)
b2chunk = np2b2(achunk[0:2, :])
b2frame = b2chunk._schunk.to_cframe()
dataset.id.write_direct_chunk((4, 0), b2frame)
b2chunk = np2b2(achunk[0:2, 0:2])
b2frame = b2chunk._schunk.to_cframe()
dataset.id.write_direct_chunk((4, 4), b2frame)
h5f.close()

h5f = tables.open_file(h5fpath, "r")
print(f"Reading with full chunks from {h5fpath}:/data_full...")
a_full = h5f.root.data_full[:]
print(f"Reading with partial chunks from {h5fpath}:/data_part...")
a_part = h5f.root.data_part[:]
h5f.close()

if not (a_full == adata).all():
    print("FULL CHUNKS FAILED (original, read):")
    print(adata)
    print(a_full)

if not (a_part == adata).all():
    print("PARTIAL CHUNKS FAILED (original, read):")
    print(adata)
    print(a_part)

print(f"Removing {h5fpath}...")
os.remove(h5fpath)
