# A simple script that illustrates how to write and read chunks directly,
# also usable for profiling the implementation of direct chunking.

import cProfile
import time

import blosc2 as b2
import numpy as np
import tables as tb


# A tomography-like array: a stack of 2D images (greyscale).
# Each image corresponds to a chunk in the array.
# The values used here result in compressed chunks of nearly 4MiB,
# which matches my CPU's L3 cache.
fname = 'direct-chunking.h5'
dtype = np.dtype('u4')
shape = (500, 25600, 19200)
# dtype = np.dtype('u2')  # for tests
# shape = (100, 256, 256)  # for tests
chunkshape = (1, *shape[1:])

np_data = (np.arange(np.prod(chunkshape), dtype=dtype)
           .reshape(chunkshape))

# Blosc2 block shape is an example of a parameter
# which cannot be specified via `tb.Filters`.
b2_blockshape = (1, *tuple(d // 2 for d in chunkshape[1:]))  # 4 blocks per chunk
b2_data = b2.asarray(np_data, chunks=chunkshape, blocks=b2_blockshape)

# The same image/chunk will be written over and over again.
wchunk = b2_data.to_cframe()


with tb.open_file(fname, mode='w') as h5f:
    array = h5f.create_earray(
        '/', 'array',
        atom=tb.Atom.from_dtype(dtype),
        shape=(0, *shape[1:]),
        # Setting both args tells others that data is compressed using Blosc2
        # and it should not be handled as plain data.
        filters=tb.Filters(complevel=1, complib='blosc2'),
        chunkshape=chunkshape,
    )

    # First, grow the array without actually storing data.
    array.truncate(shape[0])
    # Now, do store the data as raw chunks.
    coords_tail = (0,) * (len(shape) - 1)
    def do_write():
        for c in range(shape[0]):
            chunk_coords = (c,) + coords_tail
            array.write_chunk(chunk_coords, wchunk)
    start = time.time()
    do_write()  # cProfile.run('do_write()')
    elapsed = time.time() - start
    print(f"Wrote {shape[0]} chunks of {len(wchunk)} bytes ({elapsed} s).")


with tb.open_file(fname, mode='r') as h5f:
    array = h5f.root.array
    rchunk = bytearray(len(wchunk))

    coords_tail = (0,) * (len(shape) - 1)
    def do_read():
        for c in range(shape[0]):
            chunk_coords = (c,) + coords_tail
            array.read_chunk(chunk_coords, out=rchunk)
    start = time.time()
    do_read()  # cProfile.run('do_read()')
    elapsed = time.time() - start
    print(f"Read {shape[0]} chunks of {len(wchunk)} bytes ({elapsed} s).")
