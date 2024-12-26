# A simple script that illustrates how to write and read chunks directly,
# also usable for profiling the implementation of direct chunking.

import time
import cProfile

import numpy as np
import blosc2 as b2

import tables as tb

# When not profiling, chunks are compressed and decompressed on each iteration
# so as to make the example more realistic (but also way slower).
# That is skipped when profiling since we are only interested in
# the performance of write/read operations.
profile = True

# A tomography-like array: a stack of 2D images (greyscale).
# Each image corresponds to a chunk in the array.
# The values used here result in compressed chunks of nearly 4MiB,
# which matches my CPU's L3 cache.
fname = "direct-chunking.h5"
dtype = np.dtype("u2")
shape = (500, 25600, 19200)
# shape = (100, 256, 256)  # for tests
chunkshape = (1, *shape[1:])

# Blosc2 block shape is an example of a parameter
# which cannot be specified via `tb.Filters`.
b2_blockshape = (
    1,
    *tuple(d // 2 for d in chunkshape[1:]),
)  # 4 blocks per chunk

np_data = np.arange(np.prod(chunkshape), dtype=dtype).reshape(chunkshape)


def chunk_from_data(data):
    b2_data = b2.asarray(data, chunks=chunkshape, blocks=b2_blockshape)
    wchunk = b2_data.to_cframe()
    return wchunk


def data_from_chunk(rchunk):
    b2_array = b2.ndarray_from_cframe(rchunk)
    data = b2_array[:]
    return data


with tb.open_file(fname, mode="w") as h5f:
    array = h5f.create_earray(
        "/",
        "array",
        atom=tb.Atom.from_dtype(dtype),
        shape=(0, *shape[1:]),
        # Setting both args tells others that data is compressed using Blosc2
        # and it should not be handled as plain data.
        filters=tb.Filters(complevel=1, complib="blosc2"),
        chunkshape=chunkshape,
    )

    # First, grow the array without actually storing data.
    array.truncate(shape[0])
    # Now, do store the data as raw chunks.
    coords_tail = (0,) * (len(shape) - 1)
    if profile:
        wchunk = chunk_from_data(np_data)

    def do_write():
        for c in range(shape[0]):
            if profile:
                # The same image/chunk is written over and over again.
                global wchunk
            else:
                # A new image/chunk.is written.
                wchunk = chunk_from_data(np_data)
            chunk_coords = (c,) + coords_tail
            array.write_chunk(chunk_coords, wchunk)

    start = time.time()
    if profile:
        cProfile.run("do_write()")
    else:
        do_write()
    elapsed = time.time() - start
    print(f"Wrote {shape[0]} chunks ({elapsed} s).")


with tb.open_file(fname, mode="r") as h5f:
    array = h5f.root.array
    rchunk = bytearray(len(wchunk))

    coords_tail = (0,) * (len(shape) - 1)

    def do_read():
        for c in range(shape[0]):
            chunk_coords = (c,) + coords_tail
            array.read_chunk(chunk_coords, out=rchunk)
            if not profile:
                _ = data_from_chunk(rchunk)

    start = time.time()
    if profile:
        cProfile.run("do_read()")
    else:
        do_read()
    elapsed = time.time() - start
    print(f"Read {shape[0]} chunks ({elapsed} s).")
