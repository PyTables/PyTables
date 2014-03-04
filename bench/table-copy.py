from __future__ import print_function
import time

import numpy as np
import tables

N = 144000
#N = 144


def timed(func, *args, **kwargs):
    start = time.time()
    res = func(*args, **kwargs)
    print("%fs elapsed." % (time.time() - start))
    return res


def create_table(output_path):
    print("creating array...", end=' ')
    dt = np.dtype([('field%d' % i, int) for i in range(320)])
    a = np.zeros(N, dtype=dt)
    print("done.")

    output_file = tables.open_file(output_path, mode="w")
    table = output_file.create_table("/", "test", dt)  # , filters=blosc4)
    print("appending data...", end=' ')
    table.append(a)
    print("flushing...", end=' ')
    table.flush()
    print("done.")
    output_file.close()


def copy1(input_path, output_path):
    print("copying data from %s to %s..." % (input_path, output_path))
    input_file = tables.open_file(input_path, mode="r")
    output_file = tables.open_file(output_path, mode="w")

    # copy nodes as a batch
    input_file.copy_node("/", output_file.root, recursive=True)
    output_file.close()
    input_file.close()


def copy2(input_path, output_path):
    print("copying data from %s to %s..." % (input_path, output_path))
    input_file = tables.open_file(input_path, mode="r")
    input_file.copy_file(output_path, overwrite=True)
    input_file.close()


def copy3(input_path, output_path):
    print("copying data from %s to %s..." % (input_path, output_path))
    input_file = tables.open_file(input_path, mode="r")
    output_file = tables.open_file(output_path, mode="w")
    table = input_file.root.test
    table.copy(output_file.root)
    output_file.close()
    input_file.close()


def copy4(input_path, output_path, complib='zlib', complevel=0):
    print("copying data from %s to %s..." % (input_path, output_path))
    input_file = tables.open_file(input_path, mode="r")
    output_file = tables.open_file(output_path, mode="w")

    input_table = input_file.root.test
    print("reading data...", end=' ')
    data = input_file.root.test.read()
    print("done.")

    filter = tables.Filters(complevel=complevel, complib=complib)
    output_table = output_file.create_table("/", "test", input_table.dtype,
                                            filters=filter)
    print("appending data...", end=' ')
    output_table.append(data)
    print("flushing...", end=' ')
    output_table.flush()
    print("done.")

    input_file.close()
    output_file.close()


def copy5(input_path, output_path, complib='zlib', complevel=0):
    print("copying data from %s to %s..." % (input_path, output_path))
    input_file = tables.open_file(input_path, mode="r")
    output_file = tables.open_file(output_path, mode="w")

    input_table = input_file.root.test

    filter = tables.Filters(complevel=complevel, complib=complib)
    output_table = output_file.create_table("/", "test", input_table.dtype,
                                            filters=filter)
    chunksize = 10000
    rowsleft = len(input_table)
    start = 0
    for chunk in range((len(input_table) / chunksize) + 1):
        stop = start + min(chunksize, rowsleft)
        data = input_table.read(start, stop)
        output_table.append(data)
        output_table.flush()
        rowsleft -= chunksize
        start = stop

    input_file.close()
    output_file.close()


if __name__ == '__main__':
    timed(create_table, 'tmp.h5')
#    timed(copy1, 'tmp.h5', 'test1.h5')
    timed(copy2, 'tmp.h5', 'test2.h5')
#    timed(copy3, 'tmp.h5', 'test3.h5')
    timed(copy4, 'tmp.h5', 'test4.h5')
    timed(copy5, 'tmp.h5', 'test5.h5')
