from time import perf_counter as clock

import numpy as np
import tables as tb

N = 10_000_000

filters = tb.Filters(9, "blosc2", shuffle=True)

def timed(func, *args, **kwargs):
    start = clock()
    res = func(*args, **kwargs)
    print(f"{clock() - start:.3f}s elapsed.")
    return res


def create_table(output_path):
    print("creating array...", end=' ')
    dt = np.dtype([('field%d' % i, int) for i in range(32)])
    a = np.zeros(N, dtype=dt)
    print("done.")

    output_file = tb.open_file(output_path, mode="w")
    table = output_file.create_table("/", "test", dt, filters=filters)
    print("appending data...", end=' ')
    table.append(a)
    print("flushing...", end=' ')
    table.flush()
    print("done.")
    output_file.close()


def copy1(input_path, output_path):
    print(f"copying data from {input_path} to {output_path}...")
    input_file = tb.open_file(input_path, mode="r")
    output_file = tb.open_file(output_path, mode="w")

    # copy nodes as a batch
    input_file.copy_node("/", output_file.root, recursive=True, filters=filters)
    output_file.close()
    input_file.close()


def copy2(input_path, output_path):
    print(f"copying data from {input_path} to {output_path}...")
    input_file = tb.open_file(input_path, mode="r")
    input_file.copy_file(output_path, overwrite=True, filters=filters)
    input_file.close()


def copy3(input_path, output_path):
    print(f"copying data from {input_path} to {output_path}...")
    input_file = tb.open_file(input_path, mode="r")
    output_file = tb.open_file(output_path, mode="w", filters=filters)
    table = input_file.root.test
    table.copy(output_file.root)
    output_file.close()
    input_file.close()


def copy4(input_path, output_path):
    print(f"copying data from {input_path} to {output_path}...")
    input_file = tb.open_file(input_path, mode="r")
    output_file = tb.open_file(output_path, mode="w", filters=filters)

    input_table = input_file.root.test
    print("reading data...", end=' ')
    start = clock()
    data = input_file.root.test.read()
    print(f"{clock() - start:.3f}s elapsed.")
    print("done.")

    output_table = output_file.create_table("/", "test", input_table.dtype)
    print("appending data...", end=' ')
    start = clock()
    output_table.append(data)
    print("flushing...", end=' ')
    output_table.flush()
    print(f"{clock() - start:.3f}s elapsed.")
    print("done.")

    input_file.close()
    output_file.close()


def copy5(input_path, output_path):
    print(f"copying data from {input_path} to {output_path}...")
    input_file = tb.open_file(input_path, mode="r")
    output_file = tb.open_file(output_path, mode="w", filters=filters)

    input_table = input_file.root.test
    output_table = output_file.create_table("/", "test", input_table.dtype)

    chunksize = 100_000
    rowsleft = len(input_table)
    start = 0
    for chunk in range(int(len(input_table) / chunksize) + 1):
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
    timed(copy1, 'tmp.h5', 'test1.h5')
    timed(copy2, 'tmp.h5', 'test2.h5')
    timed(copy3, 'tmp.h5', 'test3.h5')
    timed(copy4, 'tmp.h5', 'test4.h5')
    timed(copy5, 'tmp.h5', 'test5.h5')
