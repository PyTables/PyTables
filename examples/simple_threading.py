#!/usr/bin/env python3

import math
import queue
import threading
from pathlib import Path

import numpy as np
import tables as tb


SIZE = 100
NTHREADS = 5
FILENAME = 'simple_threading.h5'
H5PATH = '/array'


def create_test_file(filename):
    data = np.random.rand(SIZE, SIZE)

    with tb.open_file(filename, 'w') as h5file:
        h5file.create_array('/', 'array', title="Test Array", obj=data)


def chunk_generator(data_size, nchunks):
    chunk_size = math.ceil(data_size / nchunks)
    for start in range(0, data_size, chunk_size):
        yield slice(start, start + chunk_size)


lock = threading.Lock()


def synchronized_open_file(*args, **kwargs):
    with lock:
        return tb.open_file(*args, **kwargs)


def synchronized_close_file(self, *args, **kwargs):
    with lock:
        return self.close(*args, **kwargs)


def run(filename, path, inqueue, outqueue):
    try:
        yslice = inqueue.get()
        h5file = synchronized_open_file(filename, mode='r')
        h5array = h5file.get_node(path)
        data = h5array[yslice, ...]
        psum = np.sum(data)
    except Exception as e:
        outqueue.put(e)
    else:
        outqueue.put(psum)
    finally:
        synchronized_close_file(h5file)


def main():
    # generate the test data
    if not Path(FILENAME).exists():
        create_test_file(FILENAME)

    threads = []
    inqueue = queue.Queue()
    outqueue = queue.Queue()

    # start all threads
    for _ in range(NTHREADS):
        thread = threading.Thread(target=run,
                                  args=(FILENAME, H5PATH, inqueue, outqueue))
        thread.start()
        threads.append(thread)

    # push requests in the input queue
    for yslice in chunk_generator(SIZE, len(threads)):
        inqueue.put(yslice)

    # collect results
    try:
        mean_ = 0

        for _ in range(len(threads)):
            out = outqueue.get()
            if isinstance(out, Exception):
                raise out
            else:
                mean_ += out

        mean_ /= SIZE * SIZE

    finally:
        for thread in threads:
            thread.join()

    # print results
    print(f'Mean: {mean_}')


if __name__ == '__main__':
    main()
