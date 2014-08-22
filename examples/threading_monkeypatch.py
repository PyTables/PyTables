#!/usr/bin/env python

import os
import queue
import functools
import threading

import numpy as np
import tables as tb

import tables.file as _tables_file


class ThreadsafeFileRegistry(_tables_file._FileRegistry):
    lock = threading.RLock()

    @property
    def handlers(self):
        return self._handlers.copy()

    def add(self, handler):
        with self.lock:
            return super().add(handler)

    def remove(self, handler):
        with self.lock:
            return super().remove(handler)

    def close_all(self):
        with self.lock:
            return super().close_all(handler)


class ThreadsafeFile(_tables_file.File):
    def __init__(self, *args, **kargs):
        with ThreadsafeFileRegistry.lock:
            super().__init__(*args, **kargs)

    def close(self):
        with ThreadsafeFileRegistry.lock:
            super().close()


@functools.wraps(tb.open_file)
def synchronized_open_file(*args, **kwargs):
    with ThreadsafeFileRegistry.lock:
        return _tables_file._original_open_file(*args, **kwargs)


# monkey patch the tables package
_tables_file._original_open_file = _tables_file.open_file
_tables_file.open_file = synchronized_open_file
tb.open_file = synchronized_open_file

_tables_file._original_File = _tables_file.File
_tables_file.File = ThreadsafeFile
tb.File = ThreadsafeFile

_tables_file._open_files = ThreadsafeFileRegistry()


SIZE = 100
NTHREADS = 5
FILENAME = 'simple_threading.h5'
H5PATH = '/array'


def create_test_file(filename):
    data = np.random.rand(SIZE, SIZE)

    with tb.open_file(filename, 'w') as h5file:
        h5file.create_array('/', 'array', title="Test Array", obj=data)


def chunk_generator(data_size, nchunks):
    chunk_size = int(np.ceil(data_size / nchunks))
    for start in range(0, data_size, chunk_size):
        yield slice(start, start + chunk_size)


def run(filename, path, inqueue, outqueue):
    try:
        yslice = inqueue.get()
        with tb.open_file(filename, mode='r') as h5file:
            h5array = h5file.get_node(path)
            data = h5array[yslice, ...]
        psum = np.sum(data)
    except Exception as e:
        outqueue.put(e)
    else:
        outqueue.put(psum)


def main():
    # generate the test data
    if not os.path.exists(FILENAME):
        create_test_file(FILENAME)

    threads = []
    inqueue = queue.Queue()
    outqueue = queue.Queue()

    # start all threads
    for i in range(NTHREADS):
        thread = threading.Thread(target=run,
                                  args=(FILENAME, H5PATH, inqueue, outqueue))
        thread.start()
        threads.append(thread)

    # push requests in the input queue
    for yslice in chunk_generator(SIZE, len(threads)):
        inqueue.put(yslice)

    # collect results
    try:
        mean_ = 0.

        for i in range(len(threads)):
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
    print('Mean: {}'.format(mean_))


if __name__ == '__main__':
    main()
