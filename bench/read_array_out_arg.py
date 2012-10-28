from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import time

import numpy as np
import tables

def create_file(array_size):
    array = np.ones(array_size, dtype='i8')
    with tables.openFile('test.h5', 'w') as fobj:
        array = fobj.createArray('/', 'test', array)
        print('file created, size: ' + str(array.size_on_disk / 1e6))

def standard_read(array_size):
    N = 10
    with tables.openFile('test.h5', 'r') as fobj:
        array = fobj.getNode('/', 'test')
        start = time.time()
        for i in xrange(N):
            output = array.read(0, array_size, 1)
        end = time.time()
        assert(np.all(output == 1))
        print((end - start) / N)

def pre_allocated_read(array_size):
    N = 10
    with tables.openFile('test.h5', 'r') as fobj:
        array = fobj.getNode('/', 'test')
        start = time.time()
        output = np.empty(array_size, 'i8')
        for i in xrange(N):
            array.read(0, array_size, 1, out=output)
        end = time.time()
        assert(np.all(output == 1))
        print((end - start) / N)


if __name__ == '__main__':
    array_bytes = 80000000
    array_size = int(array_bytes // 8)

    create_file(array_size)
    standard_read(array_size)
    pre_allocated_read(array_size)