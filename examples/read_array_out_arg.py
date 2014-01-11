# This script compares reading from an array in a loop using the
# tables.Array.read method.  In the first case, read is used without supplying
# an 'out' argument, which causes a new output buffer to be pre-allocated
# with each call.  In the second case, the buffer is created once, and then
# reused.


from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import time

import numpy as np
import tables


def create_file(array_size):
    array = np.ones(array_size, dtype='i8')
    with tables.open_file('test.h5', 'w') as fobj:
        array = fobj.create_array('/', 'test', array)
        print('file created, size: {0} MB'.format(array.size_on_disk / 1e6))


def standard_read(array_size):
    N = 10
    with tables.open_file('test.h5', 'r') as fobj:
        array = fobj.get_node('/', 'test')
        start = time.time()
        for i in range(N):
            output = array.read(0, array_size, 1)
        end = time.time()
        assert(np.all(output == 1))
        print('standard read   \t {0:5.5f}'.format((end - start) / N))


def pre_allocated_read(array_size):
    N = 10
    with tables.open_file('test.h5', 'r') as fobj:
        array = fobj.get_node('/', 'test')
        start = time.time()
        output = np.empty(array_size, 'i8')
        for i in range(N):
            array.read(0, array_size, 1, out=output)
        end = time.time()
        assert(np.all(output == 1))
        print('pre-allocated read\t {0:5.5f}'.format((end - start) / N))


if __name__ == '__main__':

    array_num_bytes = [int(x) for x in [1e5, 1e6, 1e7, 1e8]]

    for array_bytes in array_num_bytes:

        array_size = int(array_bytes // 8)

        create_file(array_size)
        standard_read(array_size)
        pre_allocated_read(array_size)
        print()
