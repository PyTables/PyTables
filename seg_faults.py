import numpy as np
import tables


# this seg-faults
def buffer_too_small():

    with tables.openFile('test.h5', 'w') as f:
        array = np.arange(1000)
        disk_array = f.createArray('/', 'array', array)
        out_buffer = np.empty((500, ), 'f8')
        disk_array.read(out=out_buffer)
    return out_buffer


# this does not raise an exception but probably should
def non_contiguous():

    with tables.openFile('test.h5', 'w') as f:
        array = np.arange(1000)
        disk_array = f.createArray('/', 'array', array)
        out_buffer = np.empty((1000, ), 'i8')
        out_buffer2 = out_buffer[0:1000:2]
        disk_array.read(out=out_buffer2)
    return out_buffer, out_buffer2




