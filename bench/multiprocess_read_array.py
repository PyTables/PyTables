# Benchmark three methods of using PyTables with multiple processes, where data
# is read from a PyTables file in one process and then sent to another
#
# 1. using multiprocessing.Pipe
# 2. using a memory mapped file that's shared between two processes, plus a
#    modified version of tables.Array.read that accepts an 'out' argument
# 3. using a Unix domain socket
#
# In all three cases, an array is loaded from a file in one process, sent to
# another, and then modified by incrementing each array element.  This is meant
# to simulate retrieving data and then modifying it.

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import multiprocessing
import os
import select
import socket
import time

import numpy as np
import tables


# create a PyTables file with a single int64 array with the specified number of
# elements
def create_file(array_size):
    array = np.ones(array_size, dtype='i8')
    with tables.openFile('test.h5', 'w') as fobj:
        array = fobj.createArray('/', 'test', array)
        print('file created, size: ' + str(array.size_on_disk / 1e6))


# process to receive an array using a multiprocessing.Pipe connection
class PipeReceive(multiprocessing.Process):

    def __init__(self, receiver_pipe, result_send):
        super(PipeReceive, self).__init__()
        self.receiver_pipe = receiver_pipe
        self.result_send = result_send

    def run(self):
        # block until something is received on the pipe
        array = self.receiver_pipe.recv()
        recv_timestamp = time.time()
        # perform an operation on the received array
        array += 1
        finish_timestamp = time.time()
        assert(np.all(array == 2))
        # send the measured timestamps back to the originating process
        self.result_send.send((recv_timestamp, finish_timestamp))


def read_and_send_pipe(array_size):
    # set up Pipe objects to send the actual array to the other process
    # and receive the timing results from the other process
    array_recv, array_send = multiprocessing.Pipe(False)
    result_recv, result_send = multiprocessing.Pipe(False)
    # start the other process and pause to allow it to start up
    recv_process = PipeReceive(array_recv, result_send)
    recv_process.start()
    time.sleep(0.10)
    with tables.openFile('test.h5', 'r') as fobj:
        array = fobj.getNode('/', 'test')
        start_timestamp = time.time()
        # read an array from the PyTables file and send it to the other process
        output = array.read(0, array_size, 1)
        array_send.send(output)
        assert(np.all(output + 1 == 2))
        # receive the timestamps from the other process
        recv_timestamp, finish_timestamp = result_recv.recv()
        print_results(start_timestamp, recv_timestamp, finish_timestamp)
    recv_process.join()


# process to receive an array using a shared memory mapped file
# for real use, this would require some protocol to specify the array's
# data type and shape
class MemmapReceive(multiprocessing.Process):

    def __init__(self, path_recv, result_send):
        super(MemmapReceive, self).__init__()
        self.path_recv = path_recv
        self.result_send = result_send

    def run(self):
        # block until the memmap file path is received from the other process
        path = self.path_recv.recv()
        # create a memmap array using the received file path
        array = np.memmap(path, 'i8', 'r+')
        recv_timestamp = time.time()
        # perform an operation on the array
        array += 1
        finish_timestamp = time.time()
        assert(np.all(array == 2))
        # send the timing results back to the other process
        self.result_send.send((recv_timestamp, finish_timestamp))


def read_and_send_memmap(array_size):
    # create a multiprocessing Pipe that will be used to send the memmap
    # file path to the receiving process
    path_recv, path_send = multiprocessing.Pipe(False)
    result_recv, result_send = multiprocessing.Pipe(False)
    # start the receiving process and pause to allow it to start up
    recv_process = MemmapReceive(path_recv, result_send)
    recv_process.start()
    time.sleep(0.10)
    with tables.openFile('test.h5', 'r') as fobj:
        array = fobj.getNode('/', 'test')
        start_timestamp = time.time()
        # memmap a file as a NumPy array in 'overwrite' mode
        output = np.memmap('/tmp/array1', 'i8', 'w+', shape=(array_size, ))
        # read an array from a PyTables file into the memmory mapped array
        array.read(0, array_size, 1, out=output)
        # use a multiprocessing.Pipe to send the file's path to the receiving
        # process
        path_send.send('/tmp/array1')
        # receive the timestamps from the other process
        recv_timestamp, finish_timestamp = result_recv.recv()
        # because 'output' is shared between processes, all elements should now
        # be equal to 2
        assert(np.all(output == 2))
        print_results(start_timestamp, recv_timestamp, finish_timestamp)
    recv_process.join()


# process to receive an array using a Unix domain socket
# for real use, this would require some protocol to specify the array's
# data type and shape
class UnixSocketReceive(multiprocessing.Process):

    def __init__(self, address, result_send, array_nbytes):
        super(UnixSocketReceive, self).__init__()
        self.address = address
        self.result_send = result_send
        self.array_nbytes = array_nbytes

    def run(self):
        # create the socket, listen for a connection and use select to block
        # until a connection is made
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(self.address)
        sock.listen(1)
        readable, _, _ = select.select([sock], [], [])
        # accept the connection and read the sent data into a bytearray
        connection = sock.accept()[0]
        recv_buffer = bytearray(self.array_nbytes)
        view = memoryview(recv_buffer)
        bytes_recv = 0
        while bytes_recv < self.array_nbytes:
            bytes_recv += connection.recv_into(view[bytes_recv:])
        # convert the bytearray into a NumPy array
        array = np.frombuffer(recv_buffer, dtype='i8')
        recv_timestamp = time.time()
        # perform an operation on the received array
        array += 1
        finish_timestamp = time.time()
        assert(np.all(array == 2))
        # send the timestamps back to the originating process
        self.result_send.send((recv_timestamp, finish_timestamp))


def read_and_send_socket(array_size, array_bytes):
    # create a Unix domain address in the abstract namespace
    # this will only work on Linux
    address = b'\x00' + os.urandom(5)
    # start the receiving process and pause to allow it to start up
    result_recv, result_send = multiprocessing.Pipe(False)
    recv_process = UnixSocketReceive(address, result_send, array_bytes)
    recv_process.start()
    time.sleep(0.15)
    with tables.openFile('test.h5', 'r') as fobj:
        array = fobj.getNode('/', 'test')
        start_timestamp = time.time()
        # connect to the receiving process' socket
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(address)
        # read the array from the PyTables file and send its
        # data buffer to the receiving process
        output = array.read(0, array_size, 1)
        sock.send(output.data)
        assert(np.all(output + 1 == 2))
        # receive the timestamps from the other process
        recv_timestamp, finish_timestamp = result_recv.recv()
        print_results(start_timestamp, recv_timestamp, finish_timestamp)
    recv_process.join()


def print_results(start_timestamp, recv_timestamp, finish_timestamp):
    msg = 'receive: {0}, add:{1}, total: {2}'
    print(msg.format(recv_timestamp - start_timestamp,
                     finish_timestamp - recv_timestamp,
                     finish_timestamp - start_timestamp))


if __name__ == '__main__':
    array_bytes = 100000000
    array_size = int(array_bytes // 8)

    create_file(array_size)
    read_and_send_pipe(array_size)
    read_and_send_memmap(array_size)
    read_and_send_socket(array_size, array_bytes)