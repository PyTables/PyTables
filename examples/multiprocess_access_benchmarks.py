# Benchmark three methods of using PyTables with multiple processes, where data
# is read from a PyTables file in one process and then sent to another
#
# 1. using multiprocessing.Pipe
# 2. using a memory mapped file that's shared between two processes, passed as
#    out argument to tables.Array.read.
# 3. using a Unix domain socket (this uses the "abstract namespace" and will
#    work only on Linux).
# 4. using an IPv4 socket
#
# In all three cases, an array is loaded from a file in one process, sent to
# another, and then modified by incrementing each array element.  This is meant
# to simulate retrieving data and then modifying it.


import multiprocessing
import os
import random
import select
import socket
import time
from time import perf_counter as clock

import numpy as np
import tables as tb


# create a PyTables file with a single int64 array with the specified number of
# elements
def create_file(array_size):
    array = np.ones(array_size, dtype='i8')
    with tb.open_file('test.h5', 'w') as fobj:
        array = fobj.create_array('/', 'test', array)
        print('file created, size: {} MB'.format(array.size_on_disk / 1e6))


# process to receive an array using a multiprocessing.Pipe connection
class PipeReceive(multiprocessing.Process):

    def __init__(self, receiver_pipe, result_send):
        super().__init__()
        self.receiver_pipe = receiver_pipe
        self.result_send = result_send

    def run(self):
        # block until something is received on the pipe
        array = self.receiver_pipe.recv()
        recv_timestamp = clock()
        # perform an operation on the received array
        array += 1
        finish_timestamp = clock()
        assert(np.all(array == 2))
        # send the measured timestamps back to the originating process
        self.result_send.send((recv_timestamp, finish_timestamp))


def read_and_send_pipe(send_type, array_size):
    # set up Pipe objects to send the actual array to the other process
    # and receive the timing results from the other process
    array_recv, array_send = multiprocessing.Pipe(False)
    result_recv, result_send = multiprocessing.Pipe(False)
    # start the other process and pause to allow it to start up
    recv_process = PipeReceive(array_recv, result_send)
    recv_process.start()
    time.sleep(0.15)
    with tb.open_file('test.h5', 'r') as fobj:
        array = fobj.get_node('/', 'test')
        start_timestamp = clock()
        # read an array from the PyTables file and send it to the other process
        output = array.read(0, array_size, 1)
        array_send.send(output)
        assert(np.all(output + 1 == 2))
        # receive the timestamps from the other process
        recv_timestamp, finish_timestamp = result_recv.recv()
    print_results(send_type, start_timestamp, recv_timestamp, finish_timestamp)
    recv_process.join()


# process to receive an array using a shared memory mapped file
# for real use, this would require creating some protocol to specify the
# array's data type and shape
class MemmapReceive(multiprocessing.Process):

    def __init__(self, path_recv, result_send):
        super().__init__()
        self.path_recv = path_recv
        self.result_send = result_send

    def run(self):
        # block until the memmap file path is received from the other process
        path = self.path_recv.recv()
        # create a memmap array using the received file path
        array = np.memmap(path, 'i8', 'r+')
        recv_timestamp = clock()
        # perform an operation on the array
        array += 1
        finish_timestamp = clock()
        assert(np.all(array == 2))
        # send the timing results back to the other process
        self.result_send.send((recv_timestamp, finish_timestamp))


def read_and_send_memmap(send_type, array_size):
    # create a multiprocessing Pipe that will be used to send the memmap
    # file path to the receiving process
    path_recv, path_send = multiprocessing.Pipe(False)
    result_recv, result_send = multiprocessing.Pipe(False)
    # start the receiving process and pause to allow it to start up
    recv_process = MemmapReceive(path_recv, result_send)
    recv_process.start()
    time.sleep(0.15)
    with tb.open_file('test.h5', 'r') as fobj:
        array = fobj.get_node('/', 'test')
        start_timestamp = clock()
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
    print_results(send_type, start_timestamp, recv_timestamp, finish_timestamp)
    recv_process.join()


# process to receive an array using a socket
# for real use, this would require creating some protocol to specify the
# array's data type and shape
class SocketReceive(multiprocessing.Process):

    def __init__(self, socket_family, address, result_send, array_nbytes):
        super().__init__()
        self.socket_family = socket_family
        self.address = address
        self.result_send = result_send
        self.array_nbytes = array_nbytes

    def run(self):
        # create the socket, listen for a connection and use select to block
        # until a connection is made
        sock = socket.socket(self.socket_family, socket.SOCK_STREAM)
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
        recv_timestamp = clock()
        # perform an operation on the received array
        array += 1
        finish_timestamp = clock()
        assert(np.all(array == 2))
        # send the timestamps back to the originating process
        self.result_send.send((recv_timestamp, finish_timestamp))
        connection.close()
        sock.close()


def unix_socket_address():
    # create a Unix domain address in the abstract namespace
    # this will only work on Linux
    return b'\x00' + os.urandom(5)


def ipv4_socket_address():
    # create an IPv4 socket address
    return ('127.0.0.1', random.randint(9000, 10_000))


def read_and_send_socket(send_type, array_size, array_bytes, address_func,
                         socket_family):
    address = address_func()
    # start the receiving process and pause to allow it to start up
    result_recv, result_send = multiprocessing.Pipe(False)
    recv_process = SocketReceive(socket_family, address, result_send,
                                 array_bytes)
    recv_process.start()
    time.sleep(0.15)
    with tb.open_file('test.h5', 'r') as fobj:
        array = fobj.get_node('/', 'test')
        start_timestamp = clock()
        # connect to the receiving process' socket
        sock = socket.socket(socket_family, socket.SOCK_STREAM)
        sock.connect(address)
        # read the array from the PyTables file and send its
        # data buffer to the receiving process
        output = array.read(0, array_size, 1)
        sock.send(output.data)
        assert(np.all(output + 1 == 2))
        # receive the timestamps from the other process
        recv_timestamp, finish_timestamp = result_recv.recv()
    sock.close()
    print_results(send_type, start_timestamp, recv_timestamp, finish_timestamp)
    recv_process.join()


def print_results(send_type, start_timestamp, recv_timestamp,
                  finish_timestamp):
    msg = 'type: {0}\t receive: {1:5.5f}, add:{2:5.5f}, total: {3:5.5f}'
    print(msg.format(send_type,
                     recv_timestamp - start_timestamp,
                     finish_timestamp - recv_timestamp,
                     finish_timestamp - start_timestamp))


if __name__ == '__main__':

    random.seed(os.urandom(2))
    array_num_bytes = [10**5, 10**6, 10**7, 10**8]

    for array_bytes in array_num_bytes:
        array_size = array_bytes // 8

        create_file(array_size)
        read_and_send_pipe('multiproc.Pipe', array_size)
        read_and_send_memmap('memmap     ', array_size)
        # comment out this line to run on an OS other than Linux
        read_and_send_socket('Unix socket', array_size, array_bytes,
                             unix_socket_address, socket.AF_UNIX)
        read_and_send_socket('IPv4 socket', array_size, array_bytes,
                             ipv4_socket_address, socket.AF_INET)
        print()
