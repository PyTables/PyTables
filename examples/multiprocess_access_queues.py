"""Example showing how to access a PyTables file from multiple processes using
queues."""

import queue

import multiprocessing
import random
import time
from pathlib import Path

import numpy as np
import tables as tb


# this creates an HDF5 file with one array containing n rows
def make_file(file_path, n):

    with tb.open_file(file_path, 'w') as fobj:
        array = fobj.create_carray('/', 'array', tb.Int64Atom(), (n, n))
        for i in range(n):
            array[i, :] = i


# All access to the file goes through a single instance of this class.
# It contains several queues that are used to communicate with other
# processes.
# The read_queue is used for requests to read data from the HDF5 file.
# A list of result_queues is used to send data back to client processes.
# The write_queue is used for requests to modify the HDF5 file.
# One end of a pipe (shutdown) is used to signal the process to terminate.
class FileAccess(multiprocessing.Process):

    def __init__(self, h5_path, read_queue, result_queues, write_queue,
                 shutdown):
        self.h5_path = h5_path
        self.read_queue = read_queue
        self.result_queues = result_queues
        self.write_queue = write_queue
        self.shutdown = shutdown
        self.block_period = .01
        super().__init__()

    def run(self):
        self.h5_file = tb.open_file(self.h5_path, 'r+')
        self.array = self.h5_file.get_node('/array')
        another_loop = True
        while another_loop:

            # Check if the process has received the shutdown signal.
            if self.shutdown.poll():
                another_loop = False

            # Check for any data requests in the read_queue.
            try:
                row_num, proc_num = self.read_queue.get(
                    True, self.block_period)
                # look up the appropriate result_queue for this data processor
                # instance
                result_queue = self.result_queues[proc_num]
                print('processor {} reading from row {}'.format(proc_num,
                                                                  row_num))
                result_queue.put(self.read_data(row_num))
                another_loop = True
            except queue.Empty:
                pass

            # Check for any write requests in the write_queue.
            try:
                row_num, data = self.write_queue.get(True, self.block_period)
                print('writing row', row_num)
                self.write_data(row_num, data)
                another_loop = True
            except queue.Empty:
                pass

        # close the HDF5 file before shutting down
        self.h5_file.close()

    def read_data(self, row_num):
        return self.array[row_num, :]

    def write_data(self, row_num, data):
        self.array[row_num, :] = data


# This class represents a process that does work by reading and writing to the
# HDF5 file.  It does this by sending requests to the FileAccess class instance
# through its read and write queues.  The data results are sent back through
# the result_queue.
# Its actions are logged to a text file.
class DataProcessor(multiprocessing.Process):

    def __init__(self, read_queue, result_queue, write_queue, proc_num,
                 array_size, output_file):
        self.read_queue = read_queue
        self.result_queue = result_queue
        self.write_queue = write_queue
        self.proc_num = proc_num
        self.array_size = array_size
        self.output_file = output_file
        super().__init__()

    def run(self):
        self.output_file = open(self.output_file, 'w')
        # read a random row from the file
        row_num = random.randrange(self.array_size)
        self.read_queue.put((row_num, self.proc_num))
        self.output_file.write(str(row_num) + '\n')
        self.output_file.write(str(self.result_queue.get()) + '\n')

        # modify a random row to equal 11 * (self.proc_num + 1)
        row_num = random.randrange(self.array_size)
        new_data = (np.zeros((1, self.array_size), 'i8') +
                    11 * (self.proc_num + 1))
        self.write_queue.put((row_num, new_data))

        # pause, then read the modified row
        time.sleep(0.015)
        self.read_queue.put((row_num, self.proc_num))
        self.output_file.write(str(row_num) + '\n')
        self.output_file.write(str(self.result_queue.get()) + '\n')
        self.output_file.close()


# this function starts the FileAccess class instance and
# sets up all the queues used to communicate with it
def make_queues(num_processors):
    read_queue = multiprocessing.Queue()
    write_queue = multiprocessing.Queue()
    shutdown_recv, shutdown_send = multiprocessing.Pipe(False)
    result_queues = [multiprocessing.Queue() for i in range(num_processors)]
    file_access = FileAccess(file_path, read_queue, result_queues, write_queue,
                             shutdown_recv)
    file_access.start()
    return read_queue, result_queues, write_queue, shutdown_send


if __name__ == '__main__':
    # See the discussion in :issue:`790`.
    multiprocessing.set_start_method('spawn')

    file_path = 'test.h5'
    n = 10
    make_file(file_path, n)

    num_processors = 3
    (read_queue, result_queues,
     write_queue, shutdown_send) = make_queues(num_processors)

    processors = []
    output_files = []
    for i in range(num_processors):
        result_queue = result_queues[i]
        output_file = str(i)
        processor = DataProcessor(read_queue, result_queue, write_queue, i, n,
                                  output_file)
        processors.append(processor)
        output_files.append(output_file)

    # start all DataProcessor instances
    for processor in processors:
        processor.start()

    # wait for all DataProcessor instances to finish
    for processor in processors:
        processor.join()

    # shut down the FileAccess instance
    shutdown_send.send(0)

    # print out contents of log files and delete them
    print()
    for output_file in output_files:
        print()
        print(f'contents of log file {output_file}')
        print(open(output_file).read())
        Path(output_file).unlink()

    Path('test.h5').unlink()
