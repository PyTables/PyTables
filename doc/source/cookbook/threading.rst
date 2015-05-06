=========
Threading
=========

.. py:currentmodule:: tables


Background
==========

Several bug reports have been filed in the past by the users regarding
problems related to the impossibility to use PyTables in multi-thread
programs.

The problem was mainly related to an internal registry that forced the
sharing of HDF5 file handles across multiple threads.

In PyTables 3.1.0 the code for file handles management has been completely
redesigned (see the *Backward incompatible changes* section in 
:doc:`../release-notes/RELEASE_NOTES_v3.1.x`) to be more simple and
transparent and to allow the use of PyTables in multi-thread programs.

Citing the :doc:`../release-notes/RELEASE_NOTES_v3.1.x`::

    It is important to stress that the new implementation still has an
    internal registry (implementation detail) and it is still
    **not thread safe**.
    Just now a smart enough developer should be able to use PyTables in a
    muti-thread program without too much headaches.


A common schema for concurrency
===============================

Although it is probably not the most efficient or elegant solution to solve
a certain class of problems, many users seems to like the possibility to
load a portion of data and process it inside a *thread function* using
multiple threads to process the entire dataset.

Each thread is responsible of:

* opening the (same) HDF5 file for reading,
* load data from it and
* close the HDF5 file itself

Each file handle is of exclusive use of the thread that opened it and
file handles are never shared across threads.

In order to do it in a safe way with PyTables some care should be used
during the phase of opening and closing HDF5 files in order ensure the
correct behaviour of the internal machinery used to manage HDF5 file handles.


Very simple solution
====================

A very simple solution for this kind of scenario is to use a
:class:`threading.Lock` around part of the code that are considered critical
e.g. the :func:`open_file` function and the :meth:`File.close` method::

    import threading
    
    lock = threading.Lock()

    def synchronized_open_file(*args, **kwargs):
        with lock:
            return tb.open_file(*args, **kwargs)

    def synchronized_close_file(self, *args, **kwargs):
        with lock:
            return self.close(*args, **kwargs)


The :func:`synchronized_open_file` and :func:`synchronized_close_file` can
be used in the *thread function* to open and close the HDF5 file::

    import numpy as np
    import tables as tb

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


Finally the main function of the program:

* instantiates the input and output :class:`queue.Queue`,
* starts all threads, 
* sends the processing requests on the input :class:`queue.Queue`
* collects results reading from the output :class:`queue.Queue`
* performs finalization actions (:meth:`threading.Thread.join`)

.. code-block:: python

    import os
    import queue
    import threading

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
        chunk_size = int(np.ceil(data_size / nchunks))
        for start in range(0, data_size, chunk_size):
            yield slice(start, start + chunk_size)

    def main():
        # generate the test data
        if not os.path.exists(FILENAME):
            create_test_file(FILENAME)

        threads = []
        inqueue = queue.Queue()
        outqueue = queue.Queue()

        # start all threads
        for i in range(NTHREADS):
            thread = threading.Thread(
                target=run, args=(FILENAME, H5PATH, inqueue, outqueue))
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

The program in the example computes the mean value of a potentially huge
dataset splinting the computation across :data:`NTHREADS` (5 in this case)
threads.

The complete and working code of this example (Python 3 is required) can be
found in the :file:`examples` directory:
:download:`simple_threading.py <../../../examples/simple_threading.py>`.

The approach presented in this section is very simple and readable but has
the **drawback** that the user code have to be modified to replace
:func:`open_file` and :meth:`File.close` calls with their safe version
(:func:`synchronized_open_file` and :func:`synchronized_close_file`).

Also, the solution showed in the example does not cover the entire PyTables
API (e.g. although not recommended HDF5 files can be opened using the
:class:`File` constructor) and makes it impossible to use *pythonic*
constructs like the *with* statement::

    with tb.open_file(filename) as h5file:
        do_something(h5file)


Monkey-patching PyTables
========================

An alternative implementation with respect to the `Very simple solution`_
presented in the previous section consists in monkey-patching the PyTables
package to replace some of its components with a more thread-safe version of
themselves::

    import threading

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


At this point PyTables can be used transparently in example program presented
in the previous section.
In particular the standard PyTables API (including *with* statements) can be
used in the *thread function*::

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


The complete code of this version of the example can be found in the
:file:`examples` folder:
:download:`simple_threading.py <../../../examples/threading_monkeypatch.py>`.
Python 3 is required.


