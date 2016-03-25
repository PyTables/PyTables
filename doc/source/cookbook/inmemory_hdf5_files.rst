====================
In-memory HDF5 files
====================

The HDF5 library provides functions to allow an application to work with a
file in memory for faster reads and writes. File contents are kept in memory
until the file is closed.  At closing, the memory version of the file can be
written back to disk or abandoned.


Open an existing file in memory
===============================

Assuming the :file:`sample.h5` exists in the current folder, it is possible to
open it in memory simply using the CORE driver at opening time.

The HDF5 driver that one intend to use to open/create a file can be specified
using the *driver* keyword argument of the :func:`tables.open_file` function::

    >>> import tables
    >>> h5file = tables.open_file("sample.h", driver="H5FD_CORE")

The content of the :file`sample.h5` is opened for reading. It is loaded into
memory and all reading operations are performed without disk I/O overhead.

.. note::

    the initial loading of the entire file into memory can be time expensive
    depending on the size of the opened file and on the performances of the
    disk subsystem.

.. seealso::

    general information about HDF5 drivers can be found in the `Alternate
    File Storage Layouts and Low-level File Drivers`__ section of the `HDF5
    User's Guide`_.

__ `HDF5 drivers`_


Creating a new file in memory
=============================

Creating a new file in memory is as simple as creating a regular file, just
one needs to specify to use the CORE driver::

    >>> import tables
    >>> h5file = tables.open_file("new_sample.h5", "w", driver="H5FD_CORE")
    >>> import numpy
    >>> a = h5file.create_array(h5file.root, "array", numpy.zeros((300, 300)))
    >>> h5file.close()


Backing store
=============

In the previous example contents of the in-memory `h5file` are automatically
saved to disk when the file descriptor is closed, so a new
:file:`new_sample.h5` file is created and all data are transferred to disk.

Again this can be time a time expensive action depending on the amount of
data in the HDF5 file and depending on how fast is the disk I/O.

Saving data to disk is the default behavior for the CORE driver in PyTables.

This feature can be controlled using the *driver_core_backing_store*
parameter of the :func:`tables.open_file` function.  Setting it to `False`
disables the backing store feature and all changes in the working `h5file`
are lost after closing::

    >>> h5file = tables.open_file("new_sample.h5", "w", driver="H5FD_CORE",
    ...                           driver_core_backing_store=0)

Please note that the *driver_core_backing_store* disables saving of data, not
loading.
In the following example the :file:`sample.h5` file is opened in-memory in
append mode.  All data in the existing :file:`sample.h5` file are loaded into
memory and contents can be actually modified by the user::

    >>> import tables
    >>> h5file = tables.open_file("sample.h5", "a", driver="H5FD_CORE",
                                  driver_core_backing_store=0)
    >>> import numpy
    >>> h5file.create_array(h5file.root, "new_array", numpy.arange(20),
                            title="New array")
    >>> array2 = h5file.root.array2
    >>> print(array2)
    /array2 (Array(20,)) 'New array'
    >>> h5file.close()

Modifications are lost when the `h5file` descriptor is closed.


Memory images of HDF5 files
===========================

It is possible to get a memory image of an HDF5 file (see
`HDF5 File Image Operations`_).  This feature is only available if PyTables
is build against version 1.8.9 or newer of the HDF5 library.

In particular getting a memory image of an HDF5 file is possible only if the
file has been opened with one of the following drivers: SEC2 (the default
one), STDIO or CORE.

An example of how to get an image::

    >>> import tables
    >>> h5file = tables.open_file("sample.h5")
    >>> image = h5file.get_file_image()
    >>> h5file.close()

The memory ìmage of the :file:`sample.h5` file is copied into the `ìmage`
string (of bytes).

.. note::

    the `ìmage` string contains all data stored in the HDF5 file so, of
    course, it can be quite large.

The `ìmage` string can be passed around and can also be used to initialize a
new HDF5 file descriptor::

    >>> import tables
    >>> h5file = tables.open_file("in-memory-sample.h5", driver="H5FD_CORE",
                                  driver_core_image=image,
                                  driver_core_backing_store=0)
    >>> print(h5file.root.array)
    /array (Array(300, 300)) 'Array'
    >>> h5file.setNodeAttr(h5file.root, "description", "In memory file example")



-----


.. target-notes::

.. _`HDF5 drivers`: http://www.hdfgroup.org/HDF5/doc/UG/08_TheFile.html#Drivers
.. _`HDF5 User's Guide`: http://www.hdfgroup.org/HDF5/doc/UG/index.html
.. _`HDF5 File Image Operations`: http://www.hdfgroup.org/HDF5/doc/Advanced/FileImageOperations/HDF5FileImageOperations.pdf
