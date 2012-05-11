.. _library_reference:

Library Reference
=================
PyTables implements several classes to represent the different
nodes in the object tree. They are named File,
Group, Leaf,
Table, Array,
CArray, EArray,
VLArray and UnImplemented. Another
one allows the user to complement the information on these different
objects; its name is AttributeSet. Finally, another
important class called IsDescription allows to build
a Table record description by declaring a subclass of
it. Many other classes are defined in PyTables, but they can be regarded
as helpers whose goal is mainly to declare the *data type
properties* of the different first class objects and will be
described at the end of this chapter as well.

An important function, called openFile is
responsible to create, open or append to files. In addition, a few
utility functions are defined to guess if the user supplied file is a
*PyTables* or *HDF5* file. These
are called isPyTablesFile() and
isHDF5File(), respectively. There exists also a
function called whichLibVersion() that informs about
the versions of the underlying C libraries (for example, HDF5 or
Zlib) and another called
print_versions() that prints all the versions of the
software that PyTables relies on. Finally, test()
lets you run the complete test suite from a Python console
interactively.

Let's start discussing the first-level variables and functions
available to the user, then the different classes defined in
PyTables.

.. currentmodule:: tables


tables variables and functions
------------------------------

Global variables
~~~~~~~~~~~~~~~~

.. data:: __version__

    The PyTables version number.

.. data:: hdf5Version

    The underlying HDF5 library version number.

.. data:: is_pro

    True for PyTables Professional edition, false otherwise.

    .. note:: PyTables Professional edition has been released
       under an open source license. Starting with version 2.3,
       PyTables includes all features of PyTables Pro.
       In order to reflect the presence of advanced
       features :data:`is_pro` is always
       set to True.  :data:`is_pro` should be
       considered *deprecated*.
       It will be removed in the next major release.

    .. deprecated:: 2.3

Global functions
~~~~~~~~~~~~~~~~
.. autofunction:: tables.copyFile


.. function:: isHDF5File(filename)

    Determine whether a file is in the HDF5 format.

    When successful, it returns a true value if the file is an
    HDF5 file, false otherwise. If there were problems identifying the
    file, an HDF5ExtError is raised.

.. function:: isPyTablesFile(filename)

    Determine whether a file is in the PyTables format.

    When successful, it returns the format version string if the
    file is a PyTables file, None otherwise.  If
    there were problems identifying the file,
    an HDF5ExtError is raised.

.. function:: lrange([start, ]stop[, step])

    Iterate over long ranges.

    This is similar to xrange(), but it
    allows 64-bit arguments on all platforms.  The results of the
    iteration are sequentially yielded in the form of
    numpy.int64 values, but getting random
    individual items is not supported.

    Because of the Python 32-bit limitation on object lengths,
    the length attribute (which is also a
    numpy.int64 value) should be used instead of
    the len() syntax.

    Default start and step
    arguments are supported in the same way as in
    xrange().  When the standard
    [x]range() Python objects support 64-bit
    arguments, this iterator will be deprecated.


.. autofunction:: tables.openFile


.. function:: setBloscMaxThreads(nthreads)

    Set the maximum number of threads that Blosc can use.

    This actually overrides the :data:`tables.parameters.MAX_BLOSC_THREADS`
    setting in :mod:`tables.parameters`, so the new value will be effective
    until this function is called again or a new file with a different
    :data:`tables.parameters.MAX_BLOSC_THREADS` value is specified.

    Returns the previous setting for maximum threads.

.. function:: print_versions()

    Print all the versions of software that PyTables relies on.

.. function:: restrict_flavors(keep=['python'])

    Disable all flavors except those in keep.

    Providing an empty keep sequence implies
    disabling all flavors (but the internal one).  If the sequence is
    not specified, only optional flavors are disabled.

    .. important:: Once you disable a flavor, it can not be enabled again.

.. function:: split_type(type)

    Split a PyTables type into a PyTables
    kind and an item size.

    Returns a tuple of (kind, itemsize). If
    no item size is present in the type (in the
    form of a precision), the returned item size is
    None::

        >>> split_type('int32')
        ('int', 4)
        >>> split_type('string')
        ('string', None)
        >>> split_type('int20')
        Traceback (most recent call last):
        ...
        ValueError: precision must be a multiple of 8: 20
        >>> split_type('foo bar')
        Traceback (most recent call last):
        ...
        ValueError: malformed type: 'foo bar'

.. function:: test(verbose=False, heavy=False)

    Run all the tests in the test suite.

    If verbose is set, the test suite will
    emit messages with full verbosity (not recommended unless you are
    looking into a certain problem).

    If heavy is set, the test suite will be
    run in *heavy* mode (you should be careful with
    this because it can take a lot of time and resources from your
    computer).

.. function:: whichLibVersion(name)

    Get version information about a C library.

    If the library indicated by name is
    available, this function returns a 3-tuple containing the major
    library version as an integer, its full version as a string, and
    the version date as a string. If the library is not available,
    None is returned.

    The currently supported library names are
    hdf5, zlib,
    lzo and bzip2. If another
    name is given, a ValueError is raised.

.. function:: silenceHDF5Messages(silence=True)

   Silence (or re-enable) messages from the HDF5 C library.

   The *silence* parameter can be used control the behaviour and reset the
   standard HDF5 logging.

   .. versionadded:: 2.4


.. _FileClassDescr:

The File Class
--------------
.. autoclass:: tables.File
    :members:


.. _NodeClassDescr:

The Node class
--------------
.. autoclass:: tables.Node


.. Group class documentation in tables/group.py docstring

.. automodule:: tables.group


.. Leaf class documentation in tables/leaf.py docstring

.. automodule:: tables.leaf


.. Table class, Cols class, Column, and Row class documentation in
   tables/table.py docstring

.. automodule:: tables.table


.. Description class, Col class, and IsDescription class class documentation in
   tables/description.py docstring

.. automodule:: tables.description


.. Array class documentation in tables/array.py

.. automodule:: tables.array


.. CArray class documentation in tables/carray.py

.. automodule:: tables.carray


.. EArray class documentation in tables/earray.py

.. automodule:: tables.earray


.. VLArray class documentation in tables/vlarray.py

.. automodule:: tables.vlarray


.. Link class documentation in tables/link.py

.. automodule:: tables.link


.. UnImplemented class and Unknown class documentation in tables/unimplemented.py

.. automodule:: tables.unimplemented


.. AttributeSet class documentation in tables/attributeset.py

.. automodule:: tables.attributeset


.. All Atom and Pseudoatom class documentation in tables/atom.py

.. automodule:: tables.atom


Helper classes
.. --------------
.. move this section into its own file

This section describes some classes that do not fit in any other
section and that mainly serve for ancillary purposes.


.. Filters class documentation in tables/filters.py

.. automodule:: tables.filters


.. Index class documentation in tables/index.py

.. automodule:: tables.index


.. Enum class documentation in tables/misc/enum.py

.. automodule:: tables.misc.enum


.. Expr class documentation in tables/expression.py

.. automodule:: tables.expression


.. HDF5ExtError and other PyTables specific exception classes are
   documented in tables/exceptions.py

.. automodule:: tables.exceptions

