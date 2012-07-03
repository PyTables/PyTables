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

.. autodata:: __version__

.. autodata:: hdf5Version

.. autodata:: is_pro


Global functions
~~~~~~~~~~~~~~~~
.. autofunction:: copyFile

.. autofunction:: isHDF5File

.. autofunction:: isPyTablesFile

.. autoclass:: lrange

.. autofunction:: openFile

.. autofunction:: setBloscMaxThreads

.. autofunction:: print_versions

.. autofunction:: restrict_flavors

.. autofunction:: split_type

.. autofunction:: test

.. autofunction:: whichLibVersion

.. autofunction:: silenceHDF5Messages


.. _FileClassDescr:

The File Class
--------------
.. autoclass:: File
    :members:


.. _NodeClassDescr:

The Node class
--------------
.. autoclass:: Node
    :members:
    :private-members:


.. _GroupClassDescr:

The Group class
---------------

.. autoclass:: Group
    :members:
    :private-members:


.. _LeafClassDescr:

The Leaf class
--------------

.. autoclass:: Leaf
    :members:


.. _TableClassDescr:

The Table class
---------------

.. autoclass:: Table
    :members:

.. Table class, Cols class, Column, and Row class documentation in
   tables/table.py docstring


.. _ColsClassDescr:

The Cols class
--------------

.. autoclass:: Cols
    :members:


.. _ColumnClassDescr:

The Column class
----------------

.. autoclass:: Column
    :members:


.. _RowClassDescr:

The Row class
-------------

.. autoclass:: tables.tableExtension.Row
    :members:


.. _DescriptionClassDescr:

The Description class
---------------------

.. autoclass:: Description
    :members:


.. _ColClassDescr:

The Col class and its descendants
---------------------------------

.. autoclass:: Col
    :members:

.. autoclass:: StringCol
    :members:

.. autoclass:: BoolCol
    :members:

.. autoclass:: IntCol
    :members:

.. autoclass:: Int8Col
    :members:

.. autoclass:: Int16Col
    :members:

.. autoclass:: Int32Col
    :members:

.. autoclass:: Int64Col
    :members:

.. autoclass:: UIntCol
    :members:

.. autoclass:: UInt8Col
    :members:

.. autoclass:: UInt16Col
    :members:

.. autoclass:: UInt32Col
    :members:

.. autoclass:: UInt64Col
    :members:

.. autoclass:: Float32Col
    :members:

.. autoclass:: Float64Col
    :members:

.. autoclass:: ComplexCol
    :members:

.. autoclass:: TimeCol
    :members:

.. autoclass:: Time32Col
    :members:

.. autoclass:: Time64Col
    :members:

.. autoclass:: EnumCol
    :members:


.. _IsDescriptionClassDescr:

The IsDescription class
-----------------------

.. autoclass:: IsDescription
    :members:


.. _ArrayClassDescr:

The Array class
---------------

.. autoclass:: Array
    :members:


.. _CArrayClassDescr:

The CArray class
----------------

.. autoclass:: CArray
    :members:


.. _EArrayClassDescr:

The EArray class
----------------

.. autoclass:: EArray
    :members:


.. _VLArrayClassDescr:

The VLArray class
-----------------

.. autoclass:: VLArray
    :members:


.. _LinkClassDescr:

The Link class
--------------

.. autoclass:: tables.link.Link
    :members:


.. _SoftLinkClassDescr:

The SoftLink class
------------------

.. autoclass:: tables.link.SoftLink
    :members:


.. _ExternalLinkClassDescr:

The ExternalLink class
----------------------

.. autoclass:: tables.link.ExternalLink
    :members:


.. _UnImplementedClassDescr:

The UnImplemented class
-----------------------

.. autoclass:: UnImplemented
    :members:


.. _UnknownClassDescr:

The Unknown class
-----------------

.. autoclass:: Unknown
    :members:


.. _AttributeSetClassDescr:

The AttributeSet class
----------------------

.. autoclass:: tables.attributeset.AttributeSet
    :members:


Declarative classes
-------------------
In this section a series of classes that are meant to
*declare* datatypes that are required for creating
primary PyTables datasets are described.


.. _AtomClassDescr:

The Atom class and its descendants
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: Atom
    :members:


.. _AtomConstructors:

Atom Constructors
^^^^^^^^^^^^^^^^^

.. autoclass:: StringAtom
    :members:

.. autoclass:: BoolAtom
    :members:

.. autoclass:: IntAtom
    :members:

.. autoclass:: Int8Atom
    :members:

.. autoclass:: Int16Atom
    :members:

.. autoclass:: Int32Atom
    :members:

.. autoclass:: Int64Atom
    :members:

.. autoclass:: UIntAtom
    :members:

.. autoclass:: UInt8Atom
    :members:

.. autoclass:: UInt16Atom
    :members:

.. autoclass:: UInt32Atom
    :members:

.. autoclass:: UInt64Atom
    :members:

.. autoclass:: FloatAtom
    :members:

.. autoclass:: Float32Atom
    :members:

.. autoclass:: Float64Atom
    :members:

.. autoclass:: ComplexAtom
    :members:

.. autoclass:: Time32Atom
    :members:

.. autoclass:: Time64Atom
    :members:

.. autoclass:: EnumAtom
    :members:


Pseudo atoms
^^^^^^^^^^^^
Now, there come three special classes, ObjectAtom, VLStringAtom and
VLUnicodeAtom, that actually do not descend from Atom, but which goal is so
similar that they should be described here. Pseudo-atoms can only be used with
VLArray datasets (see :ref:`VLArrayClassDescr`), and they do not support
multidimensional values, nor multiple values per row.

They can be recognised because they also have kind, type and shape attributes,
but no size, itemsize or dflt ones. Instead, they have a base atom which
defines the elements used for storage.

See :file:`examples/vlarray1.py` and :file:`examples/vlarray2.py` for further
examples on VLArray datasets, including object serialization and string
management.


Object Atom
...........

.. autoclass:: ObjectAtom
    :members:


.. _VLStringAtom:

VLStringAtom
............

.. autoclass:: VLStringAtom
    :members:


.. _VLUnicodeAtom:

VLUnicodeAtom
.............

.. autoclass:: VLUnicodeAtom
    :members:


Helper classes
--------------
.. move this section into its own file

This section describes some classes that do not fit in any other
section and that mainly serve for ancillary purposes.


.. _FiltersClassDescr:

The Filters class
-----------------

.. autoclass:: Filters
    :members:

.. _IndexClassDescr:

The Index class
---------------

.. autoclass:: tables.index.Index
    :members:


.. _EnumClassDescr:

The Enum class
--------------

.. autoclass:: tables.misc.enum.Enum
    :members:


The Expr class - a general-purpose expression evaluator
-------------------------------------------------------

.. autoclass:: Expr
    :members:


.. _ExceptionsDescr:

Exceptions module
-----------------
In the :mod:`exceptions` module exceptions and warnings that are specific
to PyTables are declared.

.. autoexception:: tables.HDF5ExtError
    :members:

.. autoexception:: ClosedNodeError

.. autoexception:: ClosedFileError

.. autoexception:: FileModeError

.. autoexception:: NodeError

.. autoexception:: NoSuchNodeError

.. autoexception:: UndoRedoError

.. autoexception:: UndoRedoWarning

.. autoexception:: NaturalNameWarning

.. autoexception:: PerformanceWarning

.. autoexception:: FlavorError

.. autoexception:: FlavorWarning

.. autoexception:: FiltersWarning

.. autoexception:: OldIndexWarning

.. autoexception:: DataTypeWarning

.. autoexception:: Incompat16Warning

.. autoexception:: ExperimentalFeatureWarning
