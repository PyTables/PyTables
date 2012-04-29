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

.. Top-level function documentation in tables/__init.py

.. automodule:: tables


.. File class documentation in tables/file.py docstring

.. automodule:: tables.file


.. Node class documentation in tables/node.py docstring

.. automodule:: tables.node


.. Group class documentation in tables/group.py docstring

.. automodule:: tables.group


.. Leaf class documentation in tables/leaf.py docstring

.. automodule:: tables.leaf


.. Table class, Cols class, and Column class documentation in tables/table.py docstring

.. automodule:: tables.table


.. Description class, Col class, IsDescription class, and Row class documentation in
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

