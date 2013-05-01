.. _library_reference:

Library Reference
=================
PyTables implements several classes to represent the different nodes in the
object tree. They are named File, Group, Leaf, Table, Array, CArray, EArray,
VLArray and UnImplemented. Another one allows the user to complement the
information on these different objects; its name is AttributeSet. Finally,
another important class called IsDescription allows to build a Table record
description by declaring a subclass of it. Many other classes are defined in
PyTables, but they can be regarded as helpers whose goal is mainly to declare
the *data type properties* of the different first class objects and will be
described at the end of this chapter as well.

An important function, called open_file is responsible to create, open or append
to files. In addition, a few utility functions are defined to guess if the user
supplied file is a *PyTables* or *HDF5* file. These are called is_pytables_file()
and is_hdf5_file(), respectively. There exists also a function called
which_lib_version() that informs about the versions of the underlying C libraries
(for example, HDF5 or Zlib) and another called print_versions() that prints all
the versions of the software that PyTables relies on. Finally, test() lets you
run the complete test suite from a Python console interactively.

.. toctree::
    :maxdepth: 2

    libref/top_level
    libref/file_class
    libref/hierarchy_classes
    libref/structured_storage
    libref/homogenous_storage
    libref/link_classes
    libref/declarative_classes
    libref/helper_classes
    libref/expr_class
    libref/filenode_classes
