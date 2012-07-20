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

An important function, called openFile is responsible to create, open or append
to files. In addition, a few utility functions are defined to guess if the user
supplied file is a *PyTables* or *HDF5* file. These are called isPyTablesFile()
and isHDF5File(), respectively. There exists also a function called
whichLibVersion() that informs about the versions of the underlying C libraries
(for example, HDF5 or Zlib) and another called print_versions() that prints all
the versions of the software that PyTables relies on. Finally, test() lets you
run the complete test suite from a Python console interactively.

Let's start discussing the first-level variables and functions available to the
user, then the different classes defined in PyTables.

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

..  These are defined in the class docstring.
    This is necessary because attributes created in a class's
    __init__ method can't be documented with autoattribute.
    See Sphinx bug #904.
    https://bitbucket.org/birkenfeld/sphinx/issue/904

    Attributes
    ~~~~~~~~~~
    .. autoattribute:: File.filename
    .. autoattribute:: File.format_version
    .. autoattribute:: File.isopen
    .. autoattribute:: File.mode
    .. autoattribute:: File.root
    .. autoattribute:: File.rootUEP


File properties
~~~~~~~~~~~~~~~
.. autoattribute:: File.title

.. autoattribute:: File.filters

.. autoattribute:: File.open_count


File methods - file handling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automethod:: File.close

.. automethod:: File.copyFile

.. automethod:: File.flush

.. automethod:: File.fileno

.. automethod:: File.__enter__

.. automethod:: File.__exit__

.. automethod:: File.__str__

.. automethod:: File.__repr__


File methods - hierarchy manipulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automethod:: File.copyChildren

.. automethod:: File.copyNode

.. automethod:: File.createArray

.. automethod:: File.createCArray

.. automethod:: File.createEArray

.. automethod:: File.createExternalLink

.. automethod:: File.createGroup

.. automethod:: File.createHardLink

.. automethod:: File.createSoftLink

.. automethod:: File.createTable

.. automethod:: File.createVLArray

.. automethod:: File.moveNode

.. automethod:: File.removeNode

.. automethod:: File.renameNode


File methods - tree traversal
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automethod:: File.getNode

.. automethod:: File.isVisibleNode

.. automethod:: File.iterNodes

.. automethod:: File.listNodes

.. automethod:: File.walkGroups

.. automethod:: File.walkNodes

.. automethod:: File.__contains__

.. automethod:: File.__iter__


File methods - Undo/Redo support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automethod:: File.disableUndo

.. automethod:: File.enableUndo

.. automethod:: File.getCurrentMark

.. automethod:: File.goto

.. automethod:: File.isUndoEnabled

.. automethod:: File.mark

.. automethod:: File.redo

.. automethod:: File.undo


File methods - attribute handling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automethod:: File.copyNodeAttrs

.. automethod:: File.delNodeAttr

.. automethod:: File.getNodeAttr

.. automethod:: File.setNodeAttr


.. _NodeClassDescr:

The Node class
--------------
.. autoclass:: Node

.. These are defined in class docstring
    .. autoattribute:: Node._v_depth
    .. autoattribute:: Node._v_file
    .. autoattribute:: Node._v_name
    .. autoattribute:: Node._v_pathname
    .. autoattribute:: Node._v_objectID (location independent)

Node instance variables - location dependent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoattribute:: Node._v_parent


Node instance variables - location independent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoattribute:: Node._v_attrs

.. autoattribute:: Node._v_isopen


Node instance variables - attribute shorthands
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoattribute:: Node._v_title


Node methods - hierarchy manipulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automethod:: Node._f_close

.. automethod:: Node._f_copy

.. automethod:: Node._f_isVisible

.. automethod:: Node._f_move

.. automethod:: Node._f_remove

.. automethod:: Node._f_rename


Node methods - attribute handling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automethod:: Node._f_delAttr

.. automethod:: Node._f_getAttr

.. automethod:: Node._f_setAttr


.. _GroupClassDescr:

The Group class
---------------
.. autoclass:: Group

..  These are defined in the class docstring
    Group instance variables
    ~~~~~~~~~~~~~~~~~~~~~~~~
    The following instance variables are provided in addition to those in Node
    (see :ref:`NodeClassDescr`):

    .. autoattribute:: Group._v_children
    .. autoattribute:: Group._v_groups
    .. autoattribute:: Group._v_hidden
    .. autoattribute:: Group._v_leaves
    .. autoattribute:: Group._v_links
    .. autoattribute:: Group._v_unknown

Group properties
~~~~~~~~~~~~~~~~
.. autoattribute:: Group._v_nchildren

.. autoattribute:: Group._v_filters


Group methods
~~~~~~~~~~~~~

.. important::

    *Caveat:* The following methods are documented for completeness, and they
    can be used without any problem. However, you should use the high-level
    counterpart methods in the File class (see :ref:`FileClassDescr`, because
    they are most used in documentation and examples, and are a bit more
    powerful than those exposed here.

The following methods are provided in addition to those in
Node (see :ref:`NodeClassDescr`):


.. automethod:: Group._f_close

.. automethod:: Group._f_copy

.. automethod:: Group._f_copyChildren

.. automethod:: Group._f_getChild

.. automethod:: Group._f_iterNodes

.. automethod:: Group._f_listNodes

.. automethod:: Group._f_walkGroups

.. automethod:: Group._f_walkNodes


Group special methods
~~~~~~~~~~~~~~~~~~~~~
Following are described the methods that automatically trigger actions when a
Group instance is accessed in a special way.

This class defines the :meth:`__setattr__`, :meth:`__getattr__` and
:meth:`__delattr__` methods, and they set, get and delete *ordinary Python
attributes* as normally intended. In addition to that, :meth:`__getattr__`
allows getting *child nodes* by their name for the sake of easy interaction
on the command line, as long as there is no Python attribute with the same
name. Groups also allow the interactive completion (when using readline) of
the names of child nodes. For instance::

    # get a Python attribute
    nchild = group._v_nchildren

    # Add a Table child called 'table' under 'group'.
    h5file.createTable(group, 'table', myDescription)
    table = group.table          # get the table child instance
    group.table = 'foo'          # set a Python attribute

    # (PyTables warns you here about using the name of a child node.)
    foo = group.table            # get a Python attribute
    del group.table              # delete a Python attribute
    table = group.table          # get the table child instance again

.. automethod:: Group.__contains__

.. automethod:: Group.__delattr__

.. automethod:: Group.__getattr__

.. automethod:: Group.__iter__

.. automethod:: Group.__repr__

.. automethod:: Group.__setattr__

.. automethod:: Group.__str__


.. _LeafClassDescr:

The Leaf class
--------------
.. autoclass:: Leaf

..  These are defined in the class docstring
    .. _LeafInstanceVariables:

    Leaf instance variables
    ~~~~~~~~~~~~~~~~~~~~~~~
    These instance variables are provided in addition to those in Node
    (see :ref:`NodeClassDescr`):

    .. autoattribute:: Leaf.byteorder
    .. autoattribute:: Leaf.dtype
    .. autoattribute:: Leaf.extdim
    .. autoattribute:: Leaf.nrows
    .. autoattribute:: Leaf.nrowsinbuf
    .. autoattribute:: Leaf.shape


Leaf properties
~~~~~~~~~~~~~~~
.. autoattribute:: Leaf.chunkshape

.. autoattribute:: Leaf.ndim

.. autoattribute:: Leaf.filters

.. autoattribute:: Leaf.maindim

.. autoattribute:: Leaf.flavor

.. attribute:: Leaf.size_in_memory

    The size of this leaf's data in bytes when it is fully loaded into
    memory.

.. autoattribute:: Leaf.size_on_disk


Leaf instance variables - aliases
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The following are just easier-to-write aliases to their Node (see
:ref:`NodeClassDescr`) counterparts (indicated between parentheses):

.. autoattribute:: Leaf.attrs

.. autoattribute:: Leaf.name

.. autoattribute:: Leaf.objectID

.. autoattribute:: Leaf.title


Leaf methods
~~~~~~~~~~~~
.. automethod:: Leaf.close

.. automethod:: Leaf.copy

.. automethod:: Leaf.delAttr

.. automethod:: Leaf.flush

.. automethod:: Leaf.getAttr

.. automethod:: Leaf.isVisible

.. automethod:: Leaf.move

.. automethod:: Leaf.rename

.. automethod:: Leaf.remove

.. automethod:: Leaf.setAttr

.. automethod:: Leaf.truncate

.. automethod:: Leaf.__len__

.. automethod:: Leaf._f_close


.. _TableClassDescr:

The Table class
---------------
.. autoclass:: Table

.. These are defined in the class docstring
    .. _TableInstanceVariablesDescr:

    Table instance variables
    ~~~~~~~~~~~~~~~~~~~~~~~~
    The following instance variables are provided in addition to those in Leaf
    (see :ref:`LeafClassDescr`).  Please note that there are several col*
    dictionaries to ease retrieving information about a column directly by its
    path name, avoiding the need to walk through Table.description or Table.cols.

    .. autoattribute:: Table.coldescrs
    .. autoattribute:: Table.coldflts
    .. autoattribute:: Table.coldtypes
    .. autoattribute:: Table.colindexed
    .. autoattribute:: Table.colinstances
    .. autoattribute:: Table.colnames
    .. autoattribute:: Table.colpathnames
    .. autoattribute:: Table.cols
    .. autoattribute:: Table.coltypes
    .. autoattribute:: Table.description
    .. autoattribute:: Table.extdim
    .. autoattribute:: Table.indexed
    .. autoattribute:: Table.nrows


Table properties
~~~~~~~~~~~~~~~~
.. autoattribute:: Table.autoIndex

.. autoattribute:: Table.colindexes

.. autoattribute:: Table.indexedcolpathnames

.. autoattribute:: Table.row

.. autoattribute:: Table.rowsize


Table methods - reading
~~~~~~~~~~~~~~~~~~~~~~~
.. automethod:: Table.col

.. automethod:: Table.iterrows

.. automethod:: Table.itersequence

.. automethod:: Table.itersorted

.. automethod:: Table.read

.. automethod:: Table.readCoordinates

.. automethod:: Table.readSorted

.. automethod:: Table.__getitem__

.. automethod:: Table.__iter__


Table methods - writing
~~~~~~~~~~~~~~~~~~~~~~~
.. automethod:: Table.append

.. automethod:: Table.modifyColumn

.. automethod:: Table.modifyColumns

.. automethod:: Table.modifyCoordinates

.. automethod:: Table.modifyRows

.. automethod:: Table.removeRows

.. automethod:: Table.__setitem__


.. _TableMethods_querying:

Table methods - querying
~~~~~~~~~~~~~~~~~~~~~~~~
.. automethod:: Table.getWhereList

.. automethod:: Table.readWhere

.. automethod:: Table.where

.. automethod:: Table.whereAppend

.. automethod:: Table.willQueryUseIndexing


Table methods - other
~~~~~~~~~~~~~~~~~~~~~
.. automethod:: Table.copy

.. automethod:: Table.flushRowsToIndex

.. automethod:: Table.getEnum

.. automethod:: Table.reIndex

.. automethod:: Table.reIndexDirty


.. _DescriptionClassDescr:

The Description class
~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: Description

..  These are defined in the class docstring
    Description instance variables
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    .. autoattribute:: Description._v_colObjects
    .. autoattribute:: Description._v_dflts
    .. autoattribute:: Description._v_dtype
    .. autoattribute:: Description._v_dtypes
    .. autoattribute:: Description._v_is_nested
    .. autoattribute:: Description._v_itemsize
    .. autoattribute:: Description._v_name
    .. autoattribute:: Description._v_names
    .. autoattribute:: Description._v_nestedDescr
    .. autoattribute:: Description._v_nestedFormats
    .. autoattribute:: Description._v_nestedlvl
    .. autoattribute:: Description._v_nestedNames
    .. autoattribute:: Description._v_pathname
    .. autoattribute:: Description._v_pathnames
    .. autoattribute:: Description._v_types


Description methods
^^^^^^^^^^^^^^^^^^^
.. automethod:: Description._f_walk


.. _RowClassDescr:

The Row class
~~~~~~~~~~~~~
.. autoclass:: tables.tableExtension.Row

..  These are defined in the class docstring
    Row instance variables
    ^^^^^^^^^^^^^^^^^^^^^^
    .. autoattribute:: tables.tableExtension.Row.nrow


Row methods
^^^^^^^^^^^
.. automethod:: tables.tableExtension.Row.append

.. automethod:: tables.tableExtension.Row.fetch_all_fields

.. automethod:: tables.tableExtension.Row.update


.. _RowSpecialMethods:

Row special methods
^^^^^^^^^^^^^^^^^^^
.. automethod:: tables.tableExtension.Row.__contains__

.. automethod:: tables.tableExtension.Row.__getitem__

.. automethod:: tables.tableExtension.Row.__setitem__


.. _ColsClassDescr:

The Cols class
~~~~~~~~~~~~~~
.. autoclass:: Cols

..  These are defined in the class docstring
    Cols instance variables
    ^^^^^^^^^^^^^^^^^^^^^^^
    .. autoattribute:: Cols._v_colnames
    .. autoattribute:: Cols._v_colpathnames
    .. autoattribute:: Cols._v_desc


Cols properties
^^^^^^^^^^^^^^^
.. autoattribute:: Cols._v_table


Cols methods
^^^^^^^^^^^^
.. automethod:: Cols._f_col

.. automethod:: Cols.__getitem__

.. automethod:: Cols.__len__

.. automethod:: Cols.__setitem__


.. _ColumnClassDescr:

The Column class
~~~~~~~~~~~~~~~~
.. autoclass:: Column

.. These are defined in the class docstring

    .. autoattribute:: Column.descr
    .. autoattribute:: Column.name
    .. autoattribute:: Column.pathname

Column instance variables
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoattribute:: Column.dtype

.. autoattribute:: Column.index

.. autoattribute:: Column.is_indexed

.. autoattribute:: Column.maindim

.. autoattribute:: Column.shape

.. autoattribute:: Column.table

.. autoattribute:: Column.type


Column methods
^^^^^^^^^^^^^^
.. automethod:: Column.createIndex

.. automethod:: Column.createCSIndex

.. automethod:: Column.reIndex

.. automethod:: Column.reIndexDirty

.. automethod:: Column.removeIndex


Column special methods
^^^^^^^^^^^^^^^^^^^^^^
.. automethod:: Column.__getitem__

.. automethod:: Column.__len__

.. automethod:: Column.__setitem__


.. _ArrayClassDescr:

The Array class
---------------
.. autoclass:: Array


Array instance variables
~~~~~~~~~~~~~~~~~~~~~~~~
.. attribute:: Array.atom

    An Atom (see :ref:`AtomClassDescr`) instance representing the *type*
    and *shape* of the atomic objects to be saved.

.. autoattribute:: Array.rowsize

.. attribute:: Array.nrow

    On iterators, this is the index of the current row.

.. autoattribute:: Array.nrows


Array methods
~~~~~~~~~~~~~
.. automethod:: Array.getEnum

.. automethod:: Array.iterrows

.. automethod:: Array.next

.. automethod:: Array.read


Array special methods
~~~~~~~~~~~~~~~~~~~~~
The following methods automatically trigger actions when an :class:`Array`
instance is accessed in a special way (e.g. ``array[2:3,...,::2]`` will be
equivalent to a call to
``array.__getitem__((slice(2, 3, None), Ellipsis, slice(None, None, 2))))``.

.. automethod:: Array.__getitem__

.. automethod:: Array.__iter__

.. automethod:: Array.__setitem__


.. _CArrayClassDescr:

The CArray class
----------------
.. autoclass:: CArray


.. _EArrayClassDescr:

The EArray class
----------------
.. autoclass:: EArray


.. _EArrayMethodsDescr:

EArray methods
~~~~~~~~~~~~~~

.. automethod:: EArray.append


.. _VLArrayClassDescr:

The VLArray class
-----------------
.. autoclass:: VLArray

..  These are defined in the class docstring
    VLArray instance variables
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
    .. autoattribute:: VLArray.atom
    .. autoattribute:: VLArray.flavor
    .. autoattribute:: VLArray.nrow
    .. autoattribute:: VLArray.nrows
    .. autoattribute:: VLArray.extdim
    .. autoattribute:: VLArray.nrows


VLArray properties
~~~~~~~~~~~~~~~~~~
.. autoattribute:: VLArray.size_on_disk

.. autoattribute:: VLArray.size_in_memory


VLArray methods
~~~~~~~~~~~~~~~
.. automethod:: VLArray.append

.. automethod:: VLArray.getEnum

.. automethod:: VLArray.iterrows

.. automethod:: VLArray.next

.. automethod:: VLArray.read


VLArray special methods
~~~~~~~~~~~~~~~~~~~~~~~
The following methods automatically trigger actions when a :class:`VLArray`
instance is accessed in a special way (e.g., vlarray[2:5] will be equivalent
to a call to vlarray.__getitem__(slice(2, 5, None)).

.. automethod:: VLArray.__getitem__

.. automethod:: VLArray.__iter__

.. automethod:: VLArray.__setitem__


.. _LinkClassDescr:

The Link class
--------------
.. autoclass:: tables.link.Link

..  These are defined in the class docstring
    .. autoattribute:: tables.link.Link.target

Link instance variables
~~~~~~~~~~~~~~~~~~~~~~~
.. autoattribute:: tables.link.Link._v_attrs


Link methods
~~~~~~~~~~~~
The following methods are useful for copying, moving, renaming and removing
links.

.. automethod:: tables.link.Link.copy

.. automethod:: tables.link.Link.move

.. automethod:: tables.link.Link.remove

.. automethod:: tables.link.Link.rename


.. _SoftLinkClassDescr:

The SoftLink class
------------------
.. autoclass:: tables.link.SoftLink


SoftLink special methods
~~~~~~~~~~~~~~~~~~~~~~~~
The following methods are specific for dereferrencing and representing soft
links.

.. automethod:: tables.link.SoftLink.__call__

.. automethod:: tables.link.SoftLink.__str__


The ExternalLink class
----------------------
.. autoclass:: tables.link.ExternalLink

..  This is defined in the class docstring
    ExternalLink instance variables
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    .. autoattribute:: tables.link.ExternalLink.extfile


ExternalLink methods
~~~~~~~~~~~~~~~~~~~~
.. automethod:: tables.link.ExternalLink.umount


ExternalLink special methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The following methods are specific for dereferrencing and representing
external links.

.. automethod:: tables.link.ExternalLink.__call__

.. automethod:: tables.link.ExternalLink.__str__


.. _UnImplementedClassDescr:

The UnImplemented class
-----------------------
.. autoclass:: UnImplemented
    :members:


The Unknown class
-----------------
.. autoclass:: Unknown
    :members:


.. _AttributeSetClassDescr:

The AttributeSet class
----------------------
.. autoclass:: tables.attributeset.AttributeSet

..  These are defined in the class docstring
    AttributeSet attributes
    ~~~~~~~~~~~~~~~~~~~~~~~
    .. autoattribute:: tables.attributeset.AttributeSet._v_attrnames
    .. autoattribute:: tables.attributeset.AttributeSet._v_attrnamessys
    .. autoattribute:: tables.attributeset.AttributeSet._v_attrnamesuser
    .. autoattribute:: tables.attributeset.AttributeSet._v_unimplemented

AttributeSet properties
~~~~~~~~~~~~~~~~~~~~~~~
.. autoattribute:: tables.attributeset.AttributeSet._v_node


AttributeSet methods
~~~~~~~~~~~~~~~~~~~~
.. automethod:: tables.attributeset.AttributeSet._f_copy

.. automethod:: tables.attributeset.AttributeSet._f_list

.. automethod:: tables.attributeset.AttributeSet._f_rename

.. automethod:: tables.attributeset.AttributeSet.__contains__


Declarative classes
-------------------
In this section a series of classes that are meant to
*declare* datatypes that are required for creating
primary PyTables datasets are described.


.. _AtomClassDescr:

The Atom class and its descendants
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: Atom

..  These are defined in the class docstring
    Atom instance variables
    ^^^^^^^^^^^^^^^^^^^^^^^
    .. autoattribute:: Atom.dflt
    .. autoattribute:: Atom.dtype
    .. autoattribute:: Atom.itemsize
    .. autoattribute:: Atom.kind
    .. autoattribute:: Atom.shape
    .. autoattribute:: Atom.type


Atom properties
^^^^^^^^^^^^^^^
.. autoattribute:: Atom.ndim

.. autoattribute:: Atom.recarrtype

.. autoattribute:: Atom.size


Atom methods
^^^^^^^^^^^^
.. automethod:: Atom.copy


Atom factory methods
^^^^^^^^^^^^^^^^^^^^
.. automethod:: Atom.from_dtype

.. automethod:: Atom.from_kind

.. automethod:: Atom.from_sctype

.. automethod:: Atom.from_type


Atom Sub-classes
^^^^^^^^^^^^^^^^
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


ObjectAtom
..........
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


.. _ColClassDescr:

The Col class and its descendants
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: Col

..
    Col instance variables
    ^^^^^^^^^^^^^^^^^^^^^^
   .. autoattribute:: _v_pos


Col instance variables
^^^^^^^^^^^^^^^^^^^^^^
In addition to the variables that they inherit from the Atom class, Col
instances have the following attributes.

.. attribute:: Col._v_pos

    The *relative* position of this column with regard to its column
    siblings.


Col factory methods
^^^^^^^^^^^^^^^^^^^
.. automethod:: Col.from_atom


Col sub-classes
^^^^^^^^^^^^^^^
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
~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: IsDescription


Description helper functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: tables.description.descr_from_dtype

.. autofunction:: tables.description.dtype_from_descr


Helper classes
--------------
This section describes some classes that do not fit in any other
section and that mainly serve for ancillary purposes.


.. _FiltersClassDescr:

The Filters class
~~~~~~~~~~~~~~~~~
.. autoclass:: Filters

..  These are defined in the class docstring.
    Filters instance variables
    ^^^^^^^^^^^^^^^^^^^^^^^^^^
    .. autoattribute:: Filters.fletcher32
    .. autoattribute:: Filters.complevel
    .. autoattribute:: Filters.complib
    .. autoattribute:: Filters.shuffle


Filters methods
^^^^^^^^^^^^^^^
.. automethod:: Filters.copy


.. _IndexClassDescr:

The Index class
~~~~~~~~~~~~~~~
.. autoclass:: tables.index.Index

..  This is defined in the class docstring
    .. autoattribute:: tables.index.Index.nelements

Index instance variables
^^^^^^^^^^^^^^^^^^^^^^^^
.. autoattribute:: tables.index.Index.column

.. autoattribute:: tables.index.Index.dirty

.. autoattribute:: tables.index.Index.filters

.. autoattribute:: tables.index.Index.is_CSI

.. attribute:: tables.index.Index.nelements

    The number of currently indexed rows for this column.


Index methods
^^^^^^^^^^^^^
.. automethod:: tables.index.Index.readSorted

.. automethod:: tables.index.Index.readIndices


Index special methods
^^^^^^^^^^^^^^^^^^^^^
.. automethod:: tables.index.Index.__getitem__


The IndexArray class
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: tables.indexes.IndexArray
    :members:


.. _EnumClassDescr:

The Enum class
~~~~~~~~~~~~~~
.. autoclass:: tables.misc.enum.Enum


Enum special methods
^^^^^^^^^^^^^^^^^^^^
.. automethod:: Enum.__call__

.. automethod:: Enum.__contains__

.. automethod:: Enum.__eq__

.. automethod:: Enum.__getattr__

.. automethod:: Enum.__getitem__

.. automethod:: Enum.__iter__

.. automethod:: Enum.__len__

.. automethod:: Enum.__repr__


The Expr class - a general-purpose expression evaluator
-------------------------------------------------------
.. autoclass:: Expr

..  These are defined in the class docstring.
    Expr instance variables
    ~~~~~~~~~~~~~~~~~~~~~~~
    .. autoattribute:: Expr.append_mode
    .. autoattribute:: Expr.maindim
    .. autoattribute:: Expr.names
    .. autoattribute:: Expr.out
    .. autoattribute:: Expr.o_start
    .. autoattribute:: Expr.o_stop
    .. autoattribute:: Expr.o_step
    .. autoattribute:: Expr.shape
    .. autoattribute:: Expr.values


Expr methods
~~~~~~~~~~~~
.. automethod:: Expr.eval

.. automethod:: Expr.setInputsRange

.. automethod:: Expr.setOutput

.. automethod:: Expr.setOutputRange


Expr special methods
~~~~~~~~~~~~~~~~~~~~
.. automethod:: Expr.__iter__


.. _ExceptionsDescr:

Exceptions module
-----------------
In the :mod:`exceptions` module exceptions and warnings that are specific
to PyTables are declared.

.. autoexception:: HDF5ExtError
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
