.. currentmodule:: tables


Helper classes
==============
This section describes some classes that do not fit in any other
section and that mainly serve for ancillary purposes.


.. _FiltersClassDescr:

The Filters class
-----------------
.. autoclass:: Filters

..  These are defined in the class docstring.
    Filters instance variables
    ^^^^^^^^^^^^^^^^^^^^^^^^^^
    .. autoattribute:: Filters.bitshuffle
    .. autoattribute:: Filters.fletcher32
    .. autoattribute:: Filters.complevel
    .. autoattribute:: Filters.complib
    .. autoattribute:: Filters.shuffle


Filters methods
~~~~~~~~~~~~~~~
.. automethod:: Filters.copy


.. _IndexClassDescr:

The Index class
---------------
.. autoclass:: tables.index.Index

..  This is defined in the class docstring
    .. autoattribute:: tables.index.Index.nelements

Index instance variables
~~~~~~~~~~~~~~~~~~~~~~~~
.. autoattribute:: tables.index.Index.column

.. autoattribute:: tables.index.Index.dirty

.. autoattribute:: tables.index.Index.filters

.. autoattribute:: tables.index.Index.is_csi

.. attribute:: tables.index.Index.nelements

    The number of currently indexed rows for this column.


Index methods
~~~~~~~~~~~~~
.. automethod:: tables.index.Index.read_sorted

.. automethod:: tables.index.Index.read_indices


Index special methods
~~~~~~~~~~~~~~~~~~~~~
.. automethod:: tables.index.Index.__getitem__


The IndexArray class
--------------------

.. autoclass:: tables.indexes.IndexArray
    :members:


.. _EnumClassDescr:

The Enum class
--------------
.. autoclass:: tables.misc.enum.Enum


Enum special methods
~~~~~~~~~~~~~~~~~~~~
.. automethod:: Enum.__call__

.. automethod:: Enum.__contains__

.. automethod:: Enum.__eq__

.. automethod:: Enum.__getattr__

.. automethod:: Enum.__getitem__

.. automethod:: Enum.__iter__

.. automethod:: Enum.__len__

.. automethod:: Enum.__repr__


.. _UnImplementedClassDescr:

The UnImplemented class
-----------------------
.. autoclass:: UnImplemented
    :members:


The Unknown class
-----------------
.. autoclass:: Unknown
    :members:


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

.. autoexception:: ExperimentalFeatureWarning
