.. currentmodule:: tables

Heterogenous dataset classes
============================

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
