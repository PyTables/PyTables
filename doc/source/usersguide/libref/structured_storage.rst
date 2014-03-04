.. currentmodule:: tables

Structured storage classes
==========================

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
    path name, avoiding the need to walk through Table.description or
    :attr:`Table.cols`.

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
.. autoattribute:: Table.autoindex

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

.. automethod:: Table.read_coordinates

.. automethod:: Table.read_sorted

.. automethod:: Table.__getitem__

.. automethod:: Table.__iter__


Table methods - writing
~~~~~~~~~~~~~~~~~~~~~~~
.. automethod:: Table.append

.. automethod:: Table.modify_column

.. automethod:: Table.modify_columns

.. automethod:: Table.modify_coordinates

.. automethod:: Table.modify_rows

.. automethod:: Table.remove_rows

.. automethod:: Table.remove_row

.. automethod:: Table.__setitem__


.. _TableMethods_querying:

Table methods - querying
~~~~~~~~~~~~~~~~~~~~~~~~
.. automethod:: Table.get_where_list

.. automethod:: Table.read_where

.. automethod:: Table.where

.. automethod:: Table.append_where

.. automethod:: Table.will_query_use_indexing


Table methods - other
~~~~~~~~~~~~~~~~~~~~~
.. automethod:: Table.copy

.. automethod:: Table.flush_rows_to_index

.. automethod:: Table.get_enum

.. automethod:: Table.reindex

.. automethod:: Table.reindex_dirty


.. _DescriptionClassDescr:

The Description class
~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: Description

..  These are defined in the class docstring
    Description instance variables
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    .. autoattribute:: Description._v_col_objects
    .. autoattribute:: Description._v_dflts
    .. autoattribute:: Description._v_dtype
    .. autoattribute:: Description._v_dtypes
    .. autoattribute:: Description._v_is_nested
    .. autoattribute:: Description._v_itemsize
    .. autoattribute:: Description._v_name
    .. autoattribute:: Description._v_names
    .. autoattribute:: Description._v_nested_descr
    .. autoattribute:: Description._v_nested_formats
    .. autoattribute:: Description._v_nestedlvl
    .. autoattribute:: Description._v_nested_names
    .. autoattribute:: Description._v_pathname
    .. autoattribute:: Description._v_pathnames
    .. autoattribute:: Description._v_types


Description methods
^^^^^^^^^^^^^^^^^^^
.. automethod:: Description._f_walk


.. _RowClassDescr:

The Row class
~~~~~~~~~~~~~
.. autoclass:: tables.tableextension.Row

..  These are defined in the class docstring
    Row instance variables
    ^^^^^^^^^^^^^^^^^^^^^^
    .. autoattribute:: tables.tableextension.Row.nrow


Row methods
^^^^^^^^^^^
.. automethod:: tables.tableextension.Row.append

.. automethod:: tables.tableextension.Row.fetch_all_fields

.. automethod:: tables.tableextension.Row.update


.. _RowSpecialMethods:

Row special methods
^^^^^^^^^^^^^^^^^^^
.. automethod:: tables.tableextension.Row.__contains__

.. automethod:: tables.tableextension.Row.__getitem__

.. automethod:: tables.tableextension.Row.__setitem__


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
.. automethod:: Column.create_index

.. automethod:: Column.create_csindex

.. automethod:: Column.reindex

.. automethod:: Column.reindex_dirty

.. automethod:: Column.remove_index


Column special methods
^^^^^^^^^^^^^^^^^^^^^^
.. automethod:: Column.__getitem__

.. automethod:: Column.__len__

.. automethod:: Column.__setitem__
