:source: http://www.pytables.org/moin/HintsForSQLUsers
:revision: 56
:date: 2012-06-18 10:15:15
:author: valhallasw

===================
Hints for SQL users
===================

This page is intended to be **a guide to new PyTables for users who are used
to writing SQL code** to access their relational databases.
It will cover the most usual SQL statements.
If you are missing a particular statement or usage example, you can ask at the
`PyTables users' list`_ for it.
If you know some examples yourself, you can also write them here!

This page is under development: you can come back frequently to check for new
examples.
Also, this is no replacement for the `User's Guide`_;
if you don't read the manual, you'll be missing lots of features not available
in relational databases!

Examples in Python assume that you have imported the PyTables package like
this::

    import tables

.. .. contents:: Table Of Contents


Creating a new database
=======================

RDBMs happen to have several syntaxes for creating a database.
A usual syntax is::

    CREATE DATABASE database_name

In PyTables, each database goes to a different HDF5_ file (much like
SQLite_ or MS Access).
To create a new HDF5_ file, you use the :func:`tables.open_file` function with
the ``'w'`` mode (which deletes the database if it already exists), like this::

    h5f = tables.open_file('database_name.h5', 'w')

In this way you get the ``h5f`` PyTables file handle (an instance of the
:class:`tables.File` class), which is a concept similar to a *database
connection*, and a new :file:`database_name.h5` file is created in the current
directory (you can use full paths here).
You can close the handle (like you close the connection) with::

    h5f.close()

This is important for PyTables to dump pending changes to the database.
In case you forget to do it, PyTables closes all open database handles for
you when you exit your program or interactive session, but it is always safer
to close your files explicitly.
If you want to use the database after closing it, you just call
:func:`open_file` again, but using the ``'r+'`` or ``'r'`` modes, depending on
whether you do or don't need to modify the database, respectively.

You may use several PyTables databases simultaneously in a program, so you
must be explicit on which database you want to act upon (by using its handle).

A note on concurrency under PyTables
------------------------------------

Unlike most RDBMs, PyTables is not intended to serve concurrent accesses to a
database.
It has no protections whatsoever against corruption for different (or even the
same) programs accessing the same database.
Opening several handles to the same database in read-only mode is safe, though.


Creating a table
================

PyTables supports some other *datasets* besides tables, and they're not
arranged in a flat namespace, but rather into a *hierarchicalé one (see an
introduction to the _ref:`object tree <ObjectTreeSection>`);
however, due to the nature of these recipes, we'll limit ourselves to tables
in the *root group*.
The basic syntax for table creation under SQL is::

    CREATE TABLE table_name (
        column_name1 column_type1,
        column_name2 column_type2,
        ...
        column_nameN column_typeN
    )


Table descriptions
------------------

In PyTables, one first *describes* the structure of a table.
PyTables allows you to *reuse a description* for creating several tables with
the same structure, just by using the description object (``description_name``
below) or getting it from a created table.
This is specially useful for creating temporary tables holding query results.

You can create a table description using a dictionary::

    description_name = {
        'column_name1': colum_type1,
        'column_name2': colum_type2,
        'column_name3': colum_type3,
        ...
        'column_nameN': colum_typeN
    }

or a subclass of :class:`tables.IsDescription`::

    class description_name(tables.IsDescription):
        column_name1 = colum_type1
        column_name2 = colum_type2
        column_name3 = colum_type3
        ...
        column_nameN = colum_typeN

Please note that dictionaries are the only way of describing structures with
names which cannot be Python identifiers.
Also, if an explicit order is desired for colums, it must be specified through
the column type declarations (see below), since dictionariy keys and class
attributes aren't ordered.
Otherwise, columns are ordered in alphabetic increasing order.
It is important to note that PyTables doesn't have a concept of primary or
foreign keys, so relationships between tables are left to the user.


Column type declarations
------------------------

PyTables supports lots of types (including nested and multidimensional
columns).
Non-nested columns are declared through instances of :class:`tables.Col`
subclasses (which you can also reuse).
These are some correspondences with SQL:

==================== ==========================
SQL type declaration PyTables type declaration
==================== ==========================
INTEGER(digits)      tables.IntCol(itemsize)
REAL                 tables.FloatCol()
VARCHAR(length)      tables.StringCol(itemsize)
DATE                 tables.Time32Col()
TIMESTAMP            tables.Time64Col()
==================== ==========================

See a complete description of :ref:`PyTables types <datatypes>`.
Note that some types admit different *item sizes*, which are specified in
bytes.
For types with a limited set of supported item sizes, you may also use specific
subclasses which are named after the type and its *precision*, e.g. ``Int32Col``
for 4-byte (32 bit) item size.

Cells in a PyTables' table always have a value of the cell type, so there is
no ``NULL``.
Instead, cells take a *default value* (zero or empty) which can be changed in
the type declaration, like this: ``col_name = StringCol(10, dflt='nothing')``
(``col_name`` takes the value ``'nothing'`` if unset).
The declaration also allows you to set *column order* via the ``pos`` argument,
like this::

    class ParticleDescription(tables.IsDescription):
        name = tables.StringCol(10, pos=1)
        x = tables.FloatCol(pos=2)
        y = tables.FloatCol(pos=3)
        temperature = tables.FloatCol(pos=4)


Using a description
===================

Once you have a table description ``description_name`` and a writeable file
handle ``h5f``, creating a table with that description is as easy as::

    tbl = h5f.create_table('/', 'table_name', description_name)

PyTables is very object-oriented, and database is usually done through
methods of :class:`tables.File`.
The first argument indicates the *path* where the table will be created,
i.e. the root path (HDF5 uses Unix-like paths).
The :meth:`tables.File.create_table` method has many options e.g. for setting
a table title or compression properties. What you get back is an instance of
:class:`tables.Table`, a handle for accessing the data in that table.

As with files, table handles can also be closed with ``tbl.close()``.
If you want to access an already created table, you can use::

    tbl = h5f.get_node('/', 'table_name')

(PyTables uses the concept of *node* for datasets -tables and others- and
groups in the object tree) or, using *natural naming*::

    tbl = h5f.root.table_name

Once you have created a table, you can access (and reuse) its description by
accessing the ``description`` attribute of its handle.


Creating an index
=================

RDBMs use to allow named indexes on any set of columns (or all of them) in a
table, using a syntax like::

    CREATE INDEX index_name
    ON table_name (column_name1, column_name2, column_name3...)

and

    DROP INDEX index_name

Indexing is supported in the versions of PyTables >= 2.3 (and in PyTablesPro).
However, indexes don't have names and they are bound to single columns.
Following the object-oriented philosophy of PyTables, index creation is a
method (:meth:`tables.Column.create_index`) of a :class:`tables.Column` object
of a table, which you can access trough its ``cols`` accessor.

::
    tbl.cols.colum_name.create_index()

For dropping an index on a column::

    tbl.cols.colum_name.remove_index()


Altering a table
================

The first case of table alteration is renaming::

    ALTER TABLE old_name RENAME TO new_name

This is accomplished in !PyTables with::

    h5f.rename_node('/', name='old_name', newname='new_name')

or through the table handle::

    tbl.rename('new_name')

A handle to a table is still usable after renaming.
The second alteration, namely column addition, is currently not supported in
PyTables.


Dropping a table
================

In SQL you can remove a table using::

    DROP TABLE table_name

In PyTables, tables are removed as other nodes, using the
:meth:`tables.File.remove_node` method::

    h5f.remove_node('/', 'table_name')

or through the table handle::

    tbl.remove()

When you remove a table, its associated indexes are automatically removed.


Inserting data
==============

In SQL you can insert data one row at a time (fetching from a selection will
be covered later) using a syntax like::

    INSERT INTO table_name (column_name1, column_name2...)
    VALUES (value1, value2...)

In PyTables, rows in a table form a *sequence*, so data isn't *inserted* into
a set, but rather *appended* to the end of the sequence.
This also implies that identical rows may exist in a table (but they have a
different *row number*).
There are two ways of appending rows: one at a time or in a block.
The first one is conceptually similar to the SQL case::

    tbl.row['column_name1'] = value1
    tbl.row['column_name2'] = value2
    ...
    tbl.row.append()

The ``tbl.row`` accessor represents a *new row* in the table.
You just set the values you want to set (the others take the default value
from their column declarations - see above) and the effectively append the
new row.
This code is usually enclosed in some kind of loop, like::

    row = tbl.row
    while some_condition:
        row['column_name1'] = value1
        ...
        row.append()

For appending a block of rows in a single shot, :meth:`tables.Table.append`
is more adequate.
You just pass a NumPy_ record array or Python sequence with elements which
match the expected columns.
For example, given the ``tbl`` handle for a table with the ``ParticleDescription``
structure described above::

    rows = [
        ('foo', 0.0, 0.0, 150.0),
        ('bar', 0.5, 0.0, 100.0),
        ('foo', 1.0, 1.0,  25.0)
    ]
    tbl.append(rows)

    # Using a NumPy container.
    import numpy
    rows = numpy.rec.array(rows)
    tbl.append(rows)


A note on transactions
----------------------

PyTables doesn't support transactions nor checkpointing or rolling back (there
is undo support for operations performed on the object tree, but this is
unrelated).
Changes to the database are optimised for maximum performance and reasonable
memory requirements, which means that you can't tell whether e.g.
``tbl.append()`` has actually committed all, some or no data to disk when it ends.

However, you can *force* PyTables to commit changes to disk using the ``flush()``
method of table and file handles::

    tbl.flush()  # flush data in the table
    h5f.flush()  # flush all pending data

Closing a table or a database actually flushes it, but it is recommended that
you explicitly flush frequently (specially with tables).


Updating data
=============

We're now looking for alternatives to the SQL ``UPDATE`` statement::

    UPDATE table_name
    SET column_name1 = expression1, column_name2 = expression2...
    [WHERE condition]

There are different ways of approaching this, depending on your needs.
If you aren't using a condition, then the ``SET`` clause updates all rows,
something you can do in PyTables by iterating over the table::

    for row in tbl:
        row['column_name1'] = expression1
        row['column_name2'] = expression2
        ...
        row.update()

Don't forget to call ``update()`` or no value will be changed!
Also, since the used iterator allows you to read values from the current row,
you can implement a simple *conditional update*, like this::

    for row in tbl:
        if condition on row['column_name1'], row['column_name2']...:
            row['column_name1'] = expression1
            row['column_name2'] = expression2
            ...
            row.update()

There are substantially more efficient ways of locating rows fulfilling a
condition.
Given the main PyTables usage scenarios, querying and modifying data are
quite decoupled operations, so we will have a look at querying later and
assume that you already know the set of rows you want to update.

If the set happens to be a slice of the table, you may use the
:`meth:`tables.Table.modify_rows` method or its equivalent
:meth:`tables.Table.__setitem__` notation::

    rows = [
        ('foo', 0.0, 0.0, 150.0),
        ('bar', 0.5, 0.0, 100.0),
        ('foo', 1.0, 1.0,  25.0)
    ]
    tbl.modifyRows(start=6, stop=13, step=3, rows=rows)
    tbl[6:13:3] = rows  # this is the same

If you just want to update some columns in the slice, use the
:meth:`tables.Table.modify_columns` or :meth:`tables.Table.modify_column`
methods::

    cols = [
        [150.0, 100.0, 25.0]
    ]
    # These are all equivalent.
    tbl.modify_columns(start=6, stop=13, step=3, columns=cols, names=['temperature'])
    tbl.modify_column(start=6, stop=13, step=3, column=cols[0], colname='temperature')
    tbl.cols.temperature[6:13:3] = cols[0]

The last line shows an example of using the ``cols`` accessor to get to the
desired :class:`tables.Column` of the table using natural naming and apply
``setitem`` on it.

If the set happens to be an array of sparse coordinates, you can also use
PyTables' extended slice notation::

    rows = [
        ('foo', 0.0, 0.0, 150.0),
        ('bar', 0.5, 0.0, 100.0),
        ('foo', 1.0, 1.0,  25.0)
    ]
    rownos = [2, 735, 371913476]
    tbl[rownos] = rows


instead of the traditional::

    for row_id, datum in zip(rownos, rows):
         tbl[row_id] = datum

Since you are modifying table data in all cases, you should also remember to
``flush()`` the table when you're done.


Deleting data
=============

Rows are deleted from a table with the following SQL syntax::

    DELETE FROM table_name
    [WHERE condition]

:meth:`tables.Table.remove_rows` is the method used for deleting rows in
PyTables.
However, it is very simple (only contiguous blocks of rows can be deleted) and
quite inefficient, and one should consider whether *dumping filtered data from
one table into another* isn't a much more convenient approach.
This is a far more optimized operation under PyTables which will be covered
later.

Anyway, using ``remove_row()`` or ``remove_rows()`` is quite straightforward::

    tbl.remove_row(12)  # delete one single row (12)
    tbl.remove_rows(12, 20)  # delete all rows from 12 to 19 (included)
    tbl.remove_rows(0, tbl.nrows)  # delete all rows unconditionally
    tbl.remove_rows(-4, tbl.nrows)  # delete the last 4 rows


Reading data
============

The most basic syntax in SQL for reading rows in a table without using a
condition is::

    SELECT (column_name1, column_name2... | *) FROM table_name

Which reads all rows (though maybe not all columns) from a table.
In PyTables there are two ways of retrieving data: *iteratively* or *at once*.
You'll notice some similarities with how we appended and updated data above,
since this dichotomy is widespread here.

For a clearer separation with conditional queries (covered further below),
and since the concept of *row number* doesn't exist in relational databases,
we'll be including here the cases where you want to read a **known** *slice*
or *sequence* of rows, besides the case of reading *all* rows.


Iterating over rows
-------------------

This is similar to using the ``fetchone()`` method of a DB ``cursor`` in a
`Python DBAPI`_-compliant package, i.e. you *iterate* over the list of wanted
rows, getting one *row handle* at a time.
In this case, the handle is an instance of the :class:`tables.Row` class,
which allows access to individual columns as items acessed by key (so there
is no special way of selecting columns: you just use the ones you want
whenever you want).

This way of reading rows is recommended when you want to perform operations
on individual rows in a simple manner, and specially if you want to process
a lot of rows in the table (i.e. when loading them all at once would take too
much memory).
Iterators are also handy for using with the ``itertools`` Python module for
grouping, sorting and other operations.

For iterating over *all* rows, use plain iteration or the
:meth:`tables.Table.iterrows` method::

    for row in tbl:  # or tbl.iterrows()
        do something with row['column_name1'], row['column_name2']...

For iterating over a *slice* of rows, use the
:meth:`tables.Table.iterrows|Table.iterrows` method::

    for row in tbl.iterrows(start=6, stop=13, step=3):
        do something with row['column_name1'], row['column_name2']...

For iterating over a *sequence* of rows, use the
:meth:`tables.Table.itersequence` method::

    for row in tbl.itersequence([6, 7, 9, 11]):
        do something with row['column_name1'], row['column_name2']...

Reading rows at once
--------------------

In contrast with iteration, you can fetch all desired rows into a single
*container* in memory (usually an efficient NumPy_ record-array) in a single
operation, like the ``fetchall()`` or ``fetchmany()`` methods of a DBAPI ``cursor``.
This is specially useful when you want to transfer the read data to another
component in your program, avoiding loops to construct your own containers.
However, you should be careful about the amount of data you are fetching into
memory, since it can be quite large (and even exceed its physical capacity).

You can choose between the ``Table.read*()`` methods or the
:meth:`tables.Table.__getitem__` syntax for this kind of reads.
The ``read*()`` methods offer you the chance to choose a single column to read
via their ``field`` argument (which isn't still as powerful as the SQL ``SELECT``
column spec).

For reading *all* rows, use ``[:]`` or the :meth:`tables.Table.read` method::

    rows = tbl.read()
    rows = tbl[:]  # equivalent

For reading a *slice* of rows, use ``[slice]`` or the
:meth:`tables.Table.read|Table.read` method::

    rows = tbl.read(start=6, stop=13, step=3)
    rows = tbl[6:13:3]  # equivalent

For reading a *sequence* of rows, use the :meth:`tables.Table.read_coordinates`
method::

    rows = tbl.read_coordinates([6, 7, 9, 11])

Please note that you can add a ``field='column_name'`` argument to ``read*()``
methods in order to get only the given column instead of them all.


Selecting data
==============

When you want to read a subset of rows which match a given condition from a
table you use a syntax like this in SQL::

    SELECT column_specification FROM table_name
    WHERE condition

The ``condition`` is an expression yielding a boolean value based on a
combination of column names and constants with functions and operators.
If the condition holds true for a given row, the ``column_specification`` is
applied on it and the resulting row is added to the result.

In PyTables, you may filter rows using two approaches: the first one is
achieved through standard Python comparisons (similar to what we used for
conditional update), like this::

    for row in tbl:
        if condition on row['column_name1'], row['column_name2']...:
            do something with row

This is easy for newcomers, but not very efficient. That's why PyTables offers
another approach: **in-kernel** searches, which are much more efficient than
standard searches, and can take advantage of indexing (under PyTables >= 2.3).

In-kernel searches are used through the *where methods* in ``Table``, which are
passed a *condition string* describing the condition in a Python-like syntax.
For instance, with the ``ParticleDescription`` we defined above, we may specify
a condition for selecting particles at most 1 unit apart from the origin with
a temperature under 100 with a condition string like this one::

    '(sqrt(x**2 + y**2) <= 1) & (temperature < 100)'

Where ``x``, ``y`` and ``temperature`` are the names of columns in the table.
The operators and functions you may use in a condition string are described
in the :ref:`appendix on condition syntax <condition_syntax>` in the
`User's Guide`_.


Iterating over selected rows
----------------------------

You can iterate over the rows in a table which fulfill a condition (a la DBAPI
``fetchone()``) by using the :meth:`tables.Table.where` method, which is very
similar to the :meth:`tables.Table.iterrows` one discussed above, and which
can be used in the same circumstances (i.e. performing operations on individual
rows or having results exceeding available memory).

Here is an example of using ``where()`` with the previous example condition::

    for row in tbl.where('(sqrt(x**2 + y**2) <= 1) & (temperature < 100)'):
        do something with row['name'], row['x']...


Reading selected rows at once
-----------------------------

Like the aforementioned :meth:`tables.Table.read`,
:meth:`tables.Table.read_where` gets all the rows fulfilling the given
condition and packs them in a single container (a la DBAPI ``fetchmany()``).
The same warning applies: be careful on how many rows you expect to retrieve,
or you may run out of memory!

Here is an example of using ``read_where()`` with the previous example
condition::

    rows = tbl.read_where('(sqrt(x**2 + y**2) <= 1) & (temperature < 100)')

Please note that both :meth:`tables.Table.where` and
:meth:`tables.Table.read_where` can also take slicing arguments.


Getting the coordinates of selected rows
----------------------------------------

There is yet another method for querying tables:
:meth:`tables.Table.get_where_list`.
It returns just a sequence of the numbers of the rows which fulfil the given
condition.
You may pass that sequence to :meth:`tables.Table.read_coordinates`, e.g. to
retrieve data from a different table where rows with the same number as the
queried one refer to the same first-class object or entity.


A note on table joins
---------------------

You may have noticed that queries in PyTables only cover one table.
In fact, there is no way of directly performing a join between two tables in
PyTables (remember that it's not a relational database).
You may however work around this limitation depending on your case:

* If one table is an *extension* of another (i.e. it contains additional
  columns for the same entities), your best bet is to arrange rows of the
  same entity so that they are placed in the same positions in both tables.
  For instance, if ``tbl1`` and ``tbl2`` follow this rule, you may do something
  like this to emulate a natural join::

    for row1 in tbl1.where('condition'):
        row2 = tbl2[row1.nrow]
        if condition on row2['column_name1'], row2['column_name2']...:
            do something with row1 and row2...

   (Note that ``row1`` is a ``Row`` instance and ``row2`` is a record of the current
   flavor.)

* If rows in both tables are linked by a common value (e.g. acting as an
  identifier), you'll need to split your condition in one for the first table
  and one for the second table, and then nest your queries, placing the most
  restrictive one first. For instance::

    SELECT clients.name, bills.item_id FROM clients, bills
    WHERE clients.id = bills.client_id and clients.age > 50 and bills.price > 200

  could be written as::

    for client in clients.where('age > 50'):
        # Note that the following query is different for each client.
        for bill in bills.where('(client_id == %r) & (price > 200)' % client['id']):
            do something with client['name'] and bill['item_id']

  In this example, indexing the ``client_id`` column of ``bills`` could speed up
  the inner query quite a lot.
  Also, you could avoid parsing the inner condition each time by using
  *condition variables*::

    for client in clients.where('age > 50'):
        for bill in bills.where('(client_id == cid) & (price > 200)', {'cid': client['id']}):
            do something with client['name'] and bill['item_id']


Summary of row selection methods
================================

+----------------------+-----------------+---------------------+-----------------------+-------------------------+
|                      | **All rows**    | **Range of rows**   | **Sequence of rows**  | **Condition**           |
+----------------------+-----------------+---------------------+-----------------------+-------------------------+
| **Iterative access** | ``__iter__()``, | ``iterrows(range)`` | ``itersequence()``    | ``where(condition)``    |
|                      | ``iterrows()``  |                     |                       |                         |
+----------------------+-----------------+---------------------+-----------------------+-------------------------+
| **Block access**     | ``[:]``,        | ``[range]``,        | ``readCoordinates()`` |``read_where(condition)``|
|                      | ``read()``      | ``read(range)``     |                       |                         |
+----------------------+-----------------+---------------------+-----------------------+-------------------------+


Sorting the results of a selection
==================================

*Do you feel like writing this section? Your contribution is welcome!*


Grouping the results of a selection
===================================

By making use of the :func:`itertools.groupby` utility, you can group results
by field::

    group = {} # dictionary to put results grouped by 'pressure'
    def pressure_selector(row):
        return row['pressure']
    for pressure, rows_grouped_by_pressure in itertools.groupby(mytable, pressure_selector):
        group[pressure] = sum((r['energy'] + r['ADCcount'] for r in rows_grouped_by_pressure))

However, :func:`itertools.groupby` assumes the incoming array is sorted by the
grouping field.
If not, there are multiple groups with the same grouper returned.
In the example, mytable thus has to be sorted on pressure, or the last line
should be changed to::

    group[pressure] += sum((r['energy'] + r['ADCcount'] for r in rows_grouped_by_pressure))


-----


.. target-notes::

.. _`PyTables users' list`: https://lists.sourceforge.net/lists/listinfo/pytables-users
.. _`User's Guide`: http://www.pytables.org/docs/manual
.. _HDF5: http://www.hdfgroup.org/HDF5
.. _SQLite: http://www.sqlite.org
.. _NumPy: http://www.numpy.org
.. _`Python DBAPI`: http://www.python.org/dev/peps/pep-0249
