=======================================
 Release notes for PyTables 2.0 series
=======================================

:Author: Francesc Alted i Abad
:Contact: faltet@pytables.com
:Author: Ivan Vilata i Balaguer
:Contact: ivan@selidor.net


Changes from 2.0.3 to 2.0.4
===========================

- Selections in tables works now in threaded environments.  The problem was in
  the Numexpr package -- the solution has been reported to the upstream
  authors too.  Fixes #164.

- PyTables had problems importing native HDF5 files with gaps in nested
  compound types.  This has been solved.  Fixes #173.

- In order to prevent a bug existing in HDF5 1.6 series, the
  ``EArray.truncate()`` method refused to accept a 0 as parameter
  (i.e. truncate an existing EArray to have zero rows did not work).  As this
  has been fixed in the recent HDF5 1.8 series, this limitation has been
  removed (but only if the user has one of these installed).  Fixes #171.

- Small fixes for allowing the test suite to pass when using the new NumPy
  1.1.  However, it remains a small issue with the way the new NumPy
  represents complex numbers.  I'm not fixing that in the PyTables suite, as
  there are chances that this can be fixed in NumPy itself (see ticket #841).


Changes from 2.0.2 to 2.0.3
===========================

- Replaced the algorithm for computing chunksizes by another that is
  more general and useful for a larger range of expected dataset sizes.
  The outcome of the new calculation is the same than before for
  dataset sizes <= 100 GB. For datasets between 100 GB <= size < 10
  TB, larger values are returned. For sizes >= 10 TB a maximum value
  of 1 MB is always returned.

- Fixed a problem when updating multidimensional cells using the
  Row.update() method in the middle of table iterators .  Fixes #149.

- Added support for the latest 1.8.0 version of the HDF5 library.
  Fixes ticket #127.

- PyTables compiles now against latest versions of Pyrex (0.9.6.4).  For the
  first time, the extensions do compile without warnings!  Fixes #159.

- Numexpr module has been put in sync with the version in SciPy sandbox.

- Added a couple of warnings in User's Guide so as to tell the user that it is
  not safe to use methods that can change the number of rows of a table in the
  middle of a row iterator. Fixes #153.


Changes from 2.0.1 to 2.0.2
===========================

- Added ``__enter__()`` and ``__exit__()`` methods to ``File``; fixes #113.
  With this, and if using Python 2.5 you can do things like:

    with tables.openFile("test.h5") as h5file:
        ...

- Carefully preserve type when converting NumPy scalar to numarray; fixes
  #125.

- Fixed a nasty bug that appeared when moving or renaming groups due to a bad
  interaction between ``Group._g_updateChildrenLocation()`` and the LRU cache.
  Solves #126.

- Return 0 when no rows are given to ``Table.modifyRows()``; fixes #128.

- Added an informative message when the ``nctoh5`` utility is run without the
  NetCDF interface of ScientificPython bening installed.

- Now, a default representation of closed nodes is provided; fixes #129.


Changes from 2.0 to 2.0.1
=========================

- The ``coords`` argument of ``Table.readCoords()`` was not checked
  for contiguousness, raising fatal errors when it was discontiguous.
  This has been fixed.

- There is an inconsistency in the way used to specify the atom shape
  in ``Atom`` constructors.  When the shape is specified as
  ``shape=()`` it means a scalar atom and when it is specified as
  ``shape=N`` it means an atom with ``shape=(N,)``.  But when the
  shape is specified as ``shape=1`` (i.e. in the default case) then a
  scalar atom is obtained instead of an atom with ``shape=(1,)``.
  This is inconsistent and not the behavior that NumPy exhibits.

  Changing this will require a migration path which includes
  deprecating the old behaviour if we want to make the change happen
  before a new major version.  The proposed path is:

   1. In PyTables 2.0.1, we are changing the default value of the
      ``shape`` argument to ``()``, and issue a ``DeprecationWarning``
      when someone uses ``shape=1`` stating that, for the time being,
      it is equivalent to ``()``, but in near future versions it will
      become equivalent to ``(1,)``, and recommending the user to pass
      ``shape=()`` if a scalar is desired.

   2. In PyTables 2.1, we will remove the previous warning and take
      ``shape=N`` to mean ``shape=(N,)`` for any value of N.

  See ticket #96 for more info.

- The info about the ``chunkshape`` attribute of a leaf is now printed
  in the ``__repr__()`` of chunked leaves (all except ``Array``).

- After some scrupulous benchmarking job, the size of the I/O buffer
  for ``Table`` objects has been reduced to the minimum that allows
  maximum performance.  This represents more than 10x of reduction in
  size for that buffer, which will benefit those programs dealing with
  many tables simultaneously (#109).

- In the ``ptrepack`` utility, when ``--complevel`` and ``--shuffle``
  were specified at the same time, the 'shuffle' filter was always set
  to 'off'.  This has been fixed (#104).

- An ugly bug related with the integrated Numexpr not being aware of
  all the variations of data arrangements in recarray objects has been
  fixed (#103).  We should stress that the bug only affected the
  Numexpr version integrated in PyTables, and *not* the original one.

- When passing a record array to a table at creation time, its real
  length is now used instead of the default value for
  ``expectedrows``.  This allows for better performance (#97).

- Added some workarounds so that NumPy scalars can be successfully
  converted to numarray objects.  Fixes #98.

- PyTables is now able to access table rows beyond 2**31 in 32-bit
  Python.  The problem was a limitation of ``xrange`` and we have
  replaced it by a new ``lrange`` class written in Pyrex.  Moreover,
  ``lrange`` has been made publicly accessible as a safe 64-bit
  replacement for ``xrange`` for 32-bit platforms users.  Fixes #99.

- If a group and a table are created in a function, and the table is
  accessed through the group, the table can be flushed now.  Fixes
  #94.

- It is now possible to directly assign a field in a nested record of
  a table using the natural naming notation (#93).


Changes from 2.0rc2 to 2.0
==========================

- Added support for recognizing native HDF5 files with datasets compressed
  with szip compressor.

- Fixed a problem when asking for the string representation (str()) of closed
  files. Fixes ticket #79.

- Do not take LZO as available when its initialisation fails.

- Fixed a glitch in ptrepack utility. When the user wants a copy of a group,
  and a group is *to be created* in destination, the attributes of the
  original group *are* copied. If it is *not to be created*, the attributes
  will *not be* copied. I think this should be what the user would expect most
  of the times.

- Fixed the check for creating intermediate groups in ptrepack utility.
  Solves ticket #83.

- Before, when reading a dataset with an unknown CLASS id, a warning was
  issued and the dataset mapped to ``UnImplemented``. This closed the door to
  have the opportunity to try to recognize the dataset and map it to a
  supported CLASS. Now, when a CLASS attribute is not recognized, an attempt
  to recognize its associated dataset is made. If it is recognized, the
  matching class is associated with the dataset. If it is not recognized, then
  a warning is issued and the dataset becomes mapped to ``UnImplemented``.

- Always pass verbose and heavy values in the common test module to test().
  Fixes ticket #85.

- Now, the ``verbose`` and ``--heavy`` flag passed to test_all.py are honored.

- All the DLL's of dependencies are included now in Windows binaries.  This
  should allow for better portability of the binaries.

- Fixed the description of Node._v_objectID that was misleading.


Changes from 2.0rc1 to 2.0rc2
=============================

- The "Optimization tips" chapter of the User's Guide has been completely
  updated to adapt to PyTables 2.0 series.  In particular, new benchmarks on
  the much improved indexed queries have been included; you will see that
  PyTables indexing is competitive (and sometimes much faster) than that of
  traditional relational databases.  With this, the manual should be fairly
  finished for 2.0 final release.

- Large refactoring done on the ``Row`` class.  The most important change is
  that ``Table.row`` is now a single object.  This allows to reuse the same
  ``Row`` instance even after ``Table.flush()`` calls, which can be convenient
  in many situations.

- I/O buffers unified in the ``Row`` class.  That allows for bigger savings in
  memory space whenever the ``Row`` extension is used.

- Improved speed (up to a 70%) with unaligned column operations (a quite
  common scenario when dealing with ``Table`` objects) through the integrated
  Numexpr.  In-kernel searches take advantage of this optimization.

- Added ``VLUnicodeAtom`` for storing variable-length Unicode strings in
  ``VLArray`` objects regardless of encoding.  Closes ticket #51.

- Added support for ``time`` datatypes to be portable between big-endian and
  low-endian architectures.  This feature is not currently supported natively
  by the HDF5 library, so the support for such conversion has been added in
  PyTables itself.  Fixes #72.

- Added slice arguments to ``Table.readWhere()`` and ``Table.getWhereList()``.
  Although API changes are frozen, this may still be seen as an inconsistency
  with other query methods.  The patch is backwards-compatible anyway.

- Added missing overwrite argument to ``File.renameNode()`` and
  ``Node._f_rename()``.  Fixes ticket #66.

- Calling ``tables.test()`` no longer exits the interpreter session.  Fixes
  ticket #67.

- Fix comparing strings where one is a prefix of the other in integrated
  Numexpr.  Fixes ticket #76.

- Added a check for avoiding an ugly HDF5 message when copying a file over
  itself (for both ``copyFile()`` and ``File.copyFile()``).  Fixes ticket #73.

- Corrected the appendix E, were it was said that PyTables doesn't support
  compounds of compounds (it does since version 1.2!).


Changes from 2.0b2 to 2.0rc1
============================

- The API Reference section of the User's Manual (and the matching docstrings)
  has been completely reviewed, expanded and corrected.  This process has
  unveiled some errors and inconsistencies which have also been fixed.

- Fixed ``VLArray.__getitem__()`` to behave as expected in Python when using
  slices, instead of following the semantics of PyTables' ``read()`` methods
  (e.g. reading just one element when no stop is provided).  Fixes ticket #50.

- Removed implicit UTF-8 encoding from ``VLArray`` data using ``vlstring``
  atoms.  Now a variable-length string is stored as is, which lets users use
  any encoding of their choice, or none of them.  A ``vlunicode`` atom will
  probably be added to the next release so as to fix ticket #51.

- Allow non-sequence objects to be passed to ``VLArray.append()`` when using
  an ``object`` atom.  This was already possible in 1.x but stopped working
  when the old append syntax was dropped in 2.0.  Fixes ticket #63.

- Changed ``Cols.__len__()`` to return the number of rows of the table or
  nested column (instead of the number of fields), like its counterparts in
  ``Table`` and ``Column``.

- Python scalars cached in ``AttributeSet`` instances are now kept as NumPy
  objects instead of Python ones, because they do become NumPy objects when
  retrieved from disk.  Fixes ticket #59.

- Avoid HDF5 error when appending an empty array to a ``Table`` (ticket #57)
  or ``EArray`` (ticket #49) dataset.

- Fix wrong implementation of the top-level ``table.description._v_dflts``
  map, which was also including the pathnames of columns inside nested
  columns.  Fixes ticket #45.

- Optimized the access to unaligned arrays in Numexpr between a 30% and a 70%.

- Fixed a die-hard bug that caused the loading of groups while closing a file.
  This only showed with certain usage patterns of the LRU cache (e.g. the one
  caused by ``ManyNodesTestCase`` in ``test_indexes.py`` under Pro).

- Avoid copious warnings about unused functions and variables when compiling
  Numexpr.

- Several fixes to Numexpr expressions with all constant values.  Fixed
  tickets #53, #54, #55, #58.  Reported bugs to mainstream developers.

- Solved an issue when trying to open one of the included test files in append
  mode on a system-wide installation by a normal user with no write privileges
  on it.  The file isn't being modified anyway, so the test is skipped then.

- Added a new benchmark to compare the I/O speed of ``Array`` and ``EArray``
  objects with that of ``cPickle``.

- The old ``Row.__call__()`` is no longer available as a public method.  It
  was not documented, anyway.  Fixes ticket #46.

- ``Cols._f_close()`` is no longer public.  Fixes ticket #47.

- ``Attributes._f_close()`` is no longer public.  Fixes ticket #52.

- The undocumented ``Description.classdict`` attribute has been completely
  removed.  Fixes ticket #44.


Changes from 2.0b1 to 2.0b2
===========================

- A very exhaustive overhauling of the User's Manual is in process.  The
  chapters 1 (Introduction), 2 (Installation), 3 (Tutorials) have been
  completed (and hopefully, the lines of code are easier to copy&paste now),
  while chapter 4 (API Reference) has been done up to (and including) the
  Table class.  During this tedious (but critical in a library) overhauling
  work, we have tried hard to synchronize the text in the User's Guide with
  that which appears on the docstrings.

- Removed the ``recursive`` argument in ``Group._f_walkNodes()``.  Using it
  with a false value was redundant with ``Group._f_iterNodes()``.  Fixes
  ticket #42.

- Removed the ``coords`` argument from ``Table.read()``.  It was undocumented
  and redundant with ``Table.readCoordinates()``.  Fixes ticket #41.

- Fixed the signature of ``Group.__iter__()`` (by removing its parameters).

- Added new ``Table.coldescrs`` and ``Table.description._v_itemsize``
  attributes.

- Added a couple of new attributes for leaves:

  * ``nrowsinbuf``: the number of rows that fit in the internal buffers.
  * ``chunkshape``: the chunk size for chunked datasets.

- Fixed setuptools so that making an egg out of the PyTables 2 package is
  possible now.

- Added a new ``tables.restrict_flavors()`` function allowing to restrict
  available flavors to a given set.  This can be useful e.g. if you want to
  force PyTables to get NumPy data out of an old, ``numarray``-flavored
  PyTables file even if the ``numarray`` package is installed.

- Fixed a bug which caused filters of unavailable compression libraries to be
  loaded as using the default Zlib library, after issuing a warning.  Added a
  new ``FiltersWarning`` and a ``Filters.copy()``.


Important changes from 1.4.x to 2.0
===================================

API additions
-------------

- ``Column.createIndex()`` has received a couple of new parameters:
  ``optlevel`` and ``filters``.  The first one sets the desired quality level
  of the index, while the second one allows the user to specify the filters
  for the index.

- ``Table.indexprops`` has been split into ``Table.indexFilters`` and
  ``Table.autoIndex``.  The later groups the functionality of the old ``auto``
  and ``reindex``.

- The new ``Table.colpathnames`` is a sequence which contains the full
  pathnames of all bottom-level columns in a table.  This can be used to walk
  all ``Column`` objects in a table when used with ``Table.colinstances``.

- The new ``Table.colinstances`` dictionary maps column pathnames to their
  associated ``Column`` or ``Cols`` object for simple or nested columns,
  respectively.  This is similar to ``Table.cols._f_col()``, but faster.

- ``Row`` has received a new ``Row.fetch_all_fields()`` method in order to
  return all the fields in the current row.  This returns a NumPy void scalar
  for each call.

- New ``tables.test(verbose=False, heavy=False)`` high level function for
  interactively running the complete test suite from the Python console.

- Added a ``tables.print_versions()`` for easily getting the versions for all
  the software on which PyTables relies on.


Backward-incompatible changes
-----------------------------

- You can no longer mark a column for indexing in a ``Col`` declaration.  The
  only way of creating an index for a column is to invoke the
  ``createIndex()`` method of the proper column object *after the table has
  been created*.

- Now the ``Table.colnames`` attribute is just a list of the names of
  top-level columns in a table.  You can still get something similar to the
  old structure by using ``Table.description._v_nestedNames``.  See also the
  new ``Table.colpathnames`` attribute.

- The ``File.objects``, ``File.leaves`` and ``File.groups`` dictionaries have
  been removed.  If you still need this functionality, please use the
  ``File.getNode()`` and ``File.walkNodes()`` instead.

- ``Table.removeIndex()`` is no longer available; to remove an index on a
  column, one must use the ``removeIndex()`` method of the associated
  ``Column`` instance.

- ``Column.dirty`` is no longer available.  If you want to check
  column index dirtiness, use ``Column.index.dirty``.

- ``complib`` and ``complevel`` parameters have been removed from
  ``File.createTable()``, ``File.createEArray()``, ``File.createCArray()`` and
  ``File.createVLArray()``.  They were already deprecated in PyTables 1.x.

- The ``shape`` and ``atom`` parameters have been swapped in
  ``File.createCArray()``.  This has been done to be consistent with
  ``Atom()`` definitions (i.e. type comes before and shape after).

Deprecated features
-------------------

- ``Node._v_rootgroup`` has been removed.  Please use ``node._v_file.root``
  instead.

- The ``Node._f_isOpen()`` and ``Leaf.isOpen()`` methods have been removed.
  Please use the ``Node._v_isopen`` attribute instead (it is much faster).

- The ``File.getAttrNode()``, ``File.setAttrNode()`` and
  ``File.delAttrNode()`` methods have been removed.  Please use
  ``File.getNodeAttr()``, ``File.setNodeAttr()`` and ``File.delNodeAttr()``
  instead.

- ``File.copyAttrs()`` has been removed.  Please use ``File.copyNodeAttrs()``
  instead.

- The ``table[colname]`` idiom is no longer supported.  You can use
  ``table.cols._f_col(column)`` for doing the same.

API refinements
---------------

- ``File.createEArray()`` received a new ``shape`` parameter.  This allows to
  not have to use the shape of the atom so as to set the shape of the
  underlying dataset on disk.

- All the leaf constructors have received a new ``chunkshape`` parameter that
  allows specifying the chunk sizes of datasets on disk.

- All ``File.create*()`` factories for ``Leaf`` nodes have received a new
  ``byteorder`` parameter that allows the user to specify the byteorder in
  which data will be written to disk (data in memory is now always handled in
  *native* order).


----

  **Enjoy data!**

  -- The PyTables Team


.. Local Variables:
.. mode: rst
.. coding: utf-8
.. fill-column: 78
.. End:
