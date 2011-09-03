=======================================
 Release notes for PyTables 2.2 series
=======================================

:Author: Francesc Alted i Abad
:Contact: faltet@pytables.org


Changes from 2.2.1rc1 to 2.2.1
==============================

- The `Row` accessor implements a new `__contains__` special method that
  allows doing things like::

    for row in table:
        if item in row:
            print "Value found in row", row.nrow
            break

  Closes #309.

- PyTables is more friendly with easy_install and pip now, as all the
  Python dependencies should be installed automatically.  Closes #298.


Changes from 2.2 to 2.2.1rc1
============================

- When using `ObjectAtom` objects in `VLArrays` the ``HIGHEST_PROTOCOL``
  is used for pickling objects.  For NumPy arrays, this simple change
  leads to space savings up to 3x and time improvements up to 30x.
  Closes #301.

- The `Row` accessor implements a new `__contains__` special method that
  allows doing things like::

    for row in table:
        if item in row:
            print "Value found in row", row.nrow
            break

  Closes #309.

- tables.Expr can perform operations on scalars now.  Thanks to Gaëtan
  de Menten for providing a patch for this.  Closes #287.

- Fixed a problem with indexes larger than 32-bit on leaf objects on
  32-bit machines.  Fixes #283.

- Merged in Blosc 1.1.2 for fixing a problem with large datatypes and
  subprocess issues.  Closes #288 and #295.

- Due to the adoption of Blosc 1.1.2, the pthreads-win32 library
  dependency is dropped on Windows platforms.

- Fixed a problem with tables.Expr and operands with vary large
  rowsizes. Closes #300.

- ``leaf[numpy.array[scalar]]`` idiom returns a NumPy array instead of
  an scalar.  This has been done for compatibility with NumPy.  Closes
  #303.

- Optimization for `Table.copy()` so that ``FIELD_*`` attrs are not
  overwritten during the copy.  This can lead to speed-ups up to 100x
  for short tables that have hundreds of columns.  Closes #304.

- For external links, its relative paths are resolved now with respect
  to the directory of the main HDF5 file, rather than with respect to
  the current directory.  Closes #306.

- ``Expr.setInputsRange()`` and ``Expr.setOutputRange()`` do support
  ``numpy.integer`` types now.  Closes #285.

- Column names in tables can start with '__' now.  Closes #291.

- Unicode empty strings are supported now as atributes.  Addresses #307.

- Cython 0.13 and higher is supported now.  Fixes #293.

- PyTables should be more 'easy_install'-able now.  Addresses #298.


Changes from 2.2rc2 to 2.2 (final)
==================================

- Updated Blosc to 1.0 (final).

- Filter ID of Blosc changed from wrong 32010 to reserved 32001.  This
  will prevent PyTables 2.2 (final) to read files created with Blosc and
  PyTables 2.2 pre-final.  `ptrepack` can be used to retrieve those
  files, if necessary.  More info in ticket #281.

- Recent benchmarks suggest a new parametrization is better in most
  scenarios:

  * The default chunksize has been doubled for every dataset size.  This
    works better in most of scenarios, specially with the new Blosc
    compressor.

  * The HDF5 CHUNK_CACHE_SIZE parameter has been raised to 2 MB in order
    to better adapt to the chunksize increase.  This provides better hit
    ratio (at the cost of consuming more memory).

  Some plots have been added to the User's Manual (chapter 5) showing
  how the new parametrization works.


Changes from 2.2rc1 to 2.2rc2
=============================

- A new version of Blosc (0.9.5) is included.  This version is now
  considered to be stable and apt for production.  Thanks for all
  PyTables users that have contributed to find and report bugs.

- Added a new `IO_BUFFER_SIZE` parameter to ``tables/parameters.py``
  that allows to set the internal PyTables' buffer for doing I/O.  This
  replaces `CHUNKTIMES` but it is more general because it affects to all
  `Leaf` objects and also the `tables.Expr` module (and not only tables
  as before).

- `BUFFERTIMES` parameter in ``tables/parameters.py`` has been
  renamed to `BUFFER_TIMES` which is more consistent with other
  parameter names.

- On Windows platforms, the path to the tables module is now appended to
  sys.path and the PATH environment variable. That way DLLs and PYDs in
  the tables directory are to be found now.  Thanks to Christoph Gohlke
  for the hint.

- A replacement for barriers for Mac OSX, or other systems not
  implementing them, has been carried out.  This allows to compile
  PyTables on such platforms.  Fixes #278

- Fixed a couple of warts that raise compatibility warnings with
  forthcoming Python 2.7.

-  HDF5 1.8.5 is used in Windows binaries.

Changes from 2.2b3 to 2.2rc1
============================

- Numexpr is not included anymore in PyTables and has become a requisite
  instead.  This is because Numexpr already has decent enough installers
  and is available in the PyPI repository also, so it should be easy for
  users to fulfill this dependency.

- When using a Numexpr package that is turbo-loaded with Intel's
  VML/MKL, the parameter `MAX_THREADS` will control the number of
  threads that VML can use during computations.  For a finer control,
  the `numexpr.set_vml_num_threads()` can always be used.

- Cython is used now instead of Pyrex for Pyrex extensions.

- Updated to 0.9 version of Blosc compressor.  This version can make use
  of threads so as to accelerate the compression/decompression process.
  In order to change the maximum number of threads that Blosc can use (2
  by default), you can modify the `MAX_THREADS` variable in
  ``tables/parameters.py`` or make use of the new `setBloscMaxThreads()`
  global function.

- Reopening already opened files is supported now, provided that there is
  not incompatibility among intended usages (for example, you cannot
  reopen in append mode an already opened file in read-only mode).

- Option ``--print-versions`` for ``test_all.py`` script is now
  preferred over the deprecated ``--show-versions``.  This is more
  consistent with the existing `print_versions()` function.

- Fixed a bug that, under some circumstances, prevented the use of table
  iterators in `itertool.groupby()`.  Now, you can safely do things
  like::

    sel_rows = table.where('(row_id >= 3)')
    for group_id, grouped_rows in itertools.groupby(sel_rows, f_group):
        group_mean = average([row['row_id'] for row in grouped_rows])

  Fixes #264.

- Copies of `Array` objects with multidimensional atoms (coming from
  native HDF5 files) work correctly now (i.e. the copy holds the atom
  dimensionality).  Fixes #275.

- The `tables.openFile()` function does not try anymore to open/close
  the file in order to guess whether it is a HDF5 or PyTables one before
  opening it definitely.  This allows the `fcntl.flock()` and
  `fcntl.lockf()` Python functions to work correctly now (that's useful
  for arbitrating access to the file by different processes).  Thanks to
  Dag Sverre Seljebotn and Ivan Vilata for their suggestions on hunting
  this one!  Fixes #185.

- The estimation of the chunksize when using multidimensional atoms in
  EArray/Carray was wrong because it did not take in account the shape
  of the atom.  Thanks to Ralf Juengling for reporting.  Fixes #273.

- Non-contiguous arrays can now safely be saved as attributes.  Before,
  if arrays were not contiguous, incorrect data was saved in attr.
  Fixes #270.

- EXTDIM attribute for CArray/EArray now saves the correct extendeable
  dimension, instead of rubbish.  This does not affected functionality,
  because extendeable dimension was retrieved directly from shape
  information, but it was providing misleading information to the user.
  Fixes #268.

API changes
-----------

- Now, `Table.Cols.__len__()` returns the number of top level columns
  instead of the number of rows in table.  This is more consistent in
  that `Table.Cols` is an accessor for *columns*.  Fixes #276.


Changes from 2.2b2 to 2.2b3
===========================

- Blosc compressor has been added as an additional filter, in addition
  to the existing Zlib, LZO and bzip2.  This new compressor is meant for
  fast compression and extremely fast decompression.  Fixes #265.

- In `File.copyFile()` method, `copyuserattrs` was set to false as
  default.  This was unconsistent with other methods where the default
  value for `copyuserattrs` is true.  The default for this is true now.
  Closes #261.

- `tables.copyFile` and `File.copyFile` recognize now the parameters
  present in ``tables/parameters.py``.  Fixes #262.

- Backported fix for issue #25 in Numexpr (OP_NEG_LL treats the argument
  as an int, not a long long).  Thanks to David Cooke for this.

- CHUNK_CACHE_NELMTS in `tables/paramters.py` set to a prime number as
  Neil Fortner suggested.

- Workaround for a problem in Python 2.6.4 (and probably other versions
  too) for pickling strings like "0" or "0.".  Fixes #253.


Changes from 2.2b1 to 2.2b2
===========================

Enhancements
------------

- Support for HDF5 hard links, soft links and external links (when
  PyTables is compiled against HDF5 1.8.x series).  A new tutorial about
  its usage has been added to the 'Tutorials' chapter of User's Manual.
  Closes #239 and #247.

- Added support for setting HDF5 chunk cache parameters in file
  opening/creating time.  'CHUNK_CACHE_NELMTS', 'CHUNK_CACHE_PREEMPT'
  and 'CHUNK_CACHE_SIZE' are the new parameters.  See "PyTables'
  parameter files" appendix in User's Manual for more info.  Closes
  #221.

- New `Unknown` class added so that objects that HDF5 identifies as
  ``H5G_UNKNOWN`` can be mapped to it and continue operations
  gracefully.

- Optimization in the indexed queries when the resulting rows increase
  monotonically.  From 3x (for medium-size query results) and 10x (for very
  large query results) speed-ups can be expected.

- Added flag `--dont-create-sysattrs` to ``ptrepack`` so as to not
  create sys attrs (default is to do it).

- Support for native compound types in attributes.  This allows for
  better compatibility with HDF5 files.  Closes #208.

- Support for native NumPy dtype in the description parameter of
  `File.createTable()`.  Closes #238.


Bugs fixed
----------

- Added missing `_c_classId` attribute to the `UnImplemented` class.
  ``ptrepack`` no longer chokes while copying `Unimplemented` classes.

- The ``FIELD_*`` sys attrs are no longer copied when the
  ``PYTABLES_SYS_ATTRS`` parameter is set to false.

- `File.createTable()` no longer segfaults if description=None.  Closes
  #248.

- Workaround for avoiding a Python issue causing a segfault when saving
  and then retrieving a string attribute with values "0" or "0.".
  Closes #253.


API changes
-----------

- `Row.__contains__()` disabled because it has little sense to query for
  a key in Row, and the correct way should be to query for it in
  `Table.colnames` or `Table.colpathnames` better.  Closes #241.

- [Semantic change] To avoid a common pitfall when asking for the string
  representation of a `Row` class, `Row.__str__()` has been redefined.
  Now, it prints something like::

      >>> for row in table:
      ...     print row
      ...
      /newgroup/table.row (Row), pointing to row #0
      /newgroup/table.row (Row), pointing to row #1
      /newgroup/table.row (Row), pointing to row #2

  instead of::

      >>> for row in table:
      ...     print row
      ...
      ('Particle:      0', 0, 10, 0.0, 0.0)
      ('Particle:      1', 1, 9, 1.0, 1.0)
      ('Particle:      2', 2, 8, 4.0, 4.0)

  Use `print row[:]` idiom if you want to reproduce the old behaviour.
  Closes #252.


Other changes
-------------

- After some improvements in both HDF5 and PyTables, the limit before
  emitting a `PerformanceWarning` on the number of children in a group
  has been raised from 4096 to 16384.


Changes from 2.1.1 to 2.2b1
===========================

Enhancements
------------

- Added `Expr`, a class for evaluating expressions containing
  array-like objects.  It can evaluate expressions (like '3*a+4*b')
  that operate on arbitrary large arrays while optimizing the
  resources (basically main memory and CPU cache memory) required to
  perform them.  It is similar to the Numexpr package, but in addition
  to NumPy objects, it also accepts disk-based homogeneous arrays,
  like the `Array`, `CArray`, `EArray` and `Column` PyTables objects.

- Added support for NumPy's extended slicing in all `Leaf` objects.
  With that, you can do the next sort of selections::

      array1 = array[4]                       # simple selection
      array2 = array[4:1000:2]                # slice selection
      array3 = array[1, ..., ::2, 1:4, 4:]    # general slice selection
      array4 = array[1, [1,5,10], ..., -1]    # fancy selection
      array5 = array[np.where(array[:] > 4)]  # point selection
      array6 = array[array[:] > 4]            # boolean selection

  Thanks to Andrew Collette for implementing this for h5py, from which
  it has been backported.  Closes #198 and #209.

- Numexpr updated to 1.3.1.  This can lead to up a 25% improvement of
  the time for both in-kernel and indexed queries for unaligned
  tables.

- HDF5 1.8.3 supported.


Bugs fixed
----------

- Fixed problems when modifying multidimensional columns in Table
  objects.  Closes #228.

- Row attribute is no longer stalled after a table move or rename.
  Fixes #224.

- Array.__getitem__(scalar) returns a NumPy scalar now, instead of a
  0-dim NumPy array.  This should not be noticed by normal users,
  unless they check for the type of returned value.  Fixes #222.


API changes
-----------

- Added a `dtype` attribute for all leaves.  This is the NumPy
  ``dtype`` that most closely matches the leaf type.  This allows for
  a quick-and-dirty check of leaf types.  Closes #230.

- Added a `shape` attribute for `Column` objects.  This is formed by
  concatenating the length of the column and the shape of its type.
  Also, the representation of columns has changed an now includes the
  length of the column as the leading dimension.  Closes #231.

- Added a new `maindim` attribute for `Column` which has the 0 value
  (the leading dimension).  This allows for a better similarity with
  other \*Array objects.

- In order to be consistent and allow the extended slicing to happen
  in `VLArray` objects too, `VLArray.__setitem__()` is not able to
  partially modify rows based on the second dimension passed as key.
  If this is tried, an `IndexError` is raised now.  Closes #210.

- The `forceCSI` flag has been replaced by `checkCSI` in the next
  `Table` methods: `copy()`, `readSorted()` and `itersorted()`.  The
  change reflects the fact that a re-index operation cannot be
  triggered from these methods anymore.  The rational for the change
  is that an indexing operation is a potentially very expensive
  operation that should be carried out explicitly instead of being
  triggered by methods that should not be in charge of this task.
  Closes #216.


Backward incompatible changes
-----------------------------

- After the introduction of the `shape` attribute for `Column`
  objects, the shape information for multidimensional columns has been
  removed from the `dtype` attribute (it is set to the base type of
  the column now).  Closes #232.


  **Enjoy data!**

  -- The PyTables Team


.. Local Variables:
.. mode: rst
.. coding: utf-8
.. fill-column: 72
.. End:
