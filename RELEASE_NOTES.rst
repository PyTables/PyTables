========================================
 Release notes for PyTables 3.12 series
========================================

:Author: PyTables Developers
:Contact: pytables-dev@googlegroups.com

.. py:currentmodule:: tables

Changes from 3.11.1 to 3.12.0
=============================

Breaking changes
----------------

* The ``tests`` folders have been moved outside the ``tables`` package
  to make it lighter once installed.

  Consequently the `tables.test` function is no longer available.

  Moreover, the ``print_versions`` function has been moved to ``tables.utils``.
* The setialization-de-serialization of attributes and datasets via ``pickle``
  now raised a :exc:`PickleSecurityWarning` (see section below).


Security fixes
--------------

* Fix crash reading a table whose compound type has an integer column with no
  matching native size (3, 5, 6 or 7 bytes).  ``get_nested_type`` builds the
  native in-memory compound sized from the on-disk type and inserts each
  member at its on-disk offset, which only holds while every member keeps its
  on-disk size.  An odd-precision integer has no NumPy counterpart, so
  ``get_native_type`` widens it and the inserted member overruns the compound;
  ``H5Dread`` then walks past the malformed datatype on read (crash in
  ``H5T__sort_value``).  Such a column is now rejected and the leaf opens as
  ``UnImplemented``.
* Fix heap buffer overflow and wild pointer dereference reading a leaf whose
  ``CLASS`` attribute disagrees with its datatype.  A ``CLASS`` attribute
  selects the reader without re-checking the datatype, so a foreign file could
  point the VLArray reader at a fixed ``H5T_ARRAY`` dataset (read as an array
  of ``hvl_t`` handles, overflowing the row buffer and dereferencing
  file-controlled bytes as pointers) or the Array/CArray/EArray readers at a
  variable-length dataset (``hvl_t`` handles written into a buffer sized for
  the base atom).  ``H5VLARRAYget_info`` now requires an ``H5T_VLEN`` datatype
  and ``Array._open_array`` rejects ``H5T_VLEN``, so the leaf opens as
  ``UnImplemented`` instead.
* Build wheels against HDF5 2.2.0; fix blosc2 filter buffer size.

  The HDF5 1.14.6 bundled in PyTables v3.11.1 wheels was affected by ~30
  published CVEs whose fixes shipped only on the HDF5 2.x line
  (see HDFGroup/cve_hdf5), including CVE-2025-44905 and CVE-2025-44904
  (both rated 8.8 HIGH by NVD) and the CVE-2026-17572/17573/17574 batch
  fixed in 2.2.0.
* Fix heap buffer overflow reading Blosc2 datasets whose filter typesize
  disagrees with the datatype.  ``get_blosc2_slice`` took the element size
  from the Blosc2 filter metadata (``cd_values[2]``) instead of the
  dataset's datatype, so a crafted file could make the optimized slice read
  copy past the receive buffer.  The optimized path now falls back to the
  plain HDF5 read when the two sizes disagree.
* Fix heap buffer overflow reading tables with a variable-length column.
  ``get_nested_type`` mapped a variable-length compound member to its base
  atom and dropped the on-disk ``hvl_t`` size, so the NumPy record was
  smaller than the on-disk row while ``H5Dread`` filled the whole ``hvl_t``,
  writing past the read buffer.  Such a column is now rejected and the leaf
  opens as ``UnImplemented``.
* Fix out-of-bounds write opening a ``Table`` or ``VLArray`` leaf whose
  dataset has more than one dimension.  ``Table._get_info`` and
  ``H5VLARRAYget_info`` read the dataset extent into a single ``hsize_t``
  (both are 1-D entities), so a chunked compound or vlen dataset of higher
  rank made ``H5Sget_simple_extent_dims`` write past that buffer, reachable
  from a forged ``CLASS`` attribute.  Both readers now reject a rank other
  than one and the leaf opens as ``UnImplemented``.
* Keep ``allow_pickle=False`` when dereferencing an :class:`ExternalLink`.
  The linked file was opened with the default policy, so a file opened with
  unpickling disabled could still have pickles loaded from a second file it
  links to.  The linked file now inherits the disabled policy unless
  ``allow_pickle`` is passed explicitly to the link call.
* Do not use plain eval in PyTables utilities (fixes security issues
  GHSA-54cf-28h8-9p23_ and GHSA-6mmx-p77c-4hmr_).
* Fix NULL-pointer dereference when opening a malformed HDF5 root attribute
  (Closes GHSA-jc4m-9v2h-9348_).
* Pickle payloads can execute arbitrary code while they are loaded.

  Untrusted files can now be opened with ``open_file(..., allow_pickle=False)``.

  The :data:`parameters.ALLOW_PICKLE` parameter controls the default for
  files opened without an explicit ``allow_pickle`` argument; set it to
  ``False`` to disable unpickling globally for new file handles.

  The same per-file opt-out can be passed through pandas_ as
  ``pandas.read_hdf(..., allow_pickle=False)`` or
  ``pandas.HDFStore(..., allow_pickle=False)``.

  Writing remains enabled and emits :exc:`PickleSecurityWarning`, since the
  resulting data requires unpickling to be read.
  The :program:`ptdump` and :program:`ptrepack` utilities keep unpickling
  disabled unless ``--allow-pickle`` is given, and :program:`pttree` never
  unpickles.

  Automatic unpickling remains enabled by default for compatibility.

  The default is expected to change to disabled in PyTables 3.13.

  See also GHSA-3r7g-mq96-7hhc_.


Bug fixed
---------

* Point selections preserve coordinate views and accept read-only index arrays.
* Atom shapes reject non-integer dimensions instead of silently truncating them.
* Array indexing rejects out-of-range and non-integer indices before converting
  coordinates or accessing data.
* Reusing an :class:`Expr` after selecting an input range or a smaller output
  no longer truncates later results.  Evaluation and iteration preserve the
  expression's input shape, so ranges can also be reset between calls.
* Columns named like a Numexpr function (e.g. ``where`` or ``abs``) can be
  used in :meth:`Table.where` and the other query methods.  They were taken
  for the function, so the query failed (:issue:`638`).
* Reading a table whose on-disk compound type has a nested compound column
  with padding no longer overflows the read buffer.  The description now
  keeps the on-disk itemsize and member offsets of nested columns (as it
  already did for the top-level type, see #765), so the NumPy container
  matches ``H5Tget_size`` and ``H5Dread`` cannot write past it.
* ``indexesextension.keysort`` now rejects a second array shorter than the
  first instead of reordering it out of bounds.  It reorders the first
  ``len(array1)`` elements of ``array2`` in place, so a shorter ``array2``
  was read and written past its end.
* Indexed queries now reject an index whose ``bounds`` or ``ranges``
  arrays do not match the dtype (and, for ``ranges``, the shape) of its
  ``sorted`` values.  The search buffers are sized after ``sorted``, so a
  wider ``bounds`` array in a crafted file made ``H5Dread`` write past them.
* Opening a leaf or attribute whose datatype is an enum with a base integer
  wider than 8 bytes no longer overflows a heap buffer.  ``get_native_type``
  and ``get_type_enum`` asked HDF5 for the native type with
  ``H5Tget_native_type``, which overruns its buffer for such a base; the
  width is now checked first and the type is treated as unsupported.
* Group and node walks, including ``ptdump``, terminate when group hard links
  form cycles (:issue:`794`). Links to an ancestor are listed without
  traversing their children again; non-cyclic aliases retain their descendants.
* Preserve pending :meth:`Row.update` changes when iteration stops early.
  :meth:`Table.flush`, :meth:`File.flush` and :meth:`File.close` now write
  them even if the iterator has been discarded (:issue:`8`, :issue:`1098`).
* :meth:`Table.where` (and hence :meth:`Table.read_where`,
  :meth:`Table.get_where_list` and :meth:`Table.append_where`) now treats a
  *start* with no *stop* like a Python slice, i.e. the rows from *start* to
  the last one are considered.  Previously only one row was considered
  (:issue:`797`).
* Preserve field values when :meth:`Table.append` converts structured arrays
  with a different dtype or byteorder (:issue:`658`), and copy strided arrays
  before appending their records.
* Fix :meth:`Table.remove_rows` with a *step* greater than 1, which removed
  the wrong rows or raised ``OverflowError``.  An empty range now removes
  nothing instead of raising ``HDF5ExtError``, and :meth:`Table.remove_row`
  accepts negative indices.
* Indexing a multidimensional :class:`Array` with a single list or 1-D
  array of integers, e.g. ``array[[0, 2]]``, now selects along the first
  axis like NumPy instead of raising ``HDF5ExtError``.
* Modifying a nested column of an indexed table (with :meth:`Row.update`,
  :meth:`Table.modify_column` or :meth:`Table.modify_columns`) no longer
  raises ``KeyError``, and indexes on its subcolumns are marked dirty
  (:issue:`699`).
* Setting a table row from a record scalar, e.g. ``table[2] = table[0]``,
  no longer raises ``ValueError``.
* Queries that compare an indexed column with another column, e.g.
  ``table.where("a > b")`` or ``table.where("a == a")``, no longer raise
  ``AttributeError: 'Column' object has no attribute 'tolist'``.  The index
  of such a column is not used for that comparison.
* :meth:`Table.iterrows` and :meth:`Table.read` with a negative *step* now
  return the same rows as the equivalent Python slice.  Before, they could
  raise ``OverflowError``, skip the last row or (for :meth:`Table.read`)
  return wrong or uninitialized values.
* :meth:`Table.append` accepts a single row given as a tuple or a record
  (e.g. ``table.append(table[0])``) instead of raising ``IndexError``, and
  :meth:`Table.modify_rows` accepts a single record.
* The error raised by :meth:`EArray.append` for an object of the wrong rank
  now shows the rank of the EArray instead of ``(myrank)``.
* :meth:`VLArray.get_row_size` accepts negative row numbers, counting from
  the end like indexing does, instead of raising ``OverflowError``.
* Fix issue with non-zero direct-chunk filter mask (:issue:`1325`).
* Fix infinite busy loop in ``ObjectCache.updateslot`` (:issue:`1254`).
* Fix ``VLArray.append`` after ``truncate()`` writing past the new extent
  and reading back empty values (:issue:`1102`).
* Fix a one byte stack buffer overflow when opening a dataset whose HDF5
  datatype has a byte order PyTables does not support, i.e. VAX ordered or
  mixed endian types.
* Fix a heap buffer overflow when reading a compound (record) attribute whose
  HDF5 datatype carries trailing padding; the read buffer now keeps the
  on-disk itemsize, matching how padded tables are already handled.
* Fix an out of bounds stack read when opening a dataset whose filter
  pipeline holds more client data values than ``get_filter_names()`` reads
  at once; the count HDF5 reports is now clamped to the buffer it filled.
  Filter client data values above ``0x7fffffff`` are also no longer reported
  as negative numbers on platforms where ``long`` is 32 bit, i.e. Windows.
* Fix a heap buffer overflow when reading an ``H5T_BITFIELD`` attribute or
  dataset wider than one byte.  PyTables mapped every bitfield to the 1-byte
  ``bool`` dtype, so the read buffer was sized below the on-disk type and
  ``H5Aread``/``H5Dread`` wrote past it.  Only 1-byte bitfields (booleans)
  are produced by PyTables, so wider ones from foreign files are now
  reported as an unsupported type instead.
* Fix a heap buffer overflow when reading a string system attribute that
  holds more than one element.  ``H5ATTRget_attribute_string`` sized its
  buffer for a single element, but ``H5Aread`` fills the whole attribute,
  so a non-scalar string attribute wrote past it.  ``is_pytables_file``
  reads ``PYTABLES_FORMAT_VERSION`` through that path, so merely
  identifying a foreign file could trigger the overflow; the buffer is now
  sized for the whole attribute.
* Fix typos in docstrings, comments and error messages.


Other changes
-------------

* Minimum Python_ version is now 3.11.
* Minimum Cython_ version is now v3.2.
* Minimum numpy_ version is nor 2.0.
* Add git submodule for hdf5-blosc2 v3.0.1.
* Fix compatibility with c-blosc2 v3.x
* Read scalar HDF5 enum attributes as NumPy scalars with the corresponding
  integer dtype, including enum attributes stored with a foreign byte order.
* Move test data to a dedicated sub-folder to ``tests``.
* Add Windows ARM64 wheels builds.
* Fix compatibility with numpy_ >= 2.5.



Thanks to:

* Teddy Tennant
* Jacob Rideout
* Adrian Altenhoff
* Max (maxtaran2010)
* YuuLuo
* Raashish Aggarwal
* Naveed
* CAOShurong
* Eric Larson
* Nikhil Dabas


.. _Python: https://www.python.org
.. _Cython: https://cython.org
.. _numpy: https://numpy.org
.. _pandas: https://pandas.pydata.org
.. _GHSA-54cf-28h8-9p23:
    https://github.com/PyTables/PyTables/security/advisories/GHSA-54cf-28h8-9p23
.. _GHSA-6mmx-p77c-4hmr:
    https://github.com/PyTables/PyTables/security/advisories/GHSA-6mmx-p77c-4hmr
.. _GHSA-jc4m-9v2h-9348:
    https://github.com/PyTables/PyTables/security/advisories/GHSA-jc4m-9v2h-9348
.. _GHSA-3r7g-mq96-7hhc:
    https://github.com/PyTables/PyTables/security/advisories/GHSA-3r7g-mq96-7hhc
