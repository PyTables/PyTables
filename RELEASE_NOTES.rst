========================================
 Release notes for PyTables 3.12 series
========================================

:Author: PyTables Developers
:Contact: pytables-dev@googlegroups.com

.. py:currentmodule:: tables

Changes from 3.11.1 to 3.12.0
=============================

* :meth:`Table.where` (and hence :meth:`Table.read_where`,
  :meth:`Table.get_where_list` and :meth:`Table.append_where`) now treats a
  *start* with no *stop* like a Python slice, i.e. the rows from *start* to
  the last one are considered.  Previously only one row was considered
  (:issue:`797`).
* Add git submodule for hdf5-blosc2 v3.0.1.
* Minimum Cython_ version is now v3.2
* Do not use plain eval in PyTables utilities (fixes security issues
  GHSA-54cf-28h8-9p23_ and GHSA-6mmx-p77c-4hmr_).
* Fix compatibility with c-blosc2 3.x.
* Build wheels against HDF5 2.2.0; fix blosc2 filter buffer size.

  The HDF5 1.14.6 bundled in PyTables v3.11.1 wheels was affected by ~30
  published CVEs whose fixes shipped only on the HDF5 2.x line
  (see HDFGroup/cve_hdf5), including CVE-2025-44905 and CVE-2025-44904
  (both rated 8.8 HIGH by NVD) and the CVE-2026-17572/17573/17574 batch
  fixed in 2.2.0.

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
* Fix typos in docstrings, comments and error messages.
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


.. _Cython: https://cython.org
.. _GHSA-54cf-28h8-9p23:
    https://github.com/PyTables/PyTables/security/advisories/GHSA-54cf-28h8-9p23
.. _GHSA-6mmx-p77c-4hmr:
    https://github.com/PyTables/PyTables/security/advisories/GHSA-6mmx-p77c-4hmr

* Automatic unpickling remains enabled by default for compatibility.
  Pickle payloads can execute arbitrary code while they are loaded.
  Untrusted files can be opened with ``open_file(..., allow_pickle=False)``.
  The :data:`parameters.ALLOW_PICKLE` parameter controls the default for
  files opened without an explicit ``allow_pickle`` argument; set it to
  ``False`` to disable unpickling globally for new file handles.

  The same per-file opt-out can be passed through pandas as
  ``pandas.read_hdf(..., allow_pickle=False)`` or
  ``pandas.HDFStore(..., allow_pickle=False)``.  Writing remains enabled
  and emits :exc:`PickleSecurityWarning`, since the resulting data
  requires unpickling to be read.  The :program:`ptdump` utility keeps
  unpickling disabled unless ``--allow-pickle`` is given.

  The default is expected to change to disabled in PyTables 3.13.


Thanks to:

* Teddy Tennant
* Jacob Rideout
* Adrian Altenhoff
* maxtaran2010
* YuuLuo
* Raashish Aggarwal
