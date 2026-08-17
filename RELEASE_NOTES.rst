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
* YuuLuo
