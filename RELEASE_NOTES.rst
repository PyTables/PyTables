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

* Automatic unpickling is now disabled by default.  Pickled attributes are
  returned as raw bytes, and reading :class:`ObjectAtom` data raises
  :exc:`PickleNotAllowedError`.

  Pickle payloads can execute arbitrary code while they are loaded.  Trusted
  legacy files can be opened explicitly with
  ``open_file(..., allow_pickle=True)``.  The
  :data:`parameters.ALLOW_PICKLE` parameter controls the default for files
  opened without an explicit ``allow_pickle`` argument.

  Trusted pandas files can be read with
  ``pandas.read_hdf(..., allow_pickle=True)`` or by constructing
  ``pandas.HDFStore(..., allow_pickle=True)``.  Writing remains enabled and
  emits :exc:`PickleSecurityWarning`, since the resulting data requires
  unpickling to be read.  The :program:`ptdump` utility provides the
  equivalent ``--allow-pickle`` option.


Thanks to:

* Teddy Tennant
* YuuLuo
