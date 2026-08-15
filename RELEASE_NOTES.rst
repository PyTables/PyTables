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

* The new :data:`parameters.ALLOW_PICKLE` parameter has been added to allow
  the user to globally enable/disable the use of pickle in PyTables\.

  Pickle is used in PyTables mostly to serialize attributes and data that are
  not representable as numpy arrays or native types.

  Unfortunately the use of pickle is not secure and can be exploited for
  remote code execution.

  For compatibility reasons, the use of pickle is enabled by default.


Thanks to:

* Teddy Tennant
