=======================================
 Release notes for PyTables 3.3 series
=======================================

Changes from 3.2.3.1 to 3.3.0
=============================

Improvements
------------

- Single codebase Python 2 and 3 support (PR #493).
- Internal Blosc version updated to 1.11.1 (closes :issue:`541`)
- Full BitShuffle support for new Blosc versions (>= 1.8).
- It is now possible to remove all rows from a table.
- It is now possible to read reference types by dereferencing them as
  numpy array of objects (closes :issue:`518` and :issue:`519`).
  Thanks to Ehsan Azar
- Get rid of the `-native` compile flag (closes :issue:`503`)
- The default number of threads to run numexpr (MAX_NUMEXPR_THREADS)
  internally has been raised from 2 to 4.  This is because we are in
  2016 and 4 core configurations are becoming common.
- In order to avoid locking issues when using PyTables concurrently in
  several process, MAX_BLOSC_THREADS has been set to 1 by default.  If
  you are running PyTables in one single process, you may want to
  experiment if higher values (like 2 or 4) bring better performance for
  you.

Bugs fixed
----------

- On python 3 try 'latin1' encoding before 'bytes' encoding during unpickling
  of node attributes pickled on python 2. Better fix for :issue:`560`.
- Fixed Windows 32 and 64-bit builds.
