=======================================
 Release notes for PyTables 3.2 series
=======================================

:Author: PyTables Developers
:Contact: pytables-dev@googlegroups.com

.. py:currentmodule:: tables


Changes from 3.2.3 to 3.2.3.1
=============================

Fixed issues with pip install.


Changes from 3.2.2 to 3.2.3
===========================

Improvements
------------

- It is now possible to use HDF5 with the new shared library naming scheme
  (>= 1.8.10, hdf5.dll instead of hdf5dll.dll) on Windows (:issue:`540`).
  Thanks to Tadeu Manoel.
- Now :program: `ptdump` sorts output by node name and does not print a
  backtrace if file cannot be opened.
  Thanks to Zbigniew Jędrzejewski-Szmek.


Bugs fixed
----------

- Only run `tables.tests.test_basics.UnicodeFilename` if the filesystem
  encoding is utf-8. Closes :issue:`485`.
- Add lib64 to posix search path. (closes :issue:`507`)
  Thanks to Mehdi Sadeghi.
- Ensure cache entries are removed if fewer than 10 (closes :issue:`529`).
  Thanks to Graham Jones.
- Fix segmentation fault in a number of test cases that use
  :class:`index.Index` (closes :issue:`532` and :issue:`533`).
  Thanks to Diane Trout.
- Fixed the evaluation of transcendental functions when numexpr is
  compiled with VML support (closes :issue:`534`, PR #536).
  Thanks to Tom Kooij.
- Make sure that index classes use buffersizes that are a multiple
  of chunkshape[0] (closes :issue:`538`, PR #538).
  Thanks to Tom Kooij.
- Ensure benchmark paths exist before benchmarks are executed (PR #544).
  Thanks to rohitjamuar.

Other changes
-------------

- Minimum Cython_ version is now v0.21


.. _Cython: http://cython.org


Changes from 3.2.1.1 to 3.2.2
=============================

Bug fixed
---------

- Fix AssertionError in Row.__init_loop. See :issue:`477`.
- Fix issues with Cython 0.23. See :issue:`481`.
- Only run `tables.tests.test_basics.UnicodeFilename` if the filesystem
  encoding is utf-8. Closes :issue:`485`.
- Fix missing PyErr_Clear. See :issue:`486`.
- Fix the C type of some numpy attributes. See :issue:`494`.
- Cast selection indices to integer. See :issue:`496`.
- Fix indexesextension._keysort_string. Closes :issue:`497` and :issue:`498`.


Changes from 3.2.1 to 3.2.1.1
=============================

- Fix permission on distributed source distribution

Other changes
-------------

- Minimum Cython_ version is now v0.21


.. _Cython: http://cython.org


Changes from 3.2.0 to 3.2.1
===========================

Bug fixed
---------

- Fix indexesextension._keysort. Fixes :issue:`455`. Thanks to Andrew Lin.


Changes from 3.1.1 to 3.2.0
===========================

Improvements
------------

- The `nrowsinbuf` is better computed now for EArray/CArray having
  a small `chunkshape` in the main dimension.  Fixes #285.

- PyTables should be installable very friendly via pip, including NumPy
  being installed automatically in the unlikely case it is not yet
  installed in the system.  Thanks to Andrea Bedini.

- setup.py has been largely simplified and now it requires *setuptools*.
  Although we think this is a good step, please keep us informed this is
  breaking some installation in a very bad manner.

- setup.py now is able to used *pkg-config*, if available, to locate required
  libraries (hdf5, bzip2, etc.). The use of *pkg-config* can be controlled
  via setup.py command line flags or via environment variables.
  Please refer to the installation guide (in the *User Manual*) for details.
  Closes :issue:`442`.

- It is now possible to create a new node whose parent is a softlink to another
  group (see :issue:`422`). Thanks to Alistair Muldal.

- :class:`link.SoftLink` objects no longer need to be explicitly dereferenced.
  Methods and attributes of the linked object are now automatically accessed
  when the user acts on a soft-link (see :issue:`399`).
  Thanks to Alistair Muldal.

- Now :program:`ptrepack` recognizes hardlinks and replicates them in the
  output (*repacked*) file. This saves disk space and makes repacked files
  more conformal to the original one. Closes :issue:`380`.

- New :program:`pttree` script for printing HDF5 file contents as a pretty
  ASCII tree (closes :issue:`400`). Thanks to Alistair Muldal.

- The internal Blosc library has been downgraded to version 1.4.4.  This
  is in order to still allow using multiple threads *inside* Blosc, even
  on multithreaded applications (see :issue:`411`, :issue:`412`,
  :issue:`437` and :issue:`448`).

- The :func:`print_versions` function now also reports the version of
  compression libraries used by Blosc.

- Now the :file:`setup.py` tries to use the '-march=native' C flag by
  default. In falls back on '-msse2' if '-march=native' is not supported
  by the compiler. Closes :issue:`379`.

- Fixed a spurious unicode comparison warning (closes :issue:`372` and
  :issue:`373`).

- Improved handling of empty string attributes. In previous versions of
  PyTables empty string were stored as scalar HDF5 attributes having size 1
  and value '\0' (an empty null terminated string).
  Now empty string are stored as HDF5 attributes having zero size

- Added a new cookbook recipe and a couple of examples for simple threading
  with PyTables.

- The redundant :func:`utilsextension.get_indices` function has been
  eliminated (replaced by :meth:`slice.indices`). Closes :issue:`195`.

- Allow negative indices in point selection (closes :issue:`360`)

- Index wasn't being used if it claimed there were no results.
  Closes :issue:`351` (see also :issue:`353`)

- Atoms and Col types are no longer generated dynamically so now it is easier
  for IDEs and static analysis tool to handle them (closes :issue:`345`)

- The keysort functions in idx-opt.c have been cythonised using fused types.
  The perfomance is mostly unchanged, but the code is much more simpler now.
  Thanks to Andrea Bedini.

- Small unit tests re-factoring:

  * :func:`print_versions` and :func:`tests.common.print_heavy` functions
     moved to the :mod:`tests.common` module

  * always use :func:`print_versions` when test modules are called as scripts

  * use the unittest2_ package in Python 2.6.x

  * removed internal machinery used to replicate unittest2_ features

  * always use :class:`tests.common.PyTablesTestCase` as base class for all
    test cases

  * code of the old :func:`tasts.common.cleanup` function has been moved to
    :meth:`tests.common.PyTablesTestCase.tearDown` method

  * new implementation of :meth:`tests.common.PyTablesTestCase.assertWarns`
    compatible with the one provided by the standard :mod:`unittest` module
    in Python >= 3.2

  * use :meth:`tests.common.PyTablesTestCase.assertWarns` as context manager
    when appropriate

  * use the :func:`unittest.skipIf` decorator when appropriate

  * new :class:tests.comon.TestFileMixin: class


.. _unittest2: https://pypi.python.org/pypi/unittest2


Bugs fixed
----------

- Fixed compatibility problems with numpy 1.9 and 1.10-dev
  (closes :issue:`362` and :issue:`366`)

- Fixed compatibility with Cython >= 0.20 (closes :issue:`386` and
  :issue:`387`)

- Fixed support for unicode node names in LRU cache (only Python 2 was
  affected). Closes :issue:`367` and :issue:`369`.

- Fixed support for unicode node titles (only Python 2 was affected).
  Closes :issue:`370` and :issue:`374`.

- Fixed a bug that caused the silent truncation of unicode attributes
  containing the '\0' character. Closes :issue:`371`.

- Fixed :func:`descr_from_dtype` to work as expected with complex types.
  Closes :issue:`381`.

- Fixed the :class:`tests.test_basics.ThreadingTestCase` test case.
  Closes :issue:`359`.

- Fix incomplete results when performing the same query twice and exhausting
  the second iterator before the first. The first one writes incomplete
  results to *seqcache* (:issue:`353`)

- Fix false results potentially going to *seqcache* if
  :meth:`tableextension.Row.update` is used during iteration
  (see :issue:`353`)

- Fix :meth:`Column.create_csindex` when there's NaNs

- Fixed handling of unicode file names on windows (closes :issue:`389`)

- No longer not modify :data:`sys.argv` at import time (closes :issue:`405`)

- Fixed a performance issue on NFS (closes :issue:`402`)

- Fixed a nasty problem affecting results of indexed queries.
  Closes :issue:`319` and probably :issue:`419` too.

- Fixed another problem affecting results of indexed queries too.
  Closes :issue:`441`.

- Replaced "len(xrange(start, stop, step))" -> "len(xrange(0, stop -
  start, step))" to fix issues with large row counts with Python 2.x.
  Fixes #447.


Other changes
-------------

- Cython is not a hard dependency anymore (although developers will need it
  so as to generated the C extension code).

- The number of threads used by default for numexpr and Blosc operation that
  was set to the number of available cores have been reduced to 2.  This is
  a much more reasonable setting for not creating too much overhead.


  **Enjoy data!**

  -- The PyTables Developers


.. Local Variables:
.. mode: rst
.. coding: utf-8
.. fill-column: 72
.. End:
