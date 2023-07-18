=======================================
 Release notes for PyTables 3.6 series
=======================================

:Author: PyTables Developers
:Contact: pytables-dev@googlegroups.com

.. py:currentmodule:: tables


Changes from 3.6.0 to 3.6.1
===========================

Maintenance release to fix packaging issues. No new features or bugfixes.


Changes from 3.5.3 to 3.6.0
===========================

PyTables 3.6 no longer supports Python 2.7 see PR #747.

Improvements
------------
- Full python 3.8 support.
- On Windows PyTables wheels on PyPI are linked to `pytables_hdf5.dll` instead
  of `hdf5.dll` to prevent collisions with other packages/wheels that also
  vendor `hdf5.dll`.
  This should prevent problems that arise when a different version of a dll
  is imported that the version to which the program was linked to.
  This problem is known as "DLL Hell".
  With the renaming of the HDF5 DLL to `pytables_hdf5.dll` these problems
  should be solved.

Bugfixes
--------
- Bugfix for HDF5 files/types with padding. For details see :issue:`734`.
- More fixes for python 3.8 compatibility: Replace deprecated time.clock
  with time.perf_counter
  Thanks to Sergio Pascual (sergiopasra). see :issue:`744` and PR #745.
- Improvements in tests as well as clean up from dropping Python 2.7 support.
  Thanks to Seth Troisi (sethtroisi).
