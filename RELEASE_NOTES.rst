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

* Fix typos in docstrings, comments and error messages.
* Fix issue with non-zero direct-chunk filter mask (:issue:`1325`).
* Fix infinite busy loop in ``ObjectCache.updateslot`` (:issue:`1254`).
* Fix ``VLArray.append`` after ``truncate()`` writing past the new extent
  and reading back empty values (:issue:`1102`).


.. _Cython: https://cython.org
.. _GHSA-54cf-28h8-9p23:
    https://github.com/PyTables/PyTables/security/advisories/GHSA-54cf-28h8-9p23
.. _GHSA-6mmx-p77c-4hmr:
    https://github.com/PyTables/PyTables/security/advisories/GHSA-6mmx-p77c-4hmr


Thanks to:

* Teddy Tennant
* Jacob Rideout
* Adrian Altenhoff
* maxtaran2010
