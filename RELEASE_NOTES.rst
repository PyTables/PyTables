=======================================
 Release notes for PyTables 3.9 series
=======================================

:Author: PyTables Developers
:Contact: pytables-dev@googlegroups.com

.. py:currentmodule:: tables


Changes from 3.8.0 to 3.9.0
===========================

- Apply optimized slice read to Blosc2-compressed `CArray` and `EArray`, with
  Blosc2 NDim 2-level partitioning for multidimensional arrays
  (:issue:`1056`).  See "Multidimensional slicing and chunk/block sizes" in
  the User's Guide.  This development was funded by a NumFOCUS grant.
- Drop wheels and automated testing for Python 3.8; users or distributions may
  still build and test with Python 3.8 on their own (see :commit:`ae1e60e` and
  :commit:`47f5946`).
- Improve `setup.py` and `blosc2` discovery mechanism.
- Update external libraries:

  * c-blosc v1.21.4
  * hdf5 v1.14.1
  * lz4 v1.9.4
  * zlib v1.2.13

- Fix compatibility with numexpr v2.8.5.
