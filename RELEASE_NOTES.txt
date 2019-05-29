=======================================
 Release notes for PyTables 3.5 series
=======================================

:Author: PyTables Developers
:Contact: pytables-dev@googlegroups.com

.. py:currentmodule:: tables


Changes from 3.5.1 to 3.5.2
===========================

- Fixed compatibility with python 3.8: Fixed `Dictonary keys changed during
  iteration` RuntimeError while moving/renameing a node.
  Thanks to Christoph Gohlke for reporting and Miro Hroncok for help with
  building PyTables for python 3.8alpha (cython compatibility).
  see :issue:`733` and PR #737.
- Fixed a bug in offset calculations producing floats instead of ints
  affecting python 3. See PR #736. Thanks to Brad Montgomery.


Changes from 3.5.0 to 3.5.1
===========================

- Maintenance release to fix how PyPi repo is handling wheel versions.


Changes from 3.4.4 to 3.5.0
===========================

Improvements
------------
 - When copying data from native HDF5 files with padding in compound types,
   the padding is not removed now by default.  This allows for better
   compatibility with existing HDF5 applications that expect the padding
   to stay there.
   Also, when the `description` is a NumPy struct array with padding, this
   is honored now.  The previous behaviour (i.e. getting rid of paddings) can
   be replicated by passing the new `allow_padding` parameter when opening
   a file.  For some examples, see the new `examples/tables-with-padding.py`
   and `examples/attrs-with-padding.py`.  For details on the implementation
   see :issue:`720`.
 - Added a new flag `--dont-allow-padding` in `ptrepack` utility so as to
   replicate the previous behaviour of removing padding during file copies.
   The default is to honor the original padding in copies.
 - Improve compatibility with numpy 1.16.
 - Improve detection of the LZO2 library at build time.
 - Suppress several warnings.
 - Add AVX2 support for Windows.  See PR #716.  Thanks to Robert McLeod.
