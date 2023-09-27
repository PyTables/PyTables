=======================================
 Release notes for PyTables 3.7 series
=======================================

:Author: PyTables Developers
:Contact: pytables-dev@googlegroups.com

.. py:currentmodule:: tables


Changes from 3.6.1 to 3.7.0
===========================

Improvements
------------
- Compatibility with Python 3.10, numpy 1.21 and HDF5 1.12.
- Support for Python 3.5 has been dropped (:issue:`840` and :issue:`850`).
- Windows: Significantly faster `import tables` (:PR:`781`).
  Thanks to Christoph Gohlke.
- Internal C-Blosc sources updated to 1.21.1 (:issue:`931`).
  Note that, starting from C-Blosc 1.19 does not include the Snappy codec
  sources anymore, so Snappy will be not available if you compile from
  included sources; other packages (like conda or wheels),
  may (or may not) include it.
- Stop using appveyor and deprecated ci-helpers (closes :issue:`827`).
- Switch to `git submodule` for the management of vendored c-blosc sources.
- CI moved to GitHub Actions (GHA).
- Drop Travis-CI.
- Improved code formatting and notation consistency (:issue:`873`,
  :issue:`868`, :issue:`865` thanks to Miroslav Šedivý).
- Improve the use of modern Python including :mod:`pathlib`, f-strings
  (:issue:`859`, :issue:`855`, :issue:`839` and :issue:`818`
  thanks to Miroslav Šedivý).
- Several improvements to wheels generation in CI
  (thanks to Andreas Motl @amotl and Matthias @xmatthias).
- Simplified management of version information.
- Drop dependency on the deprecated distutils.
- Modernize the setup script and add support for PEP517 (:issue:`907`).

Bugfixes
--------
- Fix `pkg-config` (`setup.py`) for Python 3.9 on Debian.
  Thanks to Marco Sulla (:PR:`792`).
- Fix ROFileNode fails to return the `fileno()` (:issue:`633`).
- Do not flush read only files (:issue:`915` thanks to @lrepiton).

Other changes
-------------
- Drop the deprecated `hdf5Version` and `File.open_count`.
- the :func:`get_tables_version` and :func:`get_hdf5_version` functions are
  now deprecated please use the coresponding :data:`tables.__version__` and
  :data:`tables.hdf5_version` instead.
