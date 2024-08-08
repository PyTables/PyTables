========================================
 Release notes for PyTables 3.10 series
========================================

:Author: PyTables Developers
:Contact: pytables-dev@googlegroups.com

.. py:currentmodule:: tables


Changes from 3.10.0 to 3.9.2
============================

XXX version-specific blurb XXX

Improvements
------------

- Add type hints to atom.py. This also narrows some types, only allowing bytes
  to be stored in VLStringAtom and only str in VLUnicodeAtom.

Other changes
-------------

- Add wheels for macOS ARM64 (Apple Silicon) (:PR:`1050`). Thanks to Clemens Brunner.
