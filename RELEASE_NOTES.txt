=======================================
 Release notes for PyTables 2.1 series
=======================================

:Author: Francesc Alted i Abad
:Contact: faltet@pytables.org


Changes from 2.1.1 to 2.1.2
===========================

Bug fixes
---------

- Solved problems with Table.modifyColumn() when the column(s) is
  multidimensional. Fixes #228.

- The row attribute of a table seems stalled after a table move or
  rename.  Fixes #224.

- Fixed a problem with ``len(array)`` in 32-bit platforms when array
  is large enough (> 2**31).

- Added missing `_c_classId` attribute to the `UnImplemented` class.
  ``ptrepack`` no longer chokes while copying `Unimplemented` classes.

- The ``FIELD_*`` sys attrs are no longer copied when the
  ``PYTABLES_SYS_ATTRS`` parameter is set to false.

- The ``FILTERS`` attribute is not added anymore when
  ``PYTABLES_SYS_ATTR`` parameter is set to false.

- Disable the printing of Unicode characters that cannot be printed on
  win32 platform. Fixes #235.

Other changes
-------------

- When retrieving a row of a 1-dimensional array, a 0-dim array was
  returned instead of a numpy scalar.  Now, an actuall numpy scalar is
  returned.  Closes #222.

- LZO and bzip2 filters adapted to an API fix introduced in HDF5
  1.8.3.  Closes #225.

- Unsupported HDF5 types in attributes are no longer transferred
  during copies. A new `_v_unimplemented` list have been added in
  `AttributeSet` class so as to keep track of such attributes.  Closes
  #240.

- LZO binaries have disappeared from the GnuWin32 repository.  Until
  they come eventually back, they have been put at
  http://www.pytables.org/download/lzo-win. This has been documented
  in the install chapter.


Changes from 2.1 to 2.1.1
=========================

Bug fixes
---------

- Fixed a memory leak when a lot of queries were made.  Closes #203
  and #207.

- The chunkshape="auto" parameter value of `Leaf.copy()` is honored
  now, even when the (start, stop, step) parameters are specified.
  Closes #204.

- Due to a flaw in its design, the `File` class was not able to be
  subclassed.  This has been fixed.  Closes #205.

- Default values were not correctly retrieved when opening already
  created CArray/EArray objects.  Fixed.  Closes #212.

- Fixed a problem with the installation of the ``nctoh5`` script that
  prevented it from being executed.  Closes #215.

- [Pro] The ``iterseq`` cache ignored non-indexed conditions, giving
  wrong results when those appeared in condition expressions.  This
  has been fixed.  Closes #206.

Other changes
-------------

- `openFile()`, `isHDF5File()` and `isPyTablesFile()` functions accept
  Unicode filenames now.  Closes #202 and #214.

- When creating large type sizes (exceeding 64 KB), HDF5 complained
  and refused to do so.  The HDF5 team has logged the issue as a bug,
  but meanwhile it has been implemented a workaround in PyTables that
  allows to create such large datatypes for situations that does not
  require defaults other than zero.  Addresses #211.

- In order to be consistent with how are stored the other data types,
  Unicode attributes are retrieved now as NumPy scalars instead of
  Python Unicode strings or NumPy arrays.  For the moment, I've fixed
  this through pickling the Unicode strings.  In the future, when HDF5
  1.8.x series would be a requirement, that should be done via a HDF5
  native Unicode type.  Closes #213.


----

  **Enjoy data!**

  -- The PyTables Team


.. Local Variables:
.. mode: rst
.. coding: utf-8
.. fill-column: 72
.. End:
