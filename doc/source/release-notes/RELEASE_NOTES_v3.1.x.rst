Changes from 3.1.0 to 3.1.1
===========================

Bugs fixed
----------

- Fixed a critical bug that caused an exception at import time.
  The error was triggered when a bug in long-double detection is detected
  in the HDF5 library (see :issue:`275`) and numpy_ does not expose
  `float96` or `float128`. Closes :issue:`344`.
- The internal Blosc_ library has been updated to version 1.3.5.
  This fixes a false buffer overrun condition that made c-blosc to fail,
  even if the problem was not real.


Improvements
------------

- Do not create a temporary array when the *obj* parameter is not specified
  in :meth:`File.create_array` (thanks to Francesc).
  Closes :issue:`337` and :issue:`339`).
- Added two new utility functions
  (:func:`tables.nodes.filenode.read_from_filenode` and
  :func:`tables.nodes.filenode.save_to_filenode`) for the direct copy from
  filesystem to filenode and vice versa (closes :issue:`342`).
  Thanks to Andreas Hilboll.
- Removed the :file:`examples/nested-iter.py` considered no longer useful.
  Closes :issue:`343`.
- Better detection of the `-msse2` compiler flag.


Changes from 3.0 to 3.1.0
=========================

New features
------------

- Now PyTables is able to save/restore the default value of :class:`EnumAtom`
  types (closes :issue:`234`).
- Implemented support for the H5FD_SPLIT driver (closes :issue:`288`,
  :issue:`289` and :issue:`295`). Many thanks to simleo.
- New quantization filter: the filter truncates floating point data to a
  specified precision before writing to disk. This can significantly improve
  the performance of compressors (closes :issue:`261`).
  Thanks to Andreas Hilboll.
- Added new :meth:`VLArray.get_row_size` method to :class:`VLArray` for
  querying the number of atoms of a :class:`VLArray` row.
  Closes :issue:`24` and :issue:`315`.
- The internal Blosc_ library has been updated to version 1.3.2.
  All new features introduced in the Blosc_ 1.3.x series, and in particular
  the ability to leverage different compressors within Blosc_ (see the `Blosc
  Release Notes`_), are now available in PyTables via the blosc filter
  (closes: :issue:`324`). A big thank you to Francesc.


Improvements
------------

- The node caching mechanism has been completely redesigned to be simpler and
  less dependent from specific behaviours of the ``__del__`` method.
  Now PyTables is compatible with the forthcoming Python 3.4.
  Closes :issue:`306`.
- PyTables no longer uses shared/cached file handlers. This change somewhat
  improves support for concurrent reading allowing the user to safely open the
  same file in different threads for reading (requires HDF5 >= 1.8.7).
  More details about this change can be found in the `Backward incompatible
  changes`_ section.
  See also :issue:`130`, :issue:`129` :issue:`292` and :issue:`216`.
- PyTables is now able to detect and use external installations of the Blosc_
  library (closes :issue:`104`).  If Blosc_ is not found in the system, and the
  user do not specify a custom installation directory, then it is used an internal
  copy of the Blosc_ source code.
- Automatically disable extended float support if a buggy version of HDF5
  is detected (see also `Issues with H5T_NATIVE_LDOUBLE`_).
  See also :issue:`275`, :issue:`290` and :issue:`300`.
- Documented an unexpected behaviour with string literals in query conditions
  on Python 3 (closes :issue:`265`)
- The deprecated :mod:`getopt` module has been dropped in favour of
  :mod:`argparse` in all command line utilities (close :issue:`251`)
- Improved the installation section of the :doc:`../usersguide/index`.

  * instructions for installing PyTables via pip_ have been added.
  * added a reference to the Anaconda_, Canopy_ and `Christoph Gohlke suites`_
    (closes :issue:`291`)

- Enabled `Travis-CI`_ builds for Python_ 3.3
- :meth:`Tables.read_coordinates` now also works with boolean indices input.
  Closes :issue:`287` and :issue:`298`.
- Improved compatibility with numpy_ >= 1.8 (see :issue:`259`)
- The code of the benchmark programs (bench directory) has been updated.
  Closes :issue:`114`.
- Fixed some warning related to non-unicode file names (the Windows bytes API
  has been deprecated in Python 3.4)


Bugs fixed
----------

- Fixed detection of platforms supporting Blosc_
- Fixed a crash that occurred when one attempts to write a numpy_ array to
  an :class:`Atom` (closes :issue:`209` and :issue:`296`)
- Prevent creation of a table with no columns (closes :issue:`18` and
  :issue:`299`)
- Fixed a memory leak that occured when iterating over
  :class:`CArray`/:class:`EArray` objects (closes :issue:`308`,
  see also :issue:`309`).
  Many thanks to Alistair Muldal.
- Make NaN types sort to the end. Closes :issue:`282` and :issue:`313`
- Fixed selection on float columns when NaNs are present (closes :issue:`327`
  and :issue:`330`)
- Fix computation of the buffer size for iterations on rows.
  The buffers size was overestimated resulting in a :exc:`MemoryError`
  in some cases.
  Closes :issue:`316`. Thamks to bbudescu.
- Better check of file open mode. Closes :issue:`318`.
- The Blosc filter now works correctly together with fletcher32.
  Closes :issue:`21`.
- Close the file handle before trying to delete the corresponding file.
  Fixes a test failure on Windows.
- Use integer division for computing indices (fixes some warning on Windows)


Deprecations
------------

Following the plan for the complete transition to the new (PEP8_ compliant)
API, all calls to the old API will raise a :exc:`DeprecationWarning`.

The new API has been introduced in PyTables 3.0 and is backward incompatible.
In order to guarantee a smoother transition the old API is still usable even
if it is now deprecated.

The plan for the complete transition to the new API is outlined in
:issue:`224`.


Backward incompatible changes
-----------------------------

In PyTables <= 3.0 file handles (objects that are returned by the
:func:`open_file` function) were stored in an internal registry and re-used
when possible.

Two subsequent attempts to open the same file (with compatible open mode)
returned the same file handle in PyTables <= 3.0::

    In [1]: import tables
    In [2]: print(tables.__version__)
    3.0.0
    In [3]: a = tables.open_file('test.h5', 'a')
    In [4]: b = tables.open_file('test.h5', 'a')
    In [5]: a is b
    Out[5]: True

All this is an implementation detail, it happened under the hood and the user
had no control over the process.

This kind of behaviour was considered a feature since it can speed up opening
of files in case of repeated opens and it also avoids any potential problem
related to multiple opens, a practice that the HDF5 developers recommend to
avoid (see also H5Fopen_ reference page).

The trick, of course, is that files are not opened multiple times at HDF5
level, rather an open file is referenced several times.

The big drawback of this approach is that there are really few chances to use
PyTables safely in a multi thread program.  Several bug reports have been
filed regarding this topic.

After long discussions about the possibility to actually achieve concurrent I/O
and about patterns that should be used for the I/O in concurrent programs
PyTables developers decided to remove the *black magic under the hood* and
allow the users to implement the patterns they want.

Starting from PyTables 3.1 file handles are no more re-used (*shared*) and
each call to the :func:`open_file` function returns a new file handle::

    In [1]: import tables
    In [2]: print tables.__version__
    3.1.0
    In [3]: a = tables.open_file('test.h5', 'a')
    In [4]: b = tables.open_file('test.h5', 'a')
    In [5]: a is b
    Out[5]: False

It is important to stress that the new implementation still has an internal
registry (implementation detail) and it is still **not thread safe**.
Just now a smart enough developer should be able to use PyTables in a
muti-thread program without too much headaches.

The new implementation behaves differently from the previous one, although the
API has not been changed.  Now users should pay more attention when they open a
file multiple times (as recommended in the `HDF5 reference`__ ) and they
should take care of using them in an appropriate way.

__ H5Fopen_

Please note that the :attr:`File.open_count` property was originally intended
to keep track of the number of references to the same file handle.
In PyTables >= 3.1, despite of the name, it maintains the same semantics, just
now its value should never be higher that 1.

.. note::

    HDF5 versions lower than 1.8.7 are not fully compatible with PyTables 3.1.
    A partial support to HDF5 < 1.8.7 is still provided but in that case
    multiple file opens are not allowed at all (even in read-only mode).


.. _pip: http://www.pip-installer.org
.. _Anaconda: https://store.continuum.io/cshop/anaconda
.. _Canopy: https://www.enthought.com/products/canopy
.. _`Christoph Gohlke suites`: http://www.lfd.uci.edu/~gohlke/pythonlibs
.. _`Issues with H5T_NATIVE_LDOUBLE`: http://hdf-forum.184993.n3.nabble.com/Issues-with-H5T-NATIVE-LDOUBLE-tt4026450.html
.. _Python: http://www.python.org
.. _Blosc: http://www.blosc.org
.. _numpy: http://www.numpy.org
.. _`Travis-CI`: https://travis-ci.org
.. _PEP8: http://www.python.org/dev/peps/pep-0008
.. _`Blosc Release Notes`: https://github.com/FrancescAlted/blosc/wiki/Release-notes
.. _H5Fopen: http://www.hdfgroup.org/HDF5/doc/RM/RM_H5F.html#File-Open
