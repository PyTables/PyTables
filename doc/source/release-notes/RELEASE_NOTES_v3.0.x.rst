=======================================
 Release notes for PyTables 3.0 series
=======================================

:Author: PyTables Developers
:Contact: pytables@googlemail.com

.. py:currentmodule:: tables


Changes from 2.4 to 3.0
=======================

New features
------------

- Since this release PyTables provides full support to Python_ 3
  (closes :issue:`188`).

- The entire code base is now more compliant with coding style guidelines
  describe in the PEP8_ (closes :issue:`103` and :issue:`224`).
  See `API changes`_ for more details.

- Basic support for HDF5 drivers.  Now it is possible to open/create an
  HDF5 file using one of the SEC2, DIRECT, LOG, WINDOWS, STDIO or CORE
  drivers.  Users can also set the main driver parameters (closes
  :issue:`166`).
  Thanks to Michal Slonina.

- Basic support for in-memory image files.  An HDF5 file can be set from or
  copied into a memory buffer (thanks to Michal Slonina).  This feature is
  only available if PyTables is built against HDF5 1.8.9 or newer.
  Closes :issue:`165` and :issue:`173`.

- New :meth:`File.get_filesize` method for retrieving the HDF5 file size.

- Implemented methods to get/set the user block size in a HDF5 file
  (closes :issue:`123`)

- Improved support for PyInstaller_.  Now it is easier to pack frozen
  applications that use the PyTables package (closes: :issue:`177`).
  Thanks to Stuart Mentzer and Christoph Gohlke.

- All read methods now have an optional *out* argument that allows to pass a
  pre-allocated array to store data (closes :issue:`192`)

- Added support for the floating point data types with extended precision
  (Float96, Float128, Complex192 and Complex256).  This feature is only
  available if numpy_ provides it as well.
  Closes :issue:`51` and :issue:`214`.  Many thanks to Andrea Bedini.

- Consistent ``create_xxx()`` signatures.  Now it is possible to create all
  data sets :class:`Array`, :class:`CArray`, :class:`EArray`,
  :class:`VLArray`, and :class:`Table` from existing Python objects (closes
  :issue:`61` and :issue:`249`).  See also the `API changes`_ section.

- Complete rewrite of the :mod:`nodes.filenode` module. Now it is fully
  compliant with the interfaces defined in the standard :mod:`io` module.
  Only non-buffered binary I/O is supported currently.
  See also the `API changes`_ section.  Closes :issue:`244`.

- New :program:`pt2to3` tool is provided to help users to port their
  applications to the new API (see `API changes`_ section).


Improvements
------------

- Improved runtime checks on dynamic loading of libraries: meaningful error
  messages are generated in case of failure.
  Also, now PyTables no more alters the system PATH.
  Closes :issue:`178` and :issue:`179` (thanks to Christoph Gohlke).

- Improved list of search paths for libraries as suggested by Nicholaus
  Halecky (see :issue:`219`).

- Removed deprecated Cython_ include (.pxi) files. Contents of
  :file:`convtypetables.pxi` have been moved in :file:`utilsextension.pyx`.
  Closes :issue:`217`.

- The internal Blosc_ library has been upgraded to version 1.2.3.

- Pre-load the bzip2_ library on windows (closes :issue:`205`)

- The :meth:`File.get_node` method now accepts unicode paths
  (closes :issue:`203`)

- Improved compatibility with Cython_ 0.19 (see :issue:`220` and
  :issue:`221`)

- Improved compatibility with numexpr_ 2.1 (see also :issue:`199` and
  :issue:`241`)

- Improved compatibility with development versions of numpy_
  (see :issue:`193`)

- Packaging: since this release the standard tar-ball package no more includes
  the PDF version of the "PyTables User Guide", so it is a little bit smaller
  now.  The complete and pre-build version of the documentation both in HTML
  and PDF format is available on the file `download area`_ on SourceForge.net.
  Closes: :issue:`172`.

- Now PyTables also uses `Travis-CI`_ as continuous integration service.
  All branches and all pull requests are automatically tested with different
  Python_ versions.  Closes :issue:`212`.


Other changes
-------------

- PyTables now requires Python 2.6 or newer.

- Minimum supported version of Numexpr_ is now 2.0.


API changes
-----------

The entire PyTables API as been made more PEP8_ compliant (see :issue:`224`).

This means that many methods, attributes, module global variables and also
keyword parameters have been renamed to be compliant with PEP8_ style
guidelines (e.g. the ``tables.hdf5Version`` constant has been renamed into
``tables.hdf5_version``).

We made the best effort to maintain compatibility to the old API for existing
applications.  In most cases, the old 2.x API is still available and usable
even if it is now deprecated (see the Deprecations_ section).

The only important backwards incompatible API changes are for names of
function/methods arguments.  All uses of keyword arguments should be
checked and fixed to use the new naming convention.

The new :program:`pt2to3` tool can be used to port PyTables based applications
to the new API.

Many deprecated features and support for obsolete modules has been dropped:

- The deprecated :data:`is_pro` module constant has been removed

- The nra module and support for the obsolete numarray module has been removed.
  The *numarray* flavor is no more supported as well (closes :issue:`107`).

- Support for the obsolete Numeric module has been removed.
  The *numeric* flavor is no longer available (closes :issue:`108`).

- The tables.netcdf3 module has been removed (closes :issue:`68`).

- The deprecated :exc:`exceptions.Incompat16Warning` exception has been
  removed

- The :meth:`File.create_external_link` method no longer has a keyword
  parameter named *warn16incompat*.  It was deprecated in PyTables 2.4.

Moreover:

- The :meth:`File.create_array`, :meth:`File.create_carray`,
  :meth:`File.create_earray`, :meth:`File.create_vlarray`, and
  :meth:`File.create_table` methods of the :class:`File` objects gained a
  new (optional) keyword argument named ``obj``.  It can be used to initialize
  the newly created dataset with an existing Python object, though normally
  these are numpy_ arrays.

  The *atom*/*descriptor* and *shape* parameters are now optional if the
  *obj* argument is provided.

- The :mod:`nodes.filenode` has been completely rewritten to be fully
  compliant with the interfaces defined in the :mod:`io` module.

  The FileNode classes currently implemented are intended for binary I/O.

  Main changes:

  * the FileNode base class is no more available,
  * the new version of :class:`nodes.filenode.ROFileNode` and
    :class:`nodes.filenode.RAFileNode` objects no more expose the *offset*
    attribute (the *seek* and *tell* methods can be used instead),
  * the *lineSeparator* property is no more available and the ``\n``
    character is always used as line separator.

- The `__version__` module constants has been removed from almost all the
  modules (it was not used after the switch to Git).  Of course the package
  level constant (:data:`tables.__version__`) still remains.
  Closes :issue:`112`.

- The :func:`lrange` has been dropped in favor of xrange (:issue:`181`)

- The :data:`parameters.MAX_THREADS` configuration parameter has been dropped
  in favor of :data:`parameters.MAX_BLOSC_THREADS` and
  :data:`parameters.MAX_NUMEXPR_THREADS` (closes :issue:`147`).

- The :func:`conditions.compile_condition` function no more has a *copycols*
  argument, it was no more necessary since Numexpr_ 1.3.1.
  Closes :issue:`117`.

- The *expectedsizeinMB* parameter of the :meth:`File.create_vlarray` and of
  the :meth:`VLArrsy.__init__` methods has been replaced by *expectedrows*.
  See also (:issue:`35`).

- The :meth:`Table.whereAppend` method has been renamed into
  :meth:`Table.append_where` (closes :issue:`248`).

Please refer to the :doc:`../MIGRATING_TO_3.x` document for more details about
API changes and for some useful hint about the migration process from the 2.X
API to the new one.


Other possibly incompatible changes
-----------------------------------

- All methods of the :class:`Table` class that take *start*, *stop* and
  *step* parameters (including :meth:`Table.read`, :meth:`Table.where`,
  :meth:`Table.iterrows`, etc) have been redesigned to have a consistent
  behaviour.  The meaning of the *start*, *stop* and *step* and their default
  values now always work exactly like in the standard :class:`slice` objects.
  Closes :issue:`44` and :issue:`255`.

- Unicode attributes are not stored in the HDF5 file as pickled string.
  They are now saved on the HDF5 file as UTF-8 encoded strings.

  Although this does not introduce any API breakage, files produced are
  different (for unicode attributes) from the ones produced by earlier
  versions of PyTables.

- System attributes are now stored in the HDF5 file using the character set
  that reflects the native string behaviour: ASCII for Python 2 and UTF8 for
  Python 3.  In any case, system attributes are represented as Python string.

- The :meth:`iterrows` method of :class:`*Array` and :class:`Table` as well
  as the :meth:`Table.itersorted` now behave like functions in the standard
  :mod:`itertools` module.
  If the *start* parameter is provided and *stop* is None then the
  array/table is iterated from *start* to the last line.
  In PyTables < 3.0 only one element was returned.


Deprecations
------------

- As described in `API changes`_, all functions, methods and attribute names
  that was not compliant with the PEP8_ guidelines have been changed.
  Old names are still available but they are deprecated.

- The use of upper-case keyword arguments in the :func:`open_file` function
  and the :class:`File` class initializer is now deprecated.  All parameters
  defined in the :file:`tables/parameters.py` module can still be passed as
  keyword argument to the :func:`open_file` function just using a lower-case
  version of the parameter name.


Bugs fixed
----------

- Better check access on closed files (closes :issue:`62`)

- Fix for :meth:`File.renameNode` where in certain cases
  :meth:`File._g_updateLocation` was wrongly called (closes :issue:`208`).
  Thanks to Michka Popoff.

- Fixed ptdump failure on data with nested columns (closes :issue:`213`).
  Thanks to Alexander Ford.

- Fixed an error in :func:`open_file` when *filename* is a :class:`numpy.str_`
  (closes :issue:`204`)

- Fixed :issue:`119`, :issue:`230` and :issue:`232`, where an index on
  :class:`Time64Col` (only, :class:`Time32Col` was ok) hides the data on
  selection from a Tables. Thanks to Jeff Reback.

- Fixed ``tables.tests.test_nestedtypes.ColsTestCase.test_00a_repr`` test
  method.  Now the ``repr`` of of cols on big-endian platforms is correctly
  handled  (closes :issue:`237`).

- Fixes bug with completely sorted indexes where *nrowsinbuf* must be equal
  to or greater than the *chunksize* (thanks to Thadeus Burgess).
  Closes :issue:`206` and :issue:`238`.

- Fixed an issue of the :meth:`Table.itersorted` with reverse iteration
  (closes :issue:`252` and :issue:`253`).


.. _Python: http://www.python.org
.. _PEP8: http://www.python.org/dev/peps/pep-0008
.. _PyInstaller: http://www.pyinstaller.org
.. _Blosc: https://github.com/FrancescAlted/blosc
.. _bzip2: http://www.bzip.org
.. _Cython: http://www.cython.org
.. _Numexpr: http://code.google.com/p/numexpr
.. _numpy: http://www.numpy.org
.. _`download area`: http://sourceforge.net/projects/pytables/files/pytables
.. _`Travis-CI`: https://travis-ci.org


  **Enjoy data!**

  -- The PyTables Developers


.. Local Variables:
.. mode: rst
.. coding: utf-8
.. fill-column: 72
.. End:
