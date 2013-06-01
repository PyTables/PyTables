=======================================
 Release notes for PyTables 2.4 series
=======================================

:Author: PyTables maintainers
:Contact: pytables@googlemail.com

.. py:currentmodule:: tables


Changes from 2.3.1 to 2.4
=========================

New features
------------

- Improved HDF5 error logging management:

  * added a new function, :func:`silenceHDF5Messages`, for suppressing
    (and re-enabling) HDF5 messages.  By default HDF5 error logging is now
    suppressed. Closes :issue:`87`.
  * now all HDF5 error messages and trace-backs are trapped and attached to
    the :exc:`exceptions.HDF5ExtError` exception instances.
    Closes :issue:`120`.

- Added support for the float16 data type.  It is only available if numpy_
  provides it as well (i.e. numpy_ >= 1.6).  See :issue:`51`.

- Leaf nodes now have attributes for retrieving the size of data in memory
  and on disk.  Data on disk can be compressed, so the new attributes make it
  easy to compute the data compression ration.
  Thanks to Josh Ayers (close :issue:`141`).

- The maximum number of threads for Blosc_ and Numexpr_ is now handled using
  the :data:`parameters.MAX_BLOSC_THREADS` and
  :data:`parameters.MAX_NUMEXPR_THREADS` parameters respectively.
  This allows a more fine grained configuration capability.
  Closes :issue:`142`.

- `ndim` (read-only) attribute added to :class:`Leaf`, :class:`Atom` and
  :class:`Col` objects (closes :issue:`126`).

- Added read support for variable length string attributes (non scalar
  attributes are converted into numpy_ arrays with 'O8' type).
  See :issue:`54`.


Other improvements
------------------

- Dropped support for HDF5 1.6.x. Now PyTables uses the HDF5 1.8 API
  (closes :issue:`105`).

- Blosc_ updated to v. 1.1.3.

- The Blosc_ compression library is now automatically disabled on platforms
  that do not support unaligned memory access (see also
  https://github.com/FrancescAlted/blosc/issues/3 and
  http://bugs.debian.org/cgi-bin/bugreport.cgi?bug=661286).

- Improved bzip2 detection on Windows (:issue:`116`).  Thanks to cgohlke.

- For Windows, the setup.py script now has the ability to automatically find
  the HDF5_DIR in the system PATH.  Thanks to Mark (mwiebe).

- Improved multi-arch support in GNU/Linux platforms (closes :issue:`124`)
  Thanks to Julian Taylor and Picca Frederic-Emmanuel.

- Use new style syntax for exception raising. Closes :issue:`93`.

- Fixed most of the warnings related to py3k compatibility (see :issue:`92`).

- Fixed pyflakes_ warnings (closes :issue:`102`).

- Cython_ extensions updated to use new constructs (closes :issue:`100`).

- Reduced the number of build warnings (closes :issue:`101`).

- Removed the old lrucache module. It is no more needed after the merge with
  PyTables Pro (closes :issue:`118`).

- Added explicit (import time) testing for hdf5dll.dll on Windows to improve
  diagnostics (closes :issue:`146`).  Thanks to Mark (mwiebe).


Documentation improvements
--------------------------

- new coockbook section (contents have been coming from the PyTables wiki
  on http://www.pytables.org)

- complete rework of the library reference.  Now the entire chapter is
  generated from docstrings using the sphinx autodoc extension.
  A big thank you to Josh Ayers.  Closes :issue:`148`.

- new sphinx theme based on the cloud template


Bugs fixed
----------

- Fixed a segfault on platforms that do not support unaligned memory access
  (closes: :issue:`134`).  Thanks to Julian Taylor.

- Fixed broken inheritance in :class:`IsDescription` classes (thanks to
  Andrea Bedini).  Closes :issue:`65`.

- Fixed table descriptions copy method (closes :issue:`131`).

- Fixed open failures handling (closes :issue:`158`).
  Errors that happen when one tries to open an invalid HDF5 file (e.g. an
  empty file) are now detected earlier by PyTables and a proper exception
  (:exc:`exceptions.HDF5ExtError`) is raised.
  Also, in case of open failures, invalid file descriptors are no more cached.
  Before is fix it was not possible to completely close the bad file and reopen
  the same path, even if a valid file was created in the meanwhile.
  Thanks to Daniele for reporting and for the useful test code.

- Fixed support to rich structured  numpy.dtype in
  :func:`description.descr_from_dtype`.   Closes :issue:`160`.

- Fixed sorting of nested tables that caused AttributeError.
  Closes :issue:`156` and :issue:`157`.  Thanks to Uwe Mayer.

- Fixed flavor deregistration (closes :issue:`163`)


Deprecations
------------

- The :data:`parameters.MAX_THREADS` configuration parameter is now
  deprecated.  Please use :data:`parameters.MAX_BLOSC_THREADS` and
  :data:`parameters.MAX_NUMEXPR_THREADS` instead.
  See :issue:`142`.

- Since the support for HDF5 1.6.x has been dropped, the *warn16incompat*
  argument of the :meth:`File.createExternalLink` method and the
  :exc:`exceptions.Incompat16Warning` exception class are now deprecated.

.. _pyflakes: https://launchpad.net/pyflakes
.. _numpy: http://www.numpy.org
.. _Blosc: https://github.com/FrancescAlted/blosc
.. _Numexpr: http://code.google.com/p/numexpr
.. _Cython: http://www.cython.org


  **Enjoy data!**

  -- The PyTables Team


.. Local Variables:
.. mode: rst
.. coding: utf-8
.. fill-column: 72
.. End:
