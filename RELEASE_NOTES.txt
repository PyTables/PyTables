=======================================
 Release notes for PyTables 2.3 series
=======================================

:Author: PyTables maintainers
:Contact: pytables@googlemail.com


Changes from 2.3 to 2.3.1
=========================

- Fixed a bug that prevented to read scalar datasets of UnImplemented types
  (closes :issue:`111`). Thanks to Kamil Kisiel.

- Fixed a bug in `setup.py` that caused installation of PyTables 2.3 to fail
  on hosts with multiple python versions installed (closes :issue:`113`).
  Thanks to sbinet.


Changes from 2.2.1 to 2.3
=========================

Features coming from (now liberated) PyTables Pro
-------------------------------------------------

- OPSI is a powerful and innovative indexing engine allowing PyTables to
  perform fast queries on arbitrarily large tables. Moreover, it offers a wide
  range of optimization levels for its indexes so that the user can choose the
  best one that suits her needs (more or less size, more or less performance).
  Indexation code also takes advantage of the vectorization capabilities of the
  NumPy and Numexpr packages to ensure really short indexing and search times.

- A fine-tuned LRU cache for both metadata (nodes) and regular data that lets
  you achieve maximum speed for intensive object tree browsing during data
  reads and queries. It complements the already efficient cache present in
  HDF5, although this is more geared towards high-level structures that are
  specific to PyTables and that are critical for achieving very high
  performance.

Other changes
-------------

- Indexes with no elements are now evaluated as non-CSI ones.  Closes
  #312.

- Numexpr presence is tested now in setup.py, provided that user is not
  using setuptools (i.e. ``easy_install`` or ``pip`` tools).  When using
  setuptools, numexpr continues to be a requisite (and Cython too).
  Closes #298.

- Cython is enforced now during compilation time.  Also, it is not
  required when running tests.

- Repeatedly closing a file that has been reopened several times is
  supported now.  Closes #318.

- The number of times a file has been currently reopened is available
  now in the new `File.open_count` read-only attribute.

- The entire documentation set has been converted to sphinx (close
  :issue:`85` and :issue:`86`) that now also has an index
  (closes :issue`39`).

- The entire test suite has been updated to use unittest specific
  assertions (closes :issue:`66`).

- PyTables has been tested against the latest version of numpy (v. 1.6.1
  and 2.0dev) and Cython (v, 0.15) packages. Closes :issue:`84`.

- The setup.py script has been improved to better detect runtimes
  (closes :issue:`73`).

Deprecations
------------

Support for some old packages and related features has been deprecated
and will be removed in future versions:

- Numeric (closes :issue:`76`)
- numarray (closes :issue`76` and :issue:`75`)
- HDF5 1.6.x (closes :issue`96`)

At the API level the following are now deprecated:

- the tables.is_pro constant is deprecated because PyTables Pro
  has been released under an open source license.
- the netcdf3 sub-package (closes :issue:`67`)
- the nra sub-package


  **Enjoy data!**

  -- The PyTables Team


.. Local Variables:
.. mode: rst
.. coding: utf-8
.. fill-column: 72
.. End:
