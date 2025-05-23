===========================================
 PyTables: hierarchical datasets in Python
===========================================

.. image:: https://badges.gitter.im/Join%20Chat.svg
   :alt: Join the chat at https://gitter.im/PyTables/PyTables
   :target: https://gitter.im/PyTables/PyTables

.. image:: https://github.com/PyTables/PyTables/workflows/CI/badge.svg
   :target: https://github.com/PyTables/PyTables/actions?query=workflow%3ACI

.. image:: https://img.shields.io/pypi/v/tables.svg
  :target: https://pypi.org/project/tables/

.. image:: https://img.shields.io/pypi/pyversions/tables.svg
  :target: https://pypi.org/project/tables/

.. image:: https://img.shields.io/pypi/l/tables
  :target: https://github.com/PyTables/PyTables/


:URL: http://www.pytables.org/


PyTables is a package for managing hierarchical datasets, designed
to efficiently cope with extremely large amounts of data.

It is built on top of the HDF5 library and the NumPy package. It
features an object-oriented interface that, combined with C extensions
for the performance-critical parts of the code (generated using
Cython), makes it a fast, yet extremely easy to use tool for
interactively saving and retrieving very large amounts of data. One
important feature of PyTables is that it optimizes memory and disk
resources so that they take much less space (between 3 to 5 times
and more if the data is compressible) than other solutions, like for
example, relational or object-oriented databases.

State-of-the-art compression
----------------------------

PyTables supports the `Blosc compressor <http://www.blosc.org>`_ out of the box.
This allows for extremely high compression speed, while keeping decent
compression ratios. By doing so, I/O can be accelerated by a large extent, and
you may end up achieving higher performance than the bandwidth provided by your
I/O subsystem. See the
`Tuning The Chunksize section of the Optimization Tips chapter
<http://www.pytables.org/usersguide/optimization.html#fine-tuning-the-chunksize>`_
of the user documentation for some benchmarks.

Not a RDBMS replacement
-----------------------

PyTables is not designed to work as a relational database replacement,
but rather as a teammate. If you want to work with large datasets of
multidimensional data (for example, for multidimensional analysis), or
just provide a categorized structure for some portions of your
cluttered RDBS, then give PyTables a try. It works well for storing
data from data acquisition systems, simulation software, network
data monitoring systems (for example, traffic measurements of IP
packets on routers), or as a centralized repository for system logs,
to name only a few possible use cases.

Tables
------

A table is defined as a collection of records whose values are stored
in fixed-length fields. All records have the same structure, and all
values in each field have the same data type. The terms "fixed-length"
and strict "data types" seem to be a strange requirement for an
interpreted language like Python, but they serve a useful function if
the goal is to save very large quantities of data (such as
generated by many scientific applications, for example) in an
efficient manner that reduces demand on CPU time and I/O.

Arrays
------

There are other useful objects like arrays, enlargeable arrays, or
variable-length arrays that can cope with different use cases on your
project.

Easy to use
-----------

One of the principal objectives of PyTables is to be user-friendly.
In addition, many different iterators have been implemented to
make interactive work as productive as possible.

Platforms
---------

We use Linux on top of Intel32 and Intel64 boxes as the main
development platforms, but PyTables should be easy to compile/install
on other UNIX (including macOS) or Windows machines.

Compiling
---------

To compile PyTables, you will need a recent version of the HDF5
(C flavor) library, the Zlib compression library, and the NumPy and
Numexpr packages. Besides, PyTables comes with support for the Blosc, LZO,
and bzip2 compressor libraries. Blosc is mandatory, but PyTables comes
with Blosc sources so, although it is recommended to have Blosc
installed in your system, you don't absolutely need to install it
separately. LZO and bzip2 compression libraries are, however,
optional.

Make sure you have HDF5 version 1.10.5 or above. On Debian-based Linux
distributions, you can install it with::

   $ sudo apt install libhdf5-serial-dev

Installation
------------

1. Install with `pip <https://pip.pypa.io/en/stable/>`_:

       $ python3 -m pip install tables

2. To run the test suite::

       $ python3 -m tables.tests.test_all

   If there is some test that does not pass, please send us the
   complete output using the
   `GitHub Issue Tracker <https://github.com/PyTables/PyTables/issues/new>`_.


**Enjoy data!** -- The PyTables Team

.. Local Variables:
.. mode: text
.. coding: utf-8
.. fill-column: 70
.. End:
