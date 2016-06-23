:source: http://www.pytables.org/moin/FAQ
:revision: 95
:date: 2011-06-13 08:40:20
:author: FrancescAlted

.. py:currentmodule:: tables

===
FAQ
===

General questions
=================

What is PyTables?
-----------------

PyTables is a package for managing hierarchical datasets designed to
efficiently cope with extremely large amounts of data.

It is built on top of the HDF5_  library, the `Python language`_ and the
NumPy_ package.
It features an object-oriented interface that, combined with C extensions
for the performance-critical parts of the code, makes it a fast yet
extremely easy-to-use tool for interactively storing and retrieving very
large amounts of data.


What are PyTables' licensing terms?
-----------------------------------

PyTables is free for both commercial and non-commercial use, under the terms
of the BSD license.

.. todo:

    link to the BSD license http://opensource.org/licenses/BSD-3-Clause
    or to a local copy


I'm having problems. How can I get support?
-------------------------------------------

The most common and efficient way is to subscribe (remember you *need* to
subscribe prior to send messages) to the PyTables `users mailing list`_, and
send there a brief description of your issue and, if possible, a short script
that can reproduce it.
Hopefully, someone on the list will be able to help you.
It is also a good idea to check out the `archives of the user's list`_ (you may
want to check the `Gmane archives`_ instead) so as to see if the answer to your
question has already been dealed with.


Why HDF5?
---------

HDF5_ is the underlying C library and file format that enables PyTables to
efficiently deal with the data.  It has been chosen for the following reasons:

* Designed to efficiently manage very large datasets.
* Lets you organize datasets hierarchically.
* Very flexible and well tested in scientific environments.
* Good maintenance and improvement rate.
* Technical excellence (`R&D 100 Award`_).
* **It's Open Source software**


Why Python?
-----------

1. Python is interactive.

   People familiar with data processing understand how powerful command line
   interfaces are for exploring mathematical relationships and scientific data
   sets.  Python provides an interactive environment with the added benefit of
   a full featured programming language behind it.

2. Python is productive for beginners and experts alike.

   PyTables is targeted at engineers, scientists, system analysts, financial
   analysts, and others who consider programming a necessary evil.  Any time
   spent learning a language or tracking down bugs is time spent not solving
   their real problem.  Python has a short learning curve and most people can
   do real and useful work with it in a day of learning.  Its clean syntax and
   interactive nature facilitate this.

3. Python is data-handling friendly.

   Python comes with nice idioms that make the access to data much easier:
   general slicing (i.e. ``data[start:stop:step]``), list comprehensions,
   iterators, generators ... are constructs that make the interaction with your
   data very easy.


Why NumPy?
----------

NumPy_ is a Python package to efficiently deal with large datasets
**in-memory**, providing containers for homogeneous data, heterogeneous data,
and string arrays.
PyTables uses these NumPy containers as *in-memory buffers* to push the I/O
bandwith towards the platform limits.


Where can PyTables be applied?
==============================

In all the scenarios where one needs to deal with large datasets:

* Industrial applications

  - Data acquisition in real time
  - Quality control
  - Fast data processing

* Scientific applications

  - Meteorology, oceanography
  - Numerical simulations
  - Medicine (biological sensors, general data gathering & processing)

* Information systems

  - System log monitoring & consolidation
  - Tracing of routing data
  - Alert systems in security


Is PyTables safe?
-----------------

Well, first of all, let me state that PyTables does not support transactional
features yet (we don't even know if we will ever be motivated to implement
this!), so there is always the risk that you can lose your data in case of an
unexpected event while writing (like a power outage, system shutdowns ...).
Having said that, if your typical scenarios are *write once, read many*, then
the use of PyTables is perfectly safe, even for dealing extremely large amounts
of data.


Can PyTables be used in concurrent access scenarios?
----------------------------------------------------

It depends. Concurrent reads are no problem at all. However, whenever a process
(or thread) is trying to write, then problems will start to appear.  First,
PyTables doesn't support locking at any level, so several process writing
concurrently to the same PyTables file will probably end up corrupting it, so
don't do this!  Even having only one process writing and the others reading is
a hairy thing, because the reading processes might be reading incomplete data
from a concurrent data writing operation.

The solution would be to lock the file while writing and unlock it after a
flush over the file has been performed.  Also, in order to avoid cache (HDF5_,
PyTables) problems with read apps, you would need to re-open your files
whenever you are going to issue a read operation.  If a re-opening operation is
unacceptable in terms of speed, you may want to do all your I/O operations in
one single process (or thread) and communicate the results via sockets,
:class:`Queue.Queue` objects (in case of using threads), or whatever, with the
client process/thread.

The examples directory contains two scripts demonstrating methods of accessing a
PyTables file from multiple processes.

The first, *multiprocess_access_queues.py*, uses a
:class:`multiprocessing.Queue` object to transfer read and write requests from
multiple *DataProcessor* processes to a single process responsible for all
access to the PyTables file.  The results of read requests are then transferred
back to the originating processes using other :class:`Queue` objects.

The second example script, *multiprocess_access_benchmarks.py*, demonstrates
and benchmarks four methods of transferring PyTables array data between
processes.  The four methods are:

 * Using :class:`multiprocessing.Pipe` from the Python standard library.
 * Using a memory mapped file that is shared between two processes.  The NumPy
   array associated with the file is passed as the *out* argument to the
   :meth:`tables.Array.read` method.
 * Using a Unix domain socket.  Note that this example uses the 'abstract
   namespace' and will only work under Linux.
 * Using an IPv4 socket.


What kind of containers does PyTables implement?
------------------------------------------------

PyTables does support a series of data containers that address specific needs
of the user. Below is a brief description of them:

::class:`Table`:
    Lets you deal with heterogeneous datasets. Allows compression. Enlargeable.
    Supports nested types. Good performance for read/writing data.
::class:`Array`:
    Provides quick and dirty array handling. Not compression allowed.
    Not enlargeable. Can be used only with relatively small datasets (i.e.
    those that fit in memory). It provides the fastest I/O speed.
::class:`CArray`:
    Provides compressed array support. Not enlargeable. Good speed when
    reading/writing.
::class:`EArray`:
    Most general array support. Compressible and enlargeable. It is pretty
    fast at extending, and very good at reading.
::class:`VLArray`:
    Supports collections of homogeneous data with a variable number of entries.
    Compressible and enlargeable. I/O is not very fast.
::class:`Group`:
    The structural component.
    A hierarchically-addressable container for HDF5 nodes (each of these
    containers, including Group, are nodes), similar to a directory in a
    UNIX filesystem.

Please refer to the  :doc:`usersguide/libref` for more specific information.


Cool! I'd like to see some examples of use.
-------------------------------------------

Sure. Go to the HowToUse section to find simple examples that will help you
getting started.


Can you show me some screenshots?
---------------------------------

Well, PyTables is not a graphical library by itself.  However, you may want to
check out ViTables_, a GUI tool to browse and edit PyTables & HDF5_ files.


Is PyTables a replacement for a relational database?
----------------------------------------------------

No, by no means. PyTables lacks many features that are standard in most
relational databases.  In particular, it does not have support for
relationships (beyond the hierarchical one, of course) between datasets and it
does not have transactional features.  PyTables is more focused on speed and
dealing with really large datasets, than implementing the above features.  In
that sense, PyTables can be best viewed as a *teammate* of a relational
database.

For example, if you have very large tables in your existing relational
database, they will take lots of space on disk, potentially reducing the
performance of the relational engine.  In such a case, you can move those huge
tables out of your existing relational database to PyTables, and let your
relational engine do what it does best (i.e.  manage relatively small or medium
datasets with potentially complex relationships), and use PyTables for what it
has been designed for (i.e. manage large amounts of data which are loosely
related).


How can PyTables be fast if it is written in an interpreted language like Python?
---------------------------------------------------------------------------------

Actually, all of the critical I/O code in PyTables is a thin layer of code on
top of HDF5_, which is a very efficient C library. Cython_ is used as the
*glue* language to generate "wrappers" around HDF5 calls so that they can be
used in Python.  Also, the use of an efficient numerical package such as NumPy_
makes the most costly operations effectively run at C speed.  Finally,
time-critical loops are usually implemented in Cython_ (which, if used
properly, allows to generate code that runs at almost pure C speeds).


If it is designed to deal with very large datasets, then PyTables should consume a lot of memory, shouldn't it?
---------------------------------------------------------------------------------------------------------------

Well, you already know that PyTables sits on top of HDF5, Python and NumPy_,
and if we add its own logic (~7500 lines of code in Python, ~3000 in Cython and
~4000 in C), then we should conclude that PyTables isn't effectively a paradigm
of lightness.

Having said that, PyTables (as HDF5_ itself) tries very hard to optimize the
memory consumption by implementing a series of features like dynamic
determination of buffer sizes, *Least Recently Used* cache for keeping unused
nodes out of memory, and extensive use of compact NumPy_ data containers.
Moreover, PyTables is in a relatively mature state and most memory leaks have
been already addressed and fixed.

Just to give you an idea of what you can expect, a PyTables program can deal
with a table with around 30 columns and 1 million entries using as low as 13 MB
of memory (on a 32-bit platform).  All in all, it is not that much, is it?.


Why was PyTables born?
----------------------

Because, back in August 2002, one of its authors (`Francesc Alted`_) had a need
to save lots of hierarchical data in an efficient way for later post-processing
it.  After trying out several approaches, he found that they presented distinct
inconveniences.  For example, working with file sizes larger than, say, 100 MB,
was rather painful with ZODB (it took lots of memory with the version available
by that time).

The netCDF3_ interface provided by `Scientific Python`_ was great, but it did
not allow to structure the hierarchically; besides, netCDF3_ only supports
homogeneous datasets, not heterogeneous ones (i.e. tables). (As an aside,
netCDF4_ overcomes many of the limitations of netCDF3_, although curiously
enough, it is based on top of HDF5_, the library chosen as the base for
PyTables from the very beginning.)

So, he decided to give HDF5_ a try, start doing his own wrappings to it and
voilà, this is how the first public release of PyTables (0.1) saw the light in
October 2002, three months after his itch started to eat him ;-).


Does PyTables have a client-server interface?
---------------------------------------------

Not by itself, but you may be interested in using PyTables through pydap_, a
Python implementation of the OPeNDAP_ protocol.  Have a look at the `PyTables
plugin` of pydap_.


How does PyTables compare with the h5py project?
------------------------------------------------

Well, they are similar in that both packages are Python interfaces to the HDF5_
library, but there are some important differences to be noted.  h5py_ is an
attempt to map the HDF5_ feature set to NumPy_ as closely as possible.  In
addition, it also provides access to nearly all of the HDF5_ C API.

Instead, PyTables builds up an additional abstraction layer on top of HDF5_ and
NumPy_ where it implements things like an enhanced type system, an :ref:`engine
for enabling complex queries <searchOptim>`, an `efficient computational
kernel`_, `advanced indexing capabilities`_ or an undo/redo feature, to name
just a few.  This additional layer also allows PyTables to be relatively
independent of its underlying libraries (and their possible limitations).  For
example, PyTables can support HDF5_ data types like `enumerated` or `time` that
are available in the HDF5_ library but not in the NumPy_ package; or even
perform powerful complex queries that are not implemented directly in neither
HDF5_ nor NumPy_.

Furthermore, PyTables also tries hard to be a high performance interface to
HDF5/NumPy, implementing niceties like internal LRU caches for nodes and other
data and metadata, :ref:`automatic computation of optimal chunk sizes
<chunksizeFineTune>` for the datasets, a variety of compressors, ranging from
slow but efficient (bzip2_) to extremely fast ones (Blosc_) in addition to the
standard `zlib`_.  Another difference is that PyTables makes use of numexpr_ so
as to accelerate internal computations (for example, in evaluating complex
queries) to a maximum.

For contrasting with other opinions, you may want to check the PyTables/h5py
comparison in a similar entry of the `FAQ of h5py`_.


I've found a bug.  What do I do?
--------------------------------

The PyTables development team works hard to make this eventuality as rare as
possible, but, as in any software made by human beings, bugs do occur.  If you
find any bug, please tell us by file a bug report in the `issue tracker`_ on
GitHub_.


Is it possible to get involved in PyTables development?
-------------------------------------------------------

Indeed. We are keen for more people to help out contributing code, unit tests,
documentation, and helping out maintaining this wiki. Drop us a mail on the
`users mailing list` and tell us in which area do you want to work.


How can I cite PyTables?
------------------------

The recommended way to cite PyTables in a paper or a presentation is as
following:

* Author: Francesc Alted, Ivan Vilata and others
* Title: PyTables: Hierarchical Datasets in Python
* Year: 2002 -
* URL: http://www.pytables.org

Here's an example of a BibTeX entry::

    @Misc{,
      author =    {Francesc Alted and Ivan Vilata and others},
      title =     {{PyTables}: Hierarchical Datasets in {Python}},
      year =      {2002--},
      url = "http://www.pytables.org/"
    }


PyTables 2.x issues
===================

I'm having problems migrating my apps from PyTables 1.x into PyTables 2.x. Please, help!
----------------------------------------------------------------------------------------

Sure.  However, you should first check out the :doc:`MIGRATING_TO_2.x`
document.
It should provide hints to the most frequently asked questions on this regard.


For combined searches like `table.where('(x<5) & (x>3)')`, why was a `&` operator chosen instead of an `and`?
-------------------------------------------------------------------------------------------------------------

Search expressions are in fact Python expressions written as strings, and they
are evaluated as such.  This has the advantage of not having to learn a new
syntax, but it also implies some limitations with logical `and` and `or`
operators, namely that they can not be overloaded in Python.  Thus, it is
impossible right now to get an element-wise operation out of an expression like
`'array1 and array2'`.  That's why one has to choose some other operator, being
`&` and `|` the most similar to their C counterparts `&&` and `||`, which
aren't available in Python either.

You should be careful about expressions like `'x<5 & x>3'` and others like `'3
< x < 5'` which ''won't work as expected'', because of the different operator
precedence and the absence of an overloaded logical `and` operator.  More on
this in the appendix about condition syntax in the `HDF5 manual`_.

There are quite a few packages affected by those limitations including NumPy_
themselves and SQLObject_, and there have been quite longish discussions about
adding the possibility of overloading logical operators to Python (see `PEP
335`_ and `this thread`__ for more details).

__ https://mail.python.org/pipermail/python-dev/2004-September/048763.html


I can not select rows using in-kernel queries with a condition that involves an UInt64Col. Why?
-----------------------------------------------------------------------------------------------

This turns out to be a limitation of the numexpr_ package.  Internally,
numexpr_ uses a limited set of types for doing calculations, and unsigned
integers are always upcasted to the immediate signed integer that can fit the
information.  The problem here is that there is not a (standard) signed integer
that can be used to keep the information of a 64-bit unsigned integer.

So, your best bet right now is to avoid `uint64` types if you can.  If you
absolutely need `uint64`, the only way for doing selections with this is
through regular Python selections.  For example, if your table has a `colM`
column which is declared as an `UInt64Col`, then you can still filter its
values with::

    [row['colN'] for row in table if row['colM'] < X]


However, this approach will generally lead to slow speed (specially on Win32
platforms, where the values will be converted to Python `long` values).


I'm already using PyTables 2.x but I'm still getting numarray objects instead of NumPy ones!
--------------------------------------------------------------------------------------------

This is most probably due to the fact that you are using a file created with
PyTables 1.x series.  By default, PyTables 1.x was setting an HDF5 attribute
`FLAVOR` with the value `'numarray'` to all leaves.  Now, PyTables 2.x sees
this attribute and obediently converts the internal object (truly a NumPy
object) into a `numarray` one.  For PyTables 2.x files the `FLAVOR` attribute
will only be saved when explicitly set via the `leaf.flavor` property (or when
passing data to an :class:`Array` or :class:`Table` at creation time), so you
will be able to distinguish default flavors from user-set ones by checking the
existence of the `FLAVOR` attribute.

Meanwhile, if you don't want to receive `numarray` objects when reading old
files, you have several possibilities:

* Remove the flavor for your datasets by hand::

     for leaf in h5file.walkNodes(classname='Leaf'):
         del leaf.flavor

* Use the :program:'ptrepack` utility with the flag :option:`--upgrade-flavors`
  so as to convert all flavors in old files to the default (effectively by
  removing the `FLAVOR` attribute).
* Remove the `numarray` (and/or `Numeric`) package from your system.
  Then PyTables 2.x will return you pure NumPy objects (it can't be
  otherwise!).


Installation issues
===================

Windows
-------

Error when importing tables
~~~~~~~~~~~~~~~~~~~~~~~~~~~

You have installed the binary installer for Windows and, when importing the
*tables* package you are getting an error like::

    The command in "0x6714a822" refers to memory in "0x012011a0". The
    procedure "written" could not be executed.
    Click to ok to terminate.
    Click to abort to debug the program.

This problem can be due to a series of reasons, but the most probable one is
that you have a version of a DLL library that is needed by PyTables and it is
not at the correct version.  Please, double-check the versions of the required
libraries for PyTables and install newer versions, if needed. In most cases,
this solves the issue.

In case you continue getting problems, there are situations where other
programs do install libraries in the PATH that are **optional** to PyTables
(for example BZIP2 or LZO), but that they will be used if they are found in
your system (i.e. anywhere in your :envvar:`PATH`).  So, if you find any of
these libraries in your PATH, upgrade it to the latest version available (you
don't need to re-install PyTables).


Can't find LZO binaries for Windows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Unfortunately, the LZO binaries for Windows seems to be unavailable from its
usual place at http://gnuwin32.sourceforge.net/packages/lzo.htm.  So, in order
to allow people to be able to install this excellent compressor easily, we have
packaged the LZO binaries in a zip file available at:
http://www.pytables.org/download/lzo-win.  This zip file follows the same
structure that a typical GnuWin32_ package, so it is just a matter of unpacking
it in your ``GNUWIN32`` directory and following the :ref:`instructions
<prerequisitesBinInst>` in the `PyTables Manual`_.

Hopefully somebody else will take care again of maintaining LZO for Windows
again.


Testing issues
==============

Tests fail when running from IPython
------------------------------------

You may be getting errors related with Doctest when running the test suite from
IPython.  This is a known limitation in IPython (see
http://lists.ipython.scipy.org/pipermail/ipython-dev/2007-April/002859.html).
Try running the test suite from the vanilla Python interpreter instead.


Tests fail when running from Python 2.5 and Numeric is installed
----------------------------------------------------------------

`Numeric` doesn't get well with Python 2.5, even on 32-bit platforms.  This is
a consequence of `Numeric` not being maintained anymore and you should consider
migrating to NumPy as soon as possible.  To get rid of these errors, just
uninstall `Numeric`.


-----


.. target-notes::

.. _HDF5: http://www.hdfgroup.org/HDF5
.. _`Python language`: http://www.python.org
.. _NumPy: http://www.numpy.org
.. _`users mailing list`: https://groups.google.com/group/pytables-users
.. _`archives of the user's list`: http://sourceforge.net/mailarchive/forum.php?forum_id=13760
.. _`Gmane archives`: http://www.mail-archive.com/pytables-users@lists.sourceforge.net/
.. _`R&D 100 Award`: http://www.hdfgroup.org/HDF5/RD100-2002/
.. _ViTables: http://vitables.org
.. _Cython: http://www.cython.org
.. _`Francesc Alted`: http://www.pytables.org/moin/FrancescAlted
.. _netCDF3: http://www.unidata.ucar.edu/software/netcdf
.. _`Scientific Python`: http://dirac.cnrs-orleans.fr/plone/software/scientificpython
.. _netCDF4: http://www.unidata.ucar.edu/software/netcdf
.. _pydap: http://www.pydap.org
.. _OPeNDAP: http://opendap.org
.. _`PyTables plugin`: http://pydap.org/plugins/hdf5.html
.. _`PyTables Manual`: http://www.pytables.org/docs/manual
.. _h5py: http://www.h5py.org
.. _`efficient computational kernel`: http://www.pytables.org/moin/ComputingKernel
.. _`advanced indexing capabilities`: http://www.pytables.org/moin/PyTablesPro
.. _`automatic computation of optimal chunk sizes`: http://www.pytables.org/docs/manual/ch05.html#chunksizeFineTune
.. _bzip2: http://www.bzip.org
.. _Blosc: http://blosc.pytables.org
.. _`zlib`: http://zlib.net
.. _numexpr: https://github.com/pydata/numexpr
.. _`FAQ of h5py`: http://docs.h5py.org/en/latest/faq.html#what-s-the-difference-between-h5py-and-pytables
.. _`issue tracker`: https://github.com/PyTables/PyTables/issues
.. _GitHub: https://github.com
.. _`HDF5 manual`: http://www.hdfgroup.org/HDF5/doc/RM/RM_H5T.html
.. _SQLObject: http://sqlobject.org
.. _`PEP 335`: http://www.python.org/dev/peps/pep-0335
.. _GnuWin32: http://gnuwin32.sourceforge.net


.. todo:: fix links that point to wiki pages

