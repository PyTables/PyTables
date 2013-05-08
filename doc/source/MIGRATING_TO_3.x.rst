==================================
Migrating from PyTables 2.x to 3.x
==================================

:Author: Antonio Valentino
:Author: Anthony Scopatz

This document describes the major changes in PyTables in going from the 
2.x to 3.x series and what you need to know when migrating downstream
code bases. 

Python 3 at Last!
=================

The PyTables 3.x series now ships with full compatibility for Python 3.2+.  
Additionally, we plan on maintaining compatibility with Python 2.7 for the 
foreseeable future.  Python 2.6 is no longer under actively supported but
may work in most cases.  Note that the entire 3.x series now relies on 
numexpr v2.1+, which itself is the first version of numexpr support both 
Python 2 & 3. 

Numeric, Numarray, NetCDF3, & HDF5 1.6 No More!
===============================================

PyTables no longer supports numeric and numarray. Please use numpy instead. 
Additionally, the ``tables.netcdf3`` module has been removed. Please refer 
to the `netcdf4-python`_ project for further support. Lastly, the older 
HDF5 1.6 API is no longer supported.  Please upgrade to HDF5 1.8+.


Major API Changes
=================

The PyTables developers, `by popular demand`_, have taken this opportunity 
that a major version number upgrade affords to implement significant API 
changes.  We have tried to do this in such a way that will not immediately 
break most existing code, though in some breakages may still occur.

PEP 8 Compliance
****************
The PyTables 3.x series now follows `PEP 8`_ coding standard.  This makes 
using PyTables more idiomatic with surrounding Python code that also adheres
to this standard.  The primary way that the 2.x series was *not* PEP 8 
compliant was with respect to variable naming conventions.  Approximately 450
API variables were identified and updated for PyTables 3.x.  

To ease migration, PyTables ships with a new ``pt2to3`` command line tool.
This tool will run over a file and replace any instances of the old variable
names with the 3.x version of the name.  This tool covers the overwhelming 
majority of cases was used to transition the PyTables code base itself!  However, 
it may also accidentally also pick up variable names in 3rd party codes that 
have *exactly* the same name as a PyTables' variable.  This is because ``pt2to3``
was implemented using regular expressions rather than a fancier AST-based
method. By using regexes, ``pt2to3`` works on Python and Cython code.


``pt2to3`` **help:**

.. code-block:: bash

    usage: pt2to3 [-h] [-r] [-p] [-o OUTPUT] [-i] filename

    PyTables 2.x -> 3.x API transition tool This tool displays to standard out, so
    it is common to pipe this to another file: $ pt2to3 oldfile.py > newfile.py

    positional arguments:
      filename              path to input file.

    optional arguments:
      -h, --help            show this help message and exit
      -r, --reverse         reverts changes, going from 3.x -> 2.x.
      -p, --no-ignore-previous
                            ignores previous_api() calls.
      -o OUTPUT             output file to write to.
      -i, --inplace         overwrites the file in-place.

Note that ``pt2to3`` only works on a single file, not a a directory.  However, 
a simple BASH script may be written to run ``pt2to3`` over an entire directory 
and all sub-directories:

.. code-block:: bash

    #!/bin/bash
    for f in $(find .)
    do
        echo $f
        pt2to3 $f > temp.txt
        mv temp.txt $f
    done

The old APIs and variable names will continue to be supported for the short term,
where possible.  (The major backwards incompatible changes come from the renaming
of some function and method arguments and keyword arguments.)  Using the 2.x APIs
in the 3.x series, however, will issue warnings.  The following is the release
plan for the warning types:

* 3.0 - PendingDeprecationWarning
* 3.1 - DeprecationWarning
* >=3.2 - Remove warnings, previous_api(), and _past.py; keep pt2to3,

The current plan is to maintain the old APIs for at least 2 years, though this 
is subject to change.

Consistent ``create_xxx()`` Signatures
***************************************

Also by popular demand, it is now possible to create all data sets (``Array``, 
``CArray``, ``EArray``, ``VLArray``, and ``Table``) from existing Python objects.
Constructors for these classes now accept either of the following keyword arguments:

* an ``obj`` to initialize with data
* or both ``atom`` and ``shape`` to initialize an empty structure, if possible.

These keyword arguments are also now part of the function signature for the 
corresponding ``create_xxx()`` methods on the ``File`` class.  These would be called
as follows::

    # All create methods will support the following 
    crete_xxx(where, name, obj=obj)

    # All non-variable length arrays support the following:
    crete_xxx(where, name, atom=atom, shape=shape)

Using ``obj`` or ``atom`` and ``shape`` are mutually exclusive. Previously only 
``Array`` could be created with an existing Python object using the ``object`` 
keyword argument.  

----

  **Enjoy data!**

  -- The PyTables Developers


.. Local Variables:
.. mode: rst
.. coding: utf-8
.. fill-column: 78
.. End:


.. _by popular demand: http://sourceforge.net/mailarchive/message.php?msg_id=29584752

.. _PEP 8: http://www.python.org/dev/peps/pep-0008/

.. _netcdf4-python: http://code.google.com/p/netcdf4-python/
