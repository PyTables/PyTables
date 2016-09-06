Utilities
=========
PyTables comes with a couple of utilities that make the life easier to the
user. One is called ptdump and lets you see the contents of a PyTables file
(or generic HDF5 file, if supported). The other one is named ptrepack that
allows to (recursively) copy sub-hierarchies of objects present in a file
into another one, changing, if desired, some of the filters applied to the
leaves during the copy process.

Normally, these utilities will be installed somewhere in your PATH during the
process of installation of the PyTables package, so that you can invoke them
from any place in your file system after the installation has successfully
finished.


ptdump
------
As has been said before, ptdump utility allows you look into the contents of
your PyTables files. It lets you see not only the data but also the metadata
(that is, the *structure* and additional information in the form of
*attributes*).

Usage
~~~~~
For instructions on how to use it, just pass the -h flag to the command:

.. code-block:: bash

    $ ptdump -h

to see the message usage:

.. code-block:: bash

    usage: ptdump [-h] [-v] [-d] [-a] [-s] [-c] [-i] [-R RANGE]
                  filename[:nodepath]

    The ptdump utility allows you look into the contents of your PyTables files.
    It lets you see not only the data but also the metadata (that is, the
    *structure* and additional information in the form of *attributes*).

    positional arguments:
      filename[:nodepath]   name of the HDF5 file to dump

    optional arguments:
      -h, --help            show this help message and exit
      -v, --verbose         dump more metainformation on nodes
      -d, --dump            dump data information on leaves
      -a, --showattrs       show attributes in nodes (only useful when -v or -d
                            are active)
      -s, --sort            sort output by node name
      -c, --colinfo         show info of columns in tables (only useful when -v or
                            -d are active)
      -i, --idxinfo         show info of indexed columns (only useful when -v or
                            -d are active)
      -R RANGE, --range RANGE
                            select a RANGE of rows (in the form "start,stop,step")
                            during the copy of *all* the leaves. Default values
                            are "None,None,1", which means a copy of all the rows.

Read on for a brief introduction to this utility.


A small tutorial on ptdump
~~~~~~~~~~~~~~~~~~~~~~~~~~
Let's suppose that we want to know only the *structure* of a file. In order
to do that, just don't pass any flag, just the file as parameter.

.. code-block:: bash

    $ ptdump vlarray1.h5
    / (RootGroup) ''
    /vlarray1 (VLArray(3,), shuffle, zlib(1)) 'ragged array of ints'
    /vlarray2 (VLArray(3,), shuffle, zlib(1)) 'ragged array of strings'

we can see that the file contains just a leaf object called vlarray1, that is
an instance of VLArray, has 4 rows, and two filters has been used in order to
create it: shuffle and zlib (with a compression level of 1).

Let's say we want more meta-information. Just add the -v (verbose) flag:

.. code-block:: bash

    $ ptdump -v vlarray1.h5
    / (RootGroup) ''
    /vlarray1 (VLArray(3,), shuffle, zlib(1)) 'ragged array of ints'
      atom = Int32Atom(shape=(), dflt=0)
      byteorder = 'little'
      nrows = 3
      flavor = 'numpy'
    /vlarray2 (VLArray(3,), shuffle, zlib(1)) 'ragged array of strings'
      atom = StringAtom(itemsize=2, shape=(), dflt='')
      byteorder = 'irrelevant'
      nrows = 3
      flavor = 'python'

so we can see more info about the atoms that are the components of the
vlarray1 dataset, i.e. they are scalars of type Int32 and with NumPy
*flavor*.

If we want information about the attributes on the nodes, we must add the -a
flag:

.. code-block:: bash

    $ ptdump -va vlarray1.h5
    / (RootGroup) ''
      /._v_attrs (AttributeSet), 4 attributes:
       [CLASS := 'GROUP',
        PYTABLES_FORMAT_VERSION := '2.0',
        TITLE := '',
        VERSION := '1.0']
    /vlarray1 (VLArray(3,), shuffle, zlib(1)) 'ragged array of ints'
      atom = Int32Atom(shape=(), dflt=0)
      byteorder = 'little'
      nrows = 3
      flavor = 'numpy'
      /vlarray1._v_attrs (AttributeSet), 3 attributes:
       [CLASS := 'VLARRAY',
        TITLE := 'ragged array of ints',
        VERSION := '1.3']
    /vlarray2 (VLArray(3,), shuffle, zlib(1)) 'ragged array of strings'
      atom = StringAtom(itemsize=2, shape=(), dflt='')
      byteorder = 'irrelevant'
      nrows = 3
      flavor = 'python'
      /vlarray2._v_attrs (AttributeSet), 4 attributes:
       [CLASS := 'VLARRAY',
        FLAVOR := 'python',
        TITLE := 'ragged array of strings',
        VERSION := '1.3']


Let's have a look at the real data:

.. code-block:: bash

    $ ptdump -d vlarray1.h5
    / (RootGroup) ''
    /vlarray1 (VLArray(3,), shuffle, zlib(1)) 'ragged array of ints'
      Data dump:
    [0] [5 6]
    [1] [5 6 7]
    [2] [5 6 9 8]
    /vlarray2 (VLArray(3,), shuffle, zlib(1)) 'ragged array of strings'
      Data dump:
    [0] ['5', '66']
    [1] ['5', '6', '77']
    [2] ['5', '6', '9', '88']

We see here a data dump of the 4 rows in vlarray1 object, in the form of a
list. Because the object is a VLA, we see a different number of integers on
each row.

Say that we are interested only on a specific *row range* of the /vlarray1
object:

.. code-block:: bash

    ptdump -R2,3 -d vlarray1.h5:/vlarray1
    /vlarray1 (VLArray(3,), shuffle, zlib(1)) 'ragged array of ints'
      Data dump:
    [2] [5 6 9 8]

Here, we have specified the range of rows between 2 and 4 (the upper limit
excluded, as usual in Python). See how we have selected only the /vlarray1
object for doing the dump (vlarray1.h5:/vlarray1).

Finally, you can mix several information at once:

.. code-block:: bash

    $ ptdump -R2,3 -vad vlarray1.h5:/vlarray1
    /vlarray1 (VLArray(3,), shuffle, zlib(1)) 'ragged array of ints'
      atom = Int32Atom(shape=(), dflt=0)
      byteorder = 'little'
      nrows = 3
      flavor = 'numpy'
      /vlarray1._v_attrs (AttributeSet), 3 attributes:
       [CLASS := 'VLARRAY',
        TITLE := 'ragged array of ints',
        VERSION := '1.3']
      Data dump:
    [2] [5 6 9 8]


.. _ptrepackDescr:

ptrepack
--------
This utility is a very powerful one and lets you copy any leaf, group or
complete subtree into another file. During the copy process you are allowed
to change the filter properties if you want so. Also, in the case of
duplicated pathnames, you can decide if you want to overwrite already
existing nodes on the destination file. Generally speaking, ptrepack can be
useful in may situations, like replicating a subtree in another file, change
the filters in objects and see how affect this to the compression degree or
I/O performance, consolidating specific data in repositories or even
*importing* generic HDF5 files and create true PyTables counterparts.


Usage
~~~~~
For instructions on how to use it, just pass the -h flag to the command:

.. code-block:: bash

    $ ptrepack -h

to see the message usage:

.. code-block:: bash

    usage: ptrepack [-h] [-v] [-o] [-R RANGE] [--non-recursive]
                    [--dest-title TITLE] [--dont-create-sysattrs]
                    [--dont-copy-userattrs] [--overwrite-nodes]
                    [--complevel COMPLEVEL]
                    [--complib {zlib,lzo,bzip2,blosc,blosc:blosclz,blosc:lz4,blosc:lz4hc,blosc:snappy,blosc:zlib,blosc:zstd}]
                    [--shuffle {0,1}] [--bitshuffle {0,1}] [--fletcher32 {0,1}]
                    [--keep-source-filters] [--chunkshape CHUNKSHAPE]
                    [--upgrade-flavors] [--dont-regenerate-old-indexes]
                    [--sortby COLUMN] [--checkCSI] [--propindexes]
                    sourcefile:sourcegroup destfile:destgroup

    This utility is very powerful and lets you copy any leaf, group or complete
    subtree into another file. During the copy process you are allowed to change
    the filter properties if you want so. Also, in the case of duplicated
    pathnames, you can decide if you want to overwrite already existing nodes on
    the destination file. Generally speaking, ptrepack can be useful in may
    situations, like replicating a subtree in another file, change the filters in
    objects and see how affect this to the compression degree or I/O performance,
    consolidating specific data in repositories or even *importing* generic HDF5
    files and create true PyTables counterparts.

    positional arguments:
      sourcefile:sourcegroup
                            source file/group
      destfile:destgroup    destination file/group

    optional arguments:
      -h, --help            show this help message and exit
      -v, --verbose         show verbose information
      -o, --overwrite       overwrite destination file
      -R RANGE, --range RANGE
                            select a RANGE of rows (in the form "start,stop,step")
                            during the copy of *all* the leaves. Default values
                            are "None,None,1", which means a copy of all the rows.
      --non-recursive       do not do a recursive copy. Default is to do it
      --dest-title TITLE    title for the new file (if not specified, the source
                            is copied)
      --dont-create-sysattrs
                            do not create sys attrs (default is to do it)
      --dont-copy-userattrs
                            do not copy the user attrs (default is to do it)
      --overwrite-nodes     overwrite destination nodes if they exist. Default is
                            to not overwrite them
      --complevel COMPLEVEL
                            set a compression level (0 for no compression, which
                            is the default)
      --complib {zlib,lzo,bzip2,blosc,blosc:blosclz,blosc:lz4,blosc:lz4hc,blosc:snappy,blosc:zlib,blosc:zstd}
                            set the compression library to be used during the
                            copy. Defaults to zlib
      --shuffle {0,1}       activate or not the shuffle filter (default is active
                            if complevel > 0)
      --bitshuffle {0,1}    activate or not the bitshuffle filter (not active by
                            default)
      --fletcher32 {0,1}    whether to activate or not the fletcher32 filter (not
                            active by default)
      --keep-source-filters
                            use the original filters in source files. The default
                            is not doing that if any of --complevel, --complib,
                            --shuffle --bitshuffle or --fletcher32 option is
                            specified
      --chunkshape CHUNKSHAPE
                            set a chunkshape. Possible options are: "keep" |
                            "auto" | int | tuple. A value of "auto" computes a
                            sensible value for the chunkshape of the leaves
                            copied. The default is to "keep" the original value
      --upgrade-flavors     when repacking PyTables 1.x or PyTables 2.x files, the
                            flavor of leaves will be unset. With this, such a
                            leaves will be serialized as objects with the internal
                            flavor ('numpy' for 3.x series)
      --dont-regenerate-old-indexes
                            disable regenerating old indexes. The default is to
                            regenerate old indexes as they are found
      --sortby COLUMN       do a table copy sorted by the index in "column". For
                            reversing the order, use a negative value in the
                            "step" part of "RANGE" (see "-r" flag). Only applies
                            to table objects
      --checkCSI            Force the check for a CSI index for the --sortby
                            column
      --propindexes         propagate the indexes existing in original tables. The
                            default is to not propagate them. Only applies to
                            table objects

Read on for a brief introduction to this utility.

A small tutorial on ptrepack
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Imagine that we have ended the tutorial 1 (see the output of
examples/tutorial1-1.py), and we want to copy our reduced data (i.e. those
datasets that hangs from the /column group) to another file. First, let's
remember the content of the examples/tutorial1.h5:

.. code-block:: bash

    $ ptdump tutorial1.h5
    / (RootGroup) 'Test file'
    /columns (Group) 'Pressure and Name'
    /columns/name (Array(3,)) 'Name column selection'
    /columns/pressure (Array(3,)) 'Pressure column selection'
    /detector (Group) 'Detector information'
    /detector/readout (Table(10,)) 'Readout example'

Now, copy the /columns to other non-existing file. That's easy:

.. code-block:: bash

    $ ptrepack tutorial1.h5:/columns reduced.h5

That's all. Let's see the contents of the newly created reduced.h5 file:

.. code-block:: bash

    $ ptdump reduced.h5
    / (RootGroup) ''
    /name (Array(3,)) 'Name column selection'
    /pressure (Array(3,)) 'Pressure column selection'

so, you have copied the children of /columns group into the *root* of the
reduced.h5 file.

Now, you suddenly realized that what you intended to do was to copy all the
hierarchy, the group /columns itself included. You can do that by just
specifying the destination group:

.. code-block:: bash

    $ ptrepack tutorial1.h5:/columns reduced.h5:/columns
    $ ptdump reduced.h5
    / (RootGroup) ''
    /name (Array(3,)) 'Name column selection'
    /pressure (Array(3,)) 'Pressure column selection'
    /columns (Group) ''
    /columns/name (Array(3,)) 'Name column selection'
    /columns/pressure (Array(3,)) 'Pressure column selection'

OK. Much better. But you want to get rid of the existing nodes on the new
file. You can achieve this by adding the -o flag:

.. code-block:: bash

    $ ptrepack -o tutorial1.h5:/columns reduced.h5:/columns
    $ ptdump reduced.h5
    / (RootGroup) ''
    /columns (Group) ''
    /columns/name (Array(3,)) 'Name column selection'
    /columns/pressure (Array(3,)) 'Pressure column selection'

where you can see how the old contents of the reduced.h5 file has been
overwritten.

You can copy just one single node in the repacking operation and change its
name in destination:

.. code-block:: bash

    $ ptrepack tutorial1.h5:/detector/readout reduced.h5:/rawdata
    $ ptdump reduced.h5
    / (RootGroup) ''
    /rawdata (Table(10,)) 'Readout example'
    /columns (Group) ''
    /columns/name (Array(3,)) 'Name column selection'
    /columns/pressure (Array(3,)) 'Pressure column selection'

where the /detector/readout has been copied to /rawdata in destination.

We can change the filter properties as well:

.. code-block:: bash

    $ ptrepack --complevel=1 tutorial1.h5:/detector/readout reduced.h5:/rawdata
    Problems doing the copy from 'tutorial1.h5:/detector/readout' to 'reduced.h5:/rawdata'
    The error was --> tables.exceptions.NodeError: destination group \``/\`` already has a node named \``rawdata``; you may want to use the \``overwrite`` argument
    The destination file looks like:
    / (RootGroup) ''
    /rawdata (Table(10,)) 'Readout example'
    /columns (Group) ''
    /columns/name (Array(3,)) 'Name column selection'
    /columns/pressure (Array(3,)) 'Pressure column selection'
    Traceback (most recent call last):
      File "utils/ptrepack", line 3, in ?
        main()
      File ".../tables/scripts/ptrepack.py", line 349, in main
        stats = stats, start = start, stop = stop, step = step)
      File ".../tables/scripts/ptrepack.py", line 107, in copy_leaf
        raise RuntimeError, "Please check that the node names are not
        duplicated in destination, and if so, add the --overwrite-nodes flag
        if desired."
    RuntimeError: Please check that the node names are not duplicated in
    destination, and if so, add the --overwrite-nodes flag if desired.

Ooops! We ran into problems: we forgot that the /rawdata pathname already
existed in destination file. Let's add the --overwrite-nodes, as the verbose
error suggested:

.. code-block:: bash

    $ ptrepack --overwrite-nodes --complevel=1 tutorial1.h5:/detector/readout
    reduced.h5:/rawdata
    $ ptdump reduced.h5
    / (RootGroup) ''
    /rawdata (Table(10,), shuffle, zlib(1)) 'Readout example'
    /columns (Group) ''
    /columns/name (Array(3,)) 'Name column selection'
    /columns/pressure (Array(3,)) 'Pressure column selection'

you can check how the filter properties has been changed for the /rawdata
table. Check as the other nodes still exists.

Finally, let's copy a *slice* of the readout table in origin to destination,
under a new group called /slices and with the name, for example, aslice:

.. code-block:: bash

    $ ptrepack -R1,8,3 tutorial1.h5:/detector/readout reduced.h5:/slices/aslice
    $ ptdump reduced.h5
    / (RootGroup) ''
    /rawdata (Table(10,), shuffle, zlib(1)) 'Readout example'
    /columns (Group) ''
    /columns/name (Array(3,)) 'Name column selection'
    /columns/pressure (Array(3,)) 'Pressure column selection'
    /slices (Group) ''
    /slices/aslice (Table(3,)) 'Readout example'

note how only 3 rows of the original readout table has been copied to the new
aslice destination. Note as well how the previously nonexistent slices group
has been created in the same operation.



pt2to3
------

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

.. note::

    :program:`pt2to3` uses the :mod:`argparse` module that is part of the
    Python standard library since Python 2.7.
    Users of Python 2.6 should install :mod:`argparse` separately
    (e.g. via :program:`pip`).

.. _PEP 8: http://www.python.org/dev/peps/pep-0008/
