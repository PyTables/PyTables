What's new in PyTables 0.9
==========================

On this release you will find a series of quite
exciting new features, being the most important the indexing
capabilities, in-kernel selections, support for complex datatypes and
the possibility to modify values in both tables *and* arrays (yeah,
finally :).

New features:
-------------

- Indexing of columns in tables. That allow to make data selections on
  tables up to 500 times faster than standard selections (for
  ex. doing a selection along an indexed column of 100 milion of rows
  takes less than 1 second on a modern CPU). Perhaps the most
  interesting thing about the indexing algorithm implemented by
  PyTables is that the time taken to index grows *lineraly* with the
  length of the data, so, making the indexation process to be
  *scalable* (quite differently to many relational databases). This
  means that it can index, in a relatively quick way, arbitrarily
  large table columns (for ex. indexing a column of 100 milion of rows
  takes just 100 seconds, i.e. at a rate of 1 Mrow/sec). See more
  detailed info about that in http://www.pytables.org/docs/SciPy04.pdf.

- In-kernel selections. This feature allow to make data selections on
  tables up to 5 times faster than standard selections (i.e. pre-0.9
  selections), without a need to create an index. As a hint of how
  fast these selections can be, they are up to 10 times faster than a
  traditional relational database. Again, see
  http://www.pytables.org/docs/SciPy04.pdf for some experiments on that
  matter.

- Support of complex datatypes for all the data objects (i.e. Table,
  Array, EArray and VLArray). With that, the complete set of datatypes
  of Numeric and numarray packages are supported. Thanks to Tom Hedley
  for providing the patches for Array, EArray and VLArray objects, as
  well as updating the User's Manual and adding unit tests for the new
  functionality.

- Modification of values. You can modifiy Table, Array, EArray and
  VLArray values. See Table.modifyRows, Table.modifyColumns() and the
  newly introduced __setitem__() method for Table, Array, EArray and
  VLArray entities in the Library Reference of User's Manual.

- A new sub-package called "nodes" is there. On it, there will be
  included different modules to make more easy working with different
  entities (like images, files, ...). The first module that has been
  added to this sub-package is "FileNode", whose mission is to enable
  the creation of a database of nodes which can be used like regular
  opened files in Python.  In other words, you can store a set of
  files in a PyTables database, and read and write it as you would do
  with any other file in Python. Thanks to Ivan Vilata i Balaguer for
  contributing this.

Improvements:
-------------

- New __len__(self) methods added in Arrays, Tables and Columns. This,
  in combination with __getitem__(self,key) allows to better emulate
  sequences.

- Better capabilities to import generic HDF5 files. In particular,
  Table objects (in the HDF5_HL naming schema) with "holes" in their
  compound type definition are supported. That allows to read certain
  files produced by NASA (thanks to Stephen Walton for reporting this).

- Much improved test units. More than 2000 different tests has been
  implemented which accounts for more than 13000 loc (this represents
  twice of the PyTables library code itself (!)).

Backward-incompatible API changes:
----------------------------------

- The __call__ special method has been removed from objects File,
  Group, Table, Array, EArray and VLArray. Now, you should use
  walkNodes() in File and Group and iterrows in Table, Array, EArray
  and VLArray to get the same functionality. This would provide better
  compatibility with IPython as well.

'nctoh5', a new importing utility:

- Jeff Whitaker has contributed a script to easily convert NetCDF
  files into HDF5 files using Scientific Python and PyTables. It has
  been included and documented as a new utility.

Bug fixes:
----------

- A call to File.flush() now invoke a call to H5Fflush() so to
  effectively flushing all the file contents to disk. Thanks to Shack
  Toms for reporting this and providing a patch.

- SF #1054683: Security hole in utils.checkNameValidity(). Reported in
  2004-10-26 by ivilata

- SF #1049297: Suggestion: new method File.delAttrNode(). Reported in
  2004-10-18 by ivilata

- SF #1049285: Leak in AttributeSet.__delattr__(). Reported in
  2004-10-18 by ivilata

- SF #1014298: Wrong method call in examples/tutorial1-2.py. Reported
  in 2004-08-23 by ivilata

- SF #1013202: Cryptic error appending to EArray on RO file. Reported
  in 2004-08-21 by ivilata

- SF #991715: Table.read(field="var1", flavor="List") fails. Reported
  in 2004-07-15 by falted

- SF #988547: Wrong file type assumption in File.__new__. Reported in
  2004-07-10 by ivilata


Bon profit!,

-- Francesc Altet
falted@pytables.org

