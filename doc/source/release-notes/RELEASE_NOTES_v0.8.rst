What's new in PyTables 0.8
----------------------------

On this release, many enhancements has been added and some bugs has
been fixed. Here is the (non-exhaustive) list:

- The new VLArray class enables you to store large lists of rows
  containing variable numbers of elements. The elements can
  be scalars or fully multimensional objects, in the PyTables
  tradition. This class supports two special objects as rows:
  Unicode strings (UTF-8 codification is used internally) and
  generic Python objects (through the use of cPickle).

- The new EArray class allows you to enlarge already existing
  multidimensional homogeneous data objects. Consider it
  an extension of the already existing Array class, but
  with more functionality. Online compression or other filters
  can be applied to EArray instances, for example.

  Another nice feature of EA's is their support for fully
  multidimensional data selection with extended slices.  You
  can write "earray[1,2:3,...,4:200]", for example, to get the
  desired dataset slice from the disk. This is implemented
  using the powerful selection capabilities of the HDF5
  library, which results in very highly efficient I/O
  operations. The same functionality has been added to Array
  objects as well.

- New UnImplemented class. If a dataset contains unsupported
  datatypes, it will be associated with an UnImplemented
  instance, then inserted into to the object tree as usual.
  This allows you to continue to work with supported objects
  while retaining access to attributes of unsupported
  datasets.  This has changed from previous versions, where a
  RuntimeError occurred when an unsupported object was
  encountered.

  The combination of the new UnImplemented class with the
  support for new datatypes will enable PyTables to greatly
  increase the number of types of native HDF5 files that can
  be read and modified.

- Boolean support has been added for all the Leaf objects.

- The Table class has now an append() method that allows you
  to save large buffers of data in one go (i.e. bypassing the
  Row accessor). This can greatly improve data gathering
  speed.

- The standard HDF5 shuffle filter (to further enhance the
      compression level) is supported.

- The standard HDF5 fletcher32 checksum filter is supported.

- As the supported number of filters is growing (and may be
  further increased in the future), a Filters() class has been
  introduced to handle filters more easily.  In order to add
  support for this class, it was necessary to make a change in
  the createTable() method that is not backwards compatible:
  the "compress" and "complib" parameters are deprecated now
  and the "filters" parameter should be used in their
  place. You will be able to continue using the old parameters
  (only a Deprecation warning will be issued) for the next few
  releases, but you should migrate to the new version as soon
  as possible. In general, you can easily migrate old code by
  substituting code in its place::

    table = fileh.createTable(group, 'table', Test, '', complevel, complib)

  should be replaced by::

    table = fileh.createTable(group, 'table', Test, '',
                              Filters(complevel, complib))

- A copy() method that supports slicing and modification of
  filtering capabilities has been added for all the Leaf
  objects. See the User's Manual for more information.

- A couple of new methods, namely copyFile() and copyChilds(),
  have been added to File class, to permit easy replication
  of complete hierarchies or sub-hierarchies, even to
  other files. You can change filters during the copy
  process as well.

- Two new utilities has been added: ptdump and
  ptrepack. The utility ptdump allows the user to examine
  the contents of PyTables files (both metadata and actual
  data). The powerful ptrepack utility lets you
  selectively copy (portions of) hierarchies to specific
  locations in other files. It can be also used as an
  importer for generic HDF5 files.

- The meaning of the stop parameter in read() methods has
  changed. Now a value of 'None' means the last row, and a
  value of 0 (zero) means the first row. This is more
  consistent with the range() function in python and the
  __getitem__() special method in numarray.

- The method Table.removeRows() is no longer limited by table
  size.  You can now delete rows regardless of the size of the
  table.

- The "numarray" value has been added to the flavor parameter
  in the Table.read() method for completeness.

- The attributes (.attr instance variable) are Python
  properties now. Access to their values is no longer
  lazy, i.e. you will be able to see both system or user
  attributes from the command line using the tab-completion
  capability of your python console (if enabled).

- Documentation has been greatly improved to explain all the
  new functionality. In particular, the internal format of
  PyTables is now fully described. You can now build
  "native" PyTables files using any generic HDF5 software
  by just duplicating their format.

- Many new tests have been added, not only to check new
  functionality but also to more stringently check
  existing functionality. There are more than 800 different
  tests now (and the number is increasing :).

- PyTables has a new record in the data size that fits in one
  single file: more than 5 TB (yeah, more than 5000 GB), that
  accounts for 11 GB compressed, has been created on an AMD
  Opteron machine running Linux-64 (the 64 bits version of the
  Linux kernel). See the gory details in:
  http://pytables.sf.net/html/HowFast.html.

- New platforms supported: PyTables has been compiled and tested
  under Linux32 (Intel), Linux64 (AMD Opteron and Alpha), Win32
  (Intel), MacOSX (PowerPC), FreeBSD (Intel), Solaris (6, 7, 8
  and 9 with UltraSparc), IRIX64 (IRIX 6.5 with R12000) and it
  probably works in many more architectures. In particular,
  release 0.8 is the first one that provides a relatively clean
  porting to 64-bit platforms.

- As always, some bugs have been solved (especially bugs that
  occur when deleting and/or overwriting attributes).

- And last, but definitely not least, a new donations section
  has been added to the PyTables web site
  (http://sourceforge.net/projects/pytables, then follow the
  "Donations" tag). If you like PyTables and want this effort
  to continue, please, donate!

Enjoy!,

-- Francesc Alted
falted@pytables.org

