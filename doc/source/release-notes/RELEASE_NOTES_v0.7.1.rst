PyTables 0.7.1 is out!
----------------------

This is a mainly a bug-fixing release, where the next problems has
been addressed:

- Fixed several memory leaks. After that, the memory
  consumption when using large object trees has dropped
  sensibly. However, there remains some small leaks, but
  hopefully they are not very important unless you use *huge*
  object trees.

- Fixed a bug that make the __getitem__ special method in
  table to fail when the stop parameter in a extended slice
  was not specified. That is, table[10:] now correctly returns
  table[10:table.nrows+1], and not table[10:11].

- The removeRows() method in Table did not update the NROWS
  attribute in Table objects, giving place to errors after
  doing further updating operations (removing or adding more
  rows) in the same table. This has been fixed now.

Apart of these fixes, a new lazy reading algorithm for attributes has
been activated by default. With that, the opening of objects with
large hierarchies has been improved by 60% (you can obtain another
additional 10% if using python 2.3 instead of python 2.2).  The
documentation has been updated as well, specially a more detailed
instructions on the compression (zlib) libraries installation.

Also, a stress test has been conducted in order to see if PyTables can
*really* work not only with large data tables, but also with large
object trees. In it, it has been generated and checked a file with
more than 1 TB of size and more than 100 thousand tables on it!. See
http://www.pytables.org/moin/StressTestsBck for details.

