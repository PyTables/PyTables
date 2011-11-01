What's new in PyTables 0.7.2
----------------------------

This is a mainly a maintenance release, where the next issues has
been addressed:

- Fixed a nasty memory leak located on the C libraries (It was
  occurring during attribute writes). After that, the memory
  consumption when using large object trees has dropped quite
  a bit. However, there remains some small leaks that has been
  tracked down to the underlying numarray library. These leaks
  has been reported, and hopefully they should be fixed more
  sooner than later.

- Table buffers are built dinamically now, so if Tables are
  not accessed for reading or writing this memory will not be
  booked. This will help to reduce the memory consumption.

- The opening of files with lots of nodes has been optimized
  between a factor 2 and 3. For example, a file with 10 groups
  and 3000 tables that takes 9.3 seconds to open in 0.7.1, now
  takes only 2.8 seconds.

- The Table.read() method has been refactored and optimized
  and some parts of its code has been moved to Pyrex. In
  particular, in the special case of step=1, up to a factor 5
  of speedup (reaching 160 MB/s on a Pentium4 @ 2 GHz) when
  reading table contents can be achieved now.


Enjoy!,

-- Francesc Alted
falted@openlc.org

