===============================
 Release notes for Blosc 1.3.2
===============================

:Author: Francesc Alted
:Contact: faltet@gmail.com
:URL: http://www.blosc.org


Changes from 1.3.1 to 1.3.2
===========================

* Fix for compiling Snappy sources against MSVC 2008.  Thanks to Mark
  Wiebe!

* Version for internal LZ4 and Snappy are now supported.  When compiled
  against the external libraries, this info is not available because
  they do not support the symbols (yet).


Changes from 1.3.0 to 1.3.1
===========================

* Fixes for a series of issues with the filter for HDF5 and, in
  particular, a problem in the decompression buffer size that made it
  impossible to use the blosc_filter in combination with other ones
  (e.g. fletcher32).  See
  https://github.com/PyTables/PyTables/issues/21.

  Thanks to Antonio Valentino for the fix!


Changes from 1.2.4 to 1.3.0
===========================

A nice handful of compressors have been added to Blosc:

* LZ4 (http://code.google.com/p/lz4/): A very fast
  compressor/decompressor.  Could be thought as a replacement of the
  original BloscLZ, but it can behave better is some scenarios.

* LZ4HC (http://code.google.com/p/lz4/): This is a variation of LZ4
  that achieves much better compression ratio at the cost of being
  much slower for compressing.  Decompression speed is unaffected (and
  sometimes better than when using LZ4 itself!), so this is very good
  for read-only datasets.

* Snappy (http://code.google.com/p/snappy/): A very fast
  compressor/decompressor.  Could be thought as a replacement of the
  original BloscLZ, but it can behave better is some scenarios.

* Zlib (http://www.zlib.net/): This is a classic.  It achieves very
  good compression ratios, at the cost of speed.  However,
  decompression speed is still pretty good, so it is a good candidate
  for read-only datasets.

With this, you can select the compression library with the new
function::

  int blosc_set_complib(char* complib);

where you pass the library that you want to use (currently "blosclz",
"lz4", "lz4hc", "snappy" and "zlib", but the list can grow in the
future).

You can get more info about compressors support in you Blosc build by
using these functions::

  char* blosc_list_compressors(void);
  int blosc_get_complib_info(char *compressor, char **complib, char **version);


Changes from 1.2.2 to 1.2.3
===========================

- Added a `blosc_init()` and `blosc_destroy()` so that the global lock
  can be initialized safely.  These new functions will also allow other
  kind of initializations/destructions in the future.

  Existing applications using Blosc do not need to start using the new
  functions right away, as long as they calling `blosc_set_nthreads()`
  previous to anything else.  However, using them is highly recommended.

  Thanks to Oscar Villellas for the init/destroy suggestion, it is a
  nice idea!


Changes from 1.2.1 to 1.2.2
===========================

- All important warnings removed for all tested platforms.  This will
  allow less intrusiveness compilation experiences with applications
  including Blosc source code.

- The `bench/bench.c` has been updated so that it can be compiled on
  Windows again.

- The new web site has been set to: http://www.blosc.org


Changes from 1.2 to 1.2.1
=========================

- Fixed a problem with global lock not being initialized.  This
  affected mostly to Windows platforms.  Thanks to Christoph
  Gohlke for finding the cure!


Changes from 1.1.5 to 1.2
=========================

- Now it is possible to call Blosc simultaneously from a parent threaded
  application without problems.  This has been solved by setting a
  global lock so that the different calling threads do not execute Blosc
  routines at the same time.  Of course, real threading work is still
  available *inside* Blosc itself.  Thanks to Thibault North.

- Support for cmake is now included.  Linux, Mac OSX and Windows
  platforms are supported.  Thanks to Thibault North, Antonio Valentino
  and Mark Wiebe.

- Fixed many compilers warnings (specially about unused variables).

- As a consequence of the above, as minimal change in the API has been
  introduced.  That is, the previous API::

    void blosc_free_resources(void)

  has changed to::

    int blosc_free_resources(void)

  Now, a return value of 0 means that the resources have been released
  successfully.  If the return value is negative, then it is not
  guaranteed that all the resources have been freed.

- Many typos were fixed and docs have been improved.  The script for
  generating nice plots for the included benchmarks has been improved
  too.  Thanks to Valetin Haenel.


Changes from 1.1.4 to 1.1.5
===========================

- Fix compile error with msvc compilers (Christoph Gohlke)


Changes from 1.1.3 to 1.1.4
===========================

- Redefinition of the BLOSC_MAX_BUFFERSIZE constant as (INT_MAX -
  BLOSC_MAX_OVERHEAD) instead of just INT_MAX.  This prevents to produce
  outputs larger than INT_MAX, which is not supported.

- `exit()` call has been replaced by a ``return -1`` in blosc_compress()
  when checking for buffer sizes.  Now programs will not just exit when
  the buffer is too large, but return a negative code.

- Improvements in explicit casts.  Blosc compiles without warnings
  (with GCC) now.

- Lots of improvements in docs, in particular a nice ascii-art diagram
  of the Blosc format (Valentin Haenel).

- Improvements to the plot-speeds.py (Valentin Haenel).

- [HDF5 filter] Adapted HDF5 filter to use HDF5 1.8 by default
  (Antonio Valentino).

- [HDF5 filter] New version of H5Z_class_t definition (Antonio Valentino).


Changes from 1.1.2 to 1.1.3
===========================

- Much improved compression ratio when using large blocks (> 64 KB) and
  high compression levels (> 6) under some circumstances (special data
  distribution).  Closes #7.


Changes from 1.1.1 to 1.1.2
===========================

- Fixes for small typesizes (#6 and #1 of python-blosc).


Changes from 1.1 to 1.1.1
=========================

- Added code to avoid calling blosc_set_nthreads more than necessary.
  That will improve performance up to 3x or more, specially for small
  chunksizes (< 1 MB).


Changes from 1.0 to 1.1
=======================

- Added code for emulating pthreads API on Windows.  No need to link
  explicitly with pthreads lib on Windows anymore.  However, performance
  is a somewhat worse because the new emulation layer does not support
  the `pthread_barrier_wait()` call natively.  But the big improvement
  in installation easiness is worth this penalty (most specially on
  64-bit Windows, where pthreads-win32 support is flaky).

- New BLOSC_MAX_BUFFERSIZE, BLOSC_MAX_TYPESIZE and BLOSC_MAX_THREADS
  symbols are available in blosc.h.  These can be useful for validating
  parameters in clients.  Thanks to Robert Smallshire for suggesting
  that.

- A new BLOSC_MIN_HEADER_LENGTH symbol in blosc.h tells how many bytes
  long is the minimum length of a Blosc header.  `blosc_cbuffer_sizes()`
  only needs these bytes to be passed to work correctly.

- Removed many warnings (related with potentially dangerous type-casting
  code) issued by MSVC 2008 in 64-bit mode.

- Fixed a problem with the computation of the blocksize in the Blosc
  filter for HDF5.

- Fixed a problem with large datatypes.  See
  http://www.pytables.org/trac/ticket/288 for more info.

- Now Blosc is able to work well even if you fork an existing process
  with a pool of threads.  Bug discovered when PyTables runs in
  multiprocess environments.  See http://pytables.org/trac/ticket/295
  for details.

- Added a new `blosc_getitem()` call to allow the retrieval of items in
  sizes smaller than the complete buffer.  That is useful for the carray
  project, but certainly for others too.


Changes from 0.9.5 to 1.0
=========================

- Added a filter for HDF5 so that people can use Blosc outside PyTables,
  if they want to.

- Many small improvements, specially in README files.

- Do not assume that size_t is uint_32 for every platform.

- Added more protection for large buffers or in allocation memory
  routines.

- The src/ directory has been renamed to blosc/.

- The `maxbytes` parameter in `blosc_compress()` has been renamed to
  `destsize`.  This is for consistency with the `blosc_decompress()`
  parameters.


Changes from 0.9.4 to 0.9.5
===========================

- Now, compression level 0 is allowed, meaning not compression at all.
  The overhead of this mode will be always BLOSC_MAX_OVERHEAD (16)
  bytes.  This mode actually represents using Blosc as a basic memory
  container.

- Supported a new parameter `maxbytes` for ``blosc_compress()``.  It
  represents a maximum of bytes for output.  Tests unit added too.

- Added 3 new functions for querying different metadata on compressed
  buffers.  A test suite for testing the new API has been added too.


Changes from 0.9.3 to 0.9.4
===========================

- Support for cross-platform big/little endian compatibility in Blosc
  headers has been added.

- Fixed several failures exposed by the extremesuite.  The problem was a
  bad check for limits in the buffer size while compressing.

- Added a new suite in bench.c called ``debugsuite`` that is
  appropriate for debugging purposes.  Now, the ``extremesuite`` can be
  used for running the complete (and extremely long) suite.


Changes from 0.9.0 to 0.9.3
===========================

- Fixed several nasty bugs uncovered by the new suites in bench.c.
  Thanks to Tony Theodore and Gabriel Beckers for their (very)
  responsive beta testing and feedback.

- Added several modes (suites), namely ``suite``, ``hardsuite`` and
  ``extremehardsuite`` in bench.c so as to allow different levels of
  testing.


Changes from 0.8.0 to 0.9
=========================

- Internal format version bumped to 2 in order to allow an easy way to
  indicate that a buffer is being saved uncompressed.  This is not
  supported yet, but it might be in the future.

- Blosc can use threads now for leveraging the increasing number of
  multi-core processors out there.  See README-threaded.txt for more
  info.

- Added a protection for MacOSX so that it has to not link against
  posix_memalign() funtion, which seems not available in old versions of
  MacOSX (for example, Tiger).  At nay rate, posix_memalign() is not
  necessary on Mac because 16 bytes alignment is ensured by default.
  Thanks to Ivan Vilata.  Fixes #3.




.. Local Variables:
.. mode: rst
.. coding: utf-8
.. fill-column: 72
.. End:
