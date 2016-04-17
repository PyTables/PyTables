=================================
 Release notes for C-Blosc 1.8.1
=================================

:Author: Francesc Alted
:Contact: francesc@blosc.org
:URL: http://www.blosc.org


Changes from 1.8.0 to 1.8.1
===========================

* Disable the use of __builtin_cpu_supports() for GCC 5.3.1
  compatibility.  Details in:
  https://lists.fedoraproject.org/archives/list/devel@lists.fedoraproject.org/thread/ZM2L65WIZEEQHHLFERZYD5FAG7QY2OGB/


Changes from 1.7.1 to 1.8.0
===========================

* The code is (again) compatible with VS2008 and VS2010.  This is
  important for compatibility with Python 2.6/2.7/3.3/3.4.

* Introduced a new global lock during blosc_decompress() operation.
  As the blosc_compress() was already guarded by a global lock, this
  means that the compression/decompression is again thread safe.
  However, when using C-Blosc from multi-threaded environments, it is
  important to keep using the *_ctx() functions for performance
  reasons.  NOTE: _ctx() functions will be replaced by more powerful
  ones in C-Blosc 2.0.


Changes from 1.7.0 to 1.7.1
===========================

* Fixed a bug preventing bitshuffle to work correctly on getitem().
  Now, everything with bitshuffle seems to work correctly.

* Fixed the thread initialization for blosc_decompress_ctx().  Issue
  #158.  Thanks to Chris Webers.

* Fixed a bug in the blocksize computation introduced in 1.7.0.  This
  could have been creating segfaults.

* Allow bitshuffle to run on 1-byte typesizes.

* New parametrization of the blocksize to be independent of the
  typesize.  This allows a smoother speed throughout all typesizes.

* lz4 and lz4hc codecs upgraded to 1.7.2 (from 1.7.0).

* When calling set_nthreads() but not actually changing the number of
  threads in the internal pool does not teardown and setup it anymore.
  PR #153.  Thanks to Santi Villalba.


Changes from 1.6.1 to 1.7.0
===========================

* Added a new 'bitshuffle' filter so that the shuffle takes place at a
  bit level and not just at a byte one, which is what it does the
  previous 'shuffle' filter.

  For activating this new bit-level filter you only have to pass the
  symbol BLOSC_BITSHUFFLE to `blosc_compress()`.  For the previous
  byte-level one, pass BLOSC_SHUFFLE.  For disabling the shuffle, pass
  BLOSC_NOSHUFFLE.

  This is a port of the existing filter in
  https://github.com/kiyo-masui/bitshuffle.  Thanks to Kiyo Masui for
  changing the license and allowing its inclusion here.

* New acceleration mode for LZ4 and BloscLZ codecs that enters in
  operation with complevel < 9.  This allows for an important boost in
  speed with minimal compression ratio loss.  Francesc Alted.

* LZ4 codec updated to 1.7.0 (r130).

* PREFER_EXTERNAL_COMPLIBS cmake option has been removed and replaced
  by the more fine grained PREFER_EXTERNAL_LZ4, PREFER_EXTERNAL_SNAPPY
  and PREFER_EXTERNAL_ZLIB.  In order to allow the use of the new API
  introduced in LZ4 1.7.0, PREFER_EXTERNAL_LZ4 has been set to OFF by
  default, whereas PREFER_EXTERNAL_SNAPPY and PREFER_EXTERNAL_ZLIB
  continues to be ON.

* Implemented SSE2 shuffle support for buffers containing a number of
  elements which is not a multiple of (typesize * vectorsize).  Jack
  Pappas.

* Added SSE2 shuffle/unshuffle routines for types larger than 16
  bytes.  Jack Pappas.

* 'test_basic' suite has been split in components for a much better
  granularity on what's a possibly failing test.  Also, lots of new
  tests have been added.  Jack Pappas.

* Fixed compilation on non-Intel archs (tested on ARM).  Zbyszek
  Szmek.

* Modifyied cmake files in order to inform that AVX2 on Visual Studio
  is supported only in 2013 update 2 and higher.

* Added a replacement for stdbool.h for Visual Studio < 2013.

* blosclz codec adds Win64/Intel as a platform supporting unaligned
  addressing.  That leads to a speed-up of 2.2x in decompression.

* New blosc_get_version_string() function for retrieving the version
  of the c-blosc library.  Useful when linking with dynamic libraries
  and one want to know its version.

* New example (win-dynamic-linking.c) that shows how to link a Blosc
  DLL dynamically in run-time (Windows only).

* The `context.threads_started` is initialized now when decompressing.
  This could cause crashes in case you decompressed before compressing
  (e.g. directly deserializing blosc buffers).  @atchouprakov.

* The HDF5 filter has been removed from c-blosc and moved into its own
  repo at: https://github.com/Blosc/hdf5

* The MS Visual Studio 2008 has been tested with c-blosc for ensuring
  compatibility with extensions for Python 2.6 and up.


Changes from 1.6.0 to 1.6.1
===========================

* Support for *runtime* detection of AVX2 and SSE2 SIMD instructions.
  These changes make it possible to compile one single binary that
  runs on a system that supports SSE2 or AVX2 (or neither), so the
  redistribution problem is fixed (see #101).  Thanks to Julian Taylor
  and Jack Pappas.

* Added support for MinGW and TDM-GCC compilers for Windows.  Thanks
  to yasushima-gd.

* Fixed a bug in blosclz that could potentially overwrite an area
  beyond the output buffer.  See #113.

* New computation for blocksize so that larger typesizes (> 8 bytes)
  would benefit of much better compression ratios.  Speed is not
  penalized too much.

* New parametrization of the hash table for blosclz codec.  This
  allows better compression in many scenarios, while slightly
  increasing the speed.


Changes from 1.5.4 to 1.6.0
===========================

* Support for AVX2 is here!  The benchmarks with a 4-core Intel
  Haswell machine tell that both compression and decompression are
  accelerated around a 10%, reaching peaks of 9.6 GB/s during
  compression and 26 GB/s during decompression (memcpy() speed for
  this machine is 7.5 GB/s for writes and 11.7 GB/s for reads).  Many
  thanks to @littlezhou for this nice work.

* Support for HPET (high precision timers) for the `bench` program.
  This is particularly important for microbenchmarks like bench is
  doing; since they take so little time to run, the granularity of a
  less-accurate timer may account for a significant portion of the
  runtime of the benchmark itself, skewing the results.  Thanks to
  Jack Pappas.


Changes from 1.5.3 to 1.5.4
===========================

* Updated to LZ4 1.6.0 (r128).

* Fix resource leak in t_blosc.  Jack Pappas.

* Better checks during testing.  Jack Pappas.

* Dynamically loadable HDF5 filter plugin. Kiyo Masui.


Changes from 1.5.2 to 1.5.3
===========================

* Use llabs function (where available) instead of abs to avoid
  truncating the result.  Jack Pappas.

* Use C11 aligned_alloc when it's available.  Jack Pappas.

* Use the built-in stdint.h with MSVC when available.  Jack Pappas.

* Only define the __SSE2__ symbol when compiling with MS Visual C++
  and targeting x64 or x86 with the correct /arch flag set. This
  avoids re-defining the symbol which makes other compilers issue
  warnings.  Jack Pappas.

* Reinitializing Blosc during a call to set_nthreads() so as to fix
  problems with contexts.  Francesc Alted.



Changes from 1.5.1 to 1.5.2
===========================

* Using blosc_compress_ctx() / blosc_decompress_ctx() inside the HDF5
  compressor for allowing operation in multiprocess scenarios.  See:
  https://github.com/PyTables/PyTables/issues/412

  The drawback of this quick fix is that the Blosc filter will be only
  able to use a single thread until another solution can be devised.


Changes from 1.5.0 to 1.5.1
===========================

* Updated to LZ4 1.5.0.  Closes #74.

* Added the 'const' qualifier to non SSE2 shuffle functions. Closes #75.

* Explicitly call blosc_init() in HDF5 blosc_filter.c, fixing a
  segfault.

* Quite a few improvements in cmake files for HDF5 support.  Thanks to
  Dana Robinson (The HDF Group).

* Variable 'class' caused problems compiling the HDF5 filter with g++.
  Thanks to Laurent Chapon.

* Small improvements on docstrings of c-blosc main functions.


Changes from 1.4.1 to 1.5.0
===========================

* Added new calls for allowing Blosc to be used *simultaneously*
  (i.e. lock free) from multi-threaded environments.  The new
  functions are:

  - blosc_compress_ctx(...)
  - blosc_decompress_ctx(...)

  See the new docstrings in blosc.h for how to use them.  The previous
  API should be completely unaffected.  Thanks to Christopher Speller.

* Optimized copies during BloscLZ decompression.  This can make BloscLZ
  to decompress up to 1.5x faster in some situations.

* LZ4 and LZ4HC compressors updated to version 1.3.1.

* Added an examples directory on how to link apps with Blosc.

* stdlib.h moved from blosc.c to blosc.h as suggested by Rob Lathm.

* Fix a warning for {snappy,lz4}-free compilation.  Thanks to Andrew Schaaf.

* Several improvements for CMakeLists.txt (cmake).

* Fixing C99 compatibility warnings.  Thanks to Christopher Speller.


Changes from 1.4.0 to 1.4.1
===========================

* Fixed a bug in blosc_getitem() introduced in 1.4.0.  Added a test for
  blosc_getitem() as well.


Changes from 1.3.6 to 1.4.0
===========================

* Support for non-Intel and non-SSE2 architectures has been added.  In
  particular, the Raspberry Pi platform (ARM) has been tested and all
  tests pass here.

* Architectures requiring strict access alignment are supported as well.
  Due to this, arquitectures with a high penalty in accessing unaligned
  data (e.g. Raspberry Pi, ARMv6) can compress up to 2.5x faster.

* LZ4 has been updated to r119 (1.2.0) so as to fix a possible security
  breach.


Changes from 1.3.5 to 1.3.6
===========================

* Updated to LZ4 r118 due to a (highly unlikely) security hole.  For
  details see:
 
  http://fastcompression.blogspot.fr/2014/06/debunking-lz4-20-years-old-bug-myth.html


Changes from 1.3.4 to 1.3.5
===========================

* Removed a pointer from 'pointer from integer without a cast' compiler
  warning due to a bad macro definition.


Changes from 1.3.3 to 1.3.4
===========================

* Fixed a false buffer overrun condition.  This bug made c-blosc to
  fail, even if the failure was not real.

* Fixed the type of a buffer string.


Changes from 1.3.2 to 1.3.3
===========================

* Updated to LZ4 1.1.3 (improved speed for 32-bit platforms).

* Added a new `blosc_cbuffer_complib()` for getting the compression
  library for a compressed buffer.


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

