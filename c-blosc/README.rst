===============================================================
 Blosc: A blocking, shuffling and lossless compression library
===============================================================

:Author: Francesc Alted
:Contact: francesc@blosc.io
:URL: http://www.blosc.org

What is it?
===========

Blosc [1]_ is a high performance compressor optimized for binary data.
It has been designed to transmit data to the processor cache faster
than the traditional, non-compressed, direct memory fetch approach via
a memcpy() OS call.  Blosc is the first compressor (that I'm aware of)
that is meant not only to reduce the size of large datasets on-disk or
in-memory, but also to accelerate memory-bound computations.

It uses the blocking technique (as described in [2]_) to reduce
activity on the memory bus as much as possible. In short, this
technique works by dividing datasets in blocks that are small enough
to fit in caches of modern processors and perform compression /
decompression there.  It also leverages, if available, SIMD
instructions (SSE2) and multi-threading capabilities of CPUs, in order
to accelerate the compression / decompression process to a maximum.

Blosc is actually a metacompressor, that meaning that it can use a range
of compression libraries for performing the actual
compression/decompression. Right now, it comes with integrated support
for BloscLZ (the original one), LZ4, LZ4HC, Snappy and Zlib. Blosc comes
with full sources for all compressors, so in case it does not find the
libraries installed in your system, it will compile from the included
sources and they will be integrated into the Blosc library anyway. That
means that you can trust in having all supported compressors integrated
in Blosc in all supported platforms.

You can see some benchmarks about Blosc performance in [3]_

Blosc is distributed using the MIT license, see LICENSES/BLOSC.txt for
details.

.. [1] http://www.blosc.org
.. [2] http://blosc.org/docs/StarvingCPUs-CISE-2010.pdf
.. [3] http://blosc.org/trac/wiki/SyntheticBenchmarks

Meta-compression and other advantages over existing compressors
===============================================================

Blosc is not like other compressors: it should rather be called a
meta-compressor.  This is so because it can use different compressors
and pre-conditioners (programs that generally improve compression
ratio).  At any rate, it can also be called a compressor because it
happens that it already integrates one compressor and one
pre-conditioner, so it can actually work like so.

Currently it uses BloscLZ, a compressor heavily based on FastLZ
(http://fastlz.org/), and a highly optimized (it can use SSE2
instructions, if available) Shuffle pre-conditioner. However,
different compressors or pre-conditioners may be added in the future.

Blosc is in charge of coordinating the compressor and pre-conditioners
so that they can leverage the blocking technique (described above) as
well as multi-threaded execution (if several cores are available)
automatically. That makes that every compressor and pre-conditioner
will work at very high speeds, even if it was not initially designed
for doing blocking or multi-threading.

Other advantages of Blosc are:

* Meant for binary data: can take advantage of the type size
  meta-information for improved compression ratio (using the
  integrated shuffle pre-conditioner).

* Small overhead on non-compressible data: only a maximum of 16
  additional bytes over the source buffer length are needed to
  compress *every* input.

* Maximum destination length: contrarily to many other
  compressors, both compression and decompression routines have
  support for maximum size lengths for the destination buffer.

* Replacement for memcpy(): it supports a 0 compression level that
  does not compress at all and only adds 16 bytes of overhead. In
  this mode Blosc can copy memory usually faster than a plain
  memcpy().

When taken together, all these features set Blosc apart from other
similar solutions.

Compiling your application with a minimalistic Blosc
====================================================

The minimal Blosc consists of the next files (in blosc/ directory)::

    blosc.h and blosc.c      -- the main routines
    shuffle.h and shuffle.c  -- the shuffle code
    blosclz.h and blosclz.c  -- the blosclz compressor

Just add these files to your project in order to use Blosc.  For
information on compression and decompression routines, see blosc.h.

To compile using GCC (4.4 or higher recommended) on Unix:

.. code-block:: console

   $ gcc -O3 -msse2 -o myprog myprog.c blosc/*.c -lpthread

Using Windows and MINGW:

.. code-block:: console

   $ gcc -O3 -msse2 -o myprog myprog.c blosc\*.c

Using Windows and MSVC (2010 or higher recommended):

.. code-block:: console

  $ cl /Ox /Femyprog.exe myprog.c blosc\*.c

A simple usage example is the benchmark in the bench/bench.c file.
Another example for using Blosc as a generic HDF5 filter is in the
hdf5/ directory.

I have not tried to compile this with compilers other than GCC, clang,
MINGW, Intel ICC or MSVC yet. Please report your experiences with your
own platforms.

Adding support for other compressors (LZ4, LZ4HC, Snappy, Zlib)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to add support for the LZ4, LZ4HC, Snappy or Zlib
compressors, just add the symbols HAVE_LZ4 (will include both LZ4 and
LZ4HC), HAVE_SNAPPY and HAVE_ZLIB during compilation and add the
libraries. For example, for compiling Blosc with Zlib support do:

.. code-block:: console

   $ gcc -O3 -msse2 -o myprog myprog.c blosc/*.c -lpthread -DHAVE_ZLIB -lz

In the bench/ directory there a couple of Makefile files (one for UNIX
and the other for MinGW) with more complete building examples, like
selecting between libraries or internal sources for the compressors.

Compiling the Blosc library with CMake
======================================

Blosc can also be built, tested and installed using CMake_. Although
this procedure is a bit more invloved than the one described above, it
is the most general because it allows to integrate other compressors
than BloscLZ either from libraries or from internal sources. Hence,
serious library developers should use this way.

The following procedure describes the "out of source" build.

Create the build directory and move into it:

.. code-block:: console

  $ mkdir build
  $ cd build

Now run CMake configuration and optionally specify the installation
directory (e.g. '/usr' or '/usr/local'):

.. code-block:: console

  $ cmake -DCMAKE_INSTALL_PREFIX=your_install_prefix_directory ..

CMake allows to configure Blosc in many different ways, like prefering
internal or external sources for compressors or enabling/disabling
them.  Please note that configuration can also be performed using UI
tools provided by CMake_ (ccmake or cmake-gui):

.. code-block:: console

  $ ccmake ..      # run a curses-based interface
  $ cmake-gui ..   # run a graphical interface

Build, test and install Blosc:

.. code-block:: console

  $ make
  $ make test
  $ make install

The static and dynamic version of the Blosc library, together with
header files, will be installed into the specified
CMAKE_INSTALL_PREFIX.

.. _CMake: http://www.cmake.org

Adding support for other compressors (LZ4, LZ4HC, Snappy, Zlib) with CMake
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The CMake files in Blosc are configured to automatically detect other
compressors like LZ4, LZ4HC, Snappy or Zlib by default.  So as long as
the libraries and the header files for these libraries are accessible,
these will be used by default.

*Note on Zlib*: the library should be easily found on UNIX systems,
although on Windows, you can help CMake to find it by setting the
environment variable 'ZLIB_ROOT' to where zlib 'include' and 'lib'
directories are. Also, make sure that Zlib DDL library is in your
'\Windows' directory.

However, the full sources for LZ4, LZ4HC, Snappy and Zlib have been
included in Blosc too. So, in general, you should not worry about not
having (or CMake not finding) the libraries in your system because in
this case, their sources will be automaticall compiled for you. That
effectively means that you can be confident in having a complete
support for all the supported compression libraries in all supported
platforms.

If you want to force Blosc to use the included compression sources
instead of trying to find the libraries in the system first, you can
switch off the PREFER_EXTERNAL_COMPLIBS CMake option:

.. code-block:: console

  $ cmake -DPREFER_EXTERNAL_COMPLIBS=OFF ..

You can also disable support for some compression libraries:

.. code-block:: console

  $ cmake -DDEACTIVATE_SNAPPY=ON ..

Mac OSX troubleshooting
=======================

If you run into compilation troubles when using Mac OSX, please make
sure that you have installed the command line developer tools.  You
can always install them with:

.. code-block:: console

  $ xcode-select --install

Wrapper for Python
==================

Blosc has an official wrapper for Python.  See:

https://github.com/Blosc/python-blosc

Command line interface and serialization format for Blosc
=========================================================

Blosc can be used from command line by using Bloscpack.  See:

https://github.com/Blosc/bloscpack

Filter for HDF5
===============

For those that want to use Blosc as a filter in the HDF5 library,
there is a sample implementation in the hdf5/ directory.

Mailing list
============

There is an official mailing list for Blosc at:

blosc@googlegroups.com
http://groups.google.es/group/blosc

Acknowledgments
===============

I'd like to thank the PyTables community that have collaborated in the
exhaustive testing of Blosc.  With an aggregate amount of more than
300 TB of different datasets compressed *and* decompressed
successfully, I can say that Blosc is pretty safe now and ready for
production purposes.

Other important contributions:

* Valentin Haenel did a terrific work implementing the support for the
  Snappy compression, fixing typos and improving docs and the plotting
  script.

* Thibault North, with ideas from Oscar Villellas, contributed a way
  to call Blosc from different threads in a safe way.

* The CMake support was initially contributed by Thibault North, and
  Antonio Valentino and Mark Wiebe made great enhancements to it.


----

  **Enjoy data!**
