Using the Blosc filter from HDF5
================================

In order to register Blosc into your HDF5 application, you only need
to call a function in blosc_filter.h, with the following signature:

    int register_blosc(char **version, char **date)

Calling this will register the filter with the HDF5 library and will
return info about the Blosc release in `**version` and `**date`
char pointers.

A non-negative return value indicates success.  If the registration
fails, an error is pushed onto the current error stack and a negative
value is returned.

An example C program ("example.c") is included which demonstrates the
proper use of the filter.

Alternatively, instead of registering the Blosc filter,  you can use the
automatically detectable `HDF5 filter plugin`_ which is supported in HDF5
1.8.11 and later.

This filter has been tested against HDF5 versions 1.6.5 through
1.8.10.  It is released under the MIT license (see LICENSE.txt for
details).

.. _`HDF5 filter plugin`: http://www.hdfgroup.org/HDF5/doc/Advanced/DynamicallyLoadedFilters/HDF5DynamicallyLoadedFilters.pdf


Compiling
=========

The filter consists of a single '.c' source file and '.h' header,
along with an embedded version of the BLOSC compression library.
Also, as Blosc uses SSE2 and multithreading, you must remember to use
some special flags and libraries to make sure that these features are
used (only necessary when compiling Blosc from sources).

To compile using GCC on UNIX:

  gcc -O3 -msse2 -lhdf5 ../blosc/*.c blosc_filter.c \
        example.c -o example -lpthread

or, if you have the Blosc library already installed (recommended):

  gcc -O3 -lhdf5 -lblosc blosc_filter.c example.c -o example -lpthread

Using MINGW on Windows:

  gcc -O3 -lhdf5 -lblosc blosc_filter.c example.c -o example

Using Windows and MSVC (2008 or higher recommended):

  cl /Ox /Feexample.exe example.c ..\blosc\*.c blosc_filter.c

Intel ICC compilers should work too.

On Windows, you'll need to copy hdf5.dll and possibly the msvc*.dll files
to your filter's directory if you do not have HDF5 installed in your PATH.

For activating the support for other compressors than the integrated
BloscLZ (like LZ4, LZ4HC, Snappy or Zlib) see the README file in the
main Blosc directory.


Compiling dynamically loadable filter plugin
============================================

Compile blosc_plugin.c and blosc_filter.c to a shared library and then
let HDF5 know where to find it.

To complie using GCC on UNIX:

    gcc -O3 -msse2 -lhdf5 -lpthread ../blosc/*.c \
        blosc_filter.c blosc_plugin.c -fPIC -shared \
        -o libblosch5plugin.so

Then ether move the shared library to HDF5's default search location for
plugins (on UNIX ``/usr/local/hdf5/lib/plugin``) or to a directory pointed to
by the ``HDF5_PLUGIN_PATH`` environment variable.


IMPORTANT WINDOWS NOTE
======================

On Windows, the configuration (Release vs. Debug) and Visual Studio version
of HDF5 and the c-blosc filter must match EXACTLY or you will experience
crashes. You should also ensure that the C run-time is dynamically linked
to both HDF5 and c-blosc.

This is due to the way Microsoft implements its C library. On Windows, the
standard C library is not a fundamental part of the operating system, as it
is on Unix-like systems. Instead, the C library is implemented in separate
shared libraries (dlls - called the C run-time (CRT) by Microsoft), which
differ by Visual Studio and configuration. For example, msvcr110d.dll is the
Visual Studio 2012 debug C run-time and msvcr90.dll is the Visual Studio
2008 release C run-time. Since there is no shared state between these
independent libraries, allocating memory in one library and freeing it in
another (as the c-blosc HDF5 filter does) will corrupt the heap and cause
crashes.

There is currently no way around this issue since a fix involves exposing
the HDF5 library's memory management functions for filter author use, which
would ensure that both the filter and HDF5 use the same allocation and
free functions. The HDF Group is aware of the problem and hopes to have a
fix in HDF5 1.8.15 (May 2015).

To duplicate the problem
------------------------

* Install the HDF5 binary distribution. The HDF5 binaries are built in release mode (even though they include debugging symbols) and link to the release C run-time.

* Configure and build c-blosc using the debug configuration. Ensure that CMake uses the installed release-configuration HDF5.

* You may need to copy hdf5.dll and the msvc*.dll libraries to the filter's binary directory if the HDF5 bin directory is not in your PATH.

* At this point, HDF5 will be using the release C run-time and c-blosc will be using the debug C run-time. You can confirm this using the Visual Studio tool 'dumpbin /imports'.

* Run example.exe. It should crash.

If you build the HDF5 library from source in the debug configuration,
you can confirm that debug HDF5 and release c-blosc will also cause
example.exe to fail.

Note that the crashes may not be deterministic. Your mileage may vary.
Regardless of the behavior on your particular system, this is a serious
problem and will crash many, if not most, systems.

To demonstrate proper behavior
------------------------------

* Build c-blosc in the configuration that matches HDF5.

* example.exe should now run normally.

To confirm that it is a C run-time mismatch issue, you can modify the
src/H5.c and src/H5public.h files in the HDF5 source distribution to
expose the HDF5 library's allocator (H5free_memory() already exists).
Simply copy and modify the H5free_memory() function to something like
H5malloc() that wraps malloc(). You'll need to run 'bin/trace src/H5.c'
in the source root to generate a TRACE macro for the new API call
(requires Perl). Modify the filter to use H5malloc() and H5free_memory()
in place of malloc() and free() and rebuild c-blosc. You will now be
able to combine release and debug configurations without example.exe
crashing.


Acknowledgments
===============

This HDF5 filter interface and its example is based in the LZF interface
(http://h5py.alfven.org) by Andrew Collette.

Dana Robinson made nice improvements on existing CMake files for
Windows/MSVC.
