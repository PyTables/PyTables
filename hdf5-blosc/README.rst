=====================
Blosc filter for HDF5
=====================

:Travis CI: |travis|
:And...: |powered|

.. |travis| image:: https://travis-ci.org/Blosc/hdf5.png?branch=master
        :target: https://travis-ci.org/Blosc/hdf5

.. |powered| image:: http://b.repl.ca/v1/Powered--By-Blosc-blue.png
        :target: https://blosc.org

This is an example of filter for HDF5 that uses the Blosc compressor.

You need to be a bit careful before using this filter because you
should not activate the shuffle right in HDF5, but rather from Blosc
itself.  This is because Blosc uses an SIMD shuffle internally which
is much faster.


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

An example C program ('src/example.c') is included which demonstrates
the proper use of the filter.

This filter has been tested against HDF5 versions 1.6.5 through
1.8.10.  It is released under the MIT license (see LICENSE.txt for
details).


Compiling
=========

The filter consists of a single 'src/blosc_filter.c' source file and
'src/blosc_filter.h' header, which will need the Blosc library
installed to work.


As an HDF5 plugin
=================

Also, you can use blosc as an HDF5 plugin; see 'src/blosc_plugin.c' for
details.


Acknowledgments
===============

See THANKS.rst.


----

  **Enjoy data!**
