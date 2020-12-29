===============================================================
 Announcing C-Blosc 1.21.0
 A blocking, shuffling and lossless compression library for C
===============================================================

What is new?
============

This is a maintenance release.  Vendored lz4 and zstd codecs have been
updated to 1.9.3 and 1.4.8 respectively.

Also, this should be the first release that is officially providing
binary libraries via the Python wheels in its sibling project python-blosc.
Thanks to Jeff Hammerbacher for his generous donation to make this happen.

For more info, please see the release notes in:

https://github.com/Blosc/c-blosc/blob/master/RELEASE_NOTES.rst


What is it?
===========

Blosc (http://www.blosc.org) is a high performance meta-compressor
optimized for binary data.  It has been designed to transmit data to
the processor cache faster than the traditional, non-compressed,
direct memory fetch approach via a memcpy() OS call.

Blosc has internal support for different compressors like its internal
BloscLZ, but also LZ4, LZ4HC, Snappy, Zlib and Zstd.  This way these can
automatically leverage the multithreading and pre-filtering
(shuffling) capabilities that comes with Blosc.


Download sources
================

The github repository is over here:

https://github.com/Blosc

Blosc is distributed using the BSD license, see LICENSES/BLOSC.txt for
details.


Mailing list
============

There is an official Blosc mailing list at:

blosc@googlegroups.com
http://groups.google.es/group/blosc


Enjoy Data!
