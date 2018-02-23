===============================================================
 Announcing C-Blosc 1.14.0
 A blocking, shuffling and lossless compression library for C
===============================================================

What is new?
============

The most important change is a new split mode that favors forward
compatibility.  That means that, from now on, all the buffers created
starting with blosc 1.14.0 will be forward compatible with any previous
versions of the library --at least until 1.3.0, when support for
multi-codec was introduced.

Also, a new policy about forward compatibility has been put in place.
See blog entry at: http://blosc.org/posts/new-forward-compat-policy

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
