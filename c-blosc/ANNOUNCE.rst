===============================================================
 Announcing C-Blosc 1.11.1
 A blocking, shuffling and lossless compression library for C
===============================================================

What is new?
============

Fixed a bug introduced in 1.11.0 and discovered by pandas test suite.
This basically prevented to decompress buffers compressed with previous
versions of C-Blosc.See:
https://github.com/Blosc/python-blosc/issues/115

Also, since support for Zstd has been introduced the experience with it
has been really pleasant. As an example, see how Blosc + Zstd can
collaborate compressing images delivering pretty impressive compression
ratios and extremely fast decompression:

https://github.com/Cyan4973/zstd/issues/256

There is also a blog about what you can expect of it in:

http://blosc.org/blog/zstd-has-just-landed-in-blosc.html

For more info, please see the release notes in:

https://github.com/Blosc/c-blosc/blob/master/RELEASE_NOTES.rst


What is it?
===========

Blosc (http://www.blosc.org) is a high performance meta-compressor
optimized for binary data.  It has been designed to transmit data to
the processor cache faster than the traditional, non-compressed,
direct memory fetch approach via a memcpy() OS call.

Blosc has internal support for different compressors like its internal
BloscLZ, but also LZ4, LZ4HC, Snappy and Zlib.  This way these can
automatically leverage the multithreading and pre-filtering
(shuffling) capabilities that comes with Blosc.


Download sources
================

Please go to main web site:

http://www.blosc.org/

and proceed from there.  The github repository is over here:

https://github.com/Blosc

Blosc is distributed using the MIT license, see LICENSES/BLOSC.txt for
details.


Mailing list
============

There is an official Blosc mailing list at:

blosc@googlegroups.com
http://groups.google.es/group/blosc


Enjoy Data!
