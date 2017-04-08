===============================================================
 Announcing C-Blosc 1.11.3
 A blocking, shuffling and lossless compression library for C
===============================================================

What is new?
============

Fixed an important bug in bitshuffle filter for big endian machines.
This prevented files written in bigendian machines to be read from
little endian ones.  See issue https://github.com/Blosc/c-blosc/issues/181.

Also, the internal Zstd codec has been updated to 1.1.3.

Finally, the blocksize for compression level 8 has been made 2x larger.
This should help specially Zstd codec to achieve better compression ratios.

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
