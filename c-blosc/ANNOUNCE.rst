===============================================================
 Announcing c-blosc 1.4.1
 A blocking, shuffling and lossless compression library
===============================================================

What is new?
============

This is a maintenance release, where a bug in blosc_getitem() introduced
in 1.4.0 has been tracked down and squashed.  If you use this function,
please update your c-blosc copy.

For more info, please see the release notes in:

https://github.com/Blosc/c-blosc/wiki/Release-notes


What is it?
===========

Blosc (http://www.blosc.org) is a high performance compressor
optimized for binary data.  It has been designed to transmit data to
the processor cache faster than the traditional, non-compressed,
direct memory fetch approach via a memcpy() OS call.

Blosc is the first compressor (that I'm aware of) that is meant not
only to reduce the size of large datasets on-disk or in-memory, but
also to accelerate object manipulations that are memory-bound.

Blosc has a Python wrapper called python-blosc
(https://github.com/Blosc/python-blosc) with a high-performance
interface to NumPy too.  There is also a handy command line for Blosc
called Bloscpack (https://github.com/Blosc/bloscpack) that allows you to
compress large binary datafiles on-disk.


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


.. Local Variables:
.. mode: rst
.. coding: utf-8
.. fill-column: 70
.. End:
