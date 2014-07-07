===============================================================
 Announcing c-blosc 1.4.0
 A blocking, shuffling and lossless compression library
===============================================================

What is new?
============

Support for non-Intel and non-SSE2 architectures has been added.  In
particular, c-blosc has been tested in a Raspberry Pi (ARM) and
everything seems to go smoothly, even when the kernel was configured
to crash with a SIGBUS (echo 4 > /proc/cpu/alignment) in case of an
unaligned access.

Architectures requiring strict access alignment are supported as well.
Due to this, arquitectures with a high penalty in accessing unaligned
data (e.g. Raspberry Pi, ARMv6) can compress up to 2.5x faster.

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
