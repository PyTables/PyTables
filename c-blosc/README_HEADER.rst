Blosc Header Format
===================

Blosc (as of Version 1.0.0) has the following 16 byte header that stores
information about the compressed buffer::

    |-0-|-1-|-2-|-3-|-4-|-5-|-6-|-7-|-8-|-9-|-A-|-B-|-C-|-D-|-E-|-F-|
      ^   ^   ^   ^ |     nbytes    |   blocksize   |    ctbytes    |
      |   |   |   |
      |   |   |   +--typesize
      |   |   +------flags
      |   +----------versionlz
      +--------------version

Datatypes of the Header Entries
-------------------------------

All entries are little endian.

:version:
    (``uint8``) Blosc format version.
:versionlz:
    (``uint8``) Version of the internal compressor used.
:flags and compressor enumeration:
    (``bitfield``) The flags of the buffer 

    :bit 0 (``0x01``):
        Whether the shuffle filter has been applied or not.
    :bit 1 (``0x02``):
        Whether the internal buffer is a pure memcpy or not.
    :bit 2 (``0x04``):
        Reserved
    :bit 3 (``0x08``):
        Reserved
    :bit 4 (``0x16``):
        Reserved
    :bit 5 (``0x32``):
        Part of the enumeration for compressors.
    :bit 6 (``0x64``):
        Part of the enumeration for compressors.
    :bit 7 (``0x64``):
        Part of the enumeration for compressors.

    The last three bits form an enumeration that allows to use alternative
    compressors.

    :``0``:
        ``blosclz``
    :``1``:
        ``lz4`` or ``lz4hc``
    :``2``:
        ``snappy``
    :``3``:
        ``zlib``

:typesize:
    (``uint8``) Number of bytes for the atomic type.
:nbytes:
    (``uint32``) Uncompressed size of the buffer.
:blocksize:
    (``uint32``) Size of internal blocks.
:ctbytes:
    (``uint32``) Compressed size of the buffer.

