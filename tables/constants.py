########################################################################
#
#       License: BSD
#       Created: February 25, 2005
#       Author:  Ivan Vilata - reverse:com.carabos@ivilata
#
#       $Id$
#
########################################################################

"""
Constant values for PyTables.

Variables:

`MAX_COLUMNS`
    Maximum number of columns in ``Table`` objects before a
    ``PerformanceWarning`` is issued.
`MAX_TREE_DEPTH`
    Maximum depth tree allowed in PyTables.
`MAX_GROUP_WIDTH`
    Maximum allowed number of children hanging from a group.
`MAX_NODE_ATTRS`
    Maximum allowed number of attributes in a node.
`MAX_UNDO_PATH_LENGTH`
    Maximum length of paths allowed in undo/redo operations.
`METADATA_CACHE_SIZE`
    Size (in bytes) of the HDF5 metadata cache.
`NODE_CACHE_SIZE`
    Maximum number of unreferenced to be kept in memory.
`EXPECTED_ROWS_TABLE`
    Default expected number of rows for ``Table`` objects.
`EXPECTED_ROWS_EARRAY`
    Default expected number of rows for ``EArray`` objects.

Misc variables:

`__docformat__`
    The format of documentation strings in this module.
`__version__`
    Repository version of this file.
"""

__docformat__ = 'reStructuredText'
"""The format of documentation strings in this module."""

__version__ = '$Revision$'
"""Repository version of this file."""

# The maximum recommened number of columns in a table.
# However, this limit is somewhat arbitrary and can be increased.
MAX_COLUMNS = 1024
"""Maximum number of columns in ``Table`` objects before a
``PerformanceWarning`` is issued.
"""

# Recommended values for maximum number of groups and maximum depth in tree.
# However, these limits are somewhat arbitrary and can be increased.
MAX_TREE_DEPTH = 2048
"""Maximum depth tree allowed in PyTables."""

MAX_GROUP_WIDTH = 4096
"""Maximum allowed number of children hanging from a group."""

MAX_NODE_ATTRS = 4096
"""Maximum allowed number of attributes in a node."""

MAX_UNDO_PATH_LENGTH = 10240
"""Maximum length of paths allowed in undo/redo operations."""

# Size of cache for new metadata cache system in HDF5 1.8.x
METADATA_CACHE_SIZE = 1*1024*1024  # 1 MB is the default for HDF5
"""Size (in bytes) of the HDF5 metadata cache."""

# NODE_CACHE_SIZE tells the number of nodes than fits in the cache.
#
# There are several forces driving the election of this number:
# 1.- As more nodes, better chances to re-use nodes
#     --> better performance
# 2.- As more nodes, the re-ordering of the LRU cache takes more time
#     --> less performance
# 3.- As more nodes, the memory needs for PyTables grows, specially for table
#     writings (that could take double of memory than table reads!).
#
# Some experiments has been carried out with an AMD Duron processor with
# 256 KB of secondary cache. For processors with more secondary cache
# this can be bettered. Also, if lrucache could be bettered
# (mainly the comparison code), the CPU consumption would be improved.
#
# The next experiment is for browsing a group with 1000 tables.  Look at
# bench/LRU-experiment*.py for the bench code.  In general, retrieving a
# table from LRU cache is almost 20x times faster than re-loading the
# table from disk (0.4ms vs 7.4ms). For arrays, retrieving from cache is
# 2x faster than re-loading from disk (0.4ms vs 1.0ms). These tests has
# been conducted on a Duron platform, but for faster platforms these
# speed-ups will probably increase.
#
# Warning: This cache size must not be lower than the number of indexes on
# every table that the user is dealing with. So keep this 128 or 256 at very
# least.
#
# The default value here is quite conservative. If you have a system
# with tons of memory, and if you are touching regularly a very large
# number of leaves, try increasing this value and see if it fits better for
# you. Please, get me you feedback.
#
# F. Altet 2005-10-31

#NODE_CACHE_SIZE =  1    # 24 MB, 38.6 s
#NODE_CACHE_SIZE =  2    # 24 MB, 38.9 s
#NODE_CACHE_SIZE =  4    # 24 MB, 39.1 s
#NODE_CACHE_SIZE =  8    # 25 MB, 39.2 s
#NODE_CACHE_SIZE = 16    # 26 MB, 39.9 s
#NODE_CACHE_SIZE = 32    # 28 MB, 40.9 s
#NODE_CACHE_SIZE = 64    # 30 MB, 41.1 s
#NODE_CACHE_SIZE = 128   # 35 MB, 41.6 s        , 60 MB for writes (!)
NODE_CACHE_SIZE = 256   # 42 MB, 42.3s, opt:40.9s , 64 MB for writes
                        # This is a good compromise between CPU and memory
                        # consumption.
#NODE_CACHE_SIZE = 512   # 59 MB, 43.9s, opt: 41.8s
#NODE_CACHE_SIZE = 1024  # 52 MB, 85.1s, opt: 17.0s # everything fits on cache!
#NODE_CACHE_SIZE = 2048  # 52 MB, XXXs, opt: 17.0s # everything fits on cache!
################################################################3
# Experiments with a Pentium IV with 512 KB of secondary cache
#NODE_CACHE_SIZE = 1500  # 30.1 s
#NODE_CACHE_SIZE = 1900  # 30.3 s
#NODE_CACHE_SIZE = 2000  # > 200 s
#NODE_CACHE_SIZE = 2046  # Takes lots of time! > 200 s
#NODE_CACHE_SIZE = MAX_GROUP_WIDTH  # that would be ideal, but takes ages!
"""Maximum number of unreferenced to be kept in memory."""

SORTED_CACHE_SIZE = 256
"""The maximum number of rows cached for sorted values in index lookups"""

BOUNDS_CACHE_SIZE = 256
"""The maximum number of rows cached for bounds values in index lookups"""

EXPECTED_ROWS_TABLE = 10000
"""Default expected number of rows for ``Table`` objects."""

EXPECTED_ROWS_EARRAY = 1000
"""Default expected number of rows for ``EArray`` objects."""

#CHUNKTIMES = 10 #Makes large seq writings and reads very fast. (1.28 Mrw/s)
CHUNKTIMES = 50 # Makes large seq writings and reads quite fast. (1.26 Mrw/s)
                 # Acceptable read times for small pieces (2.88 ms, no comp)
                 # --> Seems a good compromise value
#CHUNKTIMES = 100 # Makes large seq writings and reads acceptable (1.20 Mrw/s)
                 # Quite fast read times for small pieces (2.75 ms, no comp)
#CHUNKTIMES = 200 # Makes large seq writings and reads too slow
                 # Quite fast read times for small pieces (2.72 ms, no compr)
"""The ratio buffer_size/chunksize for ``Table`` and ``EArray`` objects."""




## Local Variables:
## mode: python
## py-indent-offset: 4
## tab-width: 4
## fill-column: 72
## End:
