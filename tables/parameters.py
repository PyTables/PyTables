########################################################################
#
#       License: BSD
#       Created: February 25, 2005
#       Author:  Ivan Vilata - reverse:net.selidor@ivan
#
#       $Id$
#
########################################################################

"""
Parameters for PyTables.

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

_KB = 1024
"""The size of a Kilobyte in bytes"""

_MB = 1024*_KB
"""The size of a Megabyte in bytes"""

# Tunable parameters
# ==================
# Be careful when touching these!

# Parameters for different internal caches
# ----------------------------------------

BOUNDS_MAX_SIZE = 1*_MB
"""The maximum size for bounds values cached during index lookups."""

BOUNDS_MAX_SLOTS = 4*1024
"""The maximum number of slots for the BOUNDS cache."""

ITERSEQ_MAX_ELEMENTS = 1024
"""The maximum number of iterator elements cached in data lookups."""

ITERSEQ_MAX_SIZE = 1*_MB
"""The maximum space that will take ITERSEQ cache (in bytes)."""

ITERSEQ_MAX_SLOTS = 128
"""The maximum number of slots in ITERSEQ cache."""

LIMBOUNDS_MAX_SIZE = 256*_KB
"""The maximum size for the query limits (for example, ``(lim1, lim2)``
in conditions like ``lim1 <= col < lim2``) cached during index lookups
(in bytes)."""

LIMBOUNDS_MAX_SLOTS = 128
"""The maximum number of slots for LIMBOUNDS cache."""

TABLE_MAX_SIZE = 1*_MB
"""The maximum size for table chunks cached during index queries."""

SORTED_MAX_SIZE = 1*_MB
"""The maximum size for sorted values cached during index lookups."""

SORTEDLR_MAX_SIZE = 8*_MB
"""The maximum size for chunks in last row cached in index lookups (in
bytes)."""

SORTEDLR_MAX_SLOTS = 1024
"""The maximum number of chunks for SORTEDLR cache."""


# Parameters for general cache behaviour
# --------------------------------------
#
# The next parameters will not be effective if passed to the
# `openFile()` function (so, they can only be changed in a *global*
# way).  You can change them in the file, but this is strongly
# discouraged unless you know well what you are doing.

DISABLE_EVERY_CYCLES = 10
"""The number of cycles in which a cache will be forced to be disabled
if the hit ratio is lower than the LOWEST_HIT_RATIO (see below).  This
value should provide time enough to check whether the cache is being
efficient or not."""

ENABLE_EVERY_CYCLES = 50
"""The number of cycles in which a cache will be forced to be
(re-)enabled, irregardingly of the hit ratio. This will provide a chance
for checking if we are in a better scenario for doing caching again."""

LOWEST_HIT_RATIO = 0.6
"""The minimum acceptable hit ratio for a cache to avoid disabling (and
freeing) it."""


# Tunable parameters
# ==================
# Be careful when touching these!

# Recommended maximum values
# --------------------------

# Following are the recommended values for several limits.  However,
# these limits are somewhat arbitrary and can be increased if you have
# enough resources.

MAX_COLUMNS = 512
"""Maximum number of columns in ``Table`` objects before a
``PerformanceWarning`` is issued.  This limit is somewhat arbitrary and
can be increased.
"""

MAX_NODE_ATTRS = 4096
"""Maximum allowed number of attributes in a node."""

MAX_GROUP_WIDTH = 16384
"""Maximum allowed number of children hanging from a group."""

MAX_TREE_DEPTH = 2048
"""Maximum depth in object tree allowed."""

MAX_UNDO_PATH_LENGTH = 10240
"""Maximum length of paths allowed in undo/redo operations."""


# Cache limits
# ------------

COND_CACHE_SLOTS = 128
"""Maximum number of conditions for table queries to be kept in memory.
"""

CHUNK_CACHE_NELMTS = 521
"""Number of elements for HDF5 chunk cache."""

CHUNK_CACHE_PREEMPT = 0.0
"""Chunk preemption policy.  This value should be between 0 and 1
inclusive and indicates how much chunks that have been fully read are
favored for preemption. A value of zero means fully read chunks are
treated no differently than other chunks (the preemption is strictly
LRU) while a value of one means fully read chunks are always preempted
before other chunks."""

CHUNK_CACHE_SIZE = 2*_MB
"""Size (in bytes) for HDF5 chunk cache."""

# Size for new metadata cache system in HDF5 1.8.x
METADATA_CACHE_SIZE = 1*_MB  # 1 MB is the default for HDF5
"""Size (in bytes) of the HDF5 metadata cache.  This only takes effect
if using HDF5 1.8.x series."""


# NODE_CACHE_SLOTS tells the number of nodes that fits in the cache.
#
# There are several forces driving the election of this number:
# 1.- As more nodes, better chances to re-use nodes
#     --> better performance
# 2.- As more nodes, the re-ordering of the LRU cache takes more time
#     --> less performance
# 3.- As more nodes, the memory needs for PyTables grows, specially for table
#     writings (that could take double of memory than table reads!).
#
# The default value here is quite conservative. If you have a system
# with tons of memory, and if you are touching regularly a very large
# number of leaves, try increasing this value and see if it fits better
# for you. Please report back your feedback.
NODE_CACHE_SLOTS = 64
"""Maximum number of unreferenced nodes to be kept in memory.

If positive, this is the number of *unreferenced* nodes to be kept in
the metadata cache. Least recently used nodes are unloaded from memory
when this number of loaded nodes is reached. To load a node again,
simply access it as usual. Nodes referenced by user variables are not
taken into account nor unloaded.

Negative value means that all the touched nodes will be kept in an
internal dictionary.  This is the faster way to load/retrieve nodes.
However, and in order to avoid a large memory comsumption, the user will
be warned when the number of loaded nodes will reach the
``-NODE_CACHE_SLOTS`` value.

Finally, a value of zero means that any cache mechanism is disabled.
"""


# Parameters for the I/O buffer in `Leaf` objects
# -----------------------------------------------

IO_BUFFER_SIZE = 1*_MB
"""The PyTables internal buffer size for I/O purposes.  Should not
exceed the amount of highest level cache size in your CPU."""

BUFFER_TIMES = 100
"""The maximum buffersize/rowsize ratio before issuing a
``PerformanceWarning``."""


# Miscellaneous
# -------------

EXPECTED_ROWS_EARRAY = 1000
"""Default expected number of rows for ``EArray`` objects."""

EXPECTED_ROWS_TABLE = 10000
"""Default expected number of rows for ``Table`` objects."""

PYTABLES_SYS_ATTRS = True
"""Set this to ``False`` if you don't want to create PyTables system
attributes in datasets.  Also, if set to ``False`` the possible existing
system attributes are not considered for guessing the class of the node
during its loading from disk (this work is delegated to the PyTables'
class discoverer function for general HDF5 files)."""

MAX_THREADS = None
"""The maximum number of threads that PyTables should use internally
(mainly in Blosc and Numexpr currently).  If `None`, it is automatically
set to the number of cores in your machine. In general, it is a good
idea to set this to the number of cores in your machine or, when your
machine has many of them (e.g. > 4), perhaps one less than this."""


## Local Variables:
## mode: python
## py-indent-offset: 4
## tab-width: 4
## fill-column: 72
## End:
