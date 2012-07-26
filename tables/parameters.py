########################################################################
#
#       License: BSD
#       Created: February 25, 2005
#       Author:  Ivan Vilata - reverse:net.selidor@ivan
#
#       $Id$
#
########################################################################

"""Parameters for PyTables."""

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

# Size for new metadata cache system
METADATA_CACHE_SIZE = 1*_MB  # 1 MB is the default for HDF5
"""Size (in bytes) of the HDF5 metadata cache."""


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
machine has many of them (e.g. > 4), perhaps one less than this.

.. note::

    currently MAX_THREADS is only used as a fall-back if
    :data:`tables.parameters.MAX_NUMEXPR_THREADS` or
    :data:`tables.parameters.MAX_BLOSC_THREADS` are not set.

.. deprecated:: 2.4

    Use :data:`tables.parameters.MAX_NUMEXPR_THREADS` or
    :data:`tables.parameters.MAX_BLOSC_THREADS` instead.

"""

MAX_NUMEXPR_THREADS = None
"""The maximum number of threads that PyTables should use internally in
Numexpr.  If `None`, it is automatically set to the number of cores in
your machine. In general, it is a good idea to set this to the number of
cores in your machine or, when your machine has many of them (e.g. > 4),
perhaps one less than this."""

MAX_BLOSC_THREADS = None
"""The maximum number of threads that PyTables should use internally in
Blosc.  If `None`, it is automatically set to the number of cores in
your machine. In general, it is a good idea to set this to the number of
cores in your machine or, when your machine has many of them (e.g. > 4),
perhaps one less than this."""

DRIVER = None
"""The libhdf5 driver that should be used for reading/writing to the file.
Following drivers are implemented:
	H5FD_SEC2 - This is the default driver which uses Posix file-system functions like read and write to perform I/O to a single file.
	H5FD_STDIO - This driver uses functions from 'stdio.h' to perform buffered I/O to a single file.
	H5FD_CORE - This driver performs I/O directly to memory and can be used to create small temporary files that never exist on permanent storage. If the provided filename is set to None, an in memory only file will be created. Additionaly if filename is set to None and MEMORY_IMAGE parameter is provided, the HDF5 will be read from the data provided in the MEMORY_IMAGE.	
	H5FD_CORE_INMEMORY - This is a pseudodriver. It makes it possible to read and write in-memory only images of HDF5 files. 
	H5FD_DIRECT - With this driver, data is written to or read from the file synchronously without being cached by the system.
The following drivers may be implemented in the future.
	H5FD_MPIIO - This driver is used with Parallel HDF5, and is only pre-defined if the library is compiled with parallel I/O support. Refer to the Parallel HDF5 Tutorial for more information on using Parallel HDF5.
	H5FD_MULTI- This driver enables different types of HDF5 data and metadata to be written to separate files. The H5FD_SPLIT driver is an example of what the H5FD_MULTI driver can do.
	H5FD_FAMILY - This driver partitions a large format address space into smaller chunks (separate storage of a user's choice).
	H5FD_SPLIT - This driver splits the meta data and raw data into separate storage of a user's choice.
see http://www.hdfgroup.org/HDF5/Tutor/filedrvr.html for more information
"""

H5FD_CORE_INMEMORY_IMAGE = None
"""Passing an memory image as a single dimensional numpy array will force
in-memory mode."""

H5FD_CORE_INCREMENT=65536
H5FD_CORE_BACKING_STORE=1


## Local Variables:
## mode: python
## py-indent-offset: 4
## tab-width: 4
## fill-column: 72
## End:
