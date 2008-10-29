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

from tables._parameters_common import _KB, _MB

try:
    from tables._parameters_pro import *
except ImportError:
    pass


__docformat__ = 'reStructuredText'
"""The format of documentation strings in this module."""

__version__ = '$Revision$'
"""Repository version of this file."""


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

MAX_GROUP_WIDTH = 4096
"""Maximum allowed number of children hanging from a group."""

MAX_TREE_DEPTH = 2048
"""Maximum depth in object tree allowed."""

MAX_UNDO_PATH_LENGTH = 10240
"""Maximum length of paths allowed in undo/redo operations."""


# Cache limits
# ------------

# Size of cache for new metadata cache system in HDF5 1.8.x
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


# Parameters for the I/O buffer in `Table` objects
# ------------------------------------------------

#CHUNKTIMES = 4  # Makes large seq writings and reads quite fast (4.96 Mrw/s)
CHUNKTIMES = 8   # Makes large seq writings and reads very fast (5.30 Mrw/s)
#CHUNKTIMES = 16   # Makes large seq writings and reads very fast (5.30 Mrw/s)
"""The buffersize/chunksize ratio."""

BUFFERTIMES = 100
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


## Local Variables:
## mode: python
## py-indent-offset: 4
## tab-width: 4
## fill-column: 72
## End:
