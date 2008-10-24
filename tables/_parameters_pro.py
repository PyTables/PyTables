########################################################################
#
#       License: See http://www.pytables.org/moin/PyTablesProPricing
#       Created: February 12, 2006
#       Author:  Ivan Vilata - reverse:net.selidor@ivan
#
#       $Id$
#
########################################################################

"""Parameters for PyTables (index)."""

from tables._parameters_common import _KB, _MB


__docformat__ = 'reStructuredText'
"""The format of documentation strings in this module."""

__version__ = '$Revision$'
"""Repository version of this file."""


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




## Local Variables:
## mode: python
## py-indent-offset: 4
## tab-width: 4
## fill-column: 72
## End:
