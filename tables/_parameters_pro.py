########################################################################
#
#       License: XXX
#       Created: February 12, 2006
#       Author:  Ivan Vilata - reverse:com.carabos@ivilata
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


# Mutable parameters
# ==================
# Be careful when touching these!

LIMDATA_MAX_SLOTS = 128
"""The maximum number of limits (lim1 <= col < lim2) cached in data lookups."""

LIMDATA_MAX_SIZE = 256*_KB
"""The maximum space that will take LIMDATA cache (in bytes)."""

LIMBOUNDS_MAX_SLOTS = 128
"""The maximum number of limits (lim1 <= col < lim2) cached in index lookups."""

LIMBOUNDS_MAX_SIZE = 256*_KB
"""The maximum space that will take LIMBOUNDS cache (in bytes)."""

BOUNDS_MAX_SLOTS = 4*1024
"""The maximum number of boundrows cached during index lookups."""

BOUNDS_MAX_SIZE = 1*_MB
"""The maximum size for bounds values cached during index lookups."""

SORTED_MAX_SIZE = 1*_MB
"""The maximum size for sorted values cached during index lookups."""

INDICES_MAX_SIZE = 1*_MB
"""The maximum size for indices values cached during index lookups."""

ENABLE_EVERY_CYCLES = 50
"""The number of cycles that the several LRU caches for data (not nodes)
will be forced to be (re-)enabled, irregardingly of the hit ratio. This
will provide a chance for checking if we are in a better scenario for
doing caching again."""

LOWEST_HIT_RATIO = 0.6
"""The minimum acceptable hit ratio for the several LRU caches for data
(not nodes) to avoid disabling the cache."""

## Local Variables:
## mode: python
## py-indent-offset: 4
## tab-width: 4
## fill-column: 72
## End:
