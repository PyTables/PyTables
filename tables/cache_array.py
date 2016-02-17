from .cache_array_ext import CacheArray
from .earray import EArray
from .node import NotLoggedMixin

class CacheArray(NotLoggedMixin, EArray, CacheArray):
    """Container for keeping index caches of 1st and 2nd level."""

    # Class identifier.
    _c_classid = 'CACHEARRAY'

