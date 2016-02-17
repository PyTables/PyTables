from .carray import CArray
from .last_row_array_ext import LastRowArray
from .earray import EArray
from .node import NotLoggedMixin

class LastRowArray(NotLoggedMixin, CArray, LastRowArray):
    """Container for keeping sorted and indices values of last row of an
    index."""

    # Class identifier.
    _c_classid = 'LASTROWARRAY'

