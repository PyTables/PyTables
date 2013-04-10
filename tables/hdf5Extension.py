from warnings import warn
from tables.hdf5extension import *

_warnmsg = ("hdf5extension is pending deprecation, import hdf5extension instead. "
    "You may use the pt2to3 tool to update your source code.")
warn(_warnmsg, PendingDeprecationWarning)







