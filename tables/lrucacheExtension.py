from warnings import warn
from tables.lrucacheextension import *

_warnmsg = ("lrucacheextension is pending deprecation, import lrucacheextension instead. "
    "You may use the pt2to3 tool to update your source code.")
warn(_warnmsg, PendingDeprecationWarning, stacklevel=2)







