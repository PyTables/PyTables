from warnings import warn
from tables.tableextension import *

_warnmsg = ("tableextension is pending deprecation, import tableextension instead. "
    "You may use the pt2to3 tool to update your source code.")
warn(_warnmsg, PendingDeprecationWarning, stacklevel=2)







