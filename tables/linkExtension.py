from warnings import warn
from tables.linkextension import *

_warnmsg = ("linkextension is pending deprecation, import linextension instead. "
    "You may use the pt2to3 tool to update your source code.")
warn(_warnmsg, PendingDeprecationWarning, stacklevel=2)







