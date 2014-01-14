from warnings import warn
from tables.indexesextension import *

_warnmsg = ("indexesExtension is pending deprecation, import indexesextension instead. "
            "You may use the pt2to3 tool to update your source code.")
warn(_warnmsg, DeprecationWarning, stacklevel=2)
