from warnings import warn
from tables.utilsextension import *

_warnmsg = ("utilsExtension is pending deprecation, import utilsextension instead. "
            "You may use the pt2to3 tool to update your source code.")
warn(_warnmsg, DeprecationWarning, stacklevel=2)
