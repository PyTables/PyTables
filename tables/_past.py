"""A module with no PyTables dependencies that helps with deprecation warnings.
"""
from inspect import getmembers
from warnings import warn

isname = lambda x: x == '__name__'

def previous_api(f, oldname)
    """A decorator-like function for dealing with deprecations."""
    newname = getmembers(f, isname)[0][1]
    warnmsg = "{0}() is pending deprecation, use {1}() instead."
    warnmsg = warnmsg.format(oldname, newname)
    def oldfunc(*args, **kwargs):
        warn(warnmsg, PendingDeprecationWarning)
        return f(*args, **kwargs)
    oldfunc.__doc__ = d.__doc__ + "\n\n.. warning::\n\n    " + warnmsg + '\n'
    return oldfunc
