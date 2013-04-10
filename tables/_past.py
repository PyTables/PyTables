"""A module with no PyTables dependencies that helps with deprecation warnings.
"""
from inspect import getmembers
from warnings import warn

def previous_api(f, oldname):
    """A decorator-like function for dealing with deprecations."""
    for key, value in getmembers(f):
        if key == '__name__':
            newname = value
            break
    warnmsg = "{0}() is pending deprecation, use {1}() instead."
    warnmsg = warnmsg.format(oldname, newname)
    def oldfunc(*args, **kwargs):
        warn(warnmsg, PendingDeprecationWarning)
        return f(*args, **kwargs)
    oldfunc.__doc__ = (f.__doc__ or '') + "\n\n.. warning::\n\n    " + warnmsg + "\n"
    return oldfunc
