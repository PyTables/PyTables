"""A module with no PyTables dependencies that helps with deprecation warnings.
"""
from inspect import getmembers, isclass
from warnings import warn

def previous_api(obj, oldname):
    """A decorator-like function for dealing with deprecations."""
    if isclass(obj):
        # punt if not a function or method
        return obj
    for key, value in getmembers(obj):
        if key == '__name__':
            newname = value
            break
    warnmsg = "{0}() is pending deprecation, use {1}() instead."
    warnmsg = warnmsg.format(oldname, newname)
    def oldfunc(*args, **kwargs):
        warn(warnmsg, PendingDeprecationWarning)
        return obj(*args, **kwargs)
    oldfunc.__doc__ = (obj.__doc__ or '') + "\n\n.. warning::\n\n    " + warnmsg + "\n"
    return oldfunc
