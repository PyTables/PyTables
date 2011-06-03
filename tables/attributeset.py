########################################################################
#
#       License: BSD
#       Created: May 26, 2003
#       Author:  Francesc Alted - faltet@pytables.com
#
#       $Id$
#
########################################################################

"""Here is defined the AttributeSet class.

See AttributeSet class docstring for more info.

Classes:

    AttributeSet

Functions:

    issysattrname(name)

Misc variables:

    __version__

    SYS_ATTR -- List with attributes considered as read-only
    SYS_ATTR_PREFIXES -- List with prefixes for system attributes

"""

import re
import warnings
import cPickle
import numpy

from tables import hdf5Extension
from tables.utils import SizeType
from tables.registry import classNameDict
from tables.exceptions import ClosedNodeError, PerformanceWarning
from tables.path import checkNameValidity
from tables.undoredo import attrToShadow
from tables.filters import Filters



__version__ = "$Revision$"

# System attributes
SYS_ATTRS = ["CLASS", "VERSION", "TITLE", "NROWS", "EXTDIM",
             "ENCODING", "PYTABLES_FORMAT_VERSION",
             "FLAVOR", "FILTERS", "AUTO_INDEX",
             "DIRTY", "NODE_TYPE", "NODE_TYPE_VERSION",
             "PSEUDOATOM"]
# Prefixes of other system attributes
SYS_ATTRS_PREFIXES = ["FIELD_"]
# RO_ATTRS will be disabled and let the user modify them if they
# want to. The user is still not allowed to remove or rename
# system attributes. Francesc Alted 2004-12-19
# Read-only attributes:
# RO_ATTRS = ["CLASS", "FLAVOR", "VERSION", "NROWS", "EXTDIM",
#             "PYTABLES_FORMAT_VERSION", "FILTERS",
#             "NODE_TYPE", "NODE_TYPE_VERSION"]
#RO_ATTRS = []

# The next attributes are not meant to be copied during a Node copy process
SYS_ATTRS_NOTTOBECOPIED = ["CLASS", "VERSION", "TITLE", "NROWS", "EXTDIM",
                           "PYTABLES_FORMAT_VERSION", "FILTERS", "ENCODING"]
# Attributes forced to be copied during node copies
FORCE_COPY_CLASS = ['CLASS', 'VERSION']
# Regular expression for column default values.
_field_fill_re = re.compile('^FIELD_[0-9]+_FILL$')
# Regular expression for fixing old pickled filters.
_old_filters_re = re.compile(r'\(([ic])tables\.Leaf\n')
# Fixed version of the previous string.
_new_filters_sub = r'(\1tables.filters\n'

def issysattrname(name):
    "Check if a name is a system attribute or not"

    if (name in SYS_ATTRS or
        reduce(lambda x,y: x+y,
               [name.startswith(prefix)
                for prefix in SYS_ATTRS_PREFIXES])):
        return True
    else:
        return False


class AttributeSet(hdf5Extension.AttributeSet, object):
    """
    Container for the HDF5 attributes of a `Node`.

    This class provides methods to create new HDF5 node attributes, and
    to get, rename or delete existing ones.

    Like in `Group` instances, `AttributeSet` instances make use of the
    *natural naming* convention, i.e. you can access the attributes on
    disk as if they were normal Python attributes of the `AttributeSet`
    instance.

    This offers the user a very convenient way to access HDF5 node
    attributes.  However, for this reason and in order not to pollute
    the object namespace, one can not assign *normal* attributes to
    `AttributeSet` instances, and their members use names which start by
    special prefixes as happens with `Group` objects.

    Notes on native and pickled attributes
    --------------------------------------

    The values of most basic types are saved as HDF5 native data in the
    HDF5 file.  This includes Python ``bool``, ``int``, ``float``,
    ``complex`` and ``str`` (but not ``long`` nor ``unicode``) values,
    as well as their NumPy scalar versions and homogeneous or
    *structured* NumPy arrays of them.  When read, these values are
    always loaded as NumPy scalar or array objects, as needed.

    For that reason, attributes in native HDF5 files will be always
    mapped into NumPy objects.  Specifically, a multidimensional
    attribute will be mapped into a multidimensional ``ndarray`` and a
    scalar will be mapped into a NumPy scalar object (for example, a
    scalar ``H5T_NATIVE_LLONG`` will be read and returned as a
    ``numpy.int64`` scalar).

    However, other kinds of values are serialized using ``cPickle``, so
    you only will be able to correctly retrieve them using a
    Python-aware HDF5 library.  Thus, if you want to save Python scalar
    values and make sure you are able to read them with generic HDF5
    tools, you should make use of *scalar or homogeneous/structured
    array NumPy objects* (for example, ``numpy.int64(1)`` or
    ``numpy.array([1, 2, 3], dtype='int16')``).

    One more advice: because of the various potential difficulties in
    restoring a Python object stored in an attribute, you may end up
    getting a ``cPickle`` string where a Python object is expected.  If
    this is the case, you may wish to run ``cPickle.loads()`` on that
    string to get an idea of where things went wrong, as shown in this
    example:

    >>> import os, tempfile
    >>> import tables
    >>>
    >>> class MyClass(object):
    ...   foo = 'bar'
    ...
    >>> myObject = MyClass()  # save object of custom class in HDF5 attr
    >>> h5fname = tempfile.mktemp(suffix='.h5')
    >>> h5f = tables.openFile(h5fname, 'w')
    >>> h5f.root._v_attrs.obj = myObject  # store the object
    >>> print h5f.root._v_attrs.obj.foo  # retrieve it
    bar
    >>> h5f.close()
    >>>
    >>> del MyClass, myObject  # delete class of object and reopen file
    >>> h5f = tables.openFile(h5fname, 'r')
    >>> print repr(h5f.root._v_attrs.obj)  #doctest: +ELLIPSIS
    'ccopy_reg\\n_reconstructor...
    >>> import cPickle  # let's unpickle that to see what went wrong
    >>> cPickle.loads(h5f.root._v_attrs.obj)
    Traceback (most recent call last):
      ...
    AttributeError: 'module' object has no attribute 'MyClass'
    >>> # So the problem was not in the stored object,
    ... # but in the *environment* where it was restored.
    ... h5f.close()
    >>> os.remove(h5fname)

    Public instance variables
    -------------------------

    _v_attrnames
        A list with all attribute names.
    _v_attrnamessys
        A list with system attribute names.
    _v_attrnamesuser
        A list with user attribute names.
    _v_node
        The `Node` instance this attribute set is associated with.
    _v_unimplemented
        A list of attribute names with unimplemented native HDF5 types.

    Public methods
    --------------

    Note that this class overrides the ``__setattr__()``,
    ``__getattr__()`` and ``__delattr__()`` special methods.  This
    allows you to read, assign or delete attributes on disk by just
    using the next constructs::

        leaf.attrs.myattr = 'str attr'    # set a string (native support)
        leaf.attrs.myattr2 = 3            # set an integer (native support)
        leaf.attrs.myattr3 = [3, (1, 2)]  # a generic object (Pickled)
        attrib = leaf.attrs.myattr        # get the attribute ``myattr``
        del leaf.attrs.myattr             # delete the attribute ``myattr``

    If an attribute is set on a target node that already has a large
    number of attributes, a ``PerformanceWarning`` will be issued.

    _f_copy()
        Copy attributes to the ``where`` node.
    _f_list(attrset)
        Get a list of attribute names.
    _f_rename(oldattrname, newattrname)
        Rename an attribute from ``oldattrname`` to ``newattrname``.
    __contains__(name)
        Is there an attribute with that ``name``?
    """

    def _g_getnode(self):
        return self._v__nodeFile._getNode(self._v__nodePath)

    _v_node = property(_g_getnode)


    def __init__(self, node):
        """Create the basic structures to keep the attribute information.

        Reads all the HDF5 attributes (if any) on disk for the node "node".

        node -- The parent node

        """

        # Refuse to create an instance of an already closed node
        if not node._v_isopen:
            raise ClosedNodeError("the node for attribute set is closed")

        mydict = self.__dict__

        self._g_new(node)
        mydict["_v__nodeFile"] = node._v_file
        mydict["_v__nodePath"] = node._v_pathname
        mydict["_v_attrnames"] = self._g_listAttr(node)
        # The list of unimplemented attribute names
        mydict["_v_unimplemented"] = []

        # Get the file version format. This is an optimization
        # in order to avoid accessing it too much.
        try:
            format_version = node._v_file.format_version
        except AttributeError:
            parsed_version = None
        else:
            if format_version == 'unknown':
                parsed_version = None
            else:
                parsed_version = tuple(map(int, format_version.split('.')))
        mydict["_v__format_version"] = parsed_version
        # Split the attribute list in system and user lists
        mydict["_v_attrnamessys"] = []
        mydict["_v_attrnamesuser"] = []
        for attr in self._v_attrnames:
            # put the attributes on the local dictionary to allow
            # tab-completion
            self.__getattr__(attr)
            if issysattrname(attr):
                self._v_attrnamessys.append(attr)
            else:
                self._v_attrnamesuser.append(attr)

        # Sort the attributes
        self._v_attrnames.sort()
        self._v_attrnamessys.sort()
        self._v_attrnamesuser.sort()


    def _g_updateNodeLocation(self, node):
        """Updates the location information about the associated `node`."""

        myDict = self.__dict__
        myDict['_v__nodeFile'] = node._v_file
        myDict['_v__nodePath'] = node._v_pathname
        # hdf5Extension operations:
        self._g_new(node)


    def _f_list(self, attrset='user'):
        """
        Get a list of attribute names.

        The `attrset` string selects the attribute set to be used.  A
        ``'user'`` value returns only user attributes (this is the
        default).  A ``'sys'`` value returns only system attributes.
        Finally, ``'all'`` returns both system and user attributes.
        """

        if attrset == "user":
            return self._v_attrnamesuser[:]
        elif attrset == "sys":
            return self._v_attrnamessys[:]
        elif attrset == "all":
            return self._v_attrnames[:]


    def __getattr__(self, name):
        """Get the attribute named "name"."""

        # If attribute does not exist, raise AttributeError
        if not name in self._v_attrnames:
            raise AttributeError, \
                  "Attribute '%s' does not exist in node: '%s'" % \
                  (name, self._v__nodePath)

        # Read the attribute from disk. This is an optimization to read
        # quickly system attributes that are _string_ values, but it
        # takes care of other types as well as for example NROWS for
        # Tables and EXTDIM for EArrays
        format_version = self._v__format_version
        value = self._g_getAttr(self._v_node, name)

        # Check whether the value is pickled
        # Pickled values always seems to end with a "."
        maybe_pickled = (
            isinstance(value, numpy.generic) and  # NumPy scalar?
            value.dtype.type == numpy.string_ and # string type?
            value.itemsize > 0 and value[-1] == '.' )

        if ( maybe_pickled and value in ["0", "0."] ):
            # Workaround for a bug in many versions of Python (starting
            # somewhere after Python 2.6.1).  See ticket #253.
            retval = value
        elif ( maybe_pickled and _field_fill_re.match(name)
             and format_version == (1, 5) ):
            # This format was used during the first 1.2 releases, just
            # for string defaults.
            try:
                retval = cPickle.loads(value)
                retval = numpy.array(retval)
            except ImportError:
                retval = None  # signal error avoiding exception
        elif maybe_pickled and name == 'FILTERS' and format_version < (2, 0):
            # This is a big hack, but we don't have other way to recognize
            # pickled filters of PyTables 1.x files.
            value = _old_filters_re.sub(_new_filters_sub, value, 1)
            retval = cPickle.loads(value)  # pass unpickling errors through
        elif maybe_pickled:
            try:
                retval = cPickle.loads(value)
            #except cPickle.UnpicklingError:
            # It seems that cPickle may raise other errors than UnpicklingError
            # Perhaps it would be better just an "except:" clause?
            #except (cPickle.UnpicklingError, ImportError):
            # Definitely (see SF bug #1254636)
            except:
                # ivb (2005-09-07): It is too hard to tell
                # whether the unpickling failed
                # because of the string not being a pickle one at all,
                # because of a malformed pickle string,
                # or because of some other problem in object reconstruction,
                # thus making inconvenient even the issuing of a warning here.
                # The documentation contains a note on this issue,
                # explaining how the user can tell where the problem was.
                retval = value
            # Additional check for allowing a workaround for #307
            if type(retval) is unicode and retval == u'':
                retval = numpy.array(retval)[()]
        elif name == 'FILTERS' and format_version >= (2, 0):
            retval = Filters._unpack(value)
        else:
            retval = value

        # Put this value in local directory
        self.__dict__[name] = retval
        return retval


    def _g__setattr(self, name, value):
        """
        Set a PyTables attribute.

        Sets a (maybe new) PyTables attribute with the specified `name`
        and `value`.  If the attribute already exists, it is simply
        replaced.

        It does not log the change.
        """

        # Save this attribute to disk
        # (overwriting an existing one if needed)
        stvalue = value
        if issysattrname(name):
            if name in ["EXTDIM", "AUTO_INDEX", "DIRTY", "NODE_TYPE_VERSION"]:
                stvalue = numpy.array(value, dtype=numpy.int32)
                value = stvalue[()]
            elif name == "NROWS":
                stvalue = numpy.array(value, dtype=SizeType)
                value = stvalue[()]
            elif name == "FILTERS" and self._v__format_version >= (2, 0):
                stvalue = value._pack()
                # value will remain as a Filters instance here
        # Convert value from a Python scalar into a NumPy scalar
        # (only in case it has not been converted yet)
        # Fixes ticket #59
        if (stvalue is value and
            type(value) in (bool, str, int, float, complex, unicode)):
            # Additional check for allowing a workaround for #307
            if type(value) is unicode and value == u'':
                value = numpy.array(value)[()]
            else:
                stvalue = numpy.array(value)
                value = stvalue[()]

        self._g_setAttr(self._v_node, name, stvalue)

        # New attribute or value. Introduce it into the local
        # directory
        self.__dict__[name] = value

        # Finally, add this attribute to the list if not present
        attrnames = self._v_attrnames
        if not name in attrnames:
            attrnames.append(name)
            attrnames.sort()
            if issysattrname(name):
                attrnamessys = self._v_attrnamessys
                attrnamessys.append(name)
                attrnamessys.sort()
            else:
                attrnamesuser = self._v_attrnamesuser
                attrnamesuser.append(name)
                attrnamesuser.sort()


    def __setattr__(self, name, value):
        """
        Set a PyTables attribute.

        Sets a (maybe new) PyTables attribute with the specified `name`
        and `value`.  If the attribute already exists, it is simply
        replaced.

        A ``ValueError`` is raised when the name starts with a reserved
        prefix or contains a ``/``.  A `NaturalNameWarning` is issued if
        the name is not a valid Python identifier.  A
        `PerformanceWarning` is issued when the recommended maximum
        number of attributes in a node is going to be exceeded.
        """

        nodeFile = self._v__nodeFile
        attrnames = self._v_attrnames

        # Check for name validity
        checkNameValidity(name)

        nodeFile._checkWritable()

        # Check if there are too many attributes.
        maxNodeAttrs = nodeFile.params['MAX_NODE_ATTRS']
        if len(attrnames) >= maxNodeAttrs:
            warnings.warn("""\
node ``%s`` is exceeding the recommended maximum number of attributes (%d);\
be ready to see PyTables asking for *lots* of memory and possibly slow I/O"""
                          % (self._v__nodePath, maxNodeAttrs),
                          PerformanceWarning)

        undoEnabled = nodeFile.isUndoEnabled()
        # Log old attribute removal (if any).
        if undoEnabled and (name in attrnames):
            self._g_delAndLog(name)

        # Set the attribute.
        self._g__setattr(name, value)

        # Log new attribute addition.
        if undoEnabled:
            self._g_logAdd(name)


    def _g_logAdd(self, name):
        self._v__nodeFile._log('ADDATTR', self._v__nodePath, name)


    def _g_delAndLog(self, name):
        nodeFile = self._v__nodeFile
        nodePathname = self._v__nodePath
        # Log *before* moving to use the right shadow name.
        nodeFile._log('DELATTR', nodePathname, name)
        attrToShadow(nodeFile, nodePathname, name)


    def _g__delattr(self, name):
        """
        Delete a PyTables attribute.

        Deletes the specified existing PyTables attribute.

        It does not log the change.
        """

        # Delete the attribute from disk
        self._g_remove(self._v_node, name)

        # Delete the attribute from local lists
        self._v_attrnames.remove(name)
        if name in self._v_attrnamessys:
            self._v_attrnamessys.remove(name)
        else:
            self._v_attrnamesuser.remove(name)

        # Delete the attribute from the local directory
        # closes (#1049285)
        del self.__dict__[name]


    def __delattr__(self, name):
        """
        Delete a PyTables attribute.

        Deletes the specified existing PyTables attribute from the
        attribute set.  If a nonexistent or system attribute is
        specified, an ``AttributeError`` is raised.
        """

        nodeFile = self._v__nodeFile

        # Check if attribute exists
        if name not in self._v_attrnames:
            raise AttributeError(
                "Attribute ('%s') does not exist in node '%s'"
                % (name, self._v__nodePath))

        nodeFile._checkWritable()

        # Remove the PyTables attribute or move it to shadow.
        if nodeFile.isUndoEnabled():
            self._g_delAndLog(name)
        else:
            self._g__delattr(name)


    def __getitem__(self, name):
        """The dictionary like interface for __getattr__()."""
        try:
            return self.__getattr__(name)
        except AttributeError:
            # Capture the AttributeError an re-raise a KeyError one
            raise KeyError(
                "Attribute ('%s') does not exist in node '%s'"
                % (name, self._v__nodePath))


    def __setitem__(self, name, value):
        """The dictionary like interface for __setattr__()."""
        self.__setattr__(name, value)


    def __delitem__(self, name):
        """The dictionary like interface for __delattr__()."""
        try:
            self.__delattr__(name)
        except AttributeError:
            # Capture the AttributeError an re-raise a KeyError one
            raise KeyError(
                "Attribute ('%s') does not exist in node '%s'"
                % (name, self._v__nodePath))


    def __contains__(self, name):
        """
        Is there an attribute with that `name`?

        A true value is returned if the attribute set has an attribute
        with the given name, false otherwise.
        """
        return name in self._v_attrnames


    def _f_rename(self, oldattrname, newattrname):
        """Rename an attribute from `oldattrname` to `newattrname`."""

        if oldattrname == newattrname:
            # Do nothing
            return

        # First, fetch the value of the oldattrname
        attrvalue = getattr(self, oldattrname)

        # Now, create the new attribute
        setattr(self, newattrname, attrvalue)

        # Finally, remove the old attribute
        delattr(self, oldattrname)


    def _g_copy(self, newSet, setAttr=None, copyClass=False):
        """
        Copy set attributes.

        Copies all user and allowed system PyTables attributes to the
        given attribute set, replacing the existing ones.

        You can specify a *bound* method of the destination set that
        will be used to set its attributes.  Else, its `_g__setattr`
        method will be used.

        Changes are logged depending on the chosen setting method.  The
        default setting method does not log anything.
        """

        copysysattrs = newSet._v__nodeFile.params['PYTABLES_SYS_ATTRS']
        if setAttr is None:
            setAttr = newSet._g__setattr

        for attrname in self._v_attrnamesuser:
            # Do not copy the unimplemented attributes.
            if attrname not in self._v_unimplemented:
                setAttr(attrname, getattr(self, attrname))
        # Copy the system attributes that we are allowed to.
        if copysysattrs:
            for attrname in self._v_attrnamessys:
                if ((attrname not in SYS_ATTRS_NOTTOBECOPIED) and
                    # Do not copy the FIELD_ atributes in tables as this can
                    # be really *slow* (don't know exactly the reason).
                    # See #304.
                    not attrname.startswith("FIELD_")):
                    setAttr(attrname, getattr(self, attrname))
            # Copy CLASS and VERSION attributes if requested
            if copyClass:
                for attrname in FORCE_COPY_CLASS:
                    if attrname in self._v_attrnamessys:
                        setAttr(attrname, getattr(self, attrname))


    def _f_copy(self, where):
        """
        Copy attributes to the `where` node.

        Copies all user and certain system attributes to the given
        `where` node (a `Node` instance), replacing the existing ones.
        """

        # AttributeSet must be defined in order to define a Node.
        # However, we need to know Node here.
        # Using classNameDict avoids a circular import.
        if not isinstance(where, classNameDict['Node']):
            raise TypeError("destination object is not a node: %r" % (where,))
        self._g_copy(where._v_attrs, where._v_attrs.__setattr__)


    def _g_close(self):
        # Nothing will be done here, as the existing instance is completely
        # operative now.
        pass


    def __str__(self):
        """The string representation for this object."""

        # The pathname
        pathname = self._v__nodePath
        # Get this class name
        classname = self.__class__.__name__
        # The attribute names
        attrnumber = len([ n for n in self._v_attrnames ])
        return "%s._v_attrs (%s), %s attributes" % \
               (pathname, classname, attrnumber)

    def __repr__(self):
        """A detailed string representation for this object."""

        # print additional info only if there are attributes to show
        attrnames = [ n for n in self._v_attrnames ]
        if len(attrnames):
            rep = [ '%s := %r' %  (attr, getattr(self, attr) )
                    for attr in attrnames ]
            attrlist = '[%s]' % (',\n    '.join(rep))

            return "%s:\n   %s" % \
                   (str(self), attrlist)
        else:
            return str(self)


class NotLoggedAttributeSet(AttributeSet):
    def _g_logAdd(self, name):
        pass

    def _g_delAndLog(self, name):
        self._g__delattr(name)



## Local Variables:
## mode: python
## py-indent-offset: 4
## tab-width: 4
## fill-column: 72
## End:
