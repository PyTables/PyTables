########################################################################
#
#       License: BSD
#       Created: May 26, 2003
#       Author:  Francesc Alted - falted@openlc.org
#
#       $Source: /home/ivan/_/programari/pytables/svn/cvs/pytables/pytables/tables/AttributeSet.py,v $
#       $Id: AttributeSet.py,v 1.26 2004/02/05 16:23:37 falted Exp $
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
    
    MAX_ATTRS_IN_NODE -- Maximum allowed number of attributes in a node
    SYS_ATTR -- List with attributes considered as read-only
    SYS_ATTR_PREFIXES -- List with prefixes for system attributes

"""

__version__ = "$Revision: 1.26 $"

import warnings, types, cPickle
import hdf5Extension
import Group
import Leaf
from utils import checkNameValidity

# Note: the next constant has to be syncronized with the
# MAX_ATTRS_IN_NODE constant in util.h
MAX_ATTRS_IN_NODE = 4096

# System attributes
SYS_ATTRS = ["CLASS", "VERSION", "TITLE", "NROWS", "EXTDIM",
             "FLAVOR", "ENCODING", "PYTABLES_FORMAT_VERSION",
             "FILTERS"]
# Prefixes of other system attributes
SYS_ATTRS_PREFIXES = ["FIELD_"]
# Read-only attributes:
RO_ATTRS = ["CLASS", "FLAVOR", "VERSION", "NROWS", "EXTDIM",
            "PYTABLES_FORMAT_VERSION", "FILTERS"]

def issysattrname(name):
    "Check if a name is a system attribute or not"
    
    if (name in SYS_ATTRS or
        reduce(lambda x,y: x+y,
               [name.startswith(prefix)
                for prefix in SYS_ATTRS_PREFIXES])):
        return 1
    else:
        return 0


class AttributeSet(hdf5Extension.AttributeSet, object):
    """This is a container for the HDF5 attributes of a node

    It provides methods to get, set and ask for attributes based on
    information extracted from the HDF5 attributes belonging to a
    node.

    Like with Group instances, in AttributeSet instances, a special
    feature called "natural naming" is used, i.e. the names of the
    instance attributes that represent HDF5 attributes are the
    same. This offers the user a very convenient way to access node
    attributes by simply specifying them like a "normal" attribute
    class.

    For this reason and in order to not pollute the object namespace,
    it is explicitely forbidden to assign "normal" attributes to
    AttributeSet instances, and the only ones allowed must start by
    "_c_" (for class variables), "_f_" (for methods), "_g_" (for
    private methods) or "_v_" (for instance variables) prefixes.

    Instance variables:

        _v_node -- The parent node instance
        _v_attrnames -- List with all attribute names
        _v_attrnamessys -- List with system attribute names
        _v_attrnamesuser -- List with user attribute names

    Methods:
    
        _f_list(attrset)
        __getattr__(attrname)
        __setattr__(attrname, attrvalue)
        __delattr__(attrname)
        _f_remove(attrname)
        _f_rename(oldattrname, newattrname)
        _f_close()
        
    """

    def __init__(self, node):
        """Create the basic structures to keep the attribute information.

        Reads all the HDF5 attributes (if any) on disk for the node "node".

        node -- The parent node
        
        """
        self._g_new(node)
        self.__dict__["_v_node"] = node
        self.__dict__["_v_attrnames"] = self._g_listAttr()
        # Split the attribute list in system and user lists
        self.__dict__["_v_attrnamessys"] = []
        self.__dict__["_v_attrnamesuser"] = []
        for attr in self._v_attrnames:
            # New attribute (to allow tab-completion in interactive mode)
            # Beware! From 0.7.1 on, a lazy attribute reading is on.
            # From 0.8 on, attrs is a property, so we can againg
            # put the attributes on the local dictionary to allow
            # tab-completion
            #self.__dict__[attr] = self.__getattr__(attr)
            self.__getattr__(attr)
            if issysattrname(attr):
                self._v_attrnamessys.append(attr)
            else:
                self._v_attrnamesuser.append(attr)

        # Sort the attributes
        self._v_attrnames.sort()
        self._v_attrnamessys.sort()
        self._v_attrnamesuser.sort()

    def _f_list(self, attrset="user"):
        """Return the list of attributes of the parent node

        The parameter attrset the attribute set to be returned. An
        "user" value returns only the user attributes. This is the
        default. "sys" returns only the system (read-only)
        attributes. "readonly" returns the read-only system
        attributes. Finally, "all" returns both the system and user
        attributes.

        """
        
        if attrset == "user":
            return self._v_attrnamesuser
        elif attrset == "sys":
            return self._v_attrnamessys
        elif attrset == "readonly":
            return [ name for name in self._v_attrnamessys
                     if name in RO_ATTRS]
        elif attrset == "all":
            return self._v_attrnames

    def __getattr__(self, name):
        """Get the attribute named "name"."""

        # If attribute does not exist, return None
        if not name in self._v_attrnames:
            return None

        # Read the attribute from disk
        # This is commented out temporarily until decide if it is intereseting
        # or not having system attributes distinct from strings
        # as for example NROWS for Tables and EXTDIM for EArrays
#         if name in self._v_attrnamessys:
#             # _g_getSysAttr works only for string attributes
#             # with length less than 256 bytes
#             # (all read-only atributes *must* follow these rules!)
#             value = self._g_getSysAttr(name)   # Takes only 0.6s/2.9s
#         else:
#             value = self._g_getAttr(name)   # Takes 1.3s/3.7s
        value = self._g_getAttr(name)   # Takes 1.3s/3.7s

        # Check if value is pickled
        # Pickled values always seems to end with a "."
        if type(value) is types.StringType and value and value[-1] == ".":
            try:
                retval = cPickle.loads(value)
            except:
                retval = value
        else:
            retval = value

        # Beware! From 0.7.1 on, a lazy attribute reading is on.
        #self.__dict__[name] = value
        # This is no longuer a lazy attribute
        self.__dict__[name] = retval
        return retval

    def __setattr__(self, name, value):
        """Set new attribute to node.

        name -- The name of the new attribute
        value -- The new attribute value

        A NameError is also raised when the "name" starts by a
        reserved prefix. A SyntaxError is raised if "name" is not a
        valid Python identifier. A RuntimeError is raised if a
        read-only attribute is to be overwritten or if
        MAX_ATTRS_IN_NODE is going to be exceeded.

        """

        # Check for name validity
        checkNameValidity(name)

        # Check that the attribute is not a system one (read-only)
        if name in RO_ATTRS:
            raise RuntimeError, \
                  "Read-only attribute ('%s') cannot be overwritten" % (name)
            
        # Check if we have too much numbers of attributes
        if len(self._v_attrnames) == MAX_ATTRS_IN_NODE:
            warnings.warn( \
"""'%s' node is exceeding the recommended maximum number of attrs (%d).
 Be ready to see PyTables asking for *lots* of memory and possibly slow I/O.
""" % (self.node._v_pathname, MAX_ATTRS_IN_NODE), UserWarning)

        # Save this attribute to disk
        # (overwriting an existing one if needed)
        self._g_setAttr(name, value)
            
        # New attribute. Introduce it into the local directory
        self.__dict__[name] = value

        # Finally, add this attribute to the list if not present
        if not name in self._v_attrnames:
            self._v_attrnames.append(name)
            self._v_attrnamesuser.append(name)
            # Sort the attributes
            self._v_attrnames.sort()
            self._v_attrnamesuser.sort()

    def __delattr__(self, name):
        "Remove the attribute attrname from the attribute set"

        # Check if attribute exists
        if name not in self._v_attrnames:
            raise RuntimeError, \
                  "Attribute ('%s') does not exist in node '%s'" % \
                  (name, self._v_node._v_name)

        # The system attributes are protected
        if name in RO_ATTRS:
            raise RuntimeError, \
                  "Read-only attribute ('%s') cannot be deleted" % (name)

        # Delete the attribute from disk
        self._g_remove(name)

        # Delete the attribute from local lists
        self._v_attrnames.remove(name)
        self._v_attrnamesuser.remove(name)

    def _f_rename(self, oldattrname, newattrname):
        "Rename an attribute"

        if oldattrname == newattrname:
            # Do nothing
            return
        
        # if oldattrname or newattrname are system attributes, raise an error
        for name in [oldattrname, newattrname]:
            if name in RO_ATTRS:
                raise RuntimeError, \
            "Read-only attribute ('%s') cannot be renamed" % (name)

        # First, fetch the value of the oldattrname
        attrvalue = getattr(self, oldattrname)

        # Now, create the new attribute
        setattr(self, newattrname, attrvalue)

        # Finally, remove the old attribute
        delattr(self, oldattrname)

    def _f_copy(self, where):
        "Copy the user attributes to 'where' object"
        assert (isinstance(where, Group.Group) or
                isinstance(where, Leaf.Leaf)), \
                "The where has to be a Group or Leaf instance"
        if isinstance(where, Group.Group):
            dstAttrs = where._v_attrs
        else:
            dstAttrs = where.attrs
        for attrname in self._v_attrnamesuser:
            setattr(dstAttrs, attrname, getattr(self, attrname))

    def _f_close(self):
        "Delete some back-references"
        del self.__dict__["_v_node"]
        # After the objects are disconnected, destroy the
        # object dictionary using the brute force ;-)
        # This should help to the garbage collector
        #self.__dict__.clear()

        pass

    def __str__(self):
        """The string representation for this object."""
        # Get the associated filename
        filename = self._v_node._v_rootgroup._v_filename
        # The pathname
        pathname = self._v_node._v_pathname
        # Get this class name
        classname = self.__class__.__name__
        # The attrribute names
        attrnumber = len(self._v_attrnames)
        return "%s (%s), %s attributes" % (pathname, classname, attrnumber)

    def __repr__(self):
        """A detailed string representation for this object."""

        # print additional info only if there are attributes to show
        if len(self._v_attrnames):
            rep = [ '%s := %r' %  (attr, getattr(self, attr) )
                    for attr in self._v_attrnames ]
            attrlist = '[%s]' % (',\n    '.join(rep))
        
            return "%s:\n   %s" % \
                   (str(self), attrlist)
        else:
            return str(self)
               
