########################################################################
#
#       License: BSD
#       Created: May 26, 2003
#       Author:  Francesc Alted - falted@openlc.org
#
#       $Source: /home/ivan/_/programari/pytables/svn/cvs/pytables/pytables/tables/AttributeSet.py,v $
#       $Id: AttributeSet.py,v 1.4 2003/06/04 18:25:39 falted Exp $
#
########################################################################

"""Here is defined the AttributeSet class.

See AttributeSet class docstring for more info.

Classes:

    AttributeSet

Functions:


Misc variables:

    __version__
    

    MAX_ATTRS_IN_NODE -- Maximum allowed number of attributes in a node

"""

__version__ = "$Revision: 1.4 $"

import warnings, types
import hdf5Extension
import Group
from utils import checkNameValidity

# Note: the next constant has to be syncronized with the
# MAX_ATTRS_IN_NODE constant in util.h!
MAX_ATTRS_IN_NODE = 4096

# System attributes (read only)
SYS_ATTR = ["CLASS", "FLAVOR", "VERSION", "PYTABLES_FORMAT_VERSION", "TITLE"]
# Prefixes of other system attributes
SYS_ATTR_PREFIXES = ["FIELD_"]

def issysattrname(name):
    "Check if a name is a system attribute or not"
    
    if (name in SYS_ATTR or
        reduce(lambda x,y: x+y,
               [name.startswith(prefix)
                for prefix in SYS_ATTR_PREFIXES])):
        return 1
    else:
        return 0


class AttributeSet(hdf5Extension.AttributeSet, object):
    """This is a container for the HDF5 attributes of a node

    It provides methods to get, set and ask for attributes based on
    information extracted from the HDF5 attributes belonging to a
    node.

    In AttributeSet instances, a special feature called "natural
    naming" is used, i.e. the names of the instance attributes that
    represent HDF5 attributes are the same. This offers the user a
    very convenient way to access node attributes by simply specifying
    them like a "normal" attribute class.

    For this reason and in order to not pollute the object namespace,
    it is explicitely forbidden to assign "normal" attributes to
    AttributeSet instances, and the only ones allowed must start by
    "_c_" (for class variables), "_f_" (for methods), "_g_" (for
    private methods) or "_v_" (for instance variables) prefixes.

    Methods:
    
        _f_listAttrs()
        __getattr__(attrname)
        __setattr__(attrname, attrvalue)
        __delattr__(attrname)
        _f_remove(attrname)
        _f_rename(oldattrname, newattrname)
        _f_close()
        
    Instance variables:

        attrname -- The name of an attribute in Python namespace
        _v_node -- The parent node instance
        _v_attrnames -- List with all attribute names
        _v_attrnamessys -- List with system attribute names
        _v_attrnamesuser -- List with user attribute names

    """

    def __init__(self, node):
        """Create the basic structures to keep the attribute information.

        node -- The node that contains this attributes
        
        """
        self._g_new(node)
        self.__dict__["_v_node"] = node
        self.__dict__["_v_attrnames"] = list(self._g_listAttr())
        # Split the attribute list in system and user lists
        self.__dict__["_v_attrnamessys"] = []
        self.__dict__["_v_attrnamesuser"] = []
        for attr in self._v_attrnames:
            if issysattrname(attr):
                self._v_attrnamessys.append(attr)
            else:
                self._v_attrnamesuser.append(attr)

        # Sort the attributes
        self._v_attrnames.sort()
        self._v_attrnamessys.sort()
        self._v_attrnamesuser.sort()

    def _f_listAttrs(self, set="user"):
        "Return the list of attributes of the parent node"
        
        if set == "user":
            return self._v_attrnamesuser
        elif set == "sys":
            return self._v_attrnamessys
        elif set == "all":
            return self._v_attrnames

    def __getattr__(self, name):
        """Get the HDF5 attribute named "name"."""

        # If attribute does not exists, return None
        if not name in self._v_attrnames:
            return None

        return self._g_getAttrStr(name)

    def __setattr__(self, name, value):
        """Attach new nodes to the tree.

        name -- The name of the new attribute
        value -- The new attribute value

        A NameError is also raised when the "name" starts by a
        reserved prefix. A SyntaxError is raised if "name" is not a
        valid Python identifier.

        """

        # Check for name validity
        checkNameValidity(name)

        if type(value) <> types.StringType:
            raise ValueError, \
"""Only string values are supported as attributes right now"""
        
        # Check that the attribute is not a system one (read-only)
        if issysattrname(name):
            raise RuntimeError, \
                  "System attribute ('%s') cannot be overwritten" % (name)
            
        # Check if we have too much numbers of attributes
        if len(self._v_attrnames) < MAX_ATTRS_IN_NODE:
            self._g_setAttrStr(name, value)
        else:
            raise RuntimeError, \
               "'%s' node has exceeded the maximum number of attrs (%d)" % \
               (self._v_node._v_pathname, MAX_ATTRS_IN_NODE)

        # Finally, add this attribute to the list if not present
        if not name in self._v_attrnames:
            self._v_attrnames.append(name)
            self._v_attrnamesuser.append(name)

        # Sort the attributes
        self._v_attrnames.sort()
        self._v_attrnamesuser.sort()

    def __delattr__(self, name):
        "Remove the attribute attrname from the attribute set"
        self._f_remove(name)

    def _f_remove(self, attrname):
        "Remove the attribute attrname from the attribute set"

        # Check if attribute exists
        if attrname not in self._v_attrnames:
            raise RuntimeError, \
                  "Attribute ('%s') does not exist in node '%s'" % \
                  (name, self._v_node._v_name)

        # The system attributes are protected
        if attrname in self._v_attrnamessys:
            raise RuntimeError, \
                  "System attribute ('%s') cannot be overwritten" % (attrname)

        # Delete the attribute from disk
        self._g_remove(attrname)

        # Delete the attribute from local lists
        self._v_attrnames.remove(attrname)
        self._v_attrnamesuser.remove(attrname)

    def _f_rename(self, oldattrname, newattrname):
        "Rename an attribute"

        if oldattrname == newattrname:
            # Do nothing
            return
        
        # if oldattrname or newattrname are system attributes, raise an error
        for name in [oldattrname, newattrname]:
            if issysattrname(name):
                raise RuntimeError, \
            "System attribute ('%s') cannot be overwritten or renamed" % (name)


        # First, fetch the value of the oldattrname
        attrvalue = getattr(self, oldattrname)

        # Now, create the new attribute
        setattr(self, newattrname, attrvalue)

        # Finally, remove the old attribute
        self._f_remove(oldattrname)

    def _f_close():
        "Delete all the local variables in self to free memory"

        del self._v_node
        del self._v_attrnames
        del self._v_attrnamesuser
        del self._v_attrnamessys

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
        return "%s (%s): %s attributes" % (pathname, classname, attrnumber)

    def __repr__(self):
        """A detailed string representation for this object."""
        #return str(self)
        
        rep = [ '%s := %r' %  (attr, getattr(self, attr) )
                for attr in self._v_attrnames ]
        attrlist = '[%s]' % (',\n    '.join(rep))
        
        return "%s\n   %s" % \
               (str(self), attrlist)
               
