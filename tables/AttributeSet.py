########################################################################
#
#       License: BSD
#       Created: May 26, 2003
#       Author:  Francesc Alted - falted@openlc.org
#
#       $Source: /home/ivan/_/programari/pytables/svn/cvs/pytables/pytables/tables/AttributeSet.py,v $
#       $Id: AttributeSet.py,v 1.1 2003/06/02 14:24:19 falted Exp $
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

__version__ = "$Revision: 1.1 $"

# Note: the next constant has to be syncronized with the
# MAX_ATTRS_IN_NODE constant in util.h!
MAX_ATTRS_IN_NODE = 4096

# System attributes (read only)
SYS_ATTR = ["CLASS", "FLAVOR", "VERSION", "PYTABLES_FORMAT_VERSION", "TITLE"]
# Prefixes of other system attributes
SYS_ATTR_PREFIXES = ["FIELD_"]

import warnings, types
import hdf5Extension
#import Group
from utils import checkNameValidity

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
        _f_rename(newname)
        _f_remove()
        __getattr__(attrname)
        __getAttr__(attrname, attrvalue)
        _f_close()
        
    Instance variables:

        attrname -- The name of an attribute in Python namespace
        _v_attrname_hdf5name -- The name of this attrname in HDF5 file namespace
        _v_node -- The parent node instance
        _v_attrnames -- List with attribute names

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
            if attr in SYS_ATTR:
                self._v_attrnamessys.append(attr)
            elif reduce(lambda x,y: x+y,
                        [attr.startswith(prefix) for prefix in SYS_ATTR_PREFIXES]):
                self._v_attrnamessys.append(attr)
            else:
                self._v_attrnamesuser.append(attr)

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

        #print self._v_node._v_name
        #if self._v_node._v_name != "/":
        if 0:
            print "Getting the", name, "attribute in node", self._v_node._v_pathname
            print self._v_attrnames

        if not name in self._v_attrnames:
            # raise LookupError, "'%s' node has not a \"%s\" attribute!" % \
            #                      (self._v_node._v_pathname, name)
            # Do not issue an exception here
            return None

        if hasattr(self._v_node, "_f_getAttr"):
            return self._g_getGroupAttrStr(name)
        else:
            return self._g_getLeafAttrStr(self._v_node._v_hdf5name, name)

    def __setattr__(self, name, value):
        """Attach new nodes to the tree.

        name -- The name of the new attribute
        value -- The new attribute value

        A NameError is also raised when the "name" starts by a
        reserved prefix. A SyntaxError is raised if "name" is not a
        valid Python identifier.

        """

        #print "Setting the", name, "attribute in node", self._v_node._v_pathname
        #print self._v_attrnames
        # Check for name validity
        checkNameValidity(name)

        if type(value) <> types.StringType:
            raise ValueError, \
"""Only string values are supported as attributes right now"""
        
        # Check that the attribute is not a system one (read-only)
        if name in self._v_attrnamessys:
            raise RuntimeError, \
               "System attribute ('%s') cannot be overwritten" % (name)
            
        # Check if we have too much numbers of attributes
        if len(self._v_attrnames) < MAX_ATTRS_IN_NODE:
            #if isinstance(self._v_node, Group.Group):
            if hasattr(self._v_node, "_f_setAttr"):
                #self._v_node._f_setAttr(name, value)
                self._g_setGroupAttrStr(name, value)
            else:
                self._g_setLeafAttrStr(self._v_node._v_hdf5name, name, value)

        else:
            raise RuntimeError, \
               "'%s' node has exceeded the maximum number of attrs (%d)" % \
               (self._v_node._v_pathname, MAX_ATTRS_IN_NODE)

        # Finally, add this attribute to the list if not present
        if not name in self._v_attrnames:
            self._v_attrnames.append(name)
            self._v_attrnamesuser.append(name)

    def __str__(self):
        """The string representation for this object."""
        # Get the associated filename
        filename = self._v_node._v_rootgroup._v_filename
        # The pathname
        pathname = self._v_node._v_pathname
        # Get this class name
        classname = self.__class__.__name__
        # The attrribute names
        attrnames = self._v_attrnames
        return "%s (%s) %r" % (pathname, classname, attrnames)

    def __repr__(self):
        """A detailed string representation for this object."""
        return str(self)
        
        rep = [ '%r (%s)' %  \
                (childname, child._v_class) 
                for (childname, child) in self._v_childs.items() ]
        childlist = '[%s]' % (', '.join(rep))
        
        return "%s\n  childs := %s" % \
               (str(self), childlist)
               
