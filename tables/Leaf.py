########################################################################
#
#       License: BSD
#       Created: October 14, 2002
#       Author:  Francesc Alted - falted@openlc.org
#
#       $Source: /home/ivan/_/programari/pytables/svn/cvs/pytables/pytables/tables/Leaf.py,v $
#       $Id: Leaf.py,v 1.12 2003/03/08 11:40:54 falted Exp $
#
########################################################################

"""Here is defined the Leaf class.

See Leaf class docstring for more info.

Classes:

    Leaf

Functions:


Misc variables:

    __version__


"""

__version__ = "$Revision: 1.12 $"

import types
from utils import checkNameValidity

class Leaf:
    
    """A class to place common functionality of all Leaf objects.

    A Leaf object is all the nodes that can hang directly from a
    Group, but that are not groups nor attributes. Right now this set
    is composed by Table and Array objects.

    Leaf objects (like Table or Array) will inherit these methods
    using the mix-in technique.

    Methods:

        _f_getAttr(attrname)
        _f_setAttr(attrname, attrvalue)
        _f_rename(newname)

    Instance variables:

        name -- the Leaf node name

    """

    
    def _g_putObjectInTree(self, name, parent):
        
        """Given a new Leaf object (fresh or in a HDF5 file), set
        links and attributes to include it in the object tree."""
        
        # New attributes for the this Leaf instance
        parent._g_setproperties(name, self)
        self.name = self._v_name     # This is a standard attribute for Leaves
        # Call the new method in Leaf superclass 
        self._g_new(parent, self._v_hdf5name)
        # Update this instance attributes
        parent._v_objleaves[self._v_name] = self
        # Update class variables
        parent._c_objleaves[self._v_pathname] = self
        self._v_groupId = parent._v_groupId
        if self._v_new:
            self._create()
        else:
            self._open()

    def _g_renameObject(self, newname):
        """Rename this leaf in the object tree as well as in the HDF5 file."""

        parent = self._v_parent
        newattr = self.__dict__

        # Delete references to the oldname
        del parent._c_objleaves[self._v_pathname]
        del parent._c_objects[self._v_pathname]
        del parent._v_objleaves[self._v_name]
        del parent._v_objchilds[self._v_name]
        del parent.__dict__[self._v_name]

        # Get the alternate name (if any)
        trTable = self._v_rootgroup._v_parent.trTable
        # New attributes for the this Leaf instance
        newattr["_v_name"] = newname
        newattr["_v_hdf5name"] = trTable.get(newname, newname)
        newattr["_v_pathname"] = parent._g_join(newname)
        # Update class variables
        parent._c_objects[self._v_pathname] = self
        parent._c_objleaves[self._v_pathname] = self
        self.name = newname     # This is a standard attribute for Leaves
        # Call the _g_new method in Leaf superclass 
        self._g_new(parent, self._v_hdf5name)
        # Update this instance attributes
        parent._v_objchilds[newname] = self
        parent._v_objleaves[newname] = self
        parent.__dict__[newname] = self
        
    def _f_rename(self, newname):
        """Rename an HDF5 leaf"""

        # Check for name validity
        checkNameValidity(newname)
        # Check if self has a child with the same name
        if newname in self._v_parent._v_objchilds:
            raise RuntimeError, \
        """Another sibling (%s) already has the name '%s' """ % \
                   (self._v_parent._v_objchilds[newname], newname)
        # Rename all the appearances of oldname in the object tree
        oldname = self._v_name
        self._g_renameObject(newname)
        self._v_parent._g_renameNode(oldname, newname)
        
    def _f_getAttr(self, attrname):
        """Get a leaf attribute as a string"""
        
        if attrname == "" or attrname is None:
            raise ValueError, \
"""You need to supply a valid attribute name"""            
        return self._v_parent._g_getLeafAttrStr(self._v_hdf5name, attrname)

    def _f_setAttr(self, attrname, attrvalue):
        """Set an leaf attribute as a string"""

        if attrname == "" or attrname is None:
            raise ValueError, \
"""You need to supply a valid attribute name"""            
        if type(attrvalue) == types.StringType:
            return self._v_parent._g_setLeafAttrStr(self._v_hdf5name,
                                                    attrname, attrvalue)
        else:
            raise ValueError, \
"""Only string values are supported as attributes right now"""

    def _f_remove(self, recursive):
        """Remove an HDF5 Leaf that is child of this group

        "recursive" parameter is not needed here.
        """
        parent = self._v_parent
        self.close()
        parent._g_deleteLeaf(self._v_name)

    def close(self):
        """Flush the Leaf buffers and close this object on file."""
        self.flush()
        parent = self._v_parent
        del parent._v_objleaves[self._v_name]
        del parent.__dict__[self._v_name]
        del parent._v_objchilds[self._v_name]
        del parent._c_objleaves[self._v_pathname]
        del parent._c_objects[self._v_pathname]
        del self._v_parent
        del self._v_rootgroup

    def __str__(self):
        """The string reprsentation choosed for this object is its pathname
        in the HDF5 object tree.
        """
        
        # Get the associated filename
        filename = self._v_rootgroup._v_filename
        # The pathname
        pathname = self._v_pathname
        # Get this class name
        classname = self.__class__.__name__
        # The object shape 
        shape = str(self.shape)
        # The title
        title = self.title
        # Printing the filename can be confusing in some contexts
        #return "/%s%s %s %s \"%s\"" % \
        #       (filename, pathname, classname, shape, title)
        return "%s %s%s \"%s\"" % \
               (pathname, classname, shape, title)

