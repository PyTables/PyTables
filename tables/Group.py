########################################################################
#
#       License: BSD
#       Created: September 4, 2002
#       Author:  Francesc Alted - falted@openlc.org
#
#       $Source: /home/ivan/_/programari/pytables/svn/cvs/pytables/pytables/tables/Group.py,v $
#       $Id: Group.py,v 1.21 2003/03/08 11:40:54 falted Exp $
#
########################################################################

"""Here is defined the Group class.

See Group class docstring for more info.

Classes:

    Group

Functions:


Misc variables:

    __version__
    
    MAX_DEPTH_IN_TREE -- Maximum depth tree allowed in PyTables. This
        number should be supported by all python interpreters (i.e.,
        their recursion level should be bigger that this)

    MAX_CHILDS_IN_GROUP -- Maximum allowed number of childs hanging
        from a group

"""

__version__ = "$Revision: 1.21 $"

MAX_DEPTH_IN_TREE = 512
# Note: the next constant has to be syncronized with the
# MAX_CHILDS_IN_GROUP constant in util.h!
MAX_CHILDS_IN_GROUP = 4096

from __future__ import generators

import warnings, types
import hdf5Extension
from Table import Table
from Array import Array
from utils import checkNameValidity

class Group(hdf5Extension.Group):
    """This is the python counterpart of a group in the HDF5 structure.

    It provides methods to set properties based on information
    extracted from the HDF5 files and to walk throughout the
    tree. Every group has parents and children, which are all Group
    instances, except for the root group whose parent is a File
    instance.

    In Group instances, a special feature called "natural naming" is
    used, i.e. the instance attributes that represent HDF5 groups are
    the same as the names of the children. This offers the user a very
    convenient way to access nodes in tree by simply naming all the
    path from the root group.

    For this reason and in order to not pollute the children
    namespace, it is explicitely forbidden to assign "normal"
    attributes to Group instances, and the only ones allowed must
    start by "_c_" (for class variables), "_f_" (for methods), "_g_"
    for private methods or "_v_" (for instance variables) prefixes.

    Methods:
    
        _f_listNodes([classname])
        _f_walkGroups()
        _f_close()
        _f_rename(newname)
        _f_remove()
        _f_removeLeaf()
        
    Class variables:

        _c_objgroups -- Dictionary with object groups on tree
        _c_objleaves -- Dictionaly with object leaves on tree
        _c_objects -- Dictionary with all objects (groups or leaves) on tree

    Instance variables:

        _v_title -- TITLE attribute of this group.
        _v_name -- The name of this group.
        _v_pathname -- A string representation of the group location
            in tree.
        _v_parent -- The parent Group instance.
        _v_objgroups -- Dictionary with object groups
        _v_objleaves -- Dictionaly with object leaves
        _v_objchilds -- Dictionary with object childs (groups or leaves)

    """

    # Class variables to keep track of all the childs and group objects
    # in tree object. They are dictionaries that will use the pathnames
    # as keys and the actual objects as values.
    # That way we can find objects in the object tree easily and fast. 
    _c_objgroups = {}
    _c_objleaves = {}
    _c_objects = {}

    def __init__(self, title = "", new = 1):
        """Create the basic structures to keep group information.

        title -- The title for this group
        new -- If this group is new or has to read from disk
        
        """
        self.__dict__["_v_new"] = new
        self.__dict__["_v_title"] = title
        self.__dict__["_v_objgroups"] = {}
        self.__dict__["_v_objleaves"] = {}
        self.__dict__["_v_objchilds"] = {}

    def _g_openFile(self):
        """Recusively reads an HDF5 file and generates a tree object.

        This tree is the python replica of the file hierarchy.

        """
        pgroupId =  self._v_parent._v_groupId
        (groups, leaves) = self._g_listGroup(pgroupId, self._v_hdf5name)
        for name in groups:
            objgroup = Group(new = 0)
            # Insert this group as a child of mine
            objgroup._g_putObjectInTree(name, self)
            #setattr(self, name, objgroup)
            # Call openFile recursively over the group's tree
            objgroup._g_openFile()
        for name in leaves:
            class_ = self._g_getLeafAttrStr(name, "CLASS")
            if class_ is None:
                # No CLASS attribute, try a guess
                warnings.warn( \
"""No CLASS attribute found. Trying to guess what's here.
  I can't promise getting the correct object, but I will do my best!.""",
                UserWarning)
                class_ = hdf5Extension.whichClass(self._v_groupId, name)
                if class_ == "UNSUPPORTED":
                    raise RuntimeError, \
                    """Dataset object \'%s\' in file is unsupported!.""" % \
                          name
            if class_ == "TABLE":
                objgroup = Table()
            elif class_ == "ARRAY":
                objgroup = Array()
            else:
                raise RuntimeError, \
                      """Dataset object in file is unknown!
                      class ID: %s""" % class_
            # Set some attributes to caracterize and open this object
            objgroup._g_putObjectInTree(name, self)
            #setattr(self, name, objgroup)
        
    def _g_join(self, name):
        """Helper method to correctly concatenate a name child object
        with the pathname of this group."""
        
        if self._v_pathname == "/":
            return "/" + name
        else:
            return self._v_pathname + "/" + name

    def _g_setproperties(self, name, value):
        """Set some properties for general objects (Group and Leaf) in the
        tree."""

        # New attributes for the new Group instance
        newattr = value.__dict__
        newattr["_v_" + "rootgroup"] = self._v_rootgroup
        newattr["_v_" + "parent"] = self
        newattr["_v_" + "depth"] = self._v_depth + 1
        # Get the alternate name (if any)
        trTable = self._v_rootgroup._v_parent.trTable
        if value._v_new:
            newattr["_v_name"] = name
            newattr["_v_hdf5name"] = trTable.get(name, name)
        else:
            for (namepy, namedisk) in trTable.items():
                if namedisk == name:
                    newattr["_v_name"] = namepy
                    break
            else:
                # namedisk is not in the translation table
                newattr["_v_name"] = name
            # This attribute is always the name in disk
            newattr["_v_hdf5name"] = name
                
        newattr["_v_" + "pathname"] = self._g_join(value._v_name)
        # Update instance variable
        self._v_objchilds[value._v_name] = value
        # New attribute (to allow tab-completion in interactive mode)
        self.__dict__[value._v_name] = value
        # In the future this should be read from disk in case of an opening
        # To be done when general Attribute module available
        newattr["_v_class"] = value.__class__.__name__
        newattr["_v_version"] = "1.0"
        # Update class variables
        self._c_objects[value._v_pathname] = value

    def _g_putObjectInTree(self, name, parent):
        """Set attributes for a new or existing Group instance."""
        
        # Update the parent instance attributes
        parent._g_setproperties(name, self)
        self._g_new(parent, self._v_hdf5name)
        parent._v_objgroups[self._v_name] = self
        # Update class variables
        self._c_objgroups[self._v_pathname] = self
        if self._v_new:
            self._g_create()
        else:
            self._g_open(parent, self._v_hdf5name)

    def _g_renameObject(self, newname):
        
        """Rename this group in the object tree as well as in the HDF5 file."""

        parent = self._v_parent
        newattr = self.__dict__
        name = newname

        # Falta que açò s'invoque recursivament per a refrescar les
        # _v_pathnames en l'arbre.
        # Delete references to the oldname
        del parent._v_objgroups[self._v_name]
        del parent._v_objchilds[self._v_name]
        del parent.__dict__[self._v_name]

        # Get the alternate name (if any)
        trTable = self._v_rootgroup._v_parent.trTable
        # New attributes for the this Group instance
        newattr["_v_name"] = newname
        newattr["_v_hdf5name"] = trTable.get(newname, newname)
        # Update class variables
        parent._c_objgroups[self._v_pathname] = self
        parent._c_objects[self._v_pathname] = self
        # Call the _g_new method in Group superclass 
        self._g_new(parent, self._v_hdf5name)
        # Update this instance attributes
        parent._v_objgroups[newname] = self
        parent._v_objchilds[newname] = self
        parent.__dict__[newname] = self

        # Finally, change the old pathname in the object childs recursively
        oldpathname = self._v_pathname
        newpathname = parent._g_join(newname)
        for group in self._f_walkGroups():
            oldgpathname = group._v_pathname
            newgpathname = oldgpathname.replace(oldpathname, newpathname, 1)
            group.__dict__["_v_pathname"] = newgpathname
            # Update class variables
            del parent._c_objgroups[oldgpathname]
            del parent._c_objects[oldgpathname]
            parent = group._v_parent
            parent._c_objgroups[newgpathname] = group
            parent._c_objects[newgpathname] = group
            for node in group._f_listNodes("Leaf"):
                oldgpathname = node._v_pathname
                newgpathname = oldgpathname.replace(oldpathname, newpathname, 1)
                node.__dict__["_v_pathname"] = newgpathname
                # Update class variables
                del parent._c_objleaves[oldgpathname]
                del parent._c_objects[oldgpathname]
                parent = node._v_parent
                parent._c_objleaves[newgpathname] = node
                parent._c_objects[newgpathname] = node


    def _g_open(self, parent, name):
        """Call the openGroup method in super class to open the existing
        group on disk. Also get attributes for this group. """
        
        # Call the superclass method to open the existing group
        self.__dict__["_v_groupId"] = \
                      self._g_openGroup(parent._v_groupId, name)
        # Get the title, class and version attributes
        self.__dict__["_v_title"] = \
                      self._f_getAttr('TITLE')
        self.__dict__["_v_class"] = \
                      self._f_getAttr('CLASS')
        self.__dict__["_v_version"] = \
                      self._f_getAttr('VERSION')

    def _g_create(self):
        """Call the createGroup method in super class to create the group on
        disk. Also set attributes for this group. """

        # Call the superclass method to create a new group
        self.__dict__["_v_groupId"] = \
                     self._g_createGroup()
        # Set the title, class and version attribute
        self._g_setGroupAttrStr('TITLE', self._v_title)
        self._g_setGroupAttrStr('CLASS', "Group")
        self._g_setGroupAttrStr('VERSION', "1.0")

    def _f_listNodes(self, classname = ""):
        """Return a list with all the object nodes hanging from self.

        The list is alphanumerically sorted by node name. If a
        "classname" parameter is supplied, it will only return
        instances of this class (or subclasses of it). The supported
        classes in "classname" are 'Group', 'Leaf', 'Table' and
        'Array'.

        """
        if not classname:
            # Returns all the childs alphanumerically sorted
            names = self._v_objchilds.keys()
            names.sort()
            return [ self._v_objchilds[name] for name in names ]
        elif classname == 'Group':
            # Returns all the groups alphanumerically sorted
            names = self._v_objgroups.keys()
            names.sort()
            return [ self._v_objgroups[name] for name in names ]
        elif classname == 'Leaf':
            # Returns all the leaves alphanumerically sorted
            names = self._v_objleaves.keys()
            names.sort()
            return [ self._v_objleaves[name] for name in names ]
        elif (classname == 'Table' or
              classname == 'Array'):
            listobjects = []
            # Process alphanumerically sorted 'Leaf' objects
            for leaf in self._f_listNodes('Leaf'):
                if leaf._v_class == classname:
                    listobjects.append(leaf)
            # Returns all the 'classname' objects alphanumerically sorted
            return listobjects
        else:
            raise ValueError, \
""""classname" can only take 'Group', 'Leaf', 'Table' or 'Array' values"""

    def _f_walkGroups(self):
        """Recursively obtains Groups (not Leaves) hanging from self.

        The groups are returned from top to bottom, and
        alphanumerically sorted when in the same level.

        """
        # Returns this group
        yield self
        # Iterate over the descendants
        #for group in self._v_objgroups.itervalues():
        # Sort the groups before delivering. This uses the groups names
        # for groups in tree (in order to sort() can classify them).
        groupnames = self._v_objgroups.keys()
        groupnames.sort()
        for groupname in groupnames:
            for x in self._v_objgroups[groupname]._f_walkGroups():
                yield x

#     def __delattr__(self, name):
#         """In the future, this should delete objects both in memory
#         and in the file."""
        
#         if name in self._v_objchilds:
#             #print "Add code to delete", name, "attribute"
#             pass
#             #self._v_leaves.remove(name)
#         else:
#             raise AttributeError, "%s instance has no child %s" % \
#                   (str(self.__class__), name)

    def __getattr__(self, name):
        """Get the object named "name" hanging from me."""
        
        #print "Getting the", name, "attribute in Group", self
        if name in self._v_objgroups:
            return self._v_objgroups[name]
        elif name in self._v_objleaves:
            return self._v_objleaves[name]
        else:
            raise LookupError, "'%s' group has not a \"%s\" child!" % \
                                  (self._v_pathname, name)

    def __setattr__(self, name, value):
        """Attach new nodes to the tree.

        name -- The name of the new node
        value -- The new node object

        If "name" group already exists in "self", raise the NameError
        exception. A NameError is also raised when the "name" starts
        by a reserved prefix. A SyntaxError is raised if "name" is not
        a valid Python identifier.

        """

        # Check for name validity
        checkNameValidity(name)
        
        # Check if we are too much deeper in tree
        if self._v_depth > MAX_DEPTH_IN_TREE:
            raise RuntimeError, \
               "the object tree has exceeded the maximum depth (%d)" % \
               (MAX_DEPTH_IN_TREE) 

        # Check if we have too much number of childs
        if len(self._v_objchilds.values()) < MAX_CHILDS_IN_GROUP:
            # Put value object with name name in object tree
            if name not in self._v_objchilds:
                value._g_putObjectInTree(name, self)
            else:
                raise NameError, \
                      "\"%s\" group already has a child named %s" % \
                      (self._v_pathname, name)
        else:
            raise RuntimeError, \
               "'%s' group has exceeded the maximum number of childs (%d)" % \
               (self._v_pathname, MAX_CHILDS_IN_GROUP) 

    def _g_cleanup(self):
        """Reset all class attributes"""
        self._c_objgroups.clear()
        self._c_objleaves.clear()
        self._c_objects.clear()

    def _f_close(self):
        """Close this HDF5 group"""
        self._g_closeGroup()
        # Delete the back references in Group
        if self._v_hdf5name <> "/":
            del self._v_parent._v_objgroups[self._v_name]
            del self._v_parent._v_objchilds[self._v_name]
            del self._v_parent.__dict__[self._v_name]
        del self._v_parent
        del self._v_rootgroup
        del self._c_objgroups[self._v_pathname]
        del self._c_objects[self._v_pathname]

    def _f_getAttr(self, attrname):
        """Get a group attribute as a string"""
        
        if attrname == "" or attrname is None:
            raise ValueError, \
"""You need to supply a valid attribute name"""            
        return self._g_getGroupAttrStr(attrname)

    def _f_setAttr(self, attrname, attrvalue):
        """Set an group attribute as a string"""

        if attrname == "" or attrname is None:
            raise ValueError, \
"""You need to supply a valid attribute name"""            

        if type(attrvalue) == types.StringType:
            return self._g_setGroupAttrStr(attrname, attrvalue)
        else:
            raise ValueError, \
"""Only string values are supported as attributes right now"""

    def _f_rename(self, newname):
        """Rename an HDF5 group"""

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
        
    def _f_remove(self, recursive=0):
        """Remove this HDF5 group"""
        
        if self._v_objchilds <> {}:
            if recursive:
                # First close all the childs hanging from this group
                for group in self._f_walkGroups():
                    for leaf in group._f_listNodes('Leaf'):
                        # Delete the back references in Leaf
                        leaf.close()
                    # Close this group
                    group._f_close()
                # Finally, remove this group
                self._g_deleteGroup()
            else:
                warnings.warn( \
"""\n  The group '%s' has childs, but the 'recursive' flag is not on.
  Activate it if you really want to recursively delete this group.""" % \
(self._v_pathname), UserWarning)
        else:
            # This group has no childs, so we can delete it
            # without any other measure
            self._f_close()
            self._g_deleteGroup()

    # Moved out of scope
    def _g_del__(self):
        print "Deleting Group name:", self._v_name

    def __str__(self):
        """The string representation for this object."""
        # Get the associated filename
        filename = self._v_rootgroup._v_filename
        # The pathname
        pathname = self._v_pathname
        # Get this class name
        classname = self.__class__.__name__
        # The title
        title = self._v_title
        return "%s (%s) \"%s\"" % (pathname, classname, title)

    def __repr__(self):
        """A detailed string representation for this object."""
        return str(self)
