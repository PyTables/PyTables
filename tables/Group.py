########################################################################
#
#       License: BSD
#       Created: September 4, 2002
#       Author:  Francesc Alted - falted@openlc.org
#
#       $Source: /home/ivan/_/programari/pytables/svn/cvs/pytables/pytables/tables/Group.py,v $
#       $Id: Group.py,v 1.69 2004/02/10 16:36:52 falted Exp $
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

__version__ = "$Revision: 1.69 $"

MAX_DEPTH_IN_TREE = 2048
# Note: the next constant has to be syncronized with the
# MAX_CHILDS_IN_GROUP constant in util.h!
MAX_CHILDS_IN_GROUP = 4096

from __future__ import generators

import sys, warnings, types
import hdf5Extension
from Table import Table
from Array import Array
from EArray import EArray
from VLArray import VLArray
from Leaf import Filters
from UnImplemented import UnImplemented
from AttributeSet import AttributeSet
from utils import checkNameValidity
import cPickle

class Group(hdf5Extension.Group, object):
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
    (for private methods) or "_v_" (for instance variables) prefixes.

    Methods:
    
        _f_listNodes(classname)
        _f_walkGroups()
        __delattr__(name)
        __getattr__(name)
        __setattr__(name, object)
        __iter__()
        _f_rename(newname)
        _f_remove(recursive=0)
        _f_getAttr(attrname)
        _f_setAttr(attrname, attrvalue)
        _f_copyChilds(where, recursive=0, filters=None, copyuserattrs=1)
        _f_close()
        
    Instance variables:

        _v_title -- TITLE attribute of this group
        _v_name -- The name of this group in python namespace
        _v_hdf5name -- The name of this group in HDF5 file namespace
        _v_objectID -- The HDF5 object ID of the group
        _v_pathname -- A string representation of the group location
            in tree
        _v_parent -- The parent Group instance
        _v_depth -- The depth level in tree for this group
        _v_file -- The associated File object
        _v_rootgroup - Always point to the root group object
        _v_groups -- Dictionary with object groups
        _v_leaves -- Dictionary with object leaves
        _v_childs -- Dictionary with object childs (groups or leaves)
        _v_nchilds -- Number of childs (groups or leaves) of this object 
        _v_attrs -- The associated AttributeSet instance
        _v_filters -- The associated Filters instance

    """

    def __init__(self, title = "", new = 1, filters=None):
        """Create the basic structures to keep group information.

        title -- The title for this group
        new -- If this group is new or has to be read from disk
        filters -- A Filters instance
        
        """
        self.__dict__["_v_new"] = new
        self.__dict__["_v_new_title"] = title
        self.__dict__["_v_new_filters"] = filters
        self.__dict__["_v_groups"] = {}
        self.__dict__["_v_leaves"] = {}
        self.__dict__["_v_childs"] = {}
        self.__dict__["_v_nchilds"] = 0
        return
    
    def __iter__(self, classname=None, recursive=0):
        """Iterate over the childs on self"""

        return self._f_iterGroup(classname, recursive)

    def _f_iterGroup(self, classname=None, recursive=0):

        if not recursive:
            # Non-recursive algorithm
            for leaf in self._f_listNodes(classname):
                yield leaf
        else:
            if classname == "Group":
                # Recursive algorithm
                for group in self._f_walkGroups():
                    yield group
            else:
                for group in self._f_walkGroups():
                    for leaf in group._f_listNodes(classname):
                        yield leaf
                
    def __call__(self, classname=None, recursive=0):
        """Iterate over the childs on self

	      If "classname" is supplied, only instances of this class
	      are returned. If "recursive" is false, only childs
	      hanging immediately after the group are returned. If
	      true, a recursion over all the groups hanging from it is
	      performed. """
        
        return self.__iter__(classname, recursive)

    # This iterative version of _g_openFile is due to John Nielsen
    def _g_openFile(self):
        """Recusively reads an HDF5 file and generates a tree object.
        """

        stack=[self]                
        while stack:
            objgroup=stack.pop()
            pgroupId=objgroup._v_parent._v_objectID
            locId=objgroup._v_objectID
            (groups, leaves)=self._g_listGroup(pgroupId, locId,
                                               objgroup._v_hdf5name)
            for name in groups:
                new_objgroup = Group(new = 0)
                new_objgroup._g_putObjectInTree(name, objgroup)
                stack.append(new_objgroup)
            for name in leaves:
                objleaf=objgroup._g_getLeaf(name)
                if objleaf <> None:
                    # Try if object can be loaded
                    try:
                        objleaf._g_putObjectInTree(name, objgroup)
                    except:
                        (type, value, traceback) = sys.exc_info()
                        warnings.warn( \
"""Problems loading '%s' object. The error was: <%s>. This object will be cast into an UnImplemented instance. Continuing...""" % \
(objleaf._v_pathname, value), UserWarning)
                        # If not, associate an UnImplemented object to it
                        objleaf = UnImplemented()
                        objleaf._g_putObjectInTree(name, objgroup)
                else:
                    # If objleaf is not recognized, associate an
                    # UnImplemented object to it
                    objleaf = UnImplemented()
                    objleaf._g_putObjectInTree(name, objgroup)

    def _g_getLeaf(self,name):
        """Returns a proper Leaf class depending on the object to be opened.
        """

        if self._v_file._isPTFile:
            # We can call this only if we are certain than file has
            # the attribute CLASS
            class_ = self._v_attrs._g_getChildSysAttr(name, "CLASS")
        else:
            class_ = self._v_attrs._g_getChildAttr(name, "CLASS")
        if class_ is None:
            # No CLASS attribute, try a guess
            class_ = hdf5Extension.whichClass(self._v_objectID, name)
            if class_ == "UNSUPPORTED":
                warnings.warn( \
"Leaf object '%s' in file is unsupported and will become <UnImplemented> type." % \
self._g_join(name), UserWarning)
                return None
        if class_ == "TABLE":
            return Table()
        elif class_ == "ARRAY":
            return Array()
        elif class_ == "EARRAY":
            return EArray()
        elif class_ == "VLARRAY":
            return VLArray()
        else:
            warnings.warn( \
"Class ID '%s' for Leaf %s is unknown and will become <UnImplemented> type." % \
(class_, self._g_join(name)), UserWarning)
            return None

    def _g_join(self, name):
        """Helper method to correctly concatenate a name child object
        with the pathname of this group."""

        if name == "/":
            # This case can happen when doing copies
            return self._v_pathname
        if self._v_pathname == "/":
            return "/" + name
        else:
            return self._v_pathname + "/" + name

    def _g_setproperties(self, name, value):
        """Set some properties for general objects (Group and Leaf) in the
        tree."""

        # File object
        # New attributes for the new Group instance
        newattr = value.__dict__
        newattr["_v_" + "rootgroup"] = self._v_rootgroup
        newattr["_v_" + "file"] = self._v_rootgroup._v_parent
        newattr["_v_" + "parent"] = self
        newattr["_v_" + "depth"] = self._v_depth + 1
        # Get the alternate name (if any)
        trMap = self._v_rootgroup._v_parent.trMap
        if value._v_new:
            newattr["_v_name"] = name
            newattr["_v_hdf5name"] = trMap.get(name, name)
        else:
            for (namepy, namedisk) in trMap.items():
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
        self._v_childs[value._v_name] = value
        self.__dict__["_v_nchilds"] += 1
        # New attribute (to allow tab-completion in interactive mode)
        self.__dict__[value._v_name] = value
        # Update class variables
        self._v_file.objects[value._v_pathname] = value

    def _g_putObjectInTree(self, name, parent):
        """Set attributes for a new or existing Group instance."""
        
        # Update the parent instance attributes
        parent._g_setproperties(name, self)
        self._g_new(parent, self._v_hdf5name)
        parent._v_groups[self._v_name] = self
        # Update class variables
        self._v_file.groups[self._v_pathname] = self
        if self._v_new:
            self._g_create()
        else:
            self._g_open()
        # Attach the AttributeSet attribute
        # This doesn't becomes a property because it takes more time!
        self.__dict__["_v_attrs"] = AttributeSet(self)
        if self._v_new:
            # Set the title, class and version attribute
            self._v_attrs._g_setAttrStr('TITLE',  self._v_new_title)
            self._v_attrs._g_setAttrStr('CLASS', "GROUP")
            self._v_attrs._g_setAttrStr('VERSION', "1.0")
            # Set the filters object
            if self._v_new_filters is None:
                # If not filters has been passed in the constructor,
                filters = self._v_parent._v_filters
            else:
                filters = self._v_new_filters
            filtersPickled = cPickle.dumps(filters, 0)
            self._v_attrs._g_setAttrStr('FILTERS', filtersPickled)
            # Add these attributes to the dictionary
            attrlist = ['TITLE','CLASS','VERSION','FILTERS']
            self._v_attrs._v_attrnames.extend(attrlist)
            self._v_attrs._v_attrnamessys.extend(attrlist)
            # Sort them
            self._v_attrs._v_attrnames.sort()
            self._v_attrs._v_attrnamessys.sort()
        else:
            # We don't need to get attributes on disk. The most importants
            # are defined as properties
            pass

    # Define attrs as a property.
    # In the case of groups, it is faster to not define the _v_attrs property
    # I don't know exactly why. This should be further investigated.
#     def _get_attrs (self):
#         return AttributeSet(self)
#     # attrs can't be set or deleted by the user
#     _v_attrs = property(_get_attrs, None, None, "Attrs of this object")

    # Define _v_title as a property
    def _f_get_title (self):
        return self._v_attrs.TITLE
    
    def _f_set_title (self, title):
        self._v_attrs.TITLE = title

    # _v_title can't be deleted.
    _v_title = property(_f_get_title, _f_set_title, None,
                        "Title of this object")

    # Define _v_filters as a property
    def _f_get_filters(self):
        filters = self._v_attrs.FILTERS
        if filters == None:
            filters = Filters()
        return filters
    
    # _v_filters can't be set nor deleted
    _v_filters = property(_f_get_filters, None, None,
                          "Filters of this object")

    def _g_renameObject(self, newname):
        """Rename this group in the object tree as well as in the HDF5 file."""

        parent = self._v_parent
        newattr = self.__dict__
        name = newname

        # Delete references to the oldname
        del parent._v_groups[self._v_name]
        del parent._v_childs[self._v_name]
        del parent.__dict__[self._v_name]

        # Get the alternate name (if any)
        trMap = self._v_rootgroup._v_parent.trMap
        # New attributes for the this Group instance
        newattr["_v_name"] = newname
        newattr["_v_hdf5name"] = trMap.get(newname, newname)
        # Update class variables
        parent._v_file.groups[self._v_pathname] = self
        parent._v_file.objects[self._v_pathname] = self
        # Call the _g_new method in Group superclass 
        self._g_new(parent, self._v_hdf5name)
        # Update this instance attributes
        parent._v_groups[newname] = self
        parent._v_childs[newname] = self
        parent.__dict__[newname] = self

        # Finally, change the old pathname in the object childs recursively
        oldpathname = self._v_pathname
        newpathname = parent._g_join(newname)
        for group in self._f_walkGroups():
            oldgpathname = group._v_pathname
            newgpathname = oldgpathname.replace(oldpathname, newpathname, 1)
            group.__dict__["_v_pathname"] = newgpathname
            # Update class variables
            del parent._v_file.groups[oldgpathname]
            del parent._v_file.objects[oldgpathname]
            parent = group._v_parent
            parent._v_file.groups[newgpathname] = group
            parent._v_file.objects[newgpathname] = group
            for node in group._f_listNodes("Leaf"):
                oldgpathname = node._v_pathname
                newgpathname = oldgpathname.replace(oldpathname, newpathname, 1)
                node.__dict__["_v_pathname"] = newgpathname
                # Update class variables
                del parent._v_file.leaves[oldgpathname]
                del parent._v_file.objects[oldgpathname]
                parent = node._v_parent
                parent._v_file.leaves[newgpathname] = node
                parent._v_file.objects[newgpathname] = node


    def _g_open(self):
        """Call the openGroup method in super class to open the existing
        group on disk. """
        
        # Call the superclass method to open the existing group
        self.__dict__["_v_objectID"] = self._g_openGroup()

    def _g_create(self):
        """Call the createGroup method in super class to create the group on
        disk. Also set attributes for this group. """

        # Call the superclass method to create a new group
        self.__dict__["_v_objectID"] = \
                     self._g_createGroup()

    def _f_listNodes(self, classname = ""):
        """Return a list with all the object nodes hanging from self.

        The list is alphanumerically sorted by node name. If a
        "classname" parameter is supplied, it will only return
        instances of this class (or subclasses of it). The supported
        classes in "classname" are 'Group', 'Leaf', 'Table', 'Array',
        'EArray', 'VLArray' and 'UnImplemented'.

        """
        if not classname:
            # Returns all the childs alphanumerically sorted
            names = self._v_childs.keys()
            names.sort()
            return [ self._v_childs[name] for name in names ]
        elif classname == 'Group':
            # Returns all the groups alphanumerically sorted
            names = self._v_groups.keys()
            names.sort()
            return [ self._v_groups[name] for name in names ]
        elif classname == 'Leaf':
            # Returns all the leaves alphanumerically sorted
            names = self._v_leaves.keys()
            names.sort()
            return [ self._v_leaves[name] for name in names ]
        elif (classname == 'Table' or
              classname == 'Array' or
              classname == 'EArray' or
              classname == 'VLArray' or
              classname == 'UnImplemented'):
            listobjects = []
            # Process alphanumerically sorted 'Leaf' objects
            for leaf in self._f_listNodes('Leaf'):
                if leaf.__class__.__name__ == classname:
                    listobjects.append(leaf)
            # Returns all the 'classname' objects alphanumerically sorted
            return listobjects
        else:
            raise ValueError, \
""""classname" can only take 'Group', 'Leaf', 'Table', 'Array', 'EArray', 'VLArray', 'UnImplemented' values"""

    def _f_walkGroups(self):
        """Iterate over the Groups (not Leaves) hanging from self.

        The groups are returned ordered from top to bottom, and
        alphanumerically sorted when in the same level.

        """
        
        stack = [self]
        yield self
        # Iterate over the descendants
        while stack:
            objgroup=stack.pop()
            groupnames = objgroup._v_groups.keys()
            # Sort the groups before delivering. This uses the groups names
            # for groups in tree (in order to sort() can classify them).
            groupnames.sort()
            for groupname in groupnames:
                stack.append(objgroup._v_groups[groupname])
                yield objgroup._v_groups[groupname]
                
    def _f_getListTree(self):
        """Return a list with Groups (not Leaves) hanging from self.

        The groups are returned ordered from top to bottom.

        """
        
        stack = [self]
        stack2 = [self]
        # Iterate over the descendants
        while stack:
            objgroup=stack.pop()
            groupnames = objgroup._v_groups.keys()
            for groupname in groupnames:
                stack.append(objgroup._v_groups[groupname])
                stack2.append(objgroup._v_groups[groupname])
        return stack2
    
    def __delattr__(self, name):
        """Remove *recursively* all the objects hanging from name child."""

        if name in self._v_groups:
            return self._v_groups[name]._f_remove(1)
        elif name in self._v_leaves:
            return self._v_leaves[name].remove()

    def __getattr__(self, name):
        """Get the object named "name" hanging from me."""

        if not self._v_file.isopen:
            raise RuntimeError, "You are trying to access to a closed file handler. Giving up!."
        
        if name in self._v_groups:
            return self._v_groups[name]
        elif name in self._v_leaves:
            return self._v_leaves[name]
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
        if self._v_new or not self._v_file._isPTFile:
            # Check names only for new objects or objects coming from
            # non-pytables files
            checkNameValidity(name)
        
        # Check if we are too much deeper in tree
        if self._v_depth == MAX_DEPTH_IN_TREE:
            warnings.warn( \
"""The object tree is exceeding the recommended maximum depth (%d).
 Be ready to see PyTables asking for *lots* of memory and possibly slow I/O.
""" % (self._v_pathname, MAX_DEPTH_IN_TREE), UserWarning)

        # Check if we have too much number of childs
        #print "Group %s has %d nchilds" % (self._v_name, self._v_nchilds)
        if self._v_nchilds == MAX_CHILDS_IN_GROUP:
            warnings.warn( \
"""'%s' group is exceeding the recommended maximum number of childs (%d).
 Be ready to see PyTables asking for *lots* of memory and possibly slow I/O.
""" % (self._v_pathname, MAX_CHILDS_IN_GROUP), UserWarning)
            
        # Put value object with name "name" in object tree
        if name not in self._v_childs:
            value._g_putObjectInTree(name, self)
        else:
            raise NameError, \
                  "'%s' group already has a child named '%s' in file '%s'" % \
                  (self._v_pathname, name, self._v_rootgroup._v_filename)

    def _f_close(self):
        """Close this HDF5 group"""
        self._g_closeGroup()
        # Delete the back references in Group
        if self._v_name <> "/":
            del self._v_parent._v_groups[self._v_name]  # necessary (checked)
            del self._v_parent._v_childs[self._v_name]  # necessary (checked)
            self._v_parent.__dict__["_v_nchilds"] -= 1 
            del self._v_parent.__dict__[self._v_name]
        del self._v_file.groups[self._v_pathname]  
        del self._v_file.objects[self._v_pathname]
        ##################################
        #self._v_childs.clear()
        ##################################
        # Delete back references
        #del self._v_rootgroup    # This is incorrect!!
        del self.__dict__["_v_rootgroup"]
        del self.__dict__["_v_parent"]
        # Detach the AttributeSet instance
        self._v_attrs._f_close()
        del self.__dict__["_v_attrs"]

    def _f_getAttr(self, attrname):
        """Get a group attribute as a string"""
        
        return getattr(self._v_attrs, attrname, None)

    def _f_setAttr(self, attrname, attrvalue):
        """Set an group attribute as a string"""

        setattr(self._v_attrs, attrname, attrvalue)

    def _f_rename(self, newname):
        """Rename a group"""

        # Check for name validity
        checkNameValidity(newname)
        # Check if self has a child with the same name
        if newname in self._v_parent._v_childs:
            raise RuntimeError, \
        """Another sibling (%s) already has the name '%s' """ % \
                   (self._v_parent._v_childs[newname], newname)
        # Rename all the appearances of oldname in the object tree
        oldname = self._v_name
        self._g_renameObject(newname)
        self._v_parent._g_renameNode(oldname, newname)
        
    def _f_remove(self, recursive=0):
        """Remove this group"""
        
        if self._v_childs <> {}:
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

    def _f_copyChilds(self, where, recursive=0, filters=None, copyuserattrs=1,
                      overwrite = 0, start=0, stop=None, step=1):
        "(Recursively) Copy a group into another location"

        # Get the base names of the source
        srcBasePath = self._v_pathname
        lenSrcBasePath = len(srcBasePath)+1 # To include the trailing '/'
        if lenSrcBasePath == 2:
            lenSrcBasePath = 1  # This is a protection for srcBase == "/"
        # Get the parent group of destination
        if isinstance(where, Group):
            # The destination path can be anywhere
            dstBaseGroup = where
            dstFile = dstBaseGroup._v_file
        else:
            # The destination path should be in the same file as we are now
            dstBaseGroup = self._v_file.getNode(where, classname = "Group")
            dstFile = self._v_file
        dstBasePath = dstBaseGroup._v_pathname
        ngroups = 0
        nleafs = 0
        if recursive:
            # Recursive copy
            first = 1  # Sentinel
            for group in self._f_walkGroups():
                if first:
                    # The first group itself is not copied, only its childs
                    first = 0
                    depth = group._v_depth
                    dstGroup = dstBaseGroup
                    parentDstPath = dstBasePath
                else:
                    dstName = group._v_name
                    lenDstName = len(dstName)+1  # To include the trailing '/'
                    endGName = group._v_pathname[lenSrcBasePath:-lenDstName]
                    parentDstPath = dstBaseGroup._g_join(endGName)
                    title = group._v_title
                    if title is None: title = ""
                    # Check whether we have to delete the group before copying
                    parentDstGroup = dstFile.getNode(parentDstPath)
                    if hasattr(parentDstGroup, dstName) and overwrite:
                        dstGroup = getattr(parentDstGroup, dstName)
                        dstGroup._f_remove(recursive=1)
                    dstGroup = dstFile.createGroup(parentDstGroup, dstName,
                                                   title=title,
                                                   filters=filters)
                    if copyuserattrs:
                        group._v_attrs._f_copy(dstGroup)
                        depth = group._v_depth
                    ngroups += 1
                for leaf in group._f_listNodes('Leaf'):
                    title = leaf.title
                    if title is None: title = ""
                    # Check whether we have to delete the leaf before copying
                    if hasattr(dstGroup, leaf.name) and overwrite:
                        dstLeaf = getattr(dstGroup,leaf.name)
                        dstLeaf.remove()
                    leaf.copy(dstGroup, leaf.name, title=title,
                              filters=filters,
                              copyuserattrs=copyuserattrs,
                              start=start,
                              stop=stop,
                              step=step)
                    nleafs +=1
        else:
            # Non recursive copy
            # First, copy groups
            for group in self._f_listNodes('Group'):
                title = group._v_title
                if title is None: title = ""
                if hasattr(dstBaseGroup, group._v_name) and overwrite:
                    dstGroup = getattr(dstBaseGroup, group._v_name)
                    dstGroup._f_remove(recursive=1)
                dstGroup = dstFile.createGroup(dstBaseGroup, group._v_name,
                                               title=title,
                                               filters=filters)
                if copyuserattrs:
                    group._v_attrs._f_copy(dstGroup)
                ngroups +=1
            # Then, leafs
            for leaf in self._f_listNodes('Leaf'):
                title = leaf.title
                if title is None: title = ""
                # Check whether we have to delete the leaf before copying
                if hasattr(dstBaseGroup, leaf.name) and overwrite:
                    dstLeaf = getattr(dstBaseGroup, leaf.name)
                    dstLeaf.remove()
                leaf.copy(dstBaseGroup, leaf.name, title=title,
                          filters=filters,
                          copyuserattrs=copyuserattrs,
                          start=start,
                          stop=stop,
                          step=step)
                nleafs +=1
        # return the number of objects copied
        return (ngroups, nleafs)
    
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
        return "%s (%s) %r" % (pathname, classname, title)

    def __repr__(self):
        """A detailed string representation for this object."""
        
        rep = [ '%r (%s)' %  \
                (childname, child.__class__.__name__) 
                for (childname, child) in self._v_childs.items() ]
        childlist = '[%s]' % (', '.join(rep))
        
        return "%s\n  childs := %s" % \
               (str(self), childlist)
               
