#
#       License:        BSD
#       Created:        September 4, 2002
#       Author:  Francesc Alted - falted@openlc.org
#
#       $Source: /home/ivan/_/programari/pytables/svn/cvs/pytables/pytables/tables/File.py,v $
#       $Id: File.py,v 1.77 2004/02/10 16:36:52 falted Exp $
#
########################################################################

"""Create PyTables files and the object tree.

This module support importing generic HDF5 files, on top of which
PyTables files are created, read or extended. If a file exists, an
object tree mirroring their hierarchical structure is created in
memory. File class offer methods to traverse the tree, as well as to
create new nodes.

Classes:

    File

Functions:

    copyFile(srcFilename, dstFilename [, title] [, filters]
             [, copyuserattrs] [, overwrite])
    openFile(name [, mode = "r"] [, title] [, trMap])

Misc variables:

    __version__
    format_version
    compatible_formats

"""

__version__ = "$Revision: 1.77 $"
#format_version = "1.0" # Initial format
#format_version = "1.1" # Changes in ucl compression
format_version = "1.2"  # Support for enlargeable arrays and VLA's
                        # 1.2 was introduced in pytables 0.8
compatible_formats = [] # Old format versions we can read
                        # Empty means that we support all the old formats

from __future__ import generators

import sys
import types
import warnings
import time
import os, os.path
from fnmatch import fnmatch
import cPickle

import hdf5Extension
from Group import Group
from Leaf import Leaf, Filters
from Table import Table
from Array import Array
from EArray import EArray
from VLArray import VLArray
from UnImplemented import UnImplemented
from AttributeSet import AttributeSet
import numarray

def _checkFilters(filters, compress=None, complib=None):
    if (filters is None) and ((compress is not None) or
                              (complib is not None)):
        warnings.warn("The use of compress or complib parameters is deprecated. Please, use a Filters() instance instead.", DeprecationWarning)
        fprops = Filters(complevel=compress, complib=complib)
    elif filters is None:
        fprops = None
    elif isinstance(filters, Filters):
        fprops = filters
    else:
        raise ValueError, \
              "filter parameter has to be None or a Filter instance and the passed type is: '%s'" % type(filters)
    return fprops


def copyFile(srcFilename = None, dstFilename=None, title=None,
             filters=None, copyuserattrs=1, overwrite=0):
    """Copy srcFilename to dstFilename

    The srcFilename should exist and dstFilename should not. If
    dsFilename exists and overwrite is true, it is overwritten. title
    lets you put another title to the destination file. copyuserattrs
    specifies whether the user attrs in origin nodes should be copied
    or not; the default is copy them. Finally, specifiying a filters
    parameter overwrite the original filter properties in srcFilename.

    It returns the number of copied groups and leafs in the form
    (ngroups, nleafs).

    """

    # Open the src file
    srcFileh = openFile(srcFilename, mode="r")

    # Copy it to the destination
    ngroups, nleafs = srcFileh.copyFile(dstFilename, title=title,
                                        filters=filters,
                                        copyuserattrs=copyuserattrs,
                                        overwrite=overwrite)

    # Close the source file
    srcFileh.close()
    return ngroups, nleafs


def openFile(filename, mode="r", title="", trMap={}, rootUEP="/",
             filters=None):

    """Open an HDF5 file an returns a File object.

    Arguments:

    filename -- The name of the file (supports environment variable
            expansion).

    mode -- The mode to open the file. It can be one of the following:
    
        "r" -- read-only; no data can be modified.
        
        "w" -- write; a new file is created (an existing file with the
               same name is deleted).

        "a" -- append; an existing file is opened for reading and
               writing, and if the file does not exist it is created.

        "r+" -- is similar to "a", but the file must already exist.

    title -- A TITLE string attribute will be set on the root group
             with its value.

    trMap -- A dictionary to map names in the object tree into
             different HDF5 names in file. The keys are the Python
             names, while the values are the HDF5 names. This is
             useful when you need to name HDF5 nodes with invalid or
             reserved words in Python.

    rootUEP -- The root User Entry Point. It is a group in the file
            hierarchy which is taken as the starting point to create
            the object tree. The group can be whatever existing path
            in the file. If it does not exist, a RuntimeError is
            issued.

    filters -- An instance of the Filters class that provides
            information about the desired I/O filters applicable to
            the leaves that hangs directly from root (unless other
            filters properties are specified for these leaves, of
            course). Besides, if you do not specify filter properties
            for its child groups, they will inherit these ones.
            
    """

    isPTFile = 1  # Assume a PyTables file by default
    # Expand the form '~user'
    path = os.path.expanduser(filename)
    # Expand the environment variables
    path = os.path.expandvars(path)
    
# The file extension warning commmented out by people at GL suggestion

#     if not (fnmatch(path, "*.h5") or
#             fnmatch(path, "*.hdf") or
#             fnmatch(path, "*.hdf5")):
#         warnings.warn( \
# """filename '%s'should have one of the next file extensions
#   '.h5', '.hdf' or '.hdf5'. Continuing anyway.""" % path, UserWarning)

    # Only accept modes 'w', 'r', 'r+' or 'a'
    assert mode in ['w', 'r', 'r+', 'a'], \
"""arg 2 must take one of this values: ['w', 'r', 'r+', 'a']"""

    if (mode == "r" or mode == "r+"):
        # For 'r' and 'r+' check that path exists and is a HDF5 file
        if not os.path.isfile(path):
            raise IOError, \
        """'%s' pathname does not exist or is not a regular file""" % path
        else:
            if not hdf5Extension.isHDF5(path):
                raise IOError, \
        """'%s' does exist but it is not an HDF5 file""" % path
            
            elif not hdf5Extension.isPyTablesFile(path):
                warnings.warn( \
"""'%s' does exist, is an HDF5 file, but has not a PyTables format.
  Trying to guess what's here from HDF5 metadata. I can't promise you getting
  the correct objects, but I will do my best!."""  % path, UserWarning)
                isPTFile = 0
                    
    elif (mode == "w"):
        # For 'w' check that if path exists, and if true, delete it!
        if os.path.isfile(path):
            # Delete the old file
            os.remove(path)
    elif (mode == "a"):            
        if os.path.isfile(path):
            if not hdf5Extension.isHDF5(path):
                raise IOError, \
        """'%s' does exist but it is not an HDF5 file""" % path
            
            elif not hdf5Extension.isPyTablesFile(path):
                warnings.warn( \
"""'%s' does exist, is an HDF5 file, but has not a PyTables format.
  Trying to guess what's here from HDF5 metadata. I can't promise you getting
  the correct object, but I will do my best!."""  % path, UserWarning)
                isPTFile = 0
    else:
        raise IOError, \
        """arg 2 can only take the new values: "r", "r+", "w" and "a" """
    
    # new informs if this file is old or new
    if (mode == "r" or
        mode == "r+" or
        (mode == "a" and os.path.isfile(path)) ):
        new = 0
    else:
        new = 1
            
    # Finally, create the File instance, and return it
    return File(path, mode, title, new, trMap, rootUEP, isPTFile, filters)


class File(hdf5Extension.File, object):
    """Returns an object describing the file in-memory.

    File class offer methods to browse the object tree, to create new
    nodes, to rename them, to delete as well as to assign and read
    attributes.

    Methods:

        createGroup(where, name[, title] [, filters])
        createTable(where, name, description [, title]
                    [, filters] [, expectedrows])
        createArray(where, name, arrayObject, [, title])
        createEArray(where, name, object [, title]
                     [, filters] [, expectedrows])
        createVLArray(where, name, atom [, title]
                      [, filters] [, expectedsizeinMB])
        getNode(where [, name] [,classname])
        listNodes(where [, classname])
        removeNode(where [, name] [, recursive])
        renameNode(where, newname [, name])
        getAttrNode(self, where, attrname [, name])
        setAttrNode(self, where, attrname, attrname [, name])
        walkGroups([where])
        flush()
        close()

    Instance variables:

        filename -- filename opened
        format_version -- The PyTables version number of this file
        isopen -- 1 if the underlying file is still open; 0 if not
        mode -- mode in which the filename was opened
        title -- the title of the root group in file
        root -- the root group in file
        rootUEP -- the root User Entry Point group in file
        trMap -- the mapping between python and HDF5 domain names
        objects -- dictionary with all objects (groups or leaves) on tree
        groups -- dictionary with all object groups on tree
        leaves -- dictionary with all object leaves on tree

    """

    def __init__(self, filename, mode="r", title="", new=1, trMap={},
                 rootUEP="/", isPTFile=1, filters=None):
        
        """Open an HDF5 file. The supported access modes are: "r" means
        read-only; no data can be modified. "w" means write; a new file is
        created, an existing file with the same name is deleted. "a" means
        append (in analogy with serial files); an existing file is opened
        for reading and writing, and if the file does not exist it is
        created. "r+" is similar to "a", but the file must already exist. A
        TITLE attribute will be set on the root group if optional "title"
        parameter is passed."""

        self.filename = filename
        #print "Opening the %s HDF5 file ...." % self.filename
        
        self.mode = mode
        self.title = title
        self._isPTFile = isPTFile
        
        # _v_new informs if this file is old or new
        self._v_new = new
        # Assign the trMap and build the reverse translation
        self.trMap = trMap
        
        # Filters
        if self._v_new:
            if filters is None:
                # Set the defaults
                self.filters = Filters()
            else:
                self.filters = filters

        # Get the root group from this file
        self.root = self.__getRootGroup(rootUEP)

        # Set the flag to indicate that the file has been opened
        self.isopen = 1

        return

    
    def __getRootGroup(self, rootUEP):
        
        """Returns a Group instance which will act as the root group
        in the hierarchical tree. If file is opened in "r", "r+" or
        "a" mode, and the file already exists, this method dynamically
        builds a python object tree emulating the structure present on
        file."""
          
        global format_version
        global compatible_formats
        
        self._v_objectID = self._getFileId()
        self._v_depth = 0

        if rootUEP in [None, ""]:
            rootUEP = "/"

        # Save the User Entry Point in a variable class
        self.rootUEP=rootUEP

        rootname = "/"   # Always the name of the root group

        # Global dictionaries for the file paths.
        # These are used to keep track of all the childs and group objects
        # in tree object. They are dictionaries that will use the pathnames
        # as keys and the actual objects as values.
        # That way we can find objects in the object tree easily and quickly.
        self.groups = {}
        self.leaves = {}
        self.objects = {}

        rootGroup = Group(self._v_new)
        
        # Create new attributes for the root Group instance
        newattr = rootGroup.__dict__
        newattr["_v_rootgroup"] = rootGroup  # For compatibility with Group
        newattr["_v_objectID"] = self._v_objectID
        newattr["_v_parent"] = self
        newattr["_v_file"] = self
        newattr["_v_depth"] = 1
        newattr["_v_filename"] = self.filename  # Only root group has this

        newattr["_v_name"] = rootname
        newattr["_v_hdf5name"] = rootUEP
        newattr["_v_pathname"] = rootname   # Can be rootUEP? I don't think so
        
        # Update global path variables for Group
        self.groups["/"] = rootGroup
        self.objects["/"] = rootGroup
        
        # Open the root group. We do that always, be the file new or not
        rootGroup._g_new(self, rootUEP)
        newattr["_v_objectID"] = rootGroup._g_openGroup()

        # Attach the AttributeSet attribute to the rootGroup group
        newattr["_v_attrs"] = AttributeSet(rootGroup)

        attrsRoot =  rootGroup._v_attrs   # Shortcut
        if self._v_new:
            # Set the title
            newattr["_v_title"] = self.title
            # Set the filters instance
            newattr["_v_filters"] = self.filters
            
            # Save the rootGroup attributes on disk
            attrsRoot._g_setAttrStr('TITLE',  self.title)
            attrsRoot._g_setAttrStr('CLASS', "GROUP")
            attrsRoot._g_setAttrStr('VERSION', "1.0")
            filtersPickled = cPickle.dumps(self.filters, 0)
            attrsRoot._g_setAttrStr('FILTERS', filtersPickled)

            # Finally, save the PyTables format version for this file
            self.format_version = format_version
            attrsRoot._g_setAttrStr('PYTABLES_FORMAT_VERSION', format_version)
            attrlist = ['TITLE','CLASS','VERSION','FILTERS',
                        'PYTABLES_FORMAT_VERSION']
            # Add these attributes to the dictionary
            attrsRoot._v_attrnames.extend(attrlist)
            attrsRoot._v_attrnamessys.extend(attrlist)
            # Sort them
            attrsRoot._v_attrnames.sort()
            attrsRoot._v_attrnamessys.sort()

        else:
            # Firstly, get the PyTables format version for this file
            self.format_version = hdf5Extension.read_f_attr(self._v_objectID,
                                                     'PYTABLES_FORMAT_VERSION')
            
            if not self.format_version or not self._isPTFile:
                # PYTABLES_FORMAT_VERSION attribute is not present
                self.format_version = "unknown"
                          
            # Get the title for the rootGroup group
            rootGroup.__dict__["_v_title"] = attrsRoot.TITLE
            # Get the title for the file
            #self.title = hdf5Extension.read_f_attr(self._v_objectID, 'TITLE')
            self.title = rootGroup._v_title
            # Get the filters for the file
            filters = attrsRoot.FILTERS
            if filters is None:
                filters = Filters()
            self.filters = filters
                      
            # Get all the groups recursively
            rootGroup._g_openFile()
        
        return rootGroup

    
    def _createNode(self, classname, where, name, *args, **kwargs):

        """Create a new "classname" instance with name "name" in "where"
        location.  "where" parameter can be a path string, or another group
        instance. The rest of the parameters depends on what is required by
        "classname" class constructors. See documentation on these classes
        for information on this."""

        if classname == "Group":
            object = Group(*args, **kwargs)
        elif classname == "Table":
            object = Table(*args, **kwargs)
        elif classname == "Array":
            object = Array(*args, **kwargs)
        elif classname == "EArray":
            object = EArray(*args, **kwargs)
        elif classname == "VLArray":
            object = VLArray(*args, **kwargs)
        else:
            raise ValueError,\
            """Parameter 1 can only take 'Group', 'Table', 'Array', EArray or VLArray values."""

        group = self.getNode(where, classname = 'Group')
        # Put the object on the tree
        setattr(group, name, object)
        return object

    
    def createGroup(self, where, name, title = "", filters = None):
        """Create a new Group instance with name "name" in "where" location.

        Keyword arguments:

        where -- The parent group where the new group will hang
            from. "where" parameter can be a path string (for example
            "/level1/level2"), or Group instance.

        name -- The name of the new group.

        title -- Sets a TITLE attribute on the table entity.

        filters -- An instance of the Filters class that provides
            information about the desired I/O filters applicable to
            the leaves that hangs directly from this new group (unless
            other filters properties are specified for these leaves,
            of course). Besides, if you do not specify filter
            properties for its child groups, they will inherit these
            ones.
              
        """

        group = self.getNode(where, classname = 'Group')
        setattr(group, name, Group(title, new=1, filters=filters))
        object = getattr(group, name)
        return object

    def createTable(self, where, name, description, title="",
                    filters=None, expectedrows=10000,
                    compress=None, complib=None):  # Deprecated

        """Create a new Table instance with name "name" in "where" location.
        
        "where" parameter can be a path string, or another group
        instance.

        Keyword arguments:

        where -- The parent group where the new table will hang
            from. "where" parameter can be a path string (for example
            "/level1/leaf5"), or Group instance.

        name -- The name of the new table.

        description -- A IsDescription subclass or a dictionary where
            the keys are the field names, and the values the type
            definitions. And it can be also a RecArray object (from
            numarray.records module).

        title -- Sets a TITLE attribute on the table entity.

        filters -- An instance of the Filters class that provides
            information about the desired I/O filters to be applied
            during the life of this object.

        expectedrows -- An user estimate about the number of rows that
            will be on table. If not provided, the default value is
            10000. If you plan to save bigger tables try providing a
            guess; this will optimize the HDF5 B-Tree creation and
            management process time and the amount of memory used.

        """

        group = self.getNode(where, classname = 'Group')
        filters = _checkFilters(filters, compress, complib)
        object = Table(description, title, filters, expectedrows)
        setattr(group, name, object)
        return object
    
    def createArray(self, where, name, object, title = ""):
        
        """Create a new instance Array with name "name" in "where" location.

        Keyword arguments:

        where -- The parent group where the new table will hang
            from. "where" parameter can be a path string (for example
            "/level1/leaf5"), or Group instance.

        name -- The name of the new array.

        object -- The (regular) object to be saved. It can be any of
            NumArray, CharArray, Numeric, List, Tuple, String, Int of
            Float types, provided that they are regular (i.e. they are
            not like [[1,2],2]).

        title -- Sets a TITLE attribute on the array entity.

            """
            
        group = self.getNode(where, classname = 'Group')
        Object = Array(object, title)
        setattr(group, name, Object)
        return Object


    def createEArray(self, where, name, atom, title = "",
                     filters=None, expectedrows = 1000,
                     compress=None, complib=None):
        
        """Create a new instance EArray with name "name" in "where" location.

        Keyword arguments:

        where -- The parent group where the new table will hang
            from. "where" parameter can be a path string (for example
            "/level1/leaf5"), or Group instance.

        name -- The name of the new array.

        atom -- An Atom instance representing the shape, type and
            flavor of the atomic objects to be saved. One of the shape
            dimensions must be 0. The dimension being 0 means that the
            resulting EArray object can be extended along it.

        title -- Sets a TITLE attribute on the array entity.

        filters -- An instance of the Filters class that provides
            information about the desired I/O filters to be applied
            during the life of this object.

        expectedrows -- In the case of enlargeable arrays this
            represents an user estimate about the number of row
            elements that will be added to the growable dimension in
            the EArray object. If not provided, the default value is
            1000 rows. If you plan to create both much smaller or much
            bigger EArrays try providing a guess; this will optimize
            the HDF5 B-Tree creation and management process time and
            the amount of memory used.

            """

        group = self.getNode(where, classname = 'Group')
        filters = _checkFilters(filters, compress, complib)
        Object = EArray(atom, title, filters, expectedrows)
        setattr(group, name, Object)
        return Object


    def createVLArray(self, where, name, atom=None, title="",
                      filters=None, expectedsizeinMB=1.0,
                      compress=None, complib=None):
        
        """Create a new instance VLArray with name "name" in "where" location.

        Keyword arguments:

        where -- The parent group where the new table will hang
            from. "where" parameter can be a path string (for example
            "/level1/leaf5"), or Group instance.

        name -- The name of the new array.

        title -- Sets a TITLE attribute on the array entity.

        atom -- A Atom object representing the shape, type and flavor
            of the atomic object to be saved.

        filters -- An instance of the Filters class that provides
            information about the desired I/O filters to be applied
            during the life of this object.

        expectedsizeinMB -- An user estimate about the size (in MB) in
            the final VLArray object. If not provided, the default
            value is 1 MB.  If you plan to create both much smaller or
            much bigger Arrays try providing a guess; this will
            optimize the HDF5 B-Tree creation and management process
            time and the amount of memory used.

            """

        group = self.getNode(where, classname = 'Group')
        if atom == None:
                raise ValueError, \
                      "please, expecify an atom argument."
        filters = _checkFilters(filters, compress, complib)
        Object = VLArray(atom, title, filters, expectedsizeinMB)
        setattr(group, name, Object)
        return Object


    def getNode(self, where, name = "", classname = ""):
        
        """Returns the object node "name" under "where" location.

        "where" can be a path string or Group instance. If "where"
        doesn't exists or has not a child called "name", a LookupError
        error is raised. If "name" is a null string (""), or not
        supplied, this method assumes to find the object in
        "where". If a "classname" parameter is supplied, returns only
        an instance of this class name. Allowed names in "classname"
        are: 'Group', 'Leaf', 'Table', 'Array', 'EArray', 'VLArray'
        and 'UnImplemented'."""

        # To find out the caller
        #print repr(sys._getframe(1).f_code.co_name)
        if isinstance(where, str):
            # Get rid of a possible trailing "/"
            if len(where) > 1 and where[-1] == "/":
                where = where[:-1]
            # This is a string pathname. Get the object ...
            if name:
                if where == "/":
                    strObject = "/" + name
                else:
                    strObject = where + "/" + name
            else:
                strObject = where
            # Get the object pointed by strObject path
            if strObject in self.objects:
                object = self.objects[strObject]
            else:
                # We didn't find the pathname in the object tree.
                # This should be signaled as an error!.
                raise LookupError, \
                      "\"%s\" pathname not found in file: '%s'." % \
                      (strObject, self.filename)
                      
        elif isinstance(where, Group):
            if name:
                object = getattr(where, name)
            else:
                object = where
                
        elif isinstance(where, Leaf):
            
            if name:
                raise LookupError, \
"""'where' parameter (with value '%s') is a Leaf instance in file '%s' so it cannot have a 'name' child node (with value '%s')""" % \
(where, self.filename, name)

            else:
                object = where
                
        else:
            raise TypeError, "Wrong 'where' parameter type (%s)." % \
                  (type(where))
            
        # Finally, check if this object is a classname instance
        if classname:
            classobj = eval(classname)
            if isinstance(object, classobj):
                return object
            else:
                warnings.warn( \
"""\n  A %s() instance cannot be found at "%s".
  Instead, a %s() object has been found there.""" % \
(classname, object._v_pathname, object.__class__.__name__), UserWarning)
                return -1
        return object

    def renameNode(self, where, newname, name = ""):
        """Rename the object node "name" under "where" location.

        "where" can be a path string or Group instance. If "where"
        doesn't exists or has not a child called "name", a LookupError
        error is raised. If "name" is a null string (""), or not
        supplied, this method assumes to find the object in "where".
        "newname" is the new name of be assigned to the node.
        
        """

        # Get the node to be renamed
        object = self.getNode(where, name=name)
        if isinstance(object, Group):
            object._f_rename(newname)
        else:
            object.rename(newname)
        
    def removeNode(self, where, name = "", recursive = 0):
        """Removes the object node "name" under "where" location.

        "where" can be a path string or Group instance. If "where"
        doesn't exists or has not a child called "name", a LookupError
        error is raised. If "name" is a null string (""), or not
        supplied, this method assumes to find the object in "where".
        If "recursive" is zero or not supplied, the object will be
        removed only if it has not children. If "recursive" is true,
        the object and all its descendents will be completely removed.

        """

        # Get the node to be removed
        object = self.getNode(where, name=name)
        if isinstance(object, Group):
            object._f_remove(recursive)
        else:
            object.remove()            
        
    def getAttrNode(self, where, attrname, name = ""):
        """Returns the attribute "attrname" of node "where"."name".

        "where" can be a path string or Group instance. If "where"
        doesn't exists or has not a child called "name", a LookupError
        error is raised. If "name" is a null string (""), or not
        supplied, this method assumes to find the object in "where".
        "attrname" is the name of the attribute to get.
        
        """

        # Get the node to be renamed
        object = self.getNode(where, name=name)
        if isinstance(object, Group):
            return object._f_getAttr(attrname)
        else:
            return object.getAttr(attrname)
            
    def setAttrNode(self, where, attrname, attrvalue, name=""):
        """Set the attribute "attrname" of node "where"."name".

        "where" can be a path string or Group instance. If "where"
        doesn't exists or has not a child called "name", a LookupError
        error is raised. If "name" is a null string (""), or not
        supplied, this method assumes to find the object in "where".
        "attrname" is the name of the attribute to set and "attrvalue"
        its value.
        
        """

        # Get the node to be renamed
        object = self.getNode(where, name=name)
        if isinstance(object, Group):
            object._f_setAttr(attrname, attrvalue)
        else:
            object.setAttr(attrname, attrvalue)
        
    def copyAttrs(self, where, name="", dstNode=None):
        """Copy the attributes from node "where"."name" to "dstNode".

        "where" can be a path string or Group instance. If "where"
        doesn't exists or has not a child called "name", a LookupError
        error is raised. If "name" is a null string (""), or not
        supplied, this method assumes to find the object in "where".
        "dstNode" is the destination and can be whether a path string
        or a Node object.
        
        """

        # Get the source node
        srcObject = self.getNode(where, name=name)
        dstObject = self.getNode(dstNode)
        if isinstance(srcObject, Group):
            object._v_attrs._f_copy(dstNode)
        else:
            object.attrs._f_copy(dstNode)
        
    def copyFile(self, dstFilename=None, title=None,
                 filters=None, copyuserattrs=1, overwrite=0):
        """Copy the contents of this file to "dstFilename".

        "dstFilename" must be a path string. If this file already
        exists and overwrite is 1, it is overwritten. The default is
        not overwriting. It returns a tuple (ngroups, nleafs)
        indicating the number of copied groups and leafs.

        This copy also has the effect of compacting the destination
        file during the process.
        
        """

        if os.path.isfile(dstFilename) and not overwrite:
            raise IOError, "The file '%s' already exists and will not be overwritten. Assert the overwrite parameter if you want overwrite it." % (dstFilename)

        if title == None: title = self.title
        if title == None: title = ""  # If still None, then set to empty string
        if filters == None: filters = self.filters
        dstFileh = openFile(dstFilename, mode="w", title=title)
        # Copy the user attributes of the root group
        self.root._v_attrs._f_copy(dstFileh.root)
        # Copy all the hierarchy
        ngroups, nleafs = self.root._f_copyChilds(dstFileh.root, recursive=1,
                                                  filters=filters,
                                                  copyuserattrs=copyuserattrs)
        # Finally, close the file
        dstFileh.close()
        return (ngroups, nleafs)
        
    def listNodes(self, where, classname = ""):
        
        """Returns a list with all the object nodes (Group or Leaf)
        hanging from "where". The list is alphanumerically sorted by
        node name.  "where" can be a path string or Group instance. If
        a "classname" parameter is supplied, only instances of this
        class (or subclasses of it) are returned. The only supported
        classes in "classname" are 'Group', 'Leaf', 'Table', 'Array',
        'EArray', 'VLArray' and 'UnImplemented'."""

        group = self.getNode(where, classname = 'Group')
        if group <> -1:
            return group._f_listNodes(classname)
        else:
            return []
    
    def __iter__(self, where="/", classname=""):
        """Iterate over the nodes in the object tree."""

        return self._iterTree(where, classname)

    def _iterTree(self, where="/", classname=""):
        """Iterate over the nodes in the object tree."""
        
        if classname == "Group":
            for group in self.walkGroups(where):
                yield group
        elif classname in [None, ""]:
            yield self.getNode(where, "")
            for group in self.walkGroups(where):
                for leaf in self.listNodes(group, ""):
                    yield leaf
        else:
            for group in self.walkGroups(where):
                for leaf in self.listNodes(group, classname):
                    yield leaf
                
    def __call__(self, where="/", classname=""):
        """Iterate over the nodes in the object tree.

        If "where" supplied, the iteration starts from this group.
        If "classname" is supplied, only instances of this class are
        returned.

        """

        return self.__iter__(where, classname)

    def walkGroups(self, where = "/"):
        """Returns the list of Groups (not Leaves) hanging from "where".

        If "where" is not supplied, the root object is taken as
        origin. The groups are returned from top to bottom, and
        alphanumerically sorted when in the same level. The list of
        groups returned includes "where" (or the root object) as well.

        """
        
        group = self.getNode(where, classname = 'Group')
        return group._f_walkGroups()

                    
    def flush(self):
        
        """Flush all the objects on all the HDF5 objects tree."""

        for group in self.walkGroups(self.root):
            for leaf in self.listNodes(group, classname = 'Leaf'):
                leaf.flush()
            
                
    def close(self):
        
        """Close all the objects in HDF5 file and close the file."""

        # If the file is already closed, return immediately
        if not self.isopen:
            return

        # Get a list of groups on tree
        listgroups = self.root._f_getListTree()
        for group in self.walkGroups(self.root):
            for leaf in self.listNodes(group, classname = 'Leaf'):
                leaf.close()
            #print "File: Closing Group ==>", group
            #if group._v_pathname <> "/":
            #    group._f_close()
            group._f_close()
            
        # Delete the root object (this should recursively delete the
        # object tree)
        #         del self.root    # not necessary

        # Pass Mr proper
#         for group in listgroups:
#             group.__dict__.clear()
#             group._v_childs.clear()
#         self.groups.clear()

        # Close the file
        self._closeFile()
                    
        # Set the flag to indicate that the file is closed
        self.isopen = 0

        # After the objects are disconnected, destroy the
        # object dictionary using the brute force ;-)
        # This should help to the garbage collector
        #self.__dict__.clear()

        return

    def __str__(self):
        """Returns a string representation of the object tree"""
        
        # Print all the nodes (Group and Leaf objects) on object tree
        date = time.asctime(time.localtime(os.stat(self.filename)[8]))
        astring = "Filename: " + repr(self.filename) + ' '
        if self.title <> "unknown":
            astring += "Title: '"+self.title+"'" + ' '
        astring += ", Last modif.: " + repr(date) + ' '
        astring += ', rootUEP=' + repr(self.rootUEP)
        astring += ', filters=' + repr(self.filters)

        if self.format_version <> "unknown":
            astring += ", Format version: " + self.format_version + '\n'

        for group in self.walkGroups("/"):
            astring += str(group) + '\n'
            for leaf in self.listNodes(group, 'Leaf'):
                astring += str(leaf) + '\n'
                
        return astring

    def __repr__(self):
        """Returns a more complete representation of the object tree"""
        
        # Print all the nodes (Group and Leaf objects) on object tree
        astring = 'File(filename=' + repr(self.filename) + \
                  ', title=' + repr(self.title) + \
                  ', mode=' + repr(self.mode) + \
                  ', trMap=' + repr(self.trMap) + \
                  ', rootUEP=' + repr(self.rootUEP) + \
                  ', filters=' + repr(self.filters) + \
                  ')\n'
        for group in self.walkGroups("/"):
            astring += str(group) + '\n'
            for leaf in self.listNodes(group, 'Leaf'):
                astring += repr(leaf) + '\n'
                
        return astring
