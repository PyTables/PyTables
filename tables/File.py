#
#       License:        BSD
#       Created:        September 4, 2002
#       Author:  Francesc Alted - falted@openlc.org
#
#       $Source: /home/ivan/_/programari/pytables/svn/cvs/pytables/pytables/tables/File.py,v $
#       $Id: File.py,v 1.30 2003/04/28 17:45:59 falted Exp $
#
########################################################################

"""Open PyTables files and create the object tree (if needed).

This module support HDF5 files, on top of which PyTables files are
created, read or extended. If a file exists, an object tree mirroring
their hierarchical structure is created in memory. File class offer
methods to traverse the tree, as well as to create new nodes.

Classes:

    File

Functions:

    openFile(name [, mode = "r"] [, title] [, trMap])

Misc variables:

    __version__
    format_version
    compatible_formats

"""

__version__ = "$Revision: 1.30 $"
format_version = "1.0"                     # File format version we write
compatible_formats = []                    # Old format versions we can read

import sys
import types
import warnings
import os.path
from fnmatch import fnmatch

import hdf5Extension
from Group import Group
from Leaf import Leaf
from Table import Table
from Array import Array
import numarray

def openFile(filename, mode="r", title="", trMap={}):

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

    title -- (Optional) A TITLE string attribute will be set on the
             root group with its value.

    trMap -- (Optional) A dictionary to map names in the object tree
             into different HDF5 names in file. The keys are the
             Python names, while the values are the HDF5 names. This
             is useful when you need to name HDF5 nodes with invalid
             or reserved words in Python.

    """
    
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
    return File(path, mode, title, new, trMap)


class File(hdf5Extension.File):
    """Returns an object describing the file in-memory.

    File class offer methods to browse the object tree, to create new
    nodes, to rename them, to delete as well as to assign and read
    attributes.

    Methods:

        createGroup(where, name[, title])
        createTable(where, name, description [, title]
                    [, compress] [, expectedrows])
        createArray(where, name, arrayObject, [, title])
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
        mode -- mode in which the filename was opened
        title -- the title of the root group in file
        root -- the root group in file
        trMap -- the mapping between python and HDF5 domain names
        objects -- Dictionary with all objects (groups or leaves) on tree.
        groups -- Dictionary with all object groups on tree.
        leaves -- Dictionary with all object leaves on tree.

    """

    def __init__(self, filename, mode="r", title="",
                 new=1, trMap={}):
        
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
        
        # _v_new informs if this file is old or new
        self._v_new = new
        # Assign the trMap and build the reverse translation
        self.trMap = trMap
        
        # Get the root group from this file
        self.root = self.__getRootGroup()

        # Set the flag to indicate that the file has been opened
        self._closed = 0

        return

    
    def __getRootGroup(self):
        
        """Returns a Group instance which will act as the root group
        in the hierarchical tree. If file is opened in "r", "r+" or
        "a" mode, and the file already exists, this method dynamically
        builds a python object tree emulating the structure present on
        file."""
          
        global format_version
        global compatible_formats
        
        self._v_groupId = self._getFileId()
        self._v_depth = 0

        # Global dictionaries for the file paths.
        # These are used to keep track of all the childs and group objects
        # in tree object. They are dictionaries that will use the pathnames
        # as keys and the actual objects as values.
        # That way we can find objects in the object tree easily and quickly.
        self.groups = {}
        self.leaves = {}
        self.objects = {}

        root = Group(self._v_new)
        
        # Create new attributes for the root Group instance
        newattr = root.__dict__
        newattr["_v_rootgroup"] = root  # For compatibility with Group
        newattr["_v_groupId"] = self._v_groupId
        newattr["_v_parent"] = self
        newattr["_v_file"] = self
        newattr["_v_depth"] = 1
        newattr["_v_filename"] = self.filename  # Only root group has this

        newattr["_v_name"] = "/"
        newattr["_v_hdf5name"] = "/"  # For root, this is always "/"
        newattr["_v_pathname"] = "/"
        
        # Update global path variables for Group
        self.groups["/"] = root
        self.objects["/"] = root
        
        # Open the root group. We do that always, be the file new or not
        root._g_openGroup(self._v_groupId, "/")

        if self._v_new:
            # Set the title, class and version attributes
            newattr["_v_title"] = self.title
            newattr["_v_class"] = root.__class__.__name__
            newattr["_v_version"] = "1.0"
            
            # Do the same on disk
            root._g_setGroupAttrStr('TITLE', root._v_title)
            root._g_setGroupAttrStr('CLASS', root._v_class)
            root._g_setGroupAttrStr('VERSION', root._v_version)
            
            # Finally, save the PyTables format version for this file
            self._format_version = format_version
            root._g_setGroupAttrStr('PYTABLES_FORMAT_VERSION', format_version)
        
        else:
            # Firstly, get the PyTables format version for this file
            self._format_version = \
                                root._f_getAttr('PYTABLES_FORMAT_VERSION')
            if self._format_version == None:
                # PYTABLES_FORMAT_VERSION attribute is not present
                self._format_version = "unknown"
                          
            # Get the title, class and version attributes
            # (only for root)
            root.__dict__["_v_title"] = \
                      root._f_getAttr('TITLE')
            self.title = root._v_title   # This is a standard File attribute
            root.__dict__["_v_class"] = \
                      root._f_getAttr('CLASS')
            root.__dict__["_v_version"] = \
                      root._f_getAttr('VERSION')
                      
            # Get all the groups recursively
            root._g_openFile()
        
        return root

    
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
        else:
            raise ValueError,\
            """Parameter 1 can only take 'Group', 'Table' or 'Array' values."""

        group = self.getNode(where, classname = 'Group')
        # Put the object on the tree
        setattr(group, name, object)
        return object

    
    def createGroup(self, where, name, title = ""):
    
        """Create a new Group instance with name "name" in "where" location.
        "where" parameter can be a path string (for example
        "/Particles/TParticle1"), or another Group instance. A TITLE
        attribute will be set on this group if optional "title" parameter is
        passed."""

        group = self.getNode(where, classname = 'Group')
        setattr(group, name, Group(title))
        object = getattr(group, name)
        return object


    def createTable(self, where, name, description, title = "",
                    compress = 0, complib = "zlib", expectedrows = 10000):

        """Create a new Table instance with name "name" in "where" location.
        
        "where" parameter can be a path string, or another group
        instance.

        Keyword arguments:

        where -- The parent group where the new table will
            hang. "where" parameter can be a path string (for
            example "/Particles/TParticle1"), or Group
            instance.

        name -- The name of the new table.

        description -- A IsDescription subclass or a dictionary where
            the keys are the field names, and the values the type
            definitions. And it can be also a RecArray object (from
            recarray module).

        title -- Sets a TITLE attribute on the table entity.

        compress -- Specifies a compress level for data. The allowed
            range is 0-9. A value of 0 disables compression. The
            default is compression level 3, that balances between
            compression effort and CPU consumption.

        complib -- Specifies the compression library to be used. Right
            now, "zlib", "lzo" and "ucl" values are supported.

        expectedrows -- An user estimate about the number of rows
            that will be on table. If not provided, the default value
            is appropiate for tables until 1 MB in size (more or less,
            depending on the row size). If you plan to save bigger
            tables try providing a guess; this will optimize the HDF5
            B-Tree creation and management process time and memory
            used.

        """
    
        group = self.getNode(where, classname = 'Group')
        if complib not in ["zlib","lzo","ucl"]:
            raise ValueError, "Wrong \'complib\' parameter value: '%s'" % \
                  (str(complib))
        object = Table(description, title, compress, complib, expectedrows)
        setattr(group, name, object)
        return object

    
    def createArray(self, where, name, object,
                    title = "", atomictype = 1):
        
        """Create a new instance Array with name "name" in "where" location.

        Keyword arguments:

        where -- The parent group where the new table will
            hang. "where" parameter can be a path string (for
            example "/Particles/TParticle1"), or Group
            instance.

        name -- The name of the new array.

        object -- The (regular) object to be saved. It can be any of
            NumArray, CharArray, Numeric, List, Tuple, String, Int of
            Float types, provided that they are regular (i.e. they are
            not like [[1,2],2]).

        title -- Sets a TITLE attribute on the array entity.

        atomictype -- is a boolean that specifies the underlying HDF5
            type; if 1 an atomic data type (i.e. it can't be
            decomposed in smaller types) is used; if 0 an HDF5 array
            datatype is used. The created object is returned."""

        group = self.getNode(where, classname = 'Group')
        Object = Array(object, title, atomictype)
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
        are: 'Group', 'Leaf', 'Table' and 'Array'."""

        if isinstance(where, str):
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
                      "\"%s\" pathname not found in object tree." % \
                      (strObject)
                      
        elif isinstance(where, Group):
            if name:
                object = getattr(where, name)
            else:
                object = where
                
        elif isinstance(where, Leaf):
            
            if name:
                raise LookupError, \
"""'where' parameter (with value \'%s\') is a Leaf instance so it cannot \
have a 'name' child node (with value \'%s\')""" % (where, name)

            else:
                object = where
                
        else:
            raise TypeError, "Wrong \'where\' parameter type (%s)." % \
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
        if object._v_class == "Group":
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
        if object._v_class == "Group":
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
        if object._v_class == "Group":
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
        if object._v_class == "Group":
            object._f_setAttr(attrname, attrvalue)
        else:
            object.setAttr(attrname, attrvalue)
        
    def listNodes(self, where, classname = ""):
        
        """Returns a list with all the object nodes (Group or Leaf) hanging
        from "where". The list is alphanumerically sorted by node name.
        "where" can be a path string or Group instance. If a "classname"
        parameter is supplied, the iterator will return only instances of
        this class (or subclasses of it). The only supported classes in
        "classname" are 'Group' and 'Leaf'."""

        group = self.getNode(where, classname = 'Group')
        if group <> -1:
            return group._f_listNodes(classname)
        else:
            return []
    

    def walkGroups(self, where = "/"):
        
        """Recursively obtains Groups (not Leaves) hanging from "where".

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
        
        for group in self.walkGroups(self.root):
            for leaf in self.listNodes(group, classname = 'Leaf'):
                leaf.close()
                pass
            group._f_close()

        self._closeFile()
                    
        # Delete the root object (this should recursively delete the
        # object tree)
        del self.root

        # Set the flag to indicate that the file is closed
        self._closed = 1

    def __str__(self):
        
        """Returns a string representation of the object tree"""
        
        # Print all the nodes (Group and Leaf objects) on object tree
        string = 'Filename: ' + self.filename + " " + repr(self.title) + '\n'
        # string = 'Filename: ' + self.filename + ' \\\\'
        # string += ' Title: ' + str(self.title) + ' \\\\'
        # string += ' Format version: ' + str(self._format_version) + '\n'
        for group in self.walkGroups("/"):
            string += str(group) + '\n'
            for leaf in self.listNodes(group, 'Leaf'):
                string += str(leaf) + '\n'
                
        return string

    def __repr__(self):
        
        """Returns a more complete representation of the object tree"""
        
        # Print all the nodes (Group and Leaf objects) on object tree
        string = 'Filename: ' + self.filename + " " + repr(self.title) + '\n'
        string += '  mode = ' + repr(self.mode) + '\n'
        string += '  trMap = ' + str(self.trMap) + '\n'
        for group in self.walkGroups("/"):
            string += str(group) + '\n'
            for leaf in self.listNodes(group, 'Leaf'):
                string += repr(leaf) + '\n'
                
        return string

    def _g_del__(self):
        """Delete some objects"""
        #print "Deleting File object"
        pass

