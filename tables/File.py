#
#       License:        BSD
#       Created:        September 4, 2002
#       Author:  Francesc Alted - falted@openlc.org
#
#       $Source: /home/ivan/_/programari/pytables/svn/cvs/pytables/pytables/tables/File.py,v $
#       $Id: File.py,v 1.65 2004/01/14 10:39:14 falted Exp $
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

__version__ = "$Revision: 1.65 $"
#format_version = "1.0" # Initial format
#format_version = "1.1" # Changes in ucl compression
format_version = "1.2"  # Support for enlargeable arrays and VLA's
compatible_formats = [] # Old format versions we can read
                        # Empty means that we support all the old formats

from __future__ import generators

import sys
import types
import warnings
import time
import os, os.path
from fnmatch import fnmatch

import hdf5Extension
from Group import Group
from Leaf import Leaf
from Table import Table
from Array import Array
from EArray import EArray
from VLArray import VLArray
from UnImplemented import UnImplemented
from AttributeSet import AttributeSet
import numarray

def openFile(filename, mode="r", title="", trMap={}, rootUEP="/"):

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

    rootUEP -- (Optional) The root User Entry Point. It is a group in
            the file hierarchy which is taken as the starting point to
            create the object tree. The group can be whatever existing
            path in the file. If it does not exist, a RuntimeError is
            issued.

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
    return File(path, mode, title, new, trMap, rootUEP, isPTFile)


class File(hdf5Extension.File, object):
    """Returns an object describing the file in-memory.

    File class offer methods to browse the object tree, to create new
    nodes, to rename them, to delete as well as to assign and read
    attributes.

    Methods:

        createGroup(where, name[, title])
        createTable(where, name, description [, title]
                    [, compress] [, complib] [, shuffle]
                    [, fletcher32] [, expectedrows])
        createArray(where, name, arrayObject, [, title])
        createEArray(where, name, object [, title]
                     [, compress] [, complib] [, shuffle]
                     [, fletcher32] [, expectedrows])
        createVLArray(where, name, atom [, title]
                      [, compress] [, complib] [, shuffle]
                      [, fletcher32] [, expectedrows])
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

    def __init__(self, filename, mode="r", title="",
                 new=1, trMap={}, rootUEP="/", isPTFile=1):
        
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
            
            # Save the rootGroup attributes on disk
            attrsRoot._g_setAttrStr('TITLE',  self.title)
            attrsRoot._g_setAttrStr('CLASS', "GROUP")
            attrsRoot._g_setAttrStr('VERSION', "1.0")
            
            # Finally, save the PyTables format version for this file
            self.format_version = format_version
            attrsRoot._g_setAttrStr('PYTABLES_FORMAT_VERSION', format_version)
            # Add these attributes to the dictionary
            attrsRoot._v_attrnames.extend(['TITLE','CLASS','VERSION',
                                           'PYTABLES_FORMAT_VERSION'])
            # Add these attributes to the dictionary
            attrsRoot._v_attrnames.extend(['TITLE','CLASS','VERSION',
                                           'PYTABLES_FORMAT_VERSION'])
            attrsRoot._v_attrnamessys.extend(['TITLE','CLASS',
                                              'VERSION',
                                              'PYTABLES_FORMAT_VERSION'])
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
            self.title =  hdf5Extension.read_f_attr(self._v_objectID, 'TITLE')
                      
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

    
    def createGroup(self, where, name, title = ""):
    
        """Create a new Group instance with name "name" in "where" location.
        "where" parameter can be a path string (for example
        "/level1/leaf5"), or another Group instance. A TITLE
        attribute will be set on this group if optional "title" parameter is
        passed."""

        group = self.getNode(where, classname = 'Group')
        setattr(group, name, Group(title))
        object = getattr(group, name)
        return object


    def createTable(self, where, name, description, title = "",
                    compress = 0, complib = "zlib", shuffle = 1,
                    fletcher32 = 0, expectedrows = 10000):

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

        compress -- Specifies a compress level for data. The allowed
            range is 0-9. A value of 0 disables compression and this
            is the default.

        complib -- Specifies the compression library to be used. Right
            now, "zlib", "lzo" and "ucl" values are supported.

        shuffle -- Whether or not to use the shuffle filter in the
            HDF5 library. This is normally used to improve the
            compression ratio. A value of 0 disables shuffling and 1
            makes it active. The default value depends on whether
            compression is enabled or not; if compression is enabled,
            shuffling defaults to be active, else shuffling is
            disabled.

        fletcher32 -- Whether or not to use the fletcher32 filter in
            the HDF5 library. This is used to add a checksum on each
            data chunk. A value of 0 disables the checksum and it is
            the default.

        expectedrows -- An user estimate about the number of rows that
            will be on table. If not provided, the default value is
            10000. If you plan to save bigger tables try providing a
            guess; this will optimize the HDF5 B-Tree creation and
            management process time and the amount of memory used.

        """
    
        group = self.getNode(where, classname = 'Group')
        if complib not in ["zlib","lzo","ucl"]:
            raise ValueError, "Wrong \'complib\' parameter value: '%s'" % \
                  (str(complib))
        object = Table(description, title, compress, complib, shuffle,
                       fletcher32, expectedrows)
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


    def createEArray(self, where, name, object, title = "",
                    compress = 0, complib = "zlib", shuffle = 1,
                    fletcher32 = 0, expectedrows = 1000):
        
        """Create a new instance EArray with name "name" in "where" location.

        Keyword arguments:

        where -- The parent group where the new table will hang
            from. "where" parameter can be a path string (for example
            "/level1/leaf5"), or Group instance.

        name -- The name of the new array.

        object -- An object describing the kind of objects that you
            can append to the EArray. It can be an instance of any of
            NumArray, CharArray or Numeric classes and one of its
            dimensions must be 0. The dimension being 0 means that the
            resulting EArray object can be extended along it. Multiple
            enlargeable dimensions are not supported right now.

        title -- Sets a TITLE attribute on the array entity.

        compress -- Specifies a compress level for data. The allowed
            range is 0-9. A value of 0 disables compression and this
            is the default.

        complib -- Specifies the compression library to be used. Right
            now, "zlib", "lzo" and "ucl" values are supported.

        shuffle -- Whether or not to use the shuffle filter in the
            HDF5 library. This is normally used to improve the
            compression ratio. A value of 0 disables shuffling and 1
            makes it active. The default value depends on whether
            compression is enabled or not; if compression is enabled,
            shuffling defaults to be active, else shuffling is
            disabled.

        fletcher32 -- Whether or not to use the fletcher32 filter in
            the HDF5 library. This is used to add a checksum on each
            data chunk. A value of 0 disables the checksum and it is
            the default.

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
        Object = EArray(object, title,
                        compress, complib, shuffle, fletcher32, expectedrows)
        setattr(group, name, Object)
        return Object


    def createVLArray(self, where, name, atom = None, title = "",
                      compress = 0, complib = "zlib", shuffle = 1,
                      fletcher32 = 0, expectedsizeinMB = 1.0):
        
        """Create a new instance VLArray with name "name" in "where" location.

        Keyword arguments:

        where -- The parent group where the new table will hang
            from. "where" parameter can be a path string (for example
            "/level1/leaf5"), or Group instance.

        name -- The name of the new array.

        title -- Sets a TITLE attribute on the array entity.

        atom -- A Atom object representing the shape, type and flavor
            of the atomic object to be saved.

        compress -- Specifies a compress level for data. The allowed
            range is 0-9. A value of 0 disables compression and this
            is the default.

        complib -- Specifies the compression library to be used. Right
            now, "zlib", "lzo" and "ucl" values are supported.

        shuffle -- Whether or not to use the shuffle filter in the
            HDF5 library. This is normally used to improve the
            compression ratio. A value of 0 disables shuffling and 1
            makes it active. The default value depends on whether
            compression is enabled or not; if compression is enabled,
            shuffling defaults to be active, else shuffling is
            disabled.

        fletcher32 -- Whether or not to use the fletcher32 filter in
            the HDF5 library. This is used to add a checksum on each
            data chunk. A value of 0 disables the checksum and it is
            the default.

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
        Object = VLArray(atom, title, compress, complib, shuffle,
                         fletcher32, expectedsizeinMB)
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

        # To find out the caller
        #print repr(sys._getframe(1).f_code.co_name)
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
        
    def listNodes(self, where, classname = ""):
        
        """Returns a list with all the object nodes (Group or Leaf)
        hanging from "where". The list is alphanumerically sorted by
        node name.  "where" can be a path string or Group instance. If
        a "classname" parameter is supplied, only instances of this
        class (or subclasses of it) are returned. The only supported
        classes in "classname" are 'Group' and 'Leaf'."""

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
        
        assert classname in [None, "", "Group", "Leaf", "Table", "Array"], \
               "Incorrect specification of 'classname'"
        
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
#         astring = "Filename: " + repr(self.filename) + \
#                   " Title: " + repr(self.title) + \
#                   " Last modif.: " + repr(date) + \
#                   '\n'
        astring = "Filename: " + repr(self.filename)
        if self.format_version <> "unknown":
            astring += " Fmt: " + self.format_version + " "
        if self.title <> "unknown":
            astring += self.title + " "
        astring += "Last modif.: " + repr(date) + '\n'

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
                  ')\n'
        for group in self.walkGroups("/"):
            astring += str(group) + '\n'
            for leaf in self.listNodes(group, 'Leaf'):
                astring += repr(leaf) + '\n'
                
        return astring

    def _g_del__(self):
        """Delete some objects"""
        #print "Deleting File object"
        pass

