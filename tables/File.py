#
#       License:        BSD
#       Created:        September 4, 2002
#       Author:  Francesc Altet - faltet@carabos.com
#
#       $Source: /home/ivan/_/programari/pytables/svn/cvs/pytables/pytables/tables/File.py,v $
#       $Id$
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

import warnings
import time
import os, os.path

import tables.hdf5Extension as hdf5Extension
from tables.constants import MAX_UNDO_PATH_LENGTH
from tables.registry import classNameDict
from tables.exceptions import \
     NodeError, NoSuchNodeError, UndoRedoError, UndoRedoWarning
from tables.utils import joinPath
import tables.undoredo as undoredo
from tables.IsDescription import IsDescription, UInt8Col, StringCol
from tables.Node import Node
from tables.Group import Group, RootGroup
from tables.Group import TransactionGroupG, TransactionG, MarkG
from tables.Leaf import Leaf, Filters
from tables.Table import Table
from tables.Array import Array
from tables.EArray import EArray
from tables.VLArray import VLArray



__version__ = "$Revision: 1.96 $"


#format_version = "1.0" # Initial format
#format_version = "1.1" # Changes in ucl compression
#format_version = "1.2"  # Support for enlargeable arrays and VLA's
                        # 1.2 was introduced in pytables 0.8
format_version = "1.3"  # Support for indexes in Tables
                        # 1.3 was introduced in pytables 0.9
compatible_formats = [] # Old format versions we can read
                        # Empty means that we support all the old formats


# Opcodes for do-undo actions
_opToCode = {"MARK":    0,
           "CREATE":  1,
           "REMOVE":  2,
           "MOVE":    3,
           "ADDATTR": 4,
           "DELATTR": 5,
           }

_codeToOp = ["MARK", "CREATE", "REMOVE", "MOVE", "ADDATTR", "DELATTR"]


# Paths and names for hidden nodes related with transactions.
_transVersion = '1.0'

_transGroupParent = '/'
_transGroupName   = '_p_transactions'
_transGroupPath   = joinPath(_transGroupParent, _transGroupName)

_actionLogParent = _transGroupPath
_actionLogName   = 'actionlog'
_actionLogPath   = joinPath(_actionLogParent, _actionLogName)

_transParent = _transGroupPath
_transName   = 't%d'  # %d -> transaction number
_transPath   = joinPath(_transParent, _transName)

_markParent = _transPath
_markName   = 'm%d'  # %d -> mark number
_markPath   = joinPath(_markParent, _markName)

_shadowParent = _markPath
_shadowName   = 'a%d'  # %d -> action number
_shadowPath   = joinPath(_shadowParent, _shadowName)



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
        raise TypeError, "filter parameter has to be None or a Filter instance and the passed type is: '%s'" % type(filters)
    return fprops


def copyFile(srcFilename = None, dstFilename = None, title = None,
             filters = None, copyuserattrs = True, overwrite = False,
             stats = None):
    """Copy srcFilename to dstFilename

    The "srcFilename" should exist and "dstFilename" should not. But
    if "dsFilename" exists and "overwrite" is true, it is
    overwritten. "title" lets you put another title to the destination
    file. "copyuserattrs" specifies whether the user attrs in origin
    nodes should be copied or not; the default is copy them. Finally,
    specifying a "filters" parameter overrides the original filter
    properties of nodes in "srcFilename".


    The optional keyword argument 'stats' may be used to collect statistics on
    the copy process.  When used, it should be a dictionary whith keys
    'groups', 'leaves' and 'bytes' having a numeric value.  Their values will
    be incremented to reflect the number of groups, leaves and bytes,
    respectively, that have been copied in the operation.
    """

    # Open the src file
    srcFileh = openFile(srcFilename, mode="r")

    # Copy it to the destination
    srcFileh.copyFile(
        dstFilename, title = title, filters = filters,
        copyuserattrs = copyuserattrs, overwrite = overwrite, stats = stats)

    # Close the source file
    srcFileh.close()


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

# The file extension warning commmented out because a suggestion made
# by people at GL suggestion
#     if not (fnmatch(path, "*.h5") or
#             fnmatch(path, "*.hdf") or
#             fnmatch(path, "*.hdf5")):
#         warnings.warn( \
# """filename '%s'should have one of the next file extensions
#   '.h5', '.hdf' or '.hdf5'. Continuing anyway.""" % path)

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
                warnings.warn("""\
``%s`` exists and is an HDF5 file, but does not have a PyTables format. \
Trying to guess what's there using HDF5 metadata. \
I can't promise you getting the correct objects, but I will do my best!"""
                              % (path,))

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
                warnings.warn("""\
``%s`` exists and is an HDF5 file, but does not have a PyTables format. \
Trying to guess what's there from HDF5 metadata. \
I can't promise you getting the correct object, but I will do my best!"""
                              % (path,))
                isPTFile = 0
    else:
        raise ValueError, \
            """mode can only take the new values: "r", "r+", "w" and "a" """

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

    """
    In-memory representation of a PyTables file.

    An instance of this class is returned when a PyTables file is opened
    with the `openFile()` function.  It offers methods to manipulate
    (create, rename, delete...)  nodes and handle their attributes, as
    well as methods to traverse the object tree.  The *user entry point*
    to the object tree attached to the HDF5 file is represented in the
    ``rootUEP`` attribute.  Other attributes are available.

    `File` objects support an *Undo/Redo mechanism* which can be enabled
    with the `enableUndo()` method.  Once the Undo/Redo mechanism is
    enabled, explicit *marks* (with an optional unique name) can be set
    on the state of the database using the `mark()` method.  There are
    two implicit marks which are always available: the initial mark (0)
    and the final mark (-1).  Both the identifier of a mark and its name
    can be used in *undo* and *redo* operations.

    Hierarchy manipulation operations (node creation, movement and
    removal) and attribute handling operations (attribute setting and
    deleting) made after a mark can be undone by using the `undo()`
    method, which returns the database to the state of a past mark.  If
    `undo()` is not followed by operations that modify the hierarchy or
    attributes, the `redo()` method can be used to return the database
    to the state of a future mark.  Else, future states of the database
    are forgotten.

    Please note that data handling operations can not be undone nor
    redone by now.  Also, hierarchy manipulation operations on nodes
    that do not support the Undo/Redo mechanism issue an
    `UndoRedoWarning` *before* changing the database.

    The Undo/Redo mechanism is persistent between sessions and can only
    be disabled by calling the `disableUndo()` method.

    Instance variables:

    filename
        The name of the opened file.
    format_version
        The PyTables version number of this file.
    isopen
        True if the underlying file is open, false otherwise.
    mode
        The mode in which the file was opened.
    title
        The title of the root group in the file.
    trMap
        A dictionary that maps node names between PyTables and HDF5
        domain names.  Its initial values are set from the ``trMap``
        parameter passed to the `openFile()` function.  You can change
        its contents *after* a file is opened and the new map will take
        effect over any new object added to the tree.
    rootUEP
        The UEP (user entry point) group in the file (see the
        `openFile()` function).
    filters
        Default filter properties for the root group (see the `Filters`
	    class).
    root
        The *root* of the object tree hierarchy (a `Group` instance).
    objects
        A dictionary which maps path names to objects, for every node in
	    the tree.
    groups
        A dictionary which maps path names to objects, for every group
	    in the tree.
    leaves
        A dictionary which maps path names to objects, for every leaf in
	    the tree.

    Public methods (file handling):

    * copyFile(dstFilename[, title][, filters][, copyuserattrs]
               [, overwrite][, stats])
    * flush()
    * close()

    Public methods (hierarchy manipulation):

    * createGroup(where, name[, title][, filters])
    * createTable(where, name, description[, title][, filters]
                  [, expectedrows])
    * createVLTable(where, name, description[, title][, filters]
                    [, expectedrows])
    * createArray(where, name, array[, title])
    * createEArray(where, name, atom[, title][, filters]
                   [, expectedrows])
    * createVLArray(where, name, atom[, title][, filters]
                    [, expectedsizeinMB])
    * removeNode(where[, name][, recursive])
    * renameNode(where, newname[, name])
    * moveNode(where, newparent, newname[, name][, overwrite])
    * copyNode(where, newparent, newname[, name][, overwrite]
               [, recursive][, **kwargs])
    * copyChildren(whereSrc, whereDst[, recursive][, filters]
                   [, copyuserattrs][, start][, stop ][, step]
                   [, overwrite][, stats])

    Public methods (tree traversal):

    * getNode(where[, name][,classname])
    * listNodes(where[, classname])
    * walkGroups([where])
    * walkNodes([where][, classname])
    * __contains__(path)

    Public methods (Undo/Redo support):

    isUndoEnabled()
        Is the Undo/Redo mechanism enabled?
    enableUndo([filters])
        Enable the Undo/Redo mechanism.
    disableUndo()
        Disable the Undo/Redo mechanism.
    mark([name])
        Mark the state of the database.
    getCurrentMark()
        Get the identifier of the current mark.
    undo([mark])
        Go to a past state of the database.
    redo([mark])
        Go to a future state of the database.
    goto(mark)
        Go to a specific mark of the database.

    Public methods (attribute handling):

    * getAttrNode(where, attrname[, name])
    * setAttrNode(where, attrname, attrvalue[, name])
    * delAttrNode(where, attrname[, name])
    * copyAttrs(where, dstNode[, name])
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
        self._pttoh5 = trMap
        self._h5topt = {}
        for (ptname, h5name) in self._pttoh5.iteritems():
            if h5name in self._h5topt:
                warnings.warn(
                    "the translation map has a duplicate HDF5 name %r"
                    % (h5name,))
            self._h5topt[h5name] = ptname

        # For the moment Undo/Redo is not enabled.
        self._undoEnabled = False

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

        # If the file is old, check if it has a transaction log
        #print "_allNodes-->", self._allNodes
        if new == 0 and _transGroupPath in self._allNodes:
            # It does. Enable the undo.
            self.enableUndo()

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

        if rootUEP in [None, ""]:
            rootUEP = "/"
        # Save the User Entry Point in a variable class
        self.rootUEP=rootUEP

        # Global dictionaries for the file paths.
        # These are used to keep track of all the children and group objects
        # in tree object. They are dictionaries that will use the pathnames
        # as keys and the actual objects as values.
        # That way we can find objects in the object tree easily and quickly.
        # Hidden nodes are only placed in _allNodes.
        self.groups = {}
        self.leaves = {}
        self.objects = {}
        self._allNodes = {}
        # Create new attributes for the root Group instance
        rootGroup = RootGroup(self, '/', rootUEP, new=self._v_new)
        # Update global path variables for Group
        attrsRoot =  rootGroup._v_attrs   # Shortcut
        # Get some important attributes of file from root Group attributes
        if self._v_new:
            # Finally, save the PyTables format version for this file
            self.format_version = format_version
            attrsRoot._g__setattr('PYTABLES_FORMAT_VERSION', format_version)
        else:
            # Firstly, get the PyTables format version for this file
            self.format_version = hdf5Extension.read_f_attr(
                self._v_objectID, 'PYTABLES_FORMAT_VERSION')
            if not self.format_version or not self._isPTFile:
                # PYTABLES_FORMAT_VERSION attribute is not present
                self.format_version = "unknown"
            # Get the title for the file
            self.title = rootGroup._v_title
            # Get the filters for the file
            if hasattr(attrsRoot, "FILTERS"):
                self.filters = attrsRoot.FILTERS
            else:
                self.filters = Filters()


        return rootGroup


    # This method will go away when the node constructor is in charge of
    # putting it in the tree and logging its creation.
    def _createNode(self, where, name, class_, doCreateNode, log):
        parent = self.getNode(where)  # Does the parent node exist?
        self._checkGroup(parent)  # Is it a group?

        undoEnabled = self.isUndoEnabled()
        canUndoCreate = class_._c_canUndoCreate
        if undoEnabled and not canUndoCreate:
            warnings.warn(
                "creation can not be undone nor redone for this node",
                UndoRedoWarning)

        node = doCreateNode()
        setattr(parent, name, node)

        if log and undoEnabled and canUndoCreate:
            self._log('CREATE', node._v_pathname)
        return node


    def createGroup(self, where, name, title = "", filters = None,
                    _log = True):
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
        def doCreateNode():
            return Group(title, new=True, filters=filters)
        return self._createNode(where, name, Group, doCreateNode, _log)


    def createTable(self, where, name, description, title="",
                    filters=None, expectedrows=10000,
                    compress=None, complib=None,  # Deprecated
                    _log = True):
        """Create a new Table instance with name "name" in "where" location.

        "where" parameter can be a path string, or another group
        instance.

        Keyword arguments:

        where -- The parent group where the new table will hang
            from. "where" parameter can be a path string (for example
            "/level1/leaf5"), or Group instance.

        name -- The name of the new table.

        description -- An IsDescription subclass or a dictionary where
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
        def doCreateNode():
            fprops = _checkFilters(filters, compress, complib)
            return Table(description, title, fprops, expectedrows)
        return self._createNode(where, name, Table, doCreateNode, _log)


    def createVLTable(self, where, name, description, title="",
                      filters=None, expectedrows=10000,
                      _log = True):

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
        def doCreateNode():
            fprops = _checkFilters(filters)
            return VLTable(description, title, fprops, expectedrows)
        return self._createNode(where, name, VLTable, doCreateNode, _log)


    def createArray(self, where, name, object, title = "",
                    _log = True):
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
        def doCreateNode():
            return Array(object, title)
        return self._createNode(where, name, Array, doCreateNode, _log)


    def createEArray(self, where, name, atom, title = "",
                     filters=None, expectedrows = 1000,
                     compress=None, complib=None,
                     _log = True):
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

        expectedrows -- Represents an user estimate about the number
            of row elements that will be added to the growable
            dimension in the EArray object. If not provided, the
            default value is 1000 rows. If you plan to create both
            much smaller or much bigger EArrays try providing a guess;
            this will optimize the HDF5 B-Tree creation and management
            process time and the amount of memory used.

        """
        def doCreateNode():
            fprops = _checkFilters(filters, compress, complib)
            return EArray(atom, title, fprops, expectedrows)
        return self._createNode(where, name, EArray, doCreateNode, _log)


    def createVLArray(self, where, name, atom, title="",
                      filters=None, expectedsizeinMB=1.0,
                      compress=None, complib=None,
                      _log = True):
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
        def doCreateNode():
            fprops = _checkFilters(filters, compress, complib)
            return VLArray(atom, title, fprops, expectedsizeinMB)
        return self._createNode(where, name, VLArray, doCreateNode, _log)


    def getNode(self, where, name=None, classname=None):
        """
        Get the node under `where` with the given `name`.

        `where` can be a `Node` instance or a path string leading to a
        node.  If no `name` is specified, that node is returned.

        If a `name` is specified, this must be a string with the name of
        a node under `where`.  In this case the `where` argument can
        only lead to a `Group` instance (else a `TypeError` is raised).
        The node called `name` under the group `where` is returned.

        In both cases, if the node to be returned does not exist, a
        `NoSuchNodeError` is raised.

        If the `classname` argument is specified, it must be the name of
        a class derived from `Node`.  If the node is found but it is not
        an instance of that class, a `NoSuchNodeError` is also raised.
        """

        # For compatibility with old default arguments.
        if name == '':
            name = None
        if classname == '':
            classname = None

        return self._getNodeFromDict(where, name, classname, self.objects)


    def _getNode(self, where, name=None, classname=None):
        """
        Get the node under `where` with the given `name`.

        This is equivalent to the public `getNode()` method, but it also
        looks up hidden nodes.
        """
        return self._getNodeFromDict(where, name, classname, self._allNodes)


    def _getNodeFromDict(self, where, name, classname, nodedict):
        # Get the parent node.
        if isinstance(where, Node):
            # It is a node instance: use it.
            parent = where
        elif isinstance(where, basestring):  # Pyhton >= 2.3
            # It is a path name: get the respective node.
            if where not in nodedict:
                raise NoSuchNodeError(where)
            parent = nodedict[where]
        else:
            raise TypeError(
                "``where`` is not a string nor a node: %r" % (where,))

        # Get the target node.
        if name is None:
            node = parent
        else:
            if not isinstance(parent, Group):
                raise TypeError("""\
node ``%s`` is not a group; it can not have a child named ``%s``"""
                                % (parent._v_pathname, name))
            node = getattr(parent, name)

        # Finally, check whether the desired node is an instance
        # of the expected class.
        if classname is not None:
            if classname not in classNameDict:
                raise TypeError(
                    "there is no registered node class named ``%s``"
                    % (classname,))

            if not isinstance(node, classNameDict[classname]):
                nPathname = node._v_pathname
                nClassname = node.__class__.__name__
                raise NoSuchNodeError("""\
could not find a ``%s`` node at ``%s``; \
instead, a ``%s`` node has been found there"""
                                      % (classname, nPathname, nClassname))

        return node


    def renameNode(self, where, newname, name=None):
        """
        Rename the given node in place.

        The `where` and `name` arguments work as in `getNode()`,
        referencing the node to be acted upon.  The other arguments work
        as in `Node._f_rename()`.
        """
        obj = self.getNode(where, name=name)
        obj._f_rename(newname)

    def moveNode(self, where, newparent=None, newname=None, name=None,
                 overwrite=False):
        """
        Move or rename the given node.

        The `where` and `name` arguments work as in `getNode()`,
        referencing the node to be acted upon.  The other arguments work
        as in `Node._f_move()`.
        """
        obj = self.getNode(where, name=name)
        obj._f_move(newparent, newname, overwrite)

    def copyNode(self, where, newparent=None, newname=None, name=None,
                 overwrite=False, recursive=False, **kwargs):
        """
        Copy the given node and return the new one.

        The `where` and `name` arguments work as in `getNode()`,
        referencing the node to be acted upon.  The other arguments work
        as in `Node._f_copy()`.
        """
        obj = self.getNode(where, name=name)
        return obj._f_copy(newparent, newname, overwrite, recursive, **kwargs)

    def removeNode(self, where, name=None, recursive=False):
        """
        Remove the given node from the hierarchy.


        The `where` and `name` arguments work as in `getNode()`,
        referencing the node to be acted upon.  The other arguments work
        as in `Node._f_remove()`.
        """
        obj = self.getNode(where, name=name)
        obj._f_remove(recursive)

    def getAttrNode(self, where, attrname, name=None):
        """
        Get a PyTables attribute from the given node.

        The `where` and `name` arguments work as in `getNode()`,
        referencing the node to be acted upon.  The other arguments work
        as in `Node._f_getAttr()`.
        """
        obj = self.getNode(where, name=name)
        return obj._f_getAttr(attrname)

    def setAttrNode(self, where, attrname, attrvalue, name=None):
        """
        Set a PyTables attribute for the given node.

        The `where` and `name` arguments work as in `getNode()`,
        referencing the node to be acted upon.  The other arguments work
        as in `Node._f_setAttr()`.
        """
        obj = self.getNode(where, name=name)
        obj._f_setAttr(attrname, attrvalue)

    def delAttrNode(self, where, attrname, name=None):
        """
        Delete a PyTables attribute from the given node.

        The `where` and `name` arguments work as in `getNode()`,
        referencing the node to be acted upon.  The other arguments work
        as in `Node._f_delAttr()`.
        """
        obj = self.getNode(where, name=name)
        obj._f_delAttr(attrname)

    def copyAttrs(self, where, dstNode, name=None):
        """
        Copy attributes from one node to another.

        The `where` and `name` arguments work as in `getNode()`,
        referencing the node to be acted upon.  `dstNode` is the
        destination and can be either a path string or a `Node`
        instance.
        """
        srcObject = self.getNode(where, name=name)
        dstObject = self.getNode(dstNode)
        object._v_attrs._f_copy(dstNode)


    def copyChildren(self, whereSrc, whereDst, recursive = False,
                     filters = None, copyuserattrs = True,
                     start = 0, stop = None, step = 1,
                     overwrite = False, stats = None):
        """(Recursively) Copy the children of a group into another location

        "whereSrc" is the source group and "whereDst" is the
        destination group.  Both groups should exist or a NodeError
        will be raised. They can be specified as strings or as Group
        instances. "recursive" specifies whether the copy should
        recurse into subgroups or not. The default is not
        recurse. Specifying a "filters" parameter overrides the
        original filter properties in source nodes. You can prevent
        the user attributes from being copied by setting
        "copyuserattrs" to 0; the default is copy them. "start",
        "stop" and "step" specifies the range of rows in leaves to be
        copied; the default is to copy all the rows. "overwrite"
        means whether the possible existing children hanging from
        "whereDst" and having the same names than "whereSrc" children
        should overwrite the destination nodes or not.

        The optional keyword argument 'stats' may be used to collect
        statistics on the copy process.  When used, it should be a
        dictionary whith keys 'groups', 'leaves' and 'bytes' having a
        numeric value.  Their values will be incremented to reflect
        the number of groups, leaves and bytes, respectively, that
        have been copied in the operation.
        """

        srcGroup = self.getNode(whereSrc)  # Does the source node exist?
        self._checkGroup(srcGroup)  # Is it a group?

        srcGroup._f_copyChildren(
            where = whereDst, recursive = recursive,
            filters = filters, copyuserattrs = copyuserattrs,
            start = start, stop = stop, step = step,
            overwrite = overwrite, stats = stats)


    def copyFile(self, dstFilename=None, title=None,
                 filters=None, copyuserattrs=1, overwrite=0, stats=None):
        """Copy the contents of this file to "dstFilename".

        "dstFilename" must be a path string.  Specifying a "filters"
        parameter overrides the original filter properties in source
        nodes. If "dstFilename" file already exists and overwrite is
        1, it is overwritten. The default is not overwriting. It
        returns a tuple (ngroups, nleaves, nbytes) specifying the
        number of copied groups and leaves.

        This copy also has the effect of compacting the destination
        file during the process.

        The optional keyword argument 'stats' may be used to collect
        statistics on the copy process.  When used, it should be a
        dictionary whith keys 'groups', 'leaves' and 'bytes' having a
        numeric value.  Their values will be incremented to reflect
        the number of groups, leaves and bytes, respectively, that
        have been copied in the operation.
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
        self.root._f_copyChildren(
            dstFileh.root, recursive = True, filters = filters,
            copyuserattrs = copyuserattrs, stats = stats)
        # Finally, close the file
        dstFileh.close()


    def listNodes(self, where, classname=None):
        """
        Return a list with children nodes hanging from `where`.

        The `where` argument works as in `getNode()`, referencing the
        node to be acted upon.  The other arguments work as in
        `Group._f_listNodes()`.
        """

        group = self.getNode(where)  # Does the parent exist?
        self._checkGroup(group)  # Is it a group?

        return group._f_listNodes(classname)


    def __contains__(self, path):
        """
        Is there a node with that `path`?

        Returns ``True`` if the file has a node with the given `path` (a
        string), ``False`` otherwise.
        """
        return path in self.objects

    def __iter__(self):
        """Iterate over the nodes in the object tree."""

        return self.walkNodes('/')

    def walkNodes(self, where="/", classname=None):
        """Iterate over the nodes in the object tree.
        If "where" supplied, the iteration starts from this group.
        If "classname" is supplied, only instances of this class are
        returned.
        """

        # For compatibility with old default arguments.
        if classname == '':
            classname = None

        if classname == "Group":
            for group in self.walkGroups(where):
                yield group
        elif classname is None:
            yield self.getNode(where)
            for group in self.walkGroups(where):
                for leaf in self.listNodes(group, ""):
                    yield leaf
        else:
            for group in self.walkGroups(where):
                for leaf in self.listNodes(group, classname):
                    yield leaf


    def walkGroups(self, where = "/"):
        """Returns the list of Groups (not Leaves) hanging from "where".

        If "where" is not supplied, the root object is taken as
        origin. The groups are returned from top to bottom, and
        alphanumerically sorted when in the same level. The list of
        groups returned includes "where" (or the root object) as well.

        """

        group = self.getNode(where)  # Does the parent exist?
        self._checkGroup(group)  # Is it a group?
        return group._f_walkGroups()


    def _checkGroup(self, node):
        # `node` must already be a node.
        if not isinstance(node, Group):
            raise TypeError("node ``%s`` is not a group" % (node._v_pathname,))


    # <Undo/Redo support>

    def isUndoEnabled(self):
        """
        Is the Undo/Redo mechanism enabled?

        Returns ``True`` if the Undo/Redo mechanism has been enabled for
        this file, ``False`` otherwise.  Please note that this mechanism
        is persistent, so a newly opened PyTables file may already have
        Undo/Redo support.
        """
        return self._undoEnabled


    def _checkUndoEnabled(self):
        if not self._undoEnabled:
            raise UndoRedoError("Undo/Redo feature is currently disabled!")


    def _checkMarkName(self, name):
        "Check that name is of correct type. Put it in the internal dictionary."

        if name is None:
            name = ""
        else:
            if not isinstance(name, str):
                raise TypeError, \
"Only strings are allowed as mark names. You passed object: '%s'" % name
            if name in self._markers:
                raise UndoRedoError, \
"Name '%s' is already used as a marker name. Try another one." % name
            self._markers[name] = self._curmark + 1
        return name


    def _createTransactionGroup(self):
        tgroup = TransactionGroupG("Transaction information container")
        setattr(self.root, _transGroupName, tgroup)
        # The format of the transaction container.
        tgroup._v_attrs._g__setattr('FORMATVERSION', _transVersion)
        return tgroup


    def _createTransaction(self, troot, tid):
        trans = TransactionG("Transaction number %d" % tid)
        setattr(troot, _transName % tid, trans)
        return trans


    def _createMark(self, trans, mid):
        mark = MarkG("Mark number %d" % mid)
        setattr(trans, _markName % mid, mark)
        return mark


    def enableUndo(self, filters=Filters(complevel=1)):
        """
        Enable the Undo/Redo mechanism.

        This operation prepares the database for undoing and redoing
        modifications in the node hierarchy.  This allows `mark()`,
        `undo()`, `redo()` and other methods to be called.

        The `filters` argument, when specified, must be an instance of
        class `Filters` and is meant for setting the compression values
        for the action log.  The default is having compression enabled,
        as the gains in terms of space can be considerable.  You may
        want to disable compression if you want maximum speed for
        Undo/Redo operations.

        Calling `enableUndo()` when the Undo/Redo mechanism is already
        enabled raises an `UndoRedoError`.
        """

        # Enabling several times is not allowed to avoid the user having
        # the illusion that a new implicit mark has been created
        # when calling enableUndo for the second time.

        class ActionLog(IsDescription):
            opcode = UInt8Col(pos=0)
            arg1   = StringCol(MAX_UNDO_PATH_LENGTH, pos=1, dflt="")
            arg2   = StringCol(MAX_UNDO_PATH_LENGTH, pos=2, dflt="")

        if self.isUndoEnabled():
            raise UndoRedoError, "Undo/Redo feature is already enabled!"

        self._markers = {}
        self._seqmarkers = []
        self._nmarks = 0
        self._curtransaction = 0
        self._curmark = -1  # No marks yet

        # Get the Group for keeping user actions
        try:
            tgroup = self._getNode(_transGroupPath)
        except NodeError:
            # A transaction log group does not exist. Create it
            tgroup = self._createTransactionGroup()

            # Create a transaction.
            self._trans = self._createTransaction(
                tgroup, self._curtransaction)

            # Create an action log
            self._actionlog = self.createTable(
                tgroup, _actionLogName, ActionLog, "Action log",
                filters = filters, _log = False)

            # Create an implicit mark
            self._actionlog.append([(_opToCode["MARK"], str(0), '')])
            self._nmarks += 1
            self._seqmarkers.append(0) # current action is 0

            # Create a group for mark 0
            self._createMark(self._trans, 0)
            # Initialize the marker pointer
            self._curmark = self._nmarks - 1
            # Initialize the action pointer
            self._curaction = self._actionlog.nrows - 1
        else:
            # The group seems to exist already
            # Get the default transaction
            self._trans = self._getNode(tgroup, 't'+str(self._curtransaction))
            # Open the action log and go to the end of it
            self._actionlog = self._getNode(tgroup, "actionlog")
            for row in self._actionlog:
                if row["opcode"] == _opToCode["MARK"]:
                    name = row["arg2"]
                    self._markers[name] = self._nmarks
                    self._seqmarkers.append(row.nrow())
                    self._nmarks += 1
            # Get the current mark and current action
            self._curmark = self._actionlog.attrs.CURMARK
            self._curaction = self._actionlog.attrs.CURACTION

        # The Undo/Redo mechanism has been enabled.
        self._undoEnabled = True


    def disableUndo(self):
        """
        Disable the Undo/Redo mechanism.

        Disabling the Undo/Redo mechanism leaves the database in the
        current state and forgets past and future database states.  This
        makes `mark()`, `undo()`, `redo()` and other methods fail with
        an `UndoRedoError`.

        Calling `disableUndo()` when the Undo/Redo mechanism is already
        disabled raises an `UndoRedoError`.
        """

        if not self.isUndoEnabled():
            raise UndoRedoError, "Undo/Redo feature is already disabled!"

        del self._markers
        del self._seqmarkers
        del self._curmark
        del self._curaction
        del self._curtransaction
        del self._nmarks
        del self._actionlog
        # Recursively delete the transaction group
        tnode = self._getNode(_transGroupPath)
        tnode._g_remove(recursive=1)

        # The Undo/Redo mechanism has been disabled.
        self._undoEnabled = False


    def mark(self, name=None):
        """
        Mark the state of the database.

        Creates a mark for the current state of the database.  A unique
        (and immutable) identifier for the mark is returned.  An
        optional `name` (a string) can be assigned to the mark.  Both
        the identifier of a mark and its name can be used in `undo()`
        and `redo()` operations.  When the `name` has already been used
        for another mark, an `UndoRedoError` is raised.

        This method can only be called when the Undo/Redo mechanism has
        been enabled.  Otherwise, an `UndoRedoError` is raised.
        """

        self._checkUndoEnabled()

        name = self._checkMarkName(name)

        # Create an explicit mark
        # Insert the mark in the action log
        self._log("MARK", str(self._curmark+1), name)
        self._curmark += 1
        self._nmarks = self._curmark + 1
        self._seqmarkers.append(self._curaction)
        # Create a group for the current mark
        self._createMark(self._trans, self._curmark)
        return self._curmark


    def _log(self, action, *args):
        """
        Log an action.

        The `action` must be an all-uppercase string identifying it.
        Arguments must also be strings.

        This method should be called once the action has been completed.

        This method can only be called when the Undo/Redo mechanism has
        been enabled.  Otherwise, an `UndoRedoError` is raised.
        """

        assert self.isUndoEnabled()

        # Check whether we are at the end of the action log or not
        if self._curaction <> self._actionlog.nrows - 1:
            #print "Entering destroy....................................."
            # We are not, so delete the trailing actions
            self._actionlog.removeRows(self._curaction + 1,
                                       self._actionlog.nrows)
            # Reset the current marker group
            mnode = self._getNode(_markPath % (self._curtransaction,
                                               self._curmark))
            mnode._g_reset()
            # Delete the marker groups with backup objects
            for mark in xrange(self._curmark+1, self._nmarks):
                mnode = self._getNode(_markPath % (self._curtransaction, mark))
                mnode._g_remove(recursive=1)
            # Update the new number of marks
            self._nmarks = self._curmark+1
            self._seqmarkers = self._seqmarkers[:self._nmarks]

        if action not in _opToCode:  #INTERNAL
            raise UndoRedoError, \
                  "Action ``%s`` not in ``_opToCode`` dictionary: %r" %  \
                  (action, _opToCode)

        arg1 = ""; arg2 = ""
        if len(args) <= 1:
            arg1 = args[0]
        elif len(args) <= 2:
            arg1 = args[0]
            arg2 = args[1]
        else:  #INTERNAL
            raise UndoRedoError, \
                  "Too many parameters for action log: %r", args
        if (len(arg1) > MAX_UNDO_PATH_LENGTH
            or len(arg2) > MAX_UNDO_PATH_LENGTH):  #INTERNAL
            raise UndoRedoError, \
                  "Parameter arg1 or arg2 is too long: (%r, %r)" %  \
                  (arg1, arg2)
        #print "Logging-->", (action, arg1, arg2)
        self._actionlog.append([(_opToCode[action], arg1, arg2)])
        self._curaction += 1


    def _getMarkID(self, mark):
        "Get an integer markid from a mark sequence number or name"

        if isinstance(mark, int):
            markid = mark
        elif isinstance(mark, str):
            if mark not in self._markers:
                lmarkers = self._markers.keys()
                lmarkers.sort()
                raise UndoRedoError, \
                      "The mark that you have specified has not been found in the internal marker list: %r" % lmarkers
            markid = self._markers[mark]
        else:
            raise TypeError, \
                  "Parameter mark can only be an integer or a string, and you passed a type <%s>" % type(mark)
        #print "markid, self._nmarks:", markid, self._nmarks
        return markid


    def _getFinalAction(self, markid):
        "Get the action to go. It does not touch the self private attributes"

        if markid > self._nmarks - 1:
            # The required mark is beyond the end of the action log
            # The final action is the last row
            return self._actionlog.nrows
        elif markid <= 0:
            # The required mark is the first one
            # return the first row
            return 0

        return self._seqmarkers[markid]


    # This is a workaround for reversing a RecArray until [::-1] works
    ##@staticmethod  # Python >= 2.4
    def _reverseRecArray(recarr):
        v = recarr.view()
        for f in range(recarr._nfields):
            fow = v.field(f)
            rev = fow[::-1]
            for attr in ["_shape", "_strides", "_bytestride",
                         "_itemsize", "_byteoffset"]:
                setattr(v.field(f), attr, getattr(rev, attr))
        return v
    _reverseRecArray = staticmethod(_reverseRecArray)

    def _doundo(self, finalaction, direction):
        "Undo/Redo actions up to final action in the specificed direction"

        if direction < 0:
            # Change this when reversing RecArrays will work (numarray > 1.2.2)
            #actionlog = self._actionlog[finalaction+1:self._curaction+1][::-1]
            actionlog = self._reverseRecArray(
                self._actionlog[finalaction+1:self._curaction+1])
        else:
            actionlog = self._actionlog[self._curaction:finalaction]

        # Uncomment this for debugging
#         print "curaction, finalaction, direction", \
#               self._curaction, finalaction, direction
        for i in xrange(len(actionlog)):
            if actionlog.field('opcode')[i] <> _opToCode["MARK"]:
                # undo/redo the action
                if direction > 0:
                    # Uncomment this for debugging
                    #print "redo-->", _codeToOp[opcode[i]], arg1[i], arg2[i]
                    undoredo.redo(self,
                                  _codeToOp[actionlog.field('opcode')[i]],
                                  actionlog.field('arg1')[i],
                                  actionlog.field('arg2')[i])
                else:
                    # Uncomment this for debugging
                    #print "undo-->", _codeToOp[opcode[i]], arg1[i], arg2[i]
                    undoredo.undo(self,
                                  _codeToOp[actionlog.field('opcode')[i]],
                                  actionlog.field('arg1')[i],
                                  actionlog.field('arg2')[i])
            else:
                if direction > 0:
                    self._curmark = int(actionlog.field('arg1')[i])
                else:
                    self._curmark = int(actionlog.field('arg1')[i]) - 1
                    # Protection against negative marks
                    if self._curmark < 0:
                        self._curmark = 0
            self._curaction += direction


    def undo(self, mark=None):
        """
        Go to a past state of the database.

        Returns the database to the state associated with the specified
        `mark`.  Both the identifier of a mark and its name can be used.
        If the `mark` is omitted, the last created mark is used.  If
        there are no past marks, or the specified `mark` is not older
        than the current one, an `UndoRedoError` is raised.

        This method can only be called when the Undo/Redo mechanism has
        been enabled.  Otherwise, an `UndoRedoError` is raised.
        """

        self._checkUndoEnabled()

#         print "(pre)UNDO: (curaction, curmark) = (%s,%s)" % \
#               (self._curaction, self._curmark)
        if mark is None:
            markid = self._curmark
            # Correction if we are settled on top of a mark
            opcode = self._actionlog.cols.opcode
            if opcode[self._curaction] == _opToCode["MARK"]:
                markid -= 1
        else:
            # Get the mark ID number
            markid = self._getMarkID(mark)
        # Get the final action ID to go
        finalaction = self._getFinalAction(markid)
        if finalaction > self._curaction:
            raise UndoRedoError("""\
Mark ``%s`` is newer than the current mark. Use `redo()` or `goto()` instead."""
                                % (mark,))
        # Try to reach this mark by unwinding actions in the log
        self._doundo(finalaction-1, -1)
        if self._curaction < self._actionlog.nrows-1:
            self._curaction += 1
        self._curmark = int(self._actionlog.cols.arg1[self._curaction])
#         print "(post)UNDO: (curaction, curmark) = (%s,%s)" % \
#               (self._curaction, self._curmark)


    def redo(self, mark=None):
        """
        Go to a future state of the database.

        Returns the database to the state associated with the specified
        `mark`.  Both the identifier of a mark and its name can be used.
        If the `mark` is omitted, the next created mark is used.  If
        there are no future marks, or the specified `mark` is not newer
        than the current one, an `UndoRedoError` is raised.

        This method can only be called when the Undo/Redo mechanism has
        been enabled.  Otherwise, an `UndoRedoError` is raised.
        """

        self._checkUndoEnabled()

#         print "(pre)REDO: (curaction, curmark) = (%s, %s)" % \
#               (self._curaction, self._curmark)
        # Get the final action ID to go
        if self._curaction < self._actionlog.nrows - 1:
            self._curaction += 1
        else:
            # We are at the end of log, so no action
            return
        if mark is None:
            mark = self._curmark + 1
        elif mark == -1:
            mark = self._nmarks  # Go beyond the mark bounds up to the end
        # Get the mark ID number
        markid = self._getMarkID(mark)
        finalaction = self._getFinalAction(markid)
        if finalaction < self._curaction:
            raise UndoRedoError("""\
Mark ``%s`` is older than the current mark. Use `redo()` or `goto()` instead."""
                                % (mark,))
        # Try to reach this mark by redoing the actions in the log
        self._doundo(finalaction, 1)
        # Increment the current mark only if we are not at the end of marks
        if self._curmark < self._nmarks - 1:
            self._curmark += 1
        if self._curaction > self._actionlog.nrows-1:
            self._curaction = self._actionlog.nrows-1
#         print "(post)REDO: (curaction, curmark) = (%s,%s)" % \
#               (self._curaction, self._curmark)


    def goto(self, mark):
        """
        Go to a specific mark of the database.

        Returns the database to the state associated with the specified
        `mark`.  Both the identifier of a mark and its name can be used.

        This method can only be called when the Undo/Redo mechanism has
        been enabled.  Otherwise, an `UndoRedoError` is raised.
        """

        self._checkUndoEnabled()

        if mark == -1:  # Special case
            mark = self._nmarks  # Go beyond the mark bounds up to the end
        # Get the mark ID number
        markid = self._getMarkID(mark)
        finalaction = self._getFinalAction(markid)
        if finalaction < self._curaction:
            self.undo(mark)
        else:
            self.redo(mark)


    def getCurrentMark(self):
        """
        Get the identifier of the current mark.

        Returns the identifier of the current mark.  This can be used to
        know the state of a database after an application crash, or to
        get the identifier of the initial implicit mark after a call to
        `enableUndo()`.

        This method can only be called when the Undo/Redo mechanism has
        been enabled.  Otherwise, an `UndoRedoError` is raised.
        """

        self._checkUndoEnabled()
        return self._curmark


    def _shadowName(self):
        """
        Compute and return a shadow name.

        Computes the current shadow name according to the current
        transaction, mark and action.  It returns a tuple with the
        shadow parent node and the name of the shadow in it.
        """

        parent = self._getNode(
            _shadowParent % (self._curtransaction, self._curmark))
        name = _shadowName % (self._curaction,)

        return (parent, name)

    # </Undo/Redo support>


    def flush(self):
        """Flush all the objects on all the HDF5 objects tree."""

        for group in self.walkGroups(self.root):
            for leaf in self.listNodes(group, classname = 'Leaf'):
                leaf.flush()

        # Flush the cache to disk
        self._flushFile(0)  # 0 means local scope, 1 global (virtual) scope

    def close(self):
        """Close all the objects in HDF5 file and close the file."""

        # If the file is already closed, return immediately
        if not self.isopen:
            return

        if self._undoEnabled:
            # Save the current mark and current action
            self._actionlog.attrs._g__setattr("CURMARK", self._curmark)
            self._actionlog.attrs._g__setattr("CURACTION", self._curaction)

        # Close the root group (recursively)
        self.root._f_close()
        # Close the file
        self._closeFile()
        # After the objects are disconnected, destroy the
        # object dictionary using the brute force ;-)
        # This should help to the garbage collector
        self.__dict__.clear()
        # Set the flag to indicate that the file is closed
        self.isopen = 0

        return

    def __str__(self):
        """Returns a string representation of the object tree"""
        # Print all the nodes (Group and Leaf objects) on object tree
        date = time.asctime(time.localtime(os.stat(self.filename)[8]))
        astring =  self.filename + ' (File) ' + repr(self.title) + '\n'
#         astring += 'rootUEP :=' + repr(self.rootUEP) + '; '
#         astring += 'format_version := ' + self.format_version + '\n'
#         astring += 'filters :=' + repr(self.filters) + '\n'
        astring += 'Last modif.: ' + repr(date) + '\n'
        astring += 'Object Tree: \n'

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


    def _refNode(self, node, path):
        """
        Insert references to a `node` via a `path`.

        Checks that the `path` does not exist and creates references to
        the given `node` by that `path`.
        """

        # Check if there is already a node with the same path.
        if path in self._allNodes:
            raise NodeError(
                "file already has a node with path ``%s``" % (path,))

        # Insert references to the new node.
        self._allNodes[path] = node
        if not path.startswith('/_p_'):
            # This is only done for visible nodes.
            self.objects[path] = node
            if isinstance(node, Leaf):
                self.leaves[path] = node
            if isinstance(node, Group):
                self.groups[path] = node


    def _unrefNode(self, path):
        """
        Remove references to a node.

        Removes all references to the node pointed to by the `path`.
        """

        del self._allNodes[path]
        if not path.startswith('/_p_'):
            # This is only done for visible nodes.
            if path in self.groups:
                del self.groups[path]
            if path in self.leaves:
                del self.leaves[path]
            del self.objects[path]



## Local Variables:
## mode: python
## py-indent-offset: 4
## tab-width: 4
## fill-column: 72
## End:
