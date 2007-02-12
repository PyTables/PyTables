########################################################################
#
#       License:        BSD
#       Created:        September 4, 2002
#       Author:  Francesc Altet - faltet@carabos.com
#
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

    copyFile(srcfilename, dstfilename[, overwrite][, **kwargs])
    openFile(name[, mode][, title][, trMap][, rootUEP][, filters]
             [, nodeCacheSize])

Misc variables:

    __version__
    format_version
    compatible_formats

"""

import warnings
import time
import os, os.path
import sys
import weakref

import tables.misc.proxydict
from tables import hdf5Extension
from tables import utilsExtension
from tables.parameters import \
     MAX_UNDO_PATH_LENGTH, METADATA_CACHE_SIZE, NODE_MAX_SLOTS
from tables.exceptions import \
     ClosedFileError, FileModeError, \
     NodeError, NoSuchNodeError, UndoRedoError, \
     UndoRedoWarning, PerformanceWarning
from tables.registry import getClassByName
from tables.path import joinPath, splitPath, isVisiblePath
from tables.utils import checkFileAccess
from tables import undoredo
from tables.description import IsDescription, UInt8Col, StringCol
from tables.node import Node, NotLoggedMixin
from tables.group import Group, RootGroup
from tables.group import TransactionGroupG, TransactionG, MarkG
from tables.leaf import Leaf, Filters
from tables.array import Array
from tables.carray import CArray
from tables.earray import EArray
from tables.vlarray import VLArray

try:
    from tables import lrucacheExtension
except ImportError:
    from tables.misc import lrucache
    _LRUCache = lrucache.LRUCache
else:
    _LRUCache = lrucacheExtension.NodeCache



__version__ = "$Revision$"


#format_version = "1.0" # Initial format
#format_version = "1.1" # Changes in ucl compression
#format_version = "1.2"  # Support for enlargeable arrays and VLA's
#                        # 1.2 was introduced in PyTables 0.8
#format_version = "1.3"  # Support for indexes in Tables
#                        # 1.3 was introduced in PyTables 0.9
#format_version = "1.4"  # Support for multidimensional attributes
#                        # 1.4 was introduced in PyTables 1.1
#format_version = "1.5"  # Support for persistent defaults in tables
#                        # 1.5 was introduced in PyTables 1.2
#format_version = "1.6"  # Support for NumPy objects and new flavors for objects
#                        # 1.6 was introduced in pytables 1.3
format_version = "2.0"  # Pickles are not used anymore in system attrs
                        # 2.0 was introduced in PyTables 2.0
compatible_formats = [] # Old format versions we can read
                        # Empty means that we support all the old formats

# Dict of opened files (keys are filehandlers and values filenames)
_open_files = {}

# Opcodes for do-undo actions
_opToCode = {
    "MARK":    0,
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



def _checkfilters(filters):
    if not (filters is None or
            isinstance(filters, Filters)):
        raise TypeError, "filter parameter has to be None or a Filter instance and the passed type is: '%s'" % type(filters)


def copyFile(srcfilename, dstfilename, overwrite=False, **kwargs):
    """
    An easy way of copying one PyTables file to another.

    This function allows you to copy an existing PyTables file named
    `srcfilename` to another file called `dstfilename`.  The source file
    must exist and be readable.  The destination file can be overwritten
    in place if existing by asserting the `overwrite` argument.

    This function is a shorthand for the `File.copyFile()` method, which
    acts on an already opened file.  `kwargs` takes keyword arguments
    used to customize the copying process.  See the documentation of
    `File.copyFile()` for a description of those arguments.
    """

    # Open the source file.
    srcFileh = openFile(srcfilename, mode="r")

    try:
        # Copy it to the destination file.
        srcFileh.copyFile(dstfilename, overwrite=overwrite, **kwargs)
    finally:
        # Close the source file.
        srcFileh.close()


def openFile(filename, mode="r", title="", trMap={}, rootUEP="/",
             filters=None, nodeCacheSize=NODE_MAX_SLOTS):

    """Open an HDF5 file and return a File object.

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

    trMap -- A dictionary to map names in the object tree into different
             HDF5 names in file. The keys are the Python names, while
             the values are the HDF5 names. This is useful when you need
             to name HDF5 nodes with invalid or reserved words in Python
             and wants to continue using the natural naming facility.

    rootUEP -- The root User Entry Point. It is a group in the file
            hierarchy which is taken as the starting point to create the
            object tree. The group can be whatever existing path in the
            file. If it does not exist, an HDF5ExtError is issued.

    filters -- An instance of the Filters class that provides
            information about the desired I/O filters applicable to the
            leaves that hangs directly from root (unless other filters
            properties are specified for these leaves, of
            course). Besides, if you do not specify filter properties
            for its child groups, they will inherit these ones.

    nodeCacheSize -- The number of *unreferenced* nodes to be kept in
            memory.  Least recently used nodes are unloaded from memory
            when this number of loaded nodes is reached.  To load a node
            again, simply access it as usual.  Nodes referenced by user
            variables are not taken into account nor unloaded.

    """

    # Expand the form '~user'
    path = os.path.expanduser(filename)
    # Expand the environment variables
    path = os.path.expandvars(path)

    # Finally, create the File instance, and return it
    return File(path, mode, title, trMap, rootUEP, filters,
                METADATA_CACHE_SIZE, nodeCacheSize)


# It is necessary to import Table after openFile, because it solves a circular
# import reference.
from tables.table import Table


class _AliveNodes(dict):

    """Stores strong or weak references to nodes in a transparent way."""

    def __getitem__(self, key):
        if NODE_MAX_SLOTS > 0:
            ref = super(_AliveNodes, self).__getitem__(key)()
        else:
            ref = super(_AliveNodes, self).__getitem__(key)
        return ref

    def __setitem__(self, key, value):
        if NODE_MAX_SLOTS > 0:
            ref = weakref.ref(value)
        else:
            ref = value
            # Check if we are running out of space
            if NODE_MAX_SLOTS < 0 and len(self) > -NODE_MAX_SLOTS:
                warnings.warn("""\
the dictionary of alive nodes is exceeding the recommended maximum number (%d); \
be ready to see PyTables asking for *lots* of memory and possibly slow I/O."""
                      % (-NODE_MAX_SLOTS),
                      PerformanceWarning)
        super(_AliveNodes, self).__setitem__(key, ref)



class _DeadNodes(_LRUCache):
    pass

# A dumb class that doesn't keep nothing at all
class _NoDeadNodes(object):
    def __len__(self):
        return 0
    def __contains__(self, key):
        return False
    def __iter__(self):
        return iter([])


class _NodeDict(tables.misc.proxydict.ProxyDict):

    """
    A proxy dictionary which is able to delegate access to missing items
    to the container object (a `File`).
    """

    def _getValueFromContainer(self, container, key):
        return container.getNode(key)


    def _condition(self, node):
        """Nodes fulfilling the condition are considered to belong here."""
        raise NotImplementedError


    def _warnOnGet(self):
        warnings.warn("using this mapping object is deprecated; "
                      "please use ``File.getNode()`` instead",
                      DeprecationWarning)


    def __contains__(self, key):
        self._warnOnGet()

        # If the key is here there is nothing else to check.
        if super(_NodeDict, self).__contains__(key):
            return True

        # Look if the key is in the container `File`.
        try:
            file_ = self._getContainer()
            node = file_.getNode(key)
            # Does it fullfill the condition?
            return self._condition(node)
        except NoSuchNodeError:
            # It is not in the container.
            return False


    def __getitem__(self, key):
        self._warnOnGet()
        return super(_NodeDict, self).__getitem__(key)


    # The following operations are quite underperforming
    # because they need to browse the entire tree.
    # These objects are deprecated, anyway.

    def __iter__(self):
        return self.iterkeys()


    def iterkeys(self):
        warnings.warn("using this mapping object is deprecated; "
                      "please use ``File.walkNodes()`` instead",
                      DeprecationWarning)
        for node in self._getContainer().walkNodes('/', self._className):
            yield node._v_pathname
        raise StopIteration


    def __len__(self):
        nnodes = 0
        for nodePath in self.iterkeys():
            nnodes += 1
        return nnodes


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
        parameter passed to the `openFile()` function.  You cannot change
        its contents *after* a file is opened.
    rootUEP
        The UEP (user entry point) group in the file (see the
        `openFile()` function).
    filters
        Default filter properties for the root group (see the `Filters`
            class).
    root
        The *root* of the object tree hierarchy (a `Group` instance).


    Public methods (file handling):

    * copyFile(dstfilename[, overwrite][, **kwargs])
    * flush()
    * close()

    Public methods (hierarchy manipulation):

    * createGroup(where, name[, title][, filters][, createparents])
    * createTable(where, name, description[, title][, filters]
                  [, expectedrows][, chunkshape][, createparents])
    * createArray(where, name, array[, title][, createparents])
    * createCArray(where, name, atom, shape [, title][, filters]
                   [, chunkshape][, createparents])
    * createEArray(where, name, atom, shape [, title][, filters]
                   [, expectedrows][, chunkshape][, createparents])
    * createVLArray(where, name, atom[, title][, filters]
                    [, expectedsizeinMB][, chunkshape][, createparents])
    * removeNode(where[, name][, recursive])
    * renameNode(where, newname[, name])
    * moveNode(where, newparent, newname[, name][, overwrite])
    * copyNode(where, newparent, newname[, name][, overwrite]
               [, recursive][, **kwargs])
    * copyChildren(srcgroup, dstgroup[, overwrite][, recursive]
                   [, **kwargs])

    Public methods (tree traversal):

    * getNode(where[, name][,classname])
    * isVisibleNode(path)
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

    * getNodeAttr(where, attrname[, name])
    * setNodeAttr(where, attrname, attrvalue[, name])
    * delNodeAttr(where, attrname[, name])
    * copyNodeAttrs(where, dstnode[, name])
    """

    ## <properties>

    def _gettitle(self):
        return self.root._v_title
    def _settitle(self, title):
        self.root._v_title = title
    def _deltitle(self):
        del self.root._v_title

    title = property(
        _gettitle, _settitle, _deltitle,
        "The title of the root group in the file.")

    def _getfilters(self):
        return self.root._v_filters
    def _setfilters(self, filters):
        self.root._v_filters = filters
    def _delfilters(self):
        del self.root._v_filters

    filters = property(
        _getfilters, _setfilters, _delfilters,
        "Default filter properties for the root group "
        "(see the `Filters` class).")

    trMap = property(
        lambda self: self._pttoh5, None, None,
        "Translation map between PyTables <--> HDF5 "
        "namespaces.")

    ## </properties>


    def __init__(self, filename, mode="r", title="", trMap={},
                 rootUEP="/", filters=None,
                 metadataCacheSize=METADATA_CACHE_SIZE,
                 nodeCacheSize=NODE_MAX_SLOTS):
        """Open an HDF5 file. The supported access modes are: "r" means
        read-only; no data can be modified. "w" means write; a new file is
        created, an existing file with the same name is deleted. "a" means
        append (in analogy with serial files); an existing file is opened
        for reading and writing, and if the file does not exist it is
        created. "r+" is similar to "a", but the file must already exist. A
        TITLE attribute will be set on the root group if optional "title"
        parameter is passed."""

        global _open_files

        self.filename = filename
        self.mode = mode

        # Nodes referenced by a variable are kept in `_aliveNodes`.
        # When they are no longer referenced, they move themselves
        # to `_deadNodes`, where they are kept until they are referenced again
        # or they are preempted from it by other unreferenced nodes.
        self._aliveNodes = _AliveNodes()
        if nodeCacheSize >= 0:
            self._deadNodes = _DeadNodes(nodeCacheSize)
        else:
            self._deadNodes = _NoDeadNodes()

        # Assign the trMap to a private variable
        self._pttoh5 = trMap

        # For the moment Undo/Redo is not enabled.
        self._undoEnabled = False

        new = self._v_new

        # Filters
        if new and filters is None:
            # Set the defaults
            filters = Filters()

        # Set the flag to indicate that the file has been opened.
        # It must be set before opening the root group
        # to allow some basic access to its attributes.
        self.isopen = 1

        # Append the name of the file to the global dict of files opened.
        _open_files[self] = self.filename

        # Get the root group from this file
        self.root = root = self.__getRootGroup(rootUEP, title, filters)
        # Complete the creation of the root node
        # (see the explanation in ``RootGroup.__init__()``.
        root._g_postInitHook()

        # Save the PyTables format version for this file.
        if new:
            self.format_version = format_version
            root._v_attrs._g__setattr(
                'PYTABLES_FORMAT_VERSION', format_version)

        # If the file is old, and not opened in "read-only" mode,
        # check if it has a transaction log
        if not new and self.mode != "r" and _transGroupPath in self:
            # It does. Enable the undo.
            self.enableUndo()


    def __getRootGroup(self, rootUEP, title, filters):
        """Returns a Group instance which will act as the root group
        in the hierarchical tree. If file is opened in "r", "r+" or
        "a" mode, and the file already exists, this method dynamically
        builds a python object tree emulating the structure present on
        file."""

        self._v_objectID = self._getFileId()

        if rootUEP in [None, ""]:
            rootUEP = "/"
        # Save the User Entry Point in a variable class
        self.rootUEP=rootUEP

        new = self._v_new

        # Get format version *before* getting the object tree
        if not new:
            # Firstly, get the PyTables format version for this file
            self.format_version = utilsExtension.read_f_attr(
                self._v_objectID, 'PYTABLES_FORMAT_VERSION')
            if not self.format_version or not self._isPTFile:
                # PYTABLES_FORMAT_VERSION attribute is not present
                self.format_version = "unknown"

        # Create new attributes for the root Group instance and
        # create the object tree
        return RootGroup(self, rootUEP, title=title, new=new, filters=filters)


    def _ptNameFromH5Name(self, h5Name):
        """Get the PyTables name matching the given HDF5 name."""

        ptName = h5Name
        # This code might seem inefficient but it will be rarely used.
        for (ptName_, h5Name_) in self.trMap.iteritems():
            if h5Name_ == h5Name:
                ptName = ptName_
                break
        return ptName


    def _h5NameFromPTName(self, ptName):
        """Get the HDF5 name matching the given PyTables name."""
        return self.trMap.get(ptName, ptName)


    def _getOrCreatePath(self, path, create):
        """
        Get the given `path` or create it if `create` is true.

        If `create` is true, `path` *must* be a string path and not a
        node, otherwise a `TypeError`will be raised.
        """
        if create:
            return self._createPath(path)
        else:
            return self.getNode(path)

    def _createPath(self, path):
        """
        Create the groups needed for the `path` to exist.

        The group associated with the given `path` is returned.
        """
        if not hasattr(path, 'split'):
            raise TypeError("when creating parents, parent must be a path")

        if path == '/':
            return self.root

        parent, createGroup = self.root, self.createGroup
        for pcomp in path.split('/')[1:]:
            try:
                child = parent._f_getChild(pcomp)
            except NoSuchNodeError:
                child = createGroup(parent, pcomp)
            parent = child
        return parent


    def createGroup(self, where, name, title="", filters=None,
                    createparents=False):
        """Create a new Group instance with name "name" in "where" location.

        Keyword arguments:

        where -- The parent group where the new group will hang
            from. "where" parameter can be a path string (for example
            "/level1/level2"), or Group instance.

        name -- The name of the new group.

        title -- Sets a TITLE attribute on the table entity.

        filters -- An instance of the Filters class that provides
            information about the desired I/O filters applicable to
            the leaves that hang directly from this new group (unless
            other filters properties are specified for these leaves,
            of course). Besides, if you do not specify filter
            properties for its child groups, they will inherit these
            ones.

        createparents -- Whether to create the needed groups for the
            parent path to exist (not done by default).
        """
        parentNode = self._getOrCreatePath(where, createparents)
        return Group(parentNode, name,
                     title=title, new=True, filters=filters)


    def createTable(self, where, name, description, title="",
                    filters=None, expectedrows=10000,
                    chunkshape=None, createparents=False):
        """Create a new Table instance with name "name" in "where" location.

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

        chunkshape -- The shape of the data chunk to be read or written
            as a single HDF5 I/O operation. The filters are applied to
            those chunks of data. Its rank for tables has to be 1. If
            None, a sensible value is calculated (which is recommended).

        createparents -- Whether to create the needed groups for the
            parent path to exist (not done by default).
        """
        parentNode = self._getOrCreatePath(where, createparents)
        _checkfilters(filters)
        return Table(parentNode, name,
                     description=description, title=title,
                     filters=filters, expectedrows=expectedrows,
                     chunkshape=chunkshape)


    def createArray(self, where, name, object, title="", createparents=False):
        """Create a new instance Array with name "name" in "where" location.

        Keyword arguments:

        where -- The parent group where the new table will hang
            from. "where" parameter can be a path string (for example
            "/level1/leaf5"), or Group instance.

        name -- The name of the new array.

        object -- The (regular) object to be saved. It can be any of
            NumPy, NumArray, CharArray, Numeric or other native Python
            types, provided that they are regular (i.e. they are not
            like [[1,2],2]) and homogeneous (i.e. all the elements are
            of the same type).

        title -- Sets a TITLE attribute on the array entity.

        createparents -- Whether to create the needed groups for the
            parent path to exist (not done by default).
        """
        parentNode = self._getOrCreatePath(where, createparents)
        return Array(parentNode, name,
                     object=object, title=title)


    def createCArray(self, where, name, atom, shape, title="",
                     filters=None, chunkshape=None, createparents=False):
        """Create a new instance CArray with name "name" in "where" location.

        Keyword arguments:

        where -- The parent group where the new table will hang
            from. "where" parameter can be a path string (for example
            "/level1/leaf5"), or Group instance.

        name -- The name of the new array.

        atom -- An Atom instance representing the shape and type of the
            chunks to be saved.

        shape -- The shape of the new array.

        title -- Sets a TITLE attribute on the array entity.

        filters -- An instance of the Filters class that provides
            information about the desired I/O filters to be applied
            during the life of this object.

        chunkshape -- The shape of the data chunk to be read or written
            in a single HDF5 I/O operation. Filters are applied to those
            chunks of data. The dimensionality of chunkshape must be the
            same as that of shape. If None, a sensible value is
            calculated (which is recommended).

        createparents -- Whether to create the needed groups for the
            parent path to exist (not done by default).
        """
        parentNode = self._getOrCreatePath(where, createparents)
        _checkfilters(filters)
        return CArray(parentNode, name,
                      atom=atom, shape=shape, title=title, filters=filters,
                      chunkshape=chunkshape)


    def createEArray(self, where, name, atom, shape, title="",
                     filters=None, expectedrows=1000,
                     chunkshape=None, createparents=False):
        """Create a new instance EArray with name "name" in "where" location.

        Keyword arguments:

        where -- The parent group where the new table will hang
            from. "where" parameter can be a path string (for example
            "/level1/leaf5"), or Group instance.

        name -- The name of the new array.

        atom -- An Atom instance representing the shape and type of the
            atomic objects to be saved.

        shape -- The shape of the array. One of the shape dimensions
            must be 0. The dimension being 0 means that the resulting
            EArray object can be extended along it.

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

        chunkshape -- The shape of the data chunk to be read or written
            in a single HDF5 I/O operation. Filters are applied to those
            chunks of data. The dimensionality of chunkshape must be the
            same as that of shape (beware: no dimension should be zero
            this time!).  If None, a sensible value is calculated (which
            is recommended).

        createparents -- Whether to create the needed groups for the
            parent path to exist (not done by default).
        """
        parentNode = self._getOrCreatePath(where, createparents)
        _checkfilters(filters)
        return EArray(parentNode, name,
                      atom=atom, shape=shape, title=title,
                      filters=filters, expectedrows=expectedrows,
                      chunkshape=chunkshape)


    def createVLArray(self, where, name, atom, title="",
                      filters=None, expectedsizeinMB=1.0,
                      chunkshape=None, createparents=False):
        """Create a new instance VLArray with name "name" in "where" location.

        Keyword arguments:

        where -- The parent group where the new table will hang
            from. "where" parameter can be a path string (for example
            "/level1/leaf5"), or Group instance.

        name -- The name of the new array.

        title -- Sets a TITLE attribute on the array entity.

        atom -- A Atom object representing the shape and type of the
            atomic objects to be saved.

        filters -- An instance of the Filters class that provides
            information about the desired I/O filters to be applied
            during the life of this object.

        expectedsizeinMB -- An user estimate about the size (in MB) in
            the final VLArray object. If not provided, the default
            value is 1 MB.  If you plan to create both much smaller or
            much bigger Arrays try providing a guess; this will
            optimize the HDF5 B-Tree creation and management process
            time and the amount of memory used.

        chunkshape -- The shape of the data chunk to be read or written
            in a single HDF5 I/O operation. Filters are applied to those
            chunks of data. The dimensionality of chunkshape must be
            1. If None, a sensible value is calculated (which is
            recommended).

        createparents -- Whether to create the needed groups for the
            parent path to exist (not done by default).
        """
        parentNode = self._getOrCreatePath(where, createparents)
        _checkfilters(filters)
        return VLArray(parentNode, name,
                       atom=atom, title=title, filters=filters,
                       expectedsizeinMB=expectedsizeinMB,
                       chunkshape=chunkshape)


    # There is another version of _getNode in Pyrex space, but only
    # marginally faster (5% or less, but sometimes slower!) than this one.
    # So I think it is worth to use this one instead (much easier to debug).
    def _getNode(self, nodePath):
        # The root node is always at hand.
        if nodePath == '/':
            return self.root

        aliveNodes = self._aliveNodes
        deadNodes = self._deadNodes

        # Walk up the hierarchy until a node in the path is in memory.
        parentPath = nodePath  # deepest node in memory
        pathTail = []  # subsequent children below that node
        while parentPath != '/':
            if parentPath in aliveNodes:
                # The parent node is in memory and alive, so get it.
                parentNode = aliveNodes[parentPath]
                assert parentNode is not None, \
                       "stale weak reference to dead node ``%s``" % parentPath
                break
            if parentPath in deadNodes:
                # The parent node is in memory but dead, so revive it.
                parentNode = self._reviveNode(parentPath)
                break
            # Go up one level to try again.
            (parentPath, nodeName) = splitPath(parentPath)
            pathTail.insert(0, nodeName)
        else:
            # We hit the root node and no parent was in memory.
            parentNode = self.root

        # Walk down the hierarchy until the last child in the tail is loaded.
        node = parentNode  # maybe `nodePath` was already in memory
        for childName in pathTail:
            # Load the node and use it as a parent for the next one in tail
            # (it puts itself into life via `self._refNode()` when created).
            if not isinstance(parentNode, Group):
                # This is the root group
                parentPath = parentNode._v_pathname
                raise TypeError("node ``%s`` is not a group; "
                                "it can not have a child named ``%s``"
                                % (parentPath, childName))
            node = parentNode._g_loadChild(childName)
            parentNode = node

        return node


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
        `NoSuchNodeError` is raised.  Please note thet hidden nodes are
        also considered.

        If the `classname` argument is specified, it must be the name of
        a class derived from `Node`.  If the node is found but it is not
        an instance of that class, a `NoSuchNodeError` is also raised.
        """

        self._checkOpen()

        # For compatibility with old default arguments.
        if name == '':
            name = None

        # Get the parent path (and maybe the node itself).
        if isinstance(where, Node):
            node = where
            node._g_checkOpen()  # the node object must be open
            nodePath = where._v_pathname
        elif isinstance(where, basestring):  # Pyhton >= 2.3
            node = None
            nodePath = where
        else:
            raise TypeError(
                "``where`` is not a string nor a node: %r" % (where,))

        # Get the name of the child node.
        if name is not None:
            node = None
            nodePath = joinPath(nodePath, name)

        assert node is None or node._v_pathname == nodePath

        # Now we have the definitive node path, let us try to get the node.
        if node is None:
            node = self._getNode(nodePath)

        # Finally, check whether the desired node is an instance
        # of the expected class.
        if classname:
            class_ = getClassByName(classname)
            if not isinstance(node, class_):
                nPathname = node._v_pathname
                nClassname = node.__class__.__name__
                # This error message is right since it can never be shown
                # for ``classname in [None, 'Node']``.
                raise NoSuchNodeError(
                    "could not find a ``%s`` node at ``%s``; "
                    "instead, a ``%s`` node has been found there"
                    % (classname, nPathname, nClassname))

        return node


    def isVisibleNode(self, path):
        """
        Is the node under `path` visible?

        If the node does not exist, a ``NoSuchNodeError`` is raised.
        """

        # ``util.isVisiblePath()`` is still recommended for internal use.
        return self.getNode(path)._f_isVisible()


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
                 overwrite=False, createparents=False):
        """
        Move or rename the given node.

        The `where` and `name` arguments work as in `getNode()`,
        referencing the node to be acted upon.  The other arguments work
        as in `Node._f_move()`.
        """
        obj = self.getNode(where, name=name)
        obj._f_move(newparent, newname, overwrite, createparents)

    def copyNode(self, where, newparent=None, newname=None, name=None,
                 overwrite=False, recursive=False, createparents=False,
                 **kwargs):
        """
        Copy the given node and return the new one.

        The `where` and `name` arguments work as in `getNode()`,
        referencing the node to be acted upon.  The other arguments work
        as in `Node._f_copy()`.
        """
        obj = self.getNode(where, name=name)
        return obj._f_copy( newparent, newname,
                            overwrite, recursive, createparents, **kwargs )

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

        This method is deprecated; please use `getNodeAttr()`.
        """

        warnings.warn("""\
``File.getAttrNode()`` is deprecated; please use ``File.getNodeAttr()``""",
                      DeprecationWarning)
        return self.getNodeAttr(where, attrname, name)

    def getNodeAttr(self, where, attrname, name=None):
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

        This method is deprecated; please use `setNodeAttr()`.
        """

        warnings.warn("""\
``File.setAttrNode()`` is deprecated; please use ``File.setNodeAttr()``""",
                      DeprecationWarning)
        self.setNodeAttr(where, attrname, attrvalue, name)

    def setNodeAttr(self, where, attrname, attrvalue, name=None):
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

        This method is deprecated; please use `delNodeAttr()`.
        """

        warnings.warn("""\
``File.delAttrNode()`` is deprecated; please use ``File.delNodeAttr()``""",
                      DeprecationWarning)
        self.delNodeAttr(where, attrname, name)

    def delNodeAttr(self, where, attrname, name=None):
        """
        Delete a PyTables attribute from the given node.

        The `where` and `name` arguments work as in `getNode()`,
        referencing the node to be acted upon.  The other arguments work
        as in `Node._f_delAttr()`.
        """
        obj = self.getNode(where, name=name)
        obj._f_delAttr(attrname)


    def copyAttrs(self, where, dstnode, name=None):
        """
        Copy attributes from one node to another.

        This method is deprecated; please use `copyNodeAttrs()`.
        """

        warnings.warn("""\
``File.copyAttrs()`` is deprecated; please use ``File.copyNodeAttrs()``""",
                      DeprecationWarning)
        self.copyNodeAttrs(where, dstnode, name)

    def copyNodeAttrs(self, where, dstnode, name=None):
        """
        Copy attributes from one node to another.

        The `where` and `name` arguments work as in `getNode()`,
        referencing the node to be acted upon.  `dstnode` is the
        destination and can be either a path string or a `Node`
        instance.
        """
        srcObject = self.getNode(where, name=name)
        dstObject = self.getNode(dstnode)
        srcObject._v_attrs._f_copy(dstObject)


    def copyChildren(self, srcgroup, dstgroup,
                     overwrite=False, recursive=False,
                     createparents=False, **kwargs):
        """
        Copy the children of a group into another group.

        This method copies the nodes hanging from the source group
        `srcgroup` into the destination group `dstgroup`.  Existing
        destination nodes can be replaced by asserting the `overwrite`
        argument.  If the `recursive` argument is true, all descendant
        nodes of `srcnode` are recursively copied.  If `createparents`
        is true, the needed groups for the given destination group path
        to exist will be created.

        `kwargs` takes keyword arguments used to customize the copying
        process.  See the documentation of `Group._f_copyChildren()` for
        a description of those arguments.
        """

        srcGroup = self.getNode(srcgroup)  # Does the source node exist?
        self._checkGroup(srcGroup)  # Is it a group?

        srcGroup._f_copyChildren(
            dstgroup, overwrite, recursive, createparents, **kwargs )


    def copyFile(self, dstfilename, overwrite=False, **kwargs):
        """
        Copy the contents of this file to `dstfilename`.

        `dstfilename` must be a path string indicating the name of the
        destination file.  If it already exists, the copy will fail with
        an ``IOError``, unless the `overwrite` argument is true, in
        which case the destination file will be overwritten in place.

        Additional keyword arguments may be passed to customize the
        copying process.  For instance, title and filters may be
        changed, user attributes may be or may not be copied, data may
        be subsampled, stats may be collected, etc.  Arguments unknown
        to nodes are simply ignored.  Check the documentation for
        copying operations of nodes to see which options they support.

        Copying a file usually has the beneficial side effect of
        creating a more compact and cleaner version of the original
        file.
        """

        self._checkOpen()

        # Compute default arguments.
        filters = kwargs.get('filters', self.filters)
        copyuserattrs = kwargs.get('copyuserattrs', False)
        # These are *not* passed on.
        title = kwargs.pop('title', self.title)

        if os.path.isfile(dstfilename) and not overwrite:
            raise IOError("""\
file ``%s`` already exists; \
you may want to use the ``overwrite`` argument""" % dstfilename)

        # Create destination file, overwriting it.
        dstFileh = openFile(
            dstfilename, mode="w", title=title, filters=filters)

        try:
            # Maybe copy the user attributes of the root group.
            if copyuserattrs:
                self.root._v_attrs._f_copy(dstFileh.root)

            # Copy the rest of the hierarchy.
            self.root._f_copyChildren(dstFileh.root, recursive=True, **kwargs)
        finally:
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


    def iterNodes(self, where, classname=None):
        """
        Return an iterator yielding children nodes hanging from `where`.

        The `where` argument works as in `getNode()`, referencing the
        node to be acted upon.  The other arguments work as in
        `Group._f_listNodes()`.

        This is an iterator version of File.listNodes()
        """

        group = self.getNode(where)  # Does the parent exist?
        self._checkGroup(group)  # Is it a group?

        return group._f_iterNodes(classname)


    def __contains__(self, path):
        """
        Is there a node with that `path`?

        Returns ``True`` if the file has a node with the given `path` (a
        string), ``False`` otherwise.
        """

        try:
            self.getNode(path)
        except NoSuchNodeError:
            return False
        else:
            return True


    def __iter__(self):
        """Iterate over the nodes in the object tree."""

        return self.walkNodes('/')


    def walkNodes(self, where="/", classname=None):
        """Iterate over the nodes in the object tree.
        If "where" supplied, the iteration starts from this group.
        If "classname" is supplied, only instances of this class are
        returned.

        This version iterates over the leaves in the same group in order
        to avoid having a list referencing to them and thus, preventing
        the LRU cache to remove them after their use.
        """

        class_ = getClassByName(classname)

        if class_ is Group:  # only groups
            for group in self.walkGroups(where):
                yield group
        elif class_ is Node:  # all nodes
            yield self.getNode(where)
            for group in self.walkGroups(where):
                for leaf in self.iterNodes(group):
                    yield leaf
        else:  # only nodes of the named type
            for group in self.walkGroups(where):
                for leaf in self.iterNodes(group, classname):
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


    def _checkOpen(self):
        """
        Check the state of the file.

        If the file is closed, a `ClosedFileError` is raised.
        """
        if not self.isopen:
            raise ClosedFileError("the file object is closed")


    def _isWritable(self):
        """Is this file writable?"""
        return self.mode in ('w', 'a', 'r+')


    def _checkWritable(self):
        """Check whether the file is writable.

        If the file is not writable, a `FileModeError` is raised.
        """
        if not self._isWritable():
            raise FileModeError("the file is not writable")


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

        self._checkOpen()
        return self._undoEnabled


    def _checkUndoEnabled(self):
        if not self._undoEnabled:
            raise UndoRedoError("Undo/Redo feature is currently disabled!")


    def _createTransactionGroup(self):
        tgroup = TransactionGroupG(
            self.root, _transGroupName,
            "Transaction information container", new=True)
        # The format of the transaction container.
        tgroup._v_attrs._g__setattr('FORMATVERSION', _transVersion)
        return tgroup


    def _createTransaction(self, troot, tid):
        return TransactionG(
            troot, _transName % tid,
            "Transaction number %d" % tid, new=True)


    def _createMark(self, trans, mid):
        return MarkG(
            trans, _markName % mid,
            "Mark number %d" % mid, new=True)


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

        class ActionLog(NotLoggedMixin, Table):
            pass

        class ActionLogDesc(IsDescription):
            opcode = UInt8Col(pos=0)
            arg1   = StringCol(MAX_UNDO_PATH_LENGTH, pos=1, dflt="")
            arg2   = StringCol(MAX_UNDO_PATH_LENGTH, pos=2, dflt="")

        self._checkOpen()

        # Enabling several times is not allowed to avoid the user having
        # the illusion that a new implicit mark has been created
        # when calling enableUndo for the second time.

        if self.isUndoEnabled():
            raise UndoRedoError, "Undo/Redo feature is already enabled!"

        self._markers = {}
        self._seqmarkers = []
        self._nmarks = 0
        self._curtransaction = 0
        self._curmark = -1  # No marks yet

        # Get the Group for keeping user actions
        try:
            tgroup = self.getNode(_transGroupPath)
        except NodeError:
            # The file is going to be changed.
            self._checkWritable()

            # A transaction log group does not exist. Create it
            tgroup = self._createTransactionGroup()

            # Create a transaction.
            self._trans = self._createTransaction(
                tgroup, self._curtransaction)

            # Create an action log
            self._actionlog = ActionLog(
                tgroup, _actionLogName, ActionLogDesc, "Action log",
                filters=filters)

            # Create an implicit mark
            #self._actionlog.append([(_opToCode["MARK"], str(0), '')])
            # Use '\x00' to represent a NULL string. This is a bug
            # in numarray and should be reported.
            # F. Altet 2005-09-21
            self._actionlog.append([(_opToCode["MARK"], str(0), '\x00')])
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
            self._trans = tgroup._f_getChild(
                _transName % self._curtransaction)
            # Open the action log and go to the end of it
            self._actionlog = tgroup.actionlog
            for row in self._actionlog:
                if row["opcode"] == _opToCode["MARK"]:
                    name = row["arg2"]
                    self._markers[name] = self._nmarks
                    self._seqmarkers.append(row.nrow)
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

        self._checkOpen()

        if not self.isUndoEnabled():
            raise UndoRedoError, "Undo/Redo feature is already disabled!"

        # The file is going to be changed.
        self._checkWritable()

        del self._markers
        del self._seqmarkers
        del self._curmark
        del self._curaction
        del self._curtransaction
        del self._nmarks
        del self._actionlog
        # Recursively delete the transaction group
        tnode = self.getNode(_transGroupPath)
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

        self._checkOpen()
        self._checkUndoEnabled()

        if name is None:
            name = ''
        else:
            if not isinstance(name, str):
                raise TypeError, \
"Only strings are allowed as mark names. You passed object: '%s'" % name
            if name in self._markers:
                raise UndoRedoError, \
"Name '%s' is already used as a marker name. Try another one." % name

            # The file is going to be changed.
            self._checkWritable()

            self._markers[name] = self._curmark + 1

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
            # We are not, so delete the trailing actions
            self._actionlog.removeRows(self._curaction + 1,
                                       self._actionlog.nrows)
            # Reset the current marker group
            mnode = self.getNode(_markPath % (self._curtransaction,
                                               self._curmark))
            mnode._g_reset()
            # Delete the marker groups with backup objects
            for mark in xrange(self._curmark+1, self._nmarks):
                mnode = self.getNode(_markPath % (self._curtransaction, mark))
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


    def _doundo(self, finalaction, direction):
        "Undo/Redo actions up to final action in the specificed direction"

        if direction < 0:
            actionlog = self._actionlog[finalaction+1:self._curaction+1][::-1]
        else:
            actionlog = self._actionlog[self._curaction:finalaction]

        # Uncomment this for debugging
#         print "curaction, finalaction, direction", \
#               self._curaction, finalaction, direction
        for i in xrange(len(actionlog)):
            if actionlog['opcode'][i] <> _opToCode["MARK"]:
                # undo/redo the action
                if direction > 0:
                    # Uncomment this for debugging
#                     print "redo-->", \
#                           _codeToOp[actionlog['opcode'][i]],\
#                           actionlog['arg1'][i],\
#                           actionlog['arg2'][i]
                    undoredo.redo(self,
                                  #_codeToOp[actionlog['opcode'][i]],
                                  # The next is a workaround for python < 2.5
                                  _codeToOp[int(actionlog['opcode'][i])],
                                  actionlog['arg1'][i],
                                  actionlog['arg2'][i])
                else:
                    # Uncomment this for debugging
#                     print "undo-->", \
#                           _codeToOp[actionlog['opcode'][i]],\
#                           actionlog['arg1'][i],\
#                           actionlog['arg2'][i]
                    undoredo.undo(self,
                                  #_codeToOp[actionlog['opcode'][i]],
                                  # The next is a workaround for python < 2.5
                                  _codeToOp[int(actionlog['opcode'][i])],
                                  actionlog['arg1'][i],
                                  actionlog['arg2'][i])
            else:
                if direction > 0:
                    self._curmark = int(actionlog['arg1'][i])
                else:
                    self._curmark = int(actionlog['arg1'][i]) - 1
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

        self._checkOpen()
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

        # The file is going to be changed.
        self._checkWritable()

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

        self._checkOpen()
        self._checkUndoEnabled()

#         print "(pre)REDO: (curaction, curmark) = (%s, %s)" % \
#               (self._curaction, self._curmark)
        if self._curaction >= self._actionlog.nrows - 1:
            # We are at the end of log, so no action
            return

        if mark is None:
            mark = self._curmark + 1
        elif mark == -1:
            mark = self._nmarks  # Go beyond the mark bounds up to the end
        # Get the mark ID number
        markid = self._getMarkID(mark)
        finalaction = self._getFinalAction(markid)
        if finalaction < self._curaction + 1:
            raise UndoRedoError("""\
Mark ``%s`` is older than the current mark. Use `redo()` or `goto()` instead."""
                                % (mark,))

        # The file is going to be changed.
        self._checkWritable()

        # Get the final action ID to go
        self._curaction += 1

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

        self._checkOpen()
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

        self._checkOpen()
        self._checkUndoEnabled()
        return self._curmark


    def _shadowName(self):
        """
        Compute and return a shadow name.

        Computes the current shadow name according to the current
        transaction, mark and action.  It returns a tuple with the
        shadow parent node and the name of the shadow in it.
        """

        parent = self.getNode(
            _shadowParent % (self._curtransaction, self._curmark))
        name = _shadowName % (self._curaction,)

        return (parent, name)

    # </Undo/Redo support>


    def flush(self):
        """Flush all the alive nodes and HDF5 buffers."""

        self._checkOpen()

        # First, flush PyTables buffers on alive leaves.
        # Leaves that are dead should have been flushed already (at least,
        # users are directed to do this through a PerformanceWarning!)
        for path, refnode in self._aliveNodes.iteritems():
            if '/_i_' not in path:  # Indexes are not necessary to be flushed
                node = refnode()
                if isinstance(node, Leaf):
                    node.flush()

        # Flush the cache to disk
        self._flushFile(0)  # 0 means local scope, 1 global (virtual) scope


    def _closeDescendentsOf(self, group):
        """Close all the *loaded* descendent nodes of the given `group`."""

        assert isinstance(group, Group)

        prefix = group._v_pathname + '/'
        if prefix == '//':
            prefix = '/'

        self._closeNodes(
            [path for path in self._aliveNodes if path.startswith(prefix)])

        self._closeNodes(
            [path for path in self._deadNodes if path.startswith(prefix)])


    def _closeNodes(self, nodePaths, getNode=None):
        """
        Close all nodes in the list of `nodePaths`.

        This method uses the `getNode` callable object to get the node
        object by its path.  If `getNode` is not given, `File.getNode()`
        is used.  ``KeyError`` exceptions on `getNode` invocations are
        ignored.
        """

        if getNode is None:
            getNode = self.getNode

        for nodePath in nodePaths:
            try:
                node = getNode(nodePath)
                node._f_close()
                del node
            except KeyError:
                pass


    def close(self):
        """Close all the nodes in HDF5 file and close the file."""
        global _open_files

        # If the file is already closed, return immediately
        if not self.isopen:
            return

        if self._undoEnabled and self._isWritable():
            # Save the current mark and current action
            self._actionlog.attrs._g__setattr("CURMARK", self._curmark)
            self._actionlog.attrs._g__setattr("CURACTION", self._curaction)

        # Close all loaded nodes.

        # First, close the alive nodes and delete them
        # so they are not placed in the limbo again.
        # We do not use ``getNode()`` for efficiency.
        aliveNodes = self._aliveNodes
        # These two steps ensure tables are closed *before* their indices.
        self._closeNodes([path for path in aliveNodes.keys()
                          if '/_i_' not in path],  # not indices
                         lambda path: aliveNodes[path])
        self._closeNodes(aliveNodes.keys(),  # everything else (i.e. indices)
                         lambda path: aliveNodes[path])
        assert len(aliveNodes) == 0, \
               ("alive nodes remain after closing alive nodes: %s"
                % aliveNodes.keys())

        # Next, revive the dead nodes, close and delete them
        # so they are not placed in the limbo again.
        # We do not use ``getNode()`` for efficiency
        # and to avoid accidentally loading ancestor nodes.
        deadNodes = self._deadNodes
        # These two steps ensure tables are closed *before* their indices.
        self._closeNodes([path for path in deadNodes
                          if '/_i_' not in path],  # not indices
                         lambda path: self._reviveNode(path))
        self._closeNodes([path for path in deadNodes],
                         lambda path: self._reviveNode(path))
        assert len(deadNodes) == 0, \
               ("dead nodes remain after closing dead nodes: %s"
                % [path for path in deadNodes])

        # No other nodes should have been revived.
        assert len(aliveNodes) == 0, \
               ("alive nodes remain after closing dead nodes: %s"
                % aliveNodes.keys())

        # When all other nodes have been closed, close the root group.
        # This is done at the end because some nodes
        # may still need to be loaded during the closing process;
        # thus the root node must be open until the very end.
        self.root._f_close()

        # Close the file
        self._closeFile()
        # After the objects are disconnected, destroy the
        # object dictionary using the brute force ;-)
        # This should help to the garbage collector
        self.__dict__.clear()
        # Set the flag to indicate that the file is closed
        self.isopen = 0
        # Delete the entry in the dictionary of opened files
        del _open_files[self]

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

        if not self.isopen:
            return "<closed File>"

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


    def _refNode(self, node, nodePath):
        """
        Register `node` as alive and insert references to it.
        """

        if nodePath != '/':
            # The root group does not participate in alive/dead stuff.
            aliveNodes = self._aliveNodes
            assert nodePath not in aliveNodes, \
                   "file already has a node with path ``%s``" % nodePath

            # Add the node to the set of referenced ones.
            aliveNodes[nodePath] = node


    def _unrefNode(self, nodePath):
        """Unregister `node` as alive and remove references to it."""

        if nodePath != '/':
            # The root group does not participate in alive/dead stuff.
            aliveNodes = self._aliveNodes
            assert nodePath in aliveNodes, \
                   "file does not have a node with path ``%s``" % nodePath

            # Remove the node from the set of referenced ones.
            del aliveNodes[nodePath]


    def _killNode(self, node):
        """
        Kill the `node`.

        Moves the `node` from the set of alive, referenced nodes to the
        set of dead, unreferenced ones.
        """

        nodePath = node._v_pathname
        assert nodePath in self._aliveNodes, \
               "trying to kill non-alive node ``%s``" % nodePath

        node._g_preKillHook()

        # Remove all references to the node.
        self._unrefNode(nodePath)
        # Save the dead node in the limbo.
        self._deadNodes[nodePath] = node


    def _reviveNode(self, nodePath):
        """
        Revive the node under `nodePath` and return it.

        Moves the node under `nodePath` from the set of dead,
        unreferenced nodes to the set of alive, referenced ones.
        """

        assert nodePath in self._deadNodes, \
               "trying to revive non-dead node ``%s``" % nodePath

        # Take the node out of the limbo.
        node = self._deadNodes.pop(nodePath)
        # Make references to the node.
        self._refNode(node, nodePath)

        node._g_postReviveHook()

        return node

# If a user hits ^C during a run, it is wise to gracefully close the opened files.
def close_open_files():
    global _open_files
    if len(_open_files):
        print >> sys.stderr, "Closing remaining opened files:",
    for fileh in _open_files.keys():
        print >> sys.stderr, "%s..." % (fileh.filename,),
        fileh.close()
        print >> sys.stderr, "done",
    print >> sys.stderr

import atexit
atexit.register(close_open_files)


## Local Variables:
## mode: python
## py-indent-offset: 4
## tab-width: 4
## fill-column: 72
## End:
