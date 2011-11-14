########################################################################
#
#       License:        BSD
#       Created:        September 4, 2002
#       Author:  Francesc Alted - faltet@pytables.com
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
    openFile(name[, mode][, title][, rootUEP][, filters][, **kwargs])

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

import numexpr

import tables.misc.proxydict
from tables import hdf5Extension
from tables import utilsExtension
from tables import parameters
from tables.exceptions import \
     ClosedFileError, FileModeError, \
     NodeError, NoSuchNodeError, UndoRedoError, \
     UndoRedoWarning, PerformanceWarning, Incompat16Warning
from tables.registry import getClassByName
from tables.path import joinPath, splitPath, isVisiblePath
from tables import undoredo
from tables.description import IsDescription, UInt8Col, StringCol
from tables.filters import Filters
from tables.node import Node, NotLoggedMixin
from tables.group import Group, RootGroup
from tables.group import TransactionGroupG, TransactionG, MarkG
from tables.leaf import Leaf
from tables.array import Array
from tables.carray import CArray
from tables.earray import EArray
from tables.vlarray import VLArray
from tables.table import Table
from tables import linkExtension
from utils import detectNumberOfCores
from tables import lrucacheExtension


from tables.link import SoftLink
try:
    from tables.link import ExternalLink
except ImportError:
    are_extlinks_available = False
else:
    are_extlinks_available = True


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

# Dict of opened files (keys are filenames and values filehandlers)
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


def openFile(filename, mode="r", title="", rootUEP="/", filters=None,
             **kwargs):

    """Open an HDF5 file and return a File object.

    Arguments:

    `filename` -- The name of the file (supports environment variable
        expansion).  It is suggested that file names have any of the
        ``.h5``, ``.hdf`` or ``.hdf5`` extensions, although this is not
        mandatory.

    `mode` -- The mode to open the file.  It can be one of the
        following:

        ``'r'``
            Read-only; no data can be modified.
        ``'w``'
            Write; a new file is created (an existing file with the same
            name would be deleted).
        ``'a'``
            Append; an existing file is opened for reading and writing,
            and if the file does not exist it is created.
        ``'r+'``
            It is similar to ``'a'``, but the file must already exist.

    `title` -- If the file is to be created, a ``TITLE`` string
        attribute will be set on the root group with the given value.
        Otherwise, the title will be read from disk, and this will not
        have any effect.

    `rootUEP` -- The root User Entry Point.  This is a group in the HDF5
        hierarchy which will be taken as the starting point to create
        the object tree.  It can be whatever existing group in the file,
        named by its HDF5 path. If it does not exist, an `HDF5ExtError`
        is issued.  Use this if you do not want to build the *entire*
        object tree, but rather only a *subtree* of it.

    `filters` -- An instance of the `Filters` class that provides
        information about the desired I/O filters applicable to the
        leaves that hang directly from the *root group*, unless other
        filter properties are specified for these leaves.  Besides, if
        you do not specify filter properties for child groups, they will
        inherit these ones, which will in turn propagate to child nodes.

    In addition, it recognizes the names of parameters present in
    ``tables/parameters.py`` as additional keyword arguments. Check the
    suitable appendix in User's Guide for a detailed info on the supported
    parameters.

    """

    # Get the list of already opened files
    ofiles = [fname for fname in _open_files]
    if filename in ofiles:
        filehandle = _open_files[filename]
        omode = filehandle.mode
        # 'r' is incompatible with everything except 'r' itself
        if mode == 'r' and omode != 'r':
            raise ValueError(
                "The file '%s' is already opened, but "
                "not in read-only mode (as requested)." % filename)
        # 'a' and 'r+' are compatible with everything except 'r'
        elif mode in ('a', 'r+') and omode == 'r':
            raise ValueError(
                "The file '%s' is already opened, but "
                "in read-only mode.  Please close it before "
                "reopening in append mode." % filename)
        # 'w' means that we want to destroy existing contents
        elif mode == 'w':
            raise ValueError(
                "The file '%s' is already opened.  Please "
                "close it before reopening in write mode." % filename)
        else:
            # The file is already open and modes are compatible
            # Increase the number of openings for this file
            filehandle._open_count += 1
            return filehandle
    # Finally, create the File instance, and return it
    return File(filename, mode, title, rootUEP, filters, **kwargs)


class _AliveNodes(dict):

    """Stores strong or weak references to nodes in a transparent way."""

    def __init__(self, nodeCacheSlots):
        if nodeCacheSlots > 0:
            self.hasdeadnodes = True
        else:
            self.hasdeadnodes = False
        if nodeCacheSlots >= 0:
            self.hassoftlinks = True
        else:
            self.hassoftlinks = False
        self.nodeCacheSlots = nodeCacheSlots
        super(_AliveNodes, self).__init__()

    def __getitem__(self, key):
        if self.hassoftlinks:
            ref = super(_AliveNodes, self).__getitem__(key)()
        else:
            ref = super(_AliveNodes, self).__getitem__(key)
        return ref

    def __setitem__(self, key, value):
        if self.hassoftlinks:
            ref = weakref.ref(value)
        else:
            ref = value
            # Check if we are running out of space
            if self.nodeCacheSlots < 0 and len(self) > -self.nodeCacheSlots:
                warnings.warn("""\
the dictionary of alive nodes is exceeding the recommended maximum number (%d); \
be ready to see PyTables asking for *lots* of memory and possibly slow I/O."""
                      % (-self.nodeCacheSlots),
                      PerformanceWarning)
        super(_AliveNodes, self).__setitem__(key, ref)



class _DeadNodes(lrucacheExtension.NodeCache):
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

    File objects can also act as context managers when using the
    ``with`` statement introduced in Python 2.5.  When exiting a
    context, the file is automatically closed.

    Public instance variables
    -------------------------

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
    rootUEP
        The UEP (user entry point) group in the file (see the
        `openFile()` function).
    filters
        Default filter properties for the root group (see the `Filters`
        class).
    root
        The *root* of the object tree hierarchy (a `Group` instance).

    Public methods -- file handling
    -------------------------------

    * close()
    * copyFile(dstfilename[, overwrite][, **kwargs])
    * flush()
    * fileno()
    * __enter__()
    * __exit__([*exc_info])
    * __str__()
    * __repr__()

    Public methods -- hierarchy manipulation
    ----------------------------------------

    * copyChildren(srcgroup, dstgroup[, overwrite][, recursive]
                   [, **kwargs])
    * copyNode(where, newparent, newname[, name][, overwrite]
               [, recursive][, **kwargs])
    * createArray(where, name, array[, title][, byteorder][, createparents])
    * createCArray(where, name, atom, shape [, title][, filters]
                   [, chunkshape][, byteorder][, createparents])
    * createEArray(where, name, atom, shape [, title][, filters]
                   [, expectedrows][, chunkshape][, byteorder]
                   [, createparents])
    * createGroup(where, name[, title][, filters][, createparents])
    * createTable(where, name, description[, title][, filters]
                  [, expectedrows][, chunkshape][, byteorder][, createparents])
    * createVLArray(where, name, atom[, title][, filters]
                    [, expectedsizeinMB][, chunkshape][, byteorder]
                    [, createparents])
    * moveNode(where, newparent, newname[, name][, overwrite])
    * removeNode(where[, name][, recursive])
    * renameNode(where, newname[, name][, overwrite])

    Public methods -- tree traversal
    --------------------------------

    * getNode(where[, name][,classname])
    * isVisibleNode(path)
    * iterNodes(where[, classname])
    * listNodes(where[, classname])
    * walkGroups([where])
    * walkNodes([where][, classname])
    * __contains__(path)
    * __iter__()

    Public methods -- Undo/Redo support
    -----------------------------------

    * disableUndo()
    * enableUndo([filters])
    * getCurrentMark()
    * goto(mark)
    * isUndoEnabled()
    * mark([name])
    * redo([mark])
    * undo([mark])

    Public methods -- attribute handling
    ------------------------------------

    * copyNodeAttrs(where, dstnode[, name])
    * delNodeAttr(where, attrname[, name])
    * getNodeAttr(where, attrname[, name])
    * setNodeAttr(where, attrname, attrvalue[, name])
    """

    ## <class variables>
    # The top level kinds. Group must go first!
    _node_kinds = ('Group', 'Leaf', 'Link', 'Unknown')

    ## </class variables>

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

    open_count = property(
        lambda self: self._open_count, None, None,
        "The number of times this file has been opened currently.")

    ## </properties>


    def __init__(self, filename, mode="r", title="",
                 rootUEP="/", filters=None, **kwargs):
        """Open an HDF5 file.

        See `openFile()` for info about the parameters.
        """
        self.filename = filename
        self.mode = mode

        # Expand the form '~user'
        path = os.path.expanduser(filename)
        # Expand the environment variables
        path = os.path.expandvars(path)

        # Get all the parameters in parameter file(s)
        params = dict([(k, v) for k,v in parameters.__dict__.iteritems()
                       if k.isupper() and not k.startswith('_')])
        # Update them with possible keyword arguments
        params.update(kwargs)

        # If MAX_THREADS is not set yet, set it to the number of cores
        # on this machine.
        if params['MAX_THREADS'] is None:
            params['MAX_THREADS'] = detectNumberOfCores()

        self.params = params

        # Now, it is time to initialize the File extension
        self._g_new(filename, mode, **params)

        # Check filters and set PyTables format version for new files.
        new = self._v_new
        if new:
            _checkfilters(filters)
            self.format_version = format_version

        # Nodes referenced by a variable are kept in `_aliveNodes`.
        # When they are no longer referenced, they move themselves
        # to `_deadNodes`, where they are kept until they are referenced again
        # or they are preempted from it by other unreferenced nodes.
        nodeCacheSlots = params['NODE_CACHE_SLOTS']
        self._aliveNodes = _AliveNodes(nodeCacheSlots)
        if nodeCacheSlots > 0:
            self._deadNodes = _DeadNodes(nodeCacheSlots)
        else:
            self._deadNodes = _NoDeadNodes()

        # For the moment Undo/Redo is not enabled.
        self._undoEnabled = False

        # Set the flag to indicate that the file has been opened.
        # It must be set before opening the root group
        # to allow some basic access to its attributes.
        self.isopen = 1

        # Append the name of the file to the global dict of files opened.
        _open_files[self.filename] = self

        # Set the number of times this file has been opened to 1
        self._open_count = 1

        # Get the root group from this file
        self.root = root = self.__getRootGroup(rootUEP, title, filters)
        # Complete the creation of the root node
        # (see the explanation in ``RootGroup.__init__()``.
        root._g_postInitHook()

        # Save the PyTables format version for this file.
        if new:
            if params['PYTABLES_SYS_ATTRS']:
                root._v_attrs._g__setattr(
                    'PYTABLES_FORMAT_VERSION', format_version)

        # If the file is old, and not opened in "read-only" mode,
        # check if it has a transaction log
        if not new and self.mode != "r" and _transGroupPath in self:
            # It does. Enable the undo.
            self.enableUndo()

        # Set the maximum number of threads for Numexpr
        numexpr.set_vml_num_threads(params['MAX_THREADS'])


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
            if not self.format_version:
                # PYTABLES_FORMAT_VERSION attribute is not present
                self.format_version = "unknown"
                self._isPTFile = False

        # Create new attributes for the root Group instance and
        # create the object tree
        return RootGroup(self, rootUEP, title=title, new=new, filters=filters)


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
        """
        Create a new group with the given `name` in `where` location.
        See the `Group` class for more information on groups.

        `filters`
            An instance of the `Filters` class that provides information
            about the desired I/O filters applicable to the leaves that
            hang directly from this new group (unless other filter
            properties are specified for these leaves).  Besides, if you
            do not specify filter properties for its child groups, they
            will inherit these ones.

        See `File.createTable()` for more information on the rest of
        parameters.
        """
        parentNode = self._getOrCreatePath(where, createparents)
        _checkfilters(filters)
        return Group(parentNode, name,
                     title=title, new=True, filters=filters)


    def createTable(self, where, name, description, title="",
                    filters=None, expectedrows=10000,
                    chunkshape=None, byteorder=None,
                    createparents=False):
        """
        Create a new table with the given `name` in `where` location.
        See the `Table` class for more information on tables.

        `where`
            The parent group where the new table will hang from.  It can
            be a path string (for example '/level1/leaf5'), or a `Group`
            instance.

        `name`
            The name of the new table.

        `description`
            This is an object that describes the table, i.e. how many
            columns it has, their names, types, shapes, etc.  It can be
            any of the following:

            A user-defined class
                This should inherit from the `IsDescription` class where
                table fields are specified.

            A dictionary
                For example, when you do not know beforehand which
                structure your table will have.

            A `Description` instance
                You can use the ``description`` attribute of another
                table to create a new one with the same structure.

            A NumPy dtype
                A completely general structured NumPy dtype.

            A NumPy (record) array
                The dtype of this record array will be used as the
                description.  Also, in case the array has actual data,
                it will be injected into the newly created table.

            A ``RecArray`` instance
                Object from the ``numarray`` package.  This does not
                give you the possibility to create a nested table.
                Array data is injected into the new table.

            A ``NestedRecArray`` instance
                If you want to have nested columns in your table and you
                are using ``numarray``, you can use this object.  Array
                data is injected into the new table.

        `title`
            A description for this node (it sets the ``TITLE`` HDF5
            attribute on disk).

        `filters`
            An instance of the `Filters` class that provides information
            about the desired I/O filters to be applied during the life
            of this object.

        `expectedrows`
            A user estimate about the number of rows that will be in the
            table.  If not provided, the default value is appropriate
            for tables up to 10 MB in size (more or less).  If you plan
            to create a bigger table try providing a guess; this will
            optimize the HDF5 B-Tree creation and management process
            time and the amount of memory used.  If you want to specify
            your own chunk size for I/O purposes, see also the
            `chunkshape` parameter below.

        `chunkshape`
            The shape of the data chunk to be read or written in a
            single HDF5 I/O operation.  Filters are applied to those
            chunks of data.  The rank of the `chunkshape` for tables
            must be 1.  If ``None``, a sensible value is calculated
            (which is recommended).

        `byteorder`
            The byteorder of data *on disk*, specified as 'little' or
            'big'.  If this is not specified, the byteorder is that of
            the platform, unless you passed an array as the
            `description`, in which case its byteorder will be used.

        `createparents`
            Whether to create the needed groups for the parent path to
            exist (not done by default).
        """
        parentNode = self._getOrCreatePath(where, createparents)
        if description is None:
            raise ValueError("invalid table description: None")
        _checkfilters(filters)
        return Table(parentNode, name,
                     description=description, title=title,
                     filters=filters, expectedrows=expectedrows,
                     chunkshape=chunkshape, byteorder=byteorder)


    def createArray(self, where, name, object, title="",
                    byteorder=None, createparents=False):
        """
        Create a new array with the given `name` in `where` location.
        See the `Array` class for more information on arrays.

        `object`
            The array or scalar to be saved.  Accepted types are NumPy
            arrays and scalars, ``numarray`` arrays and string arrays,
            Numeric arrays and scalars, as well as native Python
            sequences and scalars, provided that values are regular
            (i.e. they are not like ``[[1,2],2]``) and homogeneous
            (i.e. all the elements are of the same type).

            Also, objects that have some of their dimensions equal to 0
            are not supported (use an `EArray` node if you want to store
            an array with one of its dimensions equal to 0).

        `byteorder`
            The byteorder of the data *on disk*, specified as 'little'
            or 'big'.  If this is not specified, the byteorder is that
            of the given `object`.

        See `File.createTable()` for more information on the rest of
        parameters.
        """
        parentNode = self._getOrCreatePath(where, createparents)
        return Array(parentNode, name,
                     object=object, title=title, byteorder=byteorder)


    def createCArray(self, where, name, atom, shape, title="",
                     filters=None, chunkshape=None,
                     byteorder=None, createparents=False):
        """
        Create a new chunked array with the given `name` in `where`
        location.  See the `CArray` class for more information on
        chunked arrays.

        `atom`
            An `Atom` instance representing the *type* and *shape* of
            the atomic objects to be saved.

        `shape`
            The shape of the new array.

        `chunkshape`
            The shape of the data chunk to be read or written in a
            single HDF5 I/O operation.  Filters are applied to those
            chunks of data.  The dimensionality of `chunkshape` must be
            the same as that of `shape`.  If ``None``, a sensible value
            is calculated (which is recommended).

        See `File.createTable()` for more information on the rest of
        parameters.
        """
        parentNode = self._getOrCreatePath(where, createparents)
        _checkfilters(filters)
        return CArray(parentNode, name,
                      atom=atom, shape=shape, title=title, filters=filters,
                      chunkshape=chunkshape, byteorder=byteorder)


    def createEArray(self, where, name, atom, shape, title="",
                     filters=None, expectedrows=1000,
                     chunkshape=None, byteorder=None,
                     createparents=False):
        """
        Create a new enlargeable array with the given `name` in `where`
        location.  See the `EArray` class for more information on
        enlargeable arrays.

        `atom`
            An `Atom` instance representing the *type* and *shape* of
            the atomic objects to be saved.

        `shape`
            The shape of the new array.  One (and only one) of the shape
            dimensions *must* be 0.  The dimension being 0 means that
            the resulting `EArray` object can be extended along it.
            Multiple enlargeable dimensions are not supported right now.

        `expectedrows`
            A user estimate about the number of row elements that will
            be added to the growable dimension in the `EArray` node.  If
            not provided, the default value is 1000 rows.  If you plan
            to create either a much smaller or a much bigger array try
            providing a guess; this will optimize the HDF5 B-Tree
            creation and management process time and the amount of
            memory used.  If you want to specify your own chunk size for
            I/O purposes, see also the `chunkshape` parameter below.

        `chunkshape`
            The shape of the data chunk to be read or written in a
            single HDF5 I/O operation.  Filters are applied to those
            chunks of data.  The dimensionality of `chunkshape` must be
            the same as that of `shape` (beware: no dimension should be
            0 this time!).  If ``None``, a sensible value is calculated
            (which is recommended).

        `byteorder`
            The byteorder of the data *on disk*, specified as 'little'
            or 'big'. If this is not specified, the byteorder is that
            of the platform.

        See `File.createTable()` for more information on the rest of
        parameters.
        """
        parentNode = self._getOrCreatePath(where, createparents)
        _checkfilters(filters)
        return EArray(parentNode, name,
                      atom=atom, shape=shape, title=title,
                      filters=filters, expectedrows=expectedrows,
                      chunkshape=chunkshape, byteorder=byteorder)


    def createVLArray(self, where, name, atom, title="",
                      filters=None, expectedsizeinMB=1.0,
                      chunkshape=None, byteorder=None,
                      createparents=False):
        """
        Create a new variable-length array with the given `name` in
        `where` location.  See the `VLArray` class for more information
        on variable-length arrays.

        `atom`
            An `Atom` instance representing the *type* and *shape* of
            the atomic objects to be saved.

        `expectedsizeinMB`
            An user estimate about the size (in MB) of the final
            `VLArray` node.  If not provided, the default value is 1 MB.
            If you plan to create either a much smaller or a much bigger
            array try providing a guess; this will optimize the HDF5
            B-Tree creation and management process time and the amount
            of memory used.  If you want to specify your own chunk size
            for I/O purposes, see also the `chunkshape` parameter below.

        `chunkshape`
            The shape of the data chunk to be read or written in a
            single HDF5 I/O operation.  Filters are applied to those
            chunks of data.  The dimensionality of `chunkshape` must be
            1.  If ``None``, a sensible value is calculated (which is
            recommended).

        See `File.createTable()` for more information on the rest of
        parameters.
        """
        parentNode = self._getOrCreatePath(where, createparents)
        _checkfilters(filters)
        return VLArray(parentNode, name,
                       atom=atom, title=title, filters=filters,
                       expectedsizeinMB=expectedsizeinMB,
                       chunkshape=chunkshape, byteorder=byteorder)


    def createHardLink(self, where, name, target, createparents=False):
        """
        Create a hard link to a `target` node with the given `name` in
        `where` location.  `target` can be a node object or a path
        string.  If `createparents` is true, the intermediate groups
        required for reaching `where` are created (the default is not
        doing so).

        The returned node is a regular `Group` or `Leaf` instance.
        """
        targetNode = self.getNode(target)
        parentNode = self._getOrCreatePath(where, createparents)
        linkExtension._g_createHardLink(parentNode, name, targetNode)
        # Refresh children names in link's parent node
        parentNode._g_addChildrenNames()
        # Return the target node
        return self.getNode(parentNode, name)


    def createSoftLink(self, where, name, target, createparents=False):
        """
        Create a soft link (aka symbolic link) to a `target` node with
        the given `name` in `where` location.  `target` can be a node
        object or a path string.  If `createparents` is true, the
        intermediate groups required for reaching `where` are created
        (the default is not doing so).

        The returned node is a `SoftLink` instance.  See the `SoftLink`
        class for more information on soft links.
        """
        if type(target) is not str:
            if hasattr(target, '_v_pathname'):   # quacks like a Node
                target = target._v_pathname
            else:
                raise ValueError("`target` has to be a string or a node object")
        parentNode = self._getOrCreatePath(where, createparents)
        slink = SoftLink(parentNode, name, target)
        # Refresh children names in link's parent node
        parentNode._g_addChildrenNames()
        return slink


    def createExternalLink(self, where, name, target, createparents=False,
                           warn16incompat=True):
        """
        Create an external link to a `target` node with the given `name`
        in `where` location.  `target` can be a node object in another
        file or a path string in the form 'file:/path/to/node'.  If
        `createparents` is true, the intermediate groups required for
        reaching `where` are created (the default is not doing so).

        The purpose of the `warn16incompat` argument is to avoid an
        `Incompat16Warning` (see below).  The default is to issue the
        warning.

        The returned node is an `ExternalLink` instance.  See the
        `SoftLink` class for more information on external links.

        .. Warning:: External links are only supported when PyTables is
           compiled against HDF5 1.8.x series.  When using PyTables with
           HDF5 1.6.x, the *parent* group containing external link
           objects will be mapped to an `Unknown` instance and you won't
           be able to access *any* node hanging of this parent group.
           It follows that if the parent group containing the external
           link is the root group, you won't be able to read *any*
           information contained in the file when using HDF5 1.6.x.

        """
        if not are_extlinks_available:
            raise NotImplementedError(
                "External links are not available when using HDF5 1.6.x")
        if warn16incompat:
            warnings.warn("""\
external links are only supported when PyTables is compiled against HDF5 1.8.x series and they, and their parent groups, are unreadable with HDF5 1.6.x series.  You can set `warn16incompat` argument to false to disable this warning.""",
                          Incompat16Warning)

        if type(target) is not str:
            if hasattr(target, '_v_pathname'):   # quacks like a Node
                target = target._v_file.filename+':'+target._v_pathname
            else:
                raise ValueError("`target` has to be a string or a node object")
        elif target.find(':/') == -1:
            raise ValueError(
                "`target` must expressed as 'file:/path/to/node'")
        parentNode = self._getOrCreatePath(where, createparents)
        elink = ExternalLink(parentNode, name, target)
        # Refresh children names in link's parent node
        parentNode._g_addChildrenNames()
        return elink


    # There is another version of _getNode in Pyrex space, but only
    # marginally faster (5% or less, but sometimes slower!) than this one.
    # So I think it is worth to use this one instead (much easier to debug).
    def _getNode(self, nodePath):
        # The root node is always at hand.
        if nodePath == '/':
            return self.root

        aliveNodes = self._aliveNodes
        deadNodes = self._deadNodes

        if nodePath in aliveNodes:
            # The parent node is in memory and alive, so get it.
            node = aliveNodes[nodePath]
            assert node is not None, \
                   "stale weak reference to dead node ``%s``" % nodePath
            return node
        if nodePath in deadNodes:
            # The parent node is in memory but dead, so revive it.
            node = self._reviveNode(nodePath)
            return node

        # The node has not been found in alive or dead nodes.
        # Open it directly from disk.
        node = self.root._g_loadChild(nodePath)
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
        `NoSuchNodeError` is raised.  Please note that hidden nodes are
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
        elif isinstance(where, basestring):
            node = None
            if where.startswith('/'):
                nodePath = where
            else:
                raise NameError(
                    "``where`` must start with a slash ('/')")
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


    def renameNode(self, where, newname, name=None, overwrite=False):
        """
        Change the name of the node specified by `where` and `name` to
        `newname`.

        `where`, `name`
            These arguments work as in `File.getNode()`, referencing the
            node to be acted upon.

        `newname`
            The new name to be assigned to the node (a string).

        `overwrite`
            Whether to recursively remove a node with the same `newname`
            if it already exists (not done by default).
        """
        obj = self.getNode(where, name=name)
        obj._f_rename(newname, overwrite)


    def moveNode(self, where, newparent=None, newname=None, name=None,
                 overwrite=False, createparents=False):
        """
        Move the node specified by `where` and `name` to
        ``newparent/newname``.

        `where`, `name`
            These arguments work as in `File.getNode()`, referencing the
            node to be acted upon.

        `newparent`
            The destination group that the node will be moved into (a
            path name or a `Group` instance).  If it is not specified or
            ``None``, the current parent group is chosen as the new
            parent.

        `newname`
            The name to be assigned to the node in its destination (a
            string).  If it is not specified or ``None``, the current
            name is chosen as the new name.

        See `Node._f_move()` for further details on the semantics of
        moving nodes.
        """
        obj = self.getNode(where, name=name)
        obj._f_move(newparent, newname, overwrite, createparents)


    def copyNode(self, where, newparent=None, newname=None, name=None,
                 overwrite=False, recursive=False, createparents=False,
                 **kwargs):
        """
        Copy the node specified by `where` and `name` to
        ``newparent/newname``.

        `where`, `name`
            These arguments work as in `File.getNode()`, referencing the
            node to be acted upon.

        `newparent`
            The destination group that the node will be copied into (a
            path name or a `Group` instance).  If not specified or
            ``None``, the current parent group is chosen as the new
            parent.

        `newname`
            The name to be assigned to the new copy in its destination
            (a string).  If it is not specified or ``None``, the current
            name is chosen as the new name.

        Additional keyword arguments may be passed to customize the
        copying process.  The supported arguments depend on the kind of
        node being copied.  See `Group._f_copy()` and `Leaf.copy()` for
        more information on their allowed keyword arguments.

        This method returns the newly created copy of the source node
        (i.e. the destination node).  See `Node._f_copy()` for further
        details on the semantics of copying nodes.
        """
        obj = self.getNode(where, name=name)
        if obj._v_depth == 0 and newparent and not newname:
            npobj = self.getNode(newparent)
            if obj._v_file is not npobj._v_file:
                # Special case for copying file1:/ --> file2:/path
                self.root._f_copyChildren(npobj, overwrite=overwrite,
                                          recursive=recursive, **kwargs)
                return npobj
            else:
                raise IOError("You cannot copy a root group over the same file")
        return obj._f_copy( newparent, newname,
                            overwrite, recursive, createparents, **kwargs )


    def removeNode(self, where, name=None, recursive=False):
        """
        Remove the object node `name` under `where` location.

        `where`, `name`
            These arguments work as in `File.getNode()`, referencing the
            node to be acted upon.

        `recursive`
            If not supplied or false, the node will be removed only if
            it has no children; if it does, a `NodeError` will be
            raised.  If supplied with a true value, the node and all its
            descendants will be completely removed.
        """
        obj = self.getNode(where, name=name)
        obj._f_remove(recursive)


    def getNodeAttr(self, where, attrname, name=None):
        """
        Get a PyTables attribute from the given node.

        `where`, `name`
            These arguments work as in `File.getNode()`, referencing the
            node to be acted upon.

        `attrname`
            The name of the attribute to retrieve.  If the named
            attribute does not exist, an `AttributeError` is raised.
        """
        obj = self.getNode(where, name=name)
        return obj._f_getAttr(attrname)


    def setNodeAttr(self, where, attrname, attrvalue, name=None):
        """
        Set a PyTables attribute for the given node.

        `where`, `name`
            These arguments work as in `File.getNode()`, referencing the
            node to be acted upon.

        `attrname`
            The name of the attribute to set.

        `attrvalue`
            The value of the attribute to set.  Any kind of Python
            object (like strings, ints, floats, lists, tuples, dicts,
            small NumPy/Numeric/numarray objects...) can be stored as an
            attribute.  However, if necessary, ``cPickle`` is
            automatically used so as to serialize objects that you might
            want to save.  See the `AttributeSet` class for details.

        If the node already has a large number of attributes, a
        `PerformanceWarning` is issued.

        The `where` and `name` arguments work as in `getNode()`,
        referencing the node to be acted upon.  The other arguments work
        as in `Node._f_setAttr()`.
        """
        obj = self.getNode(where, name=name)
        obj._f_setAttr(attrname, attrvalue)


    def delNodeAttr(self, where, attrname, name=None):
        """
        Delete a PyTables attribute from the given node.

        `where`, `name`
            These arguments work as in `File.getNode()`, referencing the
            node to be acted upon.

        `attrname`
            The name of the attribute to delete.  If the named attribute
            does not exist, an `AttributeError` is raised.
        """
        obj = self.getNode(where, name=name)
        obj._f_delAttr(attrname)


    def copyNodeAttrs(self, where, dstnode, name=None):
        """
        Copy PyTables attributes from one node to another.

        `where`, `name`
            These arguments work as in `File.getNode()`, referencing the
            node to be acted upon.

        `dstnode`
            The destination node where the attributes will be copied to.
            It can be a path string or a `Node` instance.
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
        In this last case, the destination file should be closed or ugly
        errors will happen.

        Additional keyword arguments may be passed to customize the
        copying process.  For instance, title and filters may be
        changed, user attributes may be or may not be copied, data may
        be subsampled, stats may be collected, etc.  Arguments unknown
        to nodes are simply ignored.  Check the documentation for
        copying operations of nodes to see which options they support.

        In addition, it recognizes the names of parameters present in
        ``tables/parameters.py`` as additional keyword arguments.  Check the
        suitable appendix in User's Guide for a detailed info on the supported
        parameters.

        Copying a file usually has the beneficial side effect of
        creating a more compact and cleaner version of the original
        file.
        """

        self._checkOpen()

        # Check that we are not treading our own shoes
        if os.path.abspath(self.filename) == os.path.abspath(dstfilename):
            raise IOError("You cannot copy a file over itself")

        # Compute default arguments.
        # These are *not* passed on.
        filters = kwargs.pop('filters', None)
        if filters is None:
            # By checking the HDF5 attribute, we avoid setting filters
            # in the destination file if not explicitly set in the
            # source file.  Just by assigning ``self.filters`` we would
            # not be able to tell.
            filters = getattr(self.root._v_attrs, 'FILTERS', None)
        copyuserattrs = kwargs.get('copyuserattrs', True)
        title = kwargs.pop('title', self.title)

        if os.path.isfile(dstfilename) and not overwrite:
            raise IOError("""\
file ``%s`` already exists; \
you may want to use the ``overwrite`` argument""" % dstfilename)

        # Create destination file, overwriting it.
        dstFileh = openFile(
            dstfilename, mode="w", title=title, filters=filters, **kwargs)

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
        Return a *list* with children nodes hanging from `where`.

        This is a list-returning version of `File.iterNodes()`.
        """

        group = self.getNode(where)  # Does the parent exist?
        self._checkGroup(group)  # Is it a group?

        return group._f_listNodes(classname)


    def iterNodes(self, where, classname=None):
        """
        Iterate over children nodes hanging from `where`.

        `where`
            This argument works as in `File.getNode()`, referencing the
            group to be acted upon.

        `classname`
            If the name of a class derived from `Node` is supplied, only
            instances of that class (or subclasses of it) will be
            returned.

        The returned nodes are alphanumerically sorted by their name.
        This is an iterator version of `File.listNodes()`.
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
        """
        Recursively iterate over the nodes in the tree.

        This is equivalent to calling `File.walkNodes()` with no
        arguments.

        Example of use::

            # Recursively list all the nodes in the object tree.
            h5file = tables.openFile('vlarray1.h5')
            print \"All nodes in the object tree:\"
            for node in h5file:
                print node
        """

        return self.walkNodes('/')


    def walkNodes(self, where="/", classname=None):
        """
        Recursively iterate over nodes hanging from `where`.

        `where`
            If supplied, the iteration starts from (and includes) this
            group.  It can be a path string or a `Group` instance.

        `classname`
            If the name of a class derived from `Node` is supplied, only
            instances of that class (or subclasses of it) will be
            returned.

        Example of use::

            # Recursively print all the nodes hanging from '/detector'.
            print \"Nodes hanging from group '/detector':\"
            for node in h5file.walkNodes('/detector', classname='EArray'):
                print node


        Iterate over the nodes in the object tree.
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
        """
        Recursively iterate over groups (not leaves) hanging from
        `where`.

        The `where` group itself is listed first (preorder), then each
        of its child groups (following an alphanumerical order) is also
        traversed, following the same procedure.  If `where` is not
        supplied, the root group is used.

        The `where` argument can be a path string or a `Group` instance.
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
        Undo/Redo support enabled.
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
        modifications in the node hierarchy.  This allows `File.mark()`,
        `File.undo()`, `File.redo()` and other methods to be called.

        The `filters` argument, when specified, must be an instance of
        class `Filters` and is meant for setting the compression values
        for the action log.  The default is having compression enabled,
        as the gains in terms of space can be considerable.  You may
        want to disable compression if you want maximum speed for
        Undo/Redo operations.

        Calling this method when the Undo/Redo mechanism is already
        enabled raises an `UndoRedoError`.
        """

        maxUndo = self.params['MAX_UNDO_PATH_LENGTH']
        class ActionLog(NotLoggedMixin, Table):
            pass

        class ActionLogDesc(IsDescription):
            opcode = UInt8Col(pos=0)
            arg1   = StringCol(maxUndo, pos=1, dflt="")
            arg2   = StringCol(maxUndo, pos=2, dflt="")

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
            # F. Alted 2005-09-21
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
        makes `File.mark()`, `File.undo()`, `File.redo()` and other
        methods fail with an `UndoRedoError`.

        Calling this method when the Undo/Redo mechanism is already
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
        the identifier of a mark and its name can be used in
        `File.undo()` and `File.redo()` operations.  When the `name` has
        already been used for another mark, an `UndoRedoError` is
        raised.

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

        maxUndo = self.params['MAX_UNDO_PATH_LENGTH']
        # Check whether we are at the end of the action log or not
        if self._curaction != self._actionlog.nrows - 1:
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
        if (len(arg1) > maxUndo
            or len(arg2) > maxUndo):  #INTERNAL
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
            if actionlog['opcode'][i] != _opToCode["MARK"]:
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
        `File.enableUndo()`.

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
        """Flush all the alive leaves in the object tree."""

        self._checkOpen()

        # First, flush PyTables buffers on alive leaves.
        # Leaves that are dead should have been flushed already (at least,
        # users are directed to do this through a PerformanceWarning!)
        for path, refnode in self._aliveNodes.iteritems():
            if '/_i_' not in path:  # Indexes are not necessary to be flushed
                if (self._aliveNodes.hassoftlinks):
                    node = refnode()
                else:
                    node = refnode
                if isinstance(node, Leaf):
                    node.flush()

        # Flush the cache to disk
        self._flushFile(0)  # 0 means local scope, 1 global (virtual) scope


    def close(self):
        """Flush all the alive leaves in object tree and close the file."""

        # If the file is already closed, return immediately
        if not self.isopen:
            return

        # If this file has been opened more than once, decrease the
        # counter and return
        if self._open_count > 1:
            self._open_count -= 1
            return

        filename = self.filename

        if self._undoEnabled and self._isWritable():
            # Save the current mark and current action
            self._actionlog.attrs._g__setattr("CURMARK", self._curmark)
            self._actionlog.attrs._g__setattr("CURACTION", self._curaction)

        # Close all loaded nodes.
        self.root._f_close()

        # Post-conditions
        assert len(self._deadNodes) == 0, \
               ("dead nodes remain after closing dead nodes: %s"
                % [path for path in self._deadNodes])

        # No other nodes should have been revived.
        assert len(self._aliveNodes) == 0, \
               ("alive nodes remain after closing dead nodes: %s"
                % [path for path in self._aliveNodes])

        # Close the file
        self._closeFile()
        # After the objects are disconnected, destroy the
        # object dictionary using the brute force ;-)
        # This should help to the garbage collector
        self.__dict__.clear()
        # Set the flag to indicate that the file is closed
        self.isopen = 0
        # Delete the entry in the dictionary of opened files
        del _open_files[filename]


    def __enter__(self):
        """Enter a context and return the same file."""
        return self


    def __exit__(self, *exc_info):
        """Exit a context and close the file."""
        self.close()
        return False  # do not hide exceptions


    def __str__(self):
        """
        Return a short string representation of the object tree.

        >>> f = tables.openFile('data/test.h5')
        >>> print f
        data/test.h5 (File) 'Table Benchmark'
        Last modif.: 'Mon Sep 20 12:40:47 2004'
        Object Tree:
        / (Group) 'Table Benchmark'
        /tuple0 (Table(100,)) 'This is the table title'
        /group0 (Group) ''
        /group0/tuple1 (Table(100,)) 'This is the table title'
        /group0/group1 (Group) ''
        /group0/group1/tuple2 (Table(100,)) 'This is the table title'
        /group0/group1/group2 (Group) ''
        """

        if not self.isopen:
            return "<closed File>"

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
            for kind in self._node_kinds[1:]:
                for node in self.listNodes(group, kind):
                    astring += str(node) + '\n'
        return astring


    def __repr__(self):
        """Return a detailed string representation of the object tree."""

        if not self.isopen:
            return "<closed File>"

        # Print all the nodes (Group and Leaf objects) on object tree
        astring = 'File(filename=' + str(self.filename) + \
                  ', title=' + repr(self.title) + \
                  ', mode=' + repr(self.mode) + \
                  ', rootUEP=' + repr(self.rootUEP) + \
                  ', filters=' + repr(self.filters) + \
                  ')\n'
        for group in self.walkGroups("/"):
            astring += str(group) + '\n'
            for kind in self._node_kinds[1:]:
                for node in self.listNodes(group, kind):
                    astring += repr(node) + '\n'
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
        if self._aliveNodes.hasdeadnodes:
            self._deadNodes[nodePath] = node
        else:
            # We have not a cache for dead nodes,
            # so follow the usual deletion procedure.
            node._v__deleting = True
            node._f_close()


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


    def _updateNodeLocations(self, oldPath, newPath):
        """
        Update location information of nodes under `oldPath`.

        This only affects *already loaded* nodes.
        """
        oldPrefix = oldPath + '/'  # root node can not be renamed, anyway
        oldPrefixLen = len(oldPrefix)

        # Update alive and dead descendents.
        for cache in [self._aliveNodes, self._deadNodes]:
            for nodePath in cache:
                if nodePath.startswith(oldPrefix):
                    nodeSuffix = nodePath[oldPrefixLen:]
                    newNodePath = joinPath(newPath, nodeSuffix)
                    newNodePPath = splitPath(newNodePath)[0]
                    descendentNode = self._getNode(nodePath)
                    descendentNode._g_updateLocation(newNodePPath)


# If a user hits ^C during a run, it is wise to gracefully close the opened files.
def close_open_files():
    are_open_files = len(_open_files) > 0
    if are_open_files:
        print >> sys.stderr, "Closing remaining open files:",
    for fname, fileh in _open_files.items():
        print >> sys.stderr, "%s..." % (fname,),
        fileh.close()
        print >> sys.stderr, "done",
    if are_open_files:
        print >> sys.stderr

import atexit
atexit.register(close_open_files)


## Local Variables:
## mode: python
## py-indent-offset: 4
## tab-width: 4
## fill-column: 72
## End:
