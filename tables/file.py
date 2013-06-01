# -*- coding: utf-8 -*-

########################################################################
#
# License: BSD
# Created: September 4, 2002
# Author: Francesc Alted - faltet@pytables.com
#
# $Id$
#
########################################################################

"""Create PyTables files and the object tree.

This module support importing generic HDF5 files, on top of which
PyTables files are created, read or extended. If a file exists, an
object tree mirroring their hierarchical structure is created in
memory. File class offer methods to traverse the tree, as well as to
create new nodes.
"""

import os
import sys
import time
import weakref
import warnings

import numexpr
import numpy

import tables.misc.proxydict
from tables import hdf5extension
from tables import utilsextension
from tables import parameters
from tables.exceptions import (ClosedFileError, FileModeError, NodeError,
                               NoSuchNodeError, UndoRedoError,
                               PerformanceWarning)
from tables.registry import get_class_by_name
from tables.path import join_path, split_path
from tables import undoredo
from tables.description import (IsDescription, UInt8Col, StringCol,
                                descr_from_dtype, dtype_from_descr)
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
from tables import linkextension
from tables.utils import detect_number_of_cores
from tables import lrucacheextension
from tables.flavor import flavor_of, array_as_internal
from tables.atom import Atom

from tables.link import SoftLink, ExternalLink

from tables._past import previous_api, previous_api_property


# format_version = "1.0"  # Initial format
# format_version = "1.1"  # Changes in ucl compression
# format_version = "1.2"  # Support for enlargeable arrays and VLA's
#                         # 1.2 was introduced in PyTables 0.8
# format_version = "1.3"  # Support for indexes in Tables
#                         # 1.3 was introduced in PyTables 0.9
# format_version = "1.4"  # Support for multidimensional attributes
#                         # 1.4 was introduced in PyTables 1.1
# format_version = "1.5"  # Support for persistent defaults in tables
#                         # 1.5 was introduced in PyTables 1.2
# format_version = "1.6"  # Support for NumPy objects and new flavors for
#                         # objects.
#                         # 1.6 was introduced in pytables 1.3
#format_version = "2.0"  # Pickles are not used anymore in system attrs
#                        # 2.0 was introduced in PyTables 2.0
format_version = "2.1"  # Numeric and numarray flavors are gone.

compatible_formats = []  # Old format versions we can read
                         # Empty means that we support all the old formats

# Dict of opened files (keys are filenames and values filehandlers)
_open_files = {}

# Opcodes for do-undo actions
_op_to_code = {
    "MARK": 0,
    "CREATE": 1,
    "REMOVE": 2,
    "MOVE": 3,
    "ADDATTR": 4,
    "DELATTR": 5,
}

_code_to_op = ["MARK", "CREATE", "REMOVE", "MOVE", "ADDATTR", "DELATTR"]


# Paths and names for hidden nodes related with transactions.
_trans_version = '1.0'

_trans_group_parent = '/'
_trans_group_name = '_p_transactions'
_trans_group_path = join_path(_trans_group_parent, _trans_group_name)

_action_log_parent = _trans_group_path
_action_log_name = 'actionlog'
_action_log_path = join_path(_action_log_parent, _action_log_name)

_trans_parent = _trans_group_path
_trans_name = 't%d'  # %d -> transaction number
_trans_path = join_path(_trans_parent, _trans_name)

_markParent = _trans_path
_markName = 'm%d'  # %d -> mark number
_markPath = join_path(_markParent, _markName)

_shadow_parent = _markPath
_shadow_name = 'a%d'  # %d -> action number
_shadow_path = join_path(_shadow_parent, _shadow_name)


def _checkfilters(filters):
    if not (filters is None or
            isinstance(filters, Filters)):
        raise TypeError("filter parameter has to be None or a Filter "
                        "instance and the passed type is: '%s'" %
                        type(filters))


def copy_file(srcfilename, dstfilename, overwrite=False, **kwargs):
    """An easy way of copying one PyTables file to another.

    This function allows you to copy an existing PyTables file named
    srcfilename to another file called dstfilename. The source file
    must exist and be readable. The destination file can be
    overwritten in place if existing by asserting the overwrite
    argument.

    This function is a shorthand for the :meth:`File.copy_file` method,
    which acts on an already opened file. kwargs takes keyword
    arguments used to customize the copying process. See the
    documentation of :meth:`File.copy_file` for a description of those
    arguments.

    """

    # Open the source file.
    srcFileh = open_file(srcfilename, mode="r")

    try:
        # Copy it to the destination file.
        srcFileh.copy_file(dstfilename, overwrite=overwrite, **kwargs)
    finally:
        # Close the source file.
        srcFileh.close()

copyFile = previous_api(copy_file)


def open_file(filename, mode="r", title="", root_uep="/", filters=None,
              **kwargs):
    """Open a PyTables (or generic HDF5) file and return a File object.

    Parameters
    ----------
    filename : str
        The name of the file (supports environment variable expansion).
        It is suggested that file names have any of the .h5, .hdf or
        .hdf5 extensions, although this is not mandatory.
    mode : str
        The mode to open the file. It can be one of the
        following:

            * *'r'*: Read-only; no data can be modified.
            * *'w'*: Write; a new file is created (an existing file
              with the same name would be deleted).
            * *'a'*: Append; an existing file is opened for reading and
              writing, and if the file does not exist it is created.
            * *'r+'*: It is similar to 'a', but the file must already
              exist.

    title : str
        If the file is to be created, a TITLE string attribute will be
        set on the root group with the given value. Otherwise, the
        title will be read from disk, and this will not have any effect.
    root_uep : str
        The root User Entry Point. This is a group in the HDF5 hierarchy
        which will be taken as the starting point to create the object
        tree. It can be whatever existing group in the file, named by
        its HDF5 path. If it does not exist, an HDF5ExtError is issued.
        Use this if you do not want to build the *entire* object tree,
        but rather only a *subtree* of it.

        .. versionchanged:: 3.0
           The *rootUEP* parameter has been renamed into *root_uep*.

    filters : Filters
        An instance of the Filters (see :ref:`FiltersClassDescr`) class
        that provides information about the desired I/O filters
        applicable to the leaves that hang directly from the *root group*,
        unless other filter properties are specified for these leaves.
        Besides, if you do not specify filter properties for child groups,
        they will inherit these ones, which will in turn propagate to
        child nodes.

    Notes
    -----
    In addition, it recognizes the (lowercase) names of parameters
    present in :file:`tables/parameters.py` as additional keyword
    arguments.
    See :ref:`parameter_files` for a detailed info on the supported
    parameters.

    .. note::

        If you need to deal with a large number of nodes in an
        efficient way, please see :ref:`LRUOptim` for more info and
        advices about the integrated node cache engine.

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
    return File(filename, mode, title, root_uep, filters, **kwargs)

openFile = previous_api(open_file)


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
                warnings.warn("the dictionary of alive nodes is exceeding "
                              "the recommended maximum number (%d); "
                              "be ready to see PyTables asking for *lots* "
                              "of memory and possibly slow I/O." % (
                              -self.nodeCacheSlots), PerformanceWarning)
        super(_AliveNodes, self).__setitem__(key, ref)


class _DeadNodes(lrucacheextension.NodeCache):
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
    """A proxy dictionary which is able to delegate access to missing items
    to the container object (a `File`)."""

    def _get_value_from_container(self, container, key):
        return container.get_node(key)

    _getValueFromContainer = previous_api(_get_value_from_container)

    def _condition(self, node):
        """Nodes fulfilling the condition are considered to belong here."""
        raise NotImplementedError


    # def __len__(self):
    #    return len(list(self.iterkeys()))


class File(hdf5extension.File, object):
    """The in-memory representation of a PyTables file.

    An instance of this class is returned when a PyTables file is
    opened with the :func`tables.open_file` function. It offers methods
    to manipulate (create, rename, delete...) nodes and handle their
    attributes, as well as methods to traverse the object tree.
    The *user entry point* to the object tree attached to the HDF5 file
    is represented in the root_uep attribute.
    Other attributes are available.

    File objects support an *Undo/Redo mechanism* which can be enabled
    with the :meth:`File.enable_undo` method. Once the Undo/Redo
    mechanism is enabled, explicit *marks* (with an optional unique
    name) can be set on the state of the database using the
    :meth:`File.mark`
    method. There are two implicit marks which are always available:
    the initial mark (0) and the final mark (-1).  Both the identifier
    of a mark and its name can be used in *undo* and *redo* operations.

    Hierarchy manipulation operations (node creation, movement and
    removal) and attribute handling operations (setting and deleting)
    made after a mark can be undone by using the :meth:`File.undo`
    method, which returns the database to the state of a past mark.
    If undo() is not followed by operations that modify the hierarchy
    or attributes, the :meth:`File.redo` method can be used to return
    the database to the state of a future mark. Else, future states of
    the database are forgotten.

    Note that data handling operations can not be undone nor redone by
    now. Also, hierarchy manipulation operations on nodes that do not
    support the Undo/Redo mechanism issue an UndoRedoWarning *before*
    changing the database.

    The Undo/Redo mechanism is persistent between sessions and can
    only be disabled by calling the :meth:`File.disable_undo` method.

    File objects can also act as context managers when using the with
    statement introduced in Python 2.5.  When exiting a context, the
    file is automatically closed.

    Parameters
    ----------
    filename : str
        The name of the file (supports environment variable expansion).
        It is suggested that file names have any of the .h5, .hdf or
        .hdf5 extensions, although this is not mandatory.

    mode : str
        The mode to open the file. It can be one of the
        following:

            * *'r'*: Read-only; no data can be modified.
            * *'w'*: Write; a new file is created (an existing file
              with the same name would be deleted).
            * *'a'*: Append; an existing file is opened for reading
              and writing, and if the file does not exist it is created.
            * *'r+'*: It is similar to 'a', but the file must already
              exist.

    title : str
        If the file is to be created, a TITLE string attribute will be
        set on the root group with the given value. Otherwise, the
        title will be read from disk, and this will not have any effect.

    root_uep : str
        The root User Entry Point. This is a group in the HDF5 hierarchy
        which will be taken as the starting point to create the object
        tree. It can be whatever existing group in the file, named by
        its HDF5 path. If it does not exist, an HDF5ExtError is issued.
        Use this if you do not want to build the *entire* object tree,
        but rather only a *subtree* of it.

        .. versionchanged:: 3.0
           The *rootUEP* parameter has been renamed into *root_uep*.

    filters : Filters
        An instance of the Filters (see :ref:`FiltersClassDescr`) class that
        provides information about the desired I/O filters applicable to the
        leaves that hang directly from the *root group*, unless other filter
        properties are specified for these leaves. Besides, if you do not
        specify filter properties for child groups, they will inherit these
        ones, which will in turn propagate to child nodes.

    Notes
    -----
    In addition, it recognizes the (lowercase) names of parameters
    present in :file:`tables/parameters.py` as additional keyword
    arguments.
    See :ref:`parameter_files` for a detailed info on the supported
    parameters.


    .. rubric:: File attributes

    .. attribute:: filename

        The name of the opened file.

    .. attribute:: format_version

        The PyTables version number of this file.

    .. attribute:: isopen

        True if the underlying file is open, false otherwise.

    .. attribute:: mode

        The mode in which the file was opened.

    .. attribute:: root

        The *root* of the object tree hierarchy (a Group instance).

    .. attribute:: root_uep

        The UEP (user entry point) group name in the file (see
        the :func:`open_file` function).

        .. versionchanged:: 3.0
           The *rootUEP* attribute has been renamed into *root_uep*.

    """

    ## <class variables>
    # The top level kinds. Group must go first!
    _node_kinds = ('Group', 'Leaf', 'Link', 'Unknown')
    rootUEP = previous_api_property('root_uep')
    _v_objectId = previous_api_property('_v_objectid')

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
        ("Default filter properties for the root group "
         "(see :ref:`FiltersClassDescr`)."))

    open_count = property(
        lambda self: self._open_count, None, None,
        "The number of times this file has been opened currently.")

    ## </properties>

    def __init__(self, filename, mode="r", title="",
                 root_uep="/", filters=None, **kwargs):

        self.filename = filename
        """The name of the opened file."""

        self.mode = mode
        """The mode in which the file was opened."""

        # Get all the parameters in parameter file(s)
        params = dict([(k, v) for k, v in parameters.__dict__.iteritems()
                       if k.isupper() and not k.startswith('_')])
        # Update them with possible keyword arguments
        if [k for k in kwargs if k.isupper()]:
            warnings.warn("The use of uppercase keyword parameters is "
                          "deprecated", DeprecationWarning)

        kwargs = dict([(k.upper(), v) for k, v in kwargs.iteritems()])
        params.update(kwargs)

        # If MAX_ * _THREADS is not set yet, set it to the number of cores
        # on this machine.

        if params['MAX_NUMEXPR_THREADS'] is None:
            params['MAX_NUMEXPR_THREADS'] = detect_number_of_cores()

        if params['MAX_BLOSC_THREADS'] is None:
            params['MAX_BLOSC_THREADS'] = detect_number_of_cores()

        self.params = params

        # Now, it is time to initialize the File extension
        self._g_new(filename, mode, **params)

        # Check filters and set PyTables format version for new files.
        new = self._v_new
        if new:
            _checkfilters(filters)
            self.format_version = format_version
            """The PyTables version number of this file."""

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
        """True if the underlying file os open, False otherwise."""

        # Append the name of the file to the global dict of files opened.
        _open_files[self.filename] = self

        # Set the number of times this file has been opened to 1
        self._open_count = 1

        # Get the root group from this file
        self.root = root = self.__get_root_group(root_uep, title, filters)
        """The *root* of the object tree hierarchy (a Group instance)."""
        # Complete the creation of the root node
        # (see the explanation in ``RootGroup.__init__()``.
        root._g_post_init_hook()

        # Save the PyTables format version for this file.
        if new:
            if params['PYTABLES_SYS_ATTRS']:
                root._v_attrs._g__setattr(
                    'PYTABLES_FORMAT_VERSION', format_version)

        # If the file is old, and not opened in "read-only" mode,
        # check if it has a transaction log
        if not new and self.mode != "r" and _trans_group_path in self:
            # It does. Enable the undo.
            self.enable_undo()

        # Set the maximum number of threads for Numexpr
        numexpr.set_vml_num_threads(params['MAX_NUMEXPR_THREADS'])

    def __get_root_group(self, root_uep, title, filters):
        """Returns a Group instance which will act as the root group
        in the hierarchical tree. If file is opened in "r", "r+" or
        "a" mode, and the file already exists, this method dynamically
        builds a python object tree emulating the structure present on
        file."""

        self._v_objectid = self._get_file_id()

        if root_uep in [None, ""]:
            root_uep = "/"
        # Save the User Entry Point in a variable class
        self.root_uep = root_uep

        new = self._v_new

        # Get format version *before* getting the object tree
        if not new:
            # Firstly, get the PyTables format version for this file
            self.format_version = utilsextension.read_f_attr(
                self._v_objectid, 'PYTABLES_FORMAT_VERSION')
            if not self.format_version:
                # PYTABLES_FORMAT_VERSION attribute is not present
                self.format_version = "unknown"
                self._isPTFile = False
            elif not isinstance(self.format_version, str):
                # system attributes should always be str
                if sys.version_info[0] < 3:
                    self.format_version = self.format_version.encode()
                else:
                    self.format_version = self.format_version.decode('utf-8')

        # Create new attributes for the root Group instance and
        # create the object tree
        return RootGroup(self, root_uep, title=title, new=new, filters=filters)

    __getRootGroup = previous_api(__get_root_group)

    def _get_or_create_path(self, path, create):
        """Get the given `path` or create it if `create` is true.

        If `create` is true, `path` *must* be a string path and not a
        node, otherwise a `TypeError`will be raised.

        """

        if create:
            return self._create_path(path)
        else:
            return self.get_node(path)

    _getOrCreatePath = previous_api(_get_or_create_path)

    def _create_path(self, path):
        """Create the groups needed for the `path` to exist.

        The group associated with the given `path` is returned.

        """

        if not hasattr(path, 'split'):
            raise TypeError("when creating parents, parent must be a path")

        if path == '/':
            return self.root

        parent, create_group = self.root, self.create_group
        for pcomp in path.split('/')[1:]:
            try:
                child = parent._f_get_child(pcomp)
            except NoSuchNodeError:
                child = create_group(parent, pcomp)
            parent = child
        return parent

    _createPath = previous_api(_create_path)

    def create_group(self, where, name, title="", filters=None,
                     createparents=False):
        """Create a new group.

        Parameters
        ----------
        where : str or Group
            The parent group from which the new group will hang. It can be a
            path string (for example '/level1/leaf5'), or a Group instance
            (see :ref:`GroupClassDescr`).
        name : str
            The name of the new group.
        title : str, optional
            A description for this node (it sets the TITLE HDF5 attribute on
            disk).
        filters : Filters
            An instance of the Filters class (see :ref:`FiltersClassDescr`)
            that provides information about the desired I/O filters applicable
            to the leaves that hang directly from this new group (unless other
            filter properties are specified for these leaves). Besides, if you
            do not specify filter properties for its child groups, they will
            inherit these ones.
        createparents : bool
            Whether to create the needed groups for the parent
            path to exist (not done by default).

        See Also
        --------
        Group : for more information on groups

        """

        parentnode = self._get_or_create_path(where, createparents)
        _checkfilters(filters)
        return Group(parentnode, name,
                     title=title, new=True, filters=filters)

    createGroup = previous_api(create_group)

    def create_table(self, where, name, description=None, title="",
                     filters=None, expectedrows=10000,
                     chunkshape=None, byteorder=None,
                     createparents=False, obj=None):
        """Create a new table with the given name in where location.

        Parameters
        ----------
        where : str or Group
            The parent group from which the new table will hang. It can be a
            path string (for example '/level1/leaf5'), or a Group instance
            (see :ref:`GroupClassDescr`).
        name : str
            The name of the new table.
        description : Description
            This is an object that describes the table, i.e. how
            many columns it has, their names, types, shapes, etc.  It
            can be any of the following:

                * *A user-defined class*: This should inherit from the
                  IsDescription class (see :ref:`IsDescriptionClassDescr`)
                  where table fields are specified.
                * *A dictionary*: For example, when you do not know
                  beforehand which structure your table will have).
                * *A Description instance*: You can use the description
                  attribute of another table to create a new one with the
                  same structure.
                * *A NumPy dtype*: A completely general structured NumPy
                  dtype.
                * *A NumPy (structured) array instance*: The dtype of
                  this structured array will be used as the description.
                  Also, in case the array has actual data, it will be
                  injected into the newly created table.

            .. versionchanged:: 3.0
               The *description* parameter can be None (default) if *obj* is
               provided.  In that case the structure of the table is deduced
               by *obj*.

        title : str
            A description for this node (it sets the TITLE HDF5 attribute
            on disk).
        filters : Filters
            An instance of the Filters class (see :ref:`FiltersClassDescr`)
            that provides information about the desired I/O filters to be
            applied during the life of this object.
        expectedrows : int
            A user estimate of the number of records that will be in the table.
            If not provided, the default value is EXPECTED_ROWS_TABLE (see
            :file:`tables/parameters.py`). If you plan to create a bigger
            table try providing a guess; this will optimize the HDF5 B-Tree
            creation and management process time and memory used.
        chunkshape
            The shape of the data chunk to be read or written in a
            single HDF5 I/O operation. Filters are applied to those
            chunks of data. The rank of the chunkshape for tables must
            be 1. If None, a sensible value is calculated based on the
            expectedrows parameter (which is recommended).
        byteorder : str
            The byteorder of data *on disk*, specified as 'little' or 'big'.
            If this is not specified, the byteorder is that of the platform,
            unless you passed an array as the description, in which case
            its byteorder will be used.
        createparents : bool
            Whether to create the needed groups for the parent path to exist
            (not done by default).
        obj : python object
            The recarray to be saved.  Accepted types are NumPy record
            arrays, as well as native Python sequences convertible to numpy
            record arrays.

            The *obj* parameter is optional and it can be provided in
            alternative to the *description* parameter.
            If both *obj* and *description* are provided they must
            be consistent with each other.

            .. versionadded:: 3.0

        See Also
        --------
        Table : for more information on tables

        """

        if obj is not None:
            if not isinstance(obj, numpy.ndarray):
                raise TypeError('invalid obj parameter %r' % obj)

            descr, _ = descr_from_dtype(obj.dtype)
            if (description is not None and
                    dtype_from_descr(description) != obj.dtype):
                raise TypeError('the desctiption parameter is not consistent '
                                'with the data type of the obj parameter')
            elif description is None:
                description = descr

        parentnode = self._get_or_create_path(where, createparents)
        if description is None:
            raise ValueError("invalid table description: None")
        _checkfilters(filters)

        ptobj = Table(parentnode, name,
                      description=description, title=title,
                      filters=filters, expectedrows=expectedrows,
                      chunkshape=chunkshape, byteorder=byteorder)

        if obj is not None:
            ptobj.append(obj)

        return ptobj

    createTable = previous_api(create_table)

    def create_array(self, where, name, obj=None, title="",
                     byteorder=None, createparents=False,
                     atom=None, shape=None):
        """Create a new array.

        Parameters
        ----------
        where : str or Group
            The parent group from which the new array will hang. It can be a
            path string (for example '/level1/leaf5'), or a Group instance
            (see :ref:`GroupClassDescr`).
        name : str
            The name of the new array
        obj : python object
            The array or scalar to be saved.  Accepted types are NumPy
            arrays and scalars, as well as native Python sequences and
            scalars, provided that values are regular (i.e. they are
            not like ``[[1,2],2]``) and homogeneous (i.e. all the
            elements are of the same type).

            Also, objects that have some of their dimensions equal to 0
            are not supported (use an EArray node (see
            :ref:`EArrayClassDescr`) if you want to store an array with
            one of its dimensions equal to 0).

            .. versionchanged:: 3.0
               The *Object parameter has been renamed into *obj*.*

        title : str
            A description for this node (it sets the TITLE HDF5 attribute on
            disk).
        byteorder : str
            The byteorder of the data *on disk*, specified as 'little' or
            'big'.  If this is not specified, the byteorder is that of the
            given object.
        createparents : bool, optional
            Whether to create the needed groups for the parent path to exist
            (not done by default).
        atom : Atom
            An Atom (see :ref:`AtomClassDescr`) instance representing
            the *type* and *shape* of the atomic objects to be saved.

            .. versionadded:: 3.0

        shape : tuple of ints
            The shape of the stored array.

            .. versionadded:: 3.0

        See Also
        --------
        Array : for more information on arrays
        create_table : for more information on the rest of parameters

        """

        if obj is None:
            if atom is None or shape is None:
                raise TypeError('if the obj parameter is not specified '
                                '(or None) then both the atom and shape '
                                'parametes should be provided.')
            else:
                obj = numpy.zeros(shape, atom.dtype)
        else:
            flavor = flavor_of(obj)
            # use a temporary object because converting obj at this stage
            # breaks some test. This is soultion performs a double,
            # potentially expensive, conversion of the obj parameter.
            _obj = array_as_internal(obj, flavor)

            if shape is not None and shape != _obj.shape:
                raise TypeError('the shape parameter do not match obj.shape')

            if atom is not None and atom.dtype != _obj.dtype:
                raise TypeError('the atom parameter is not consistent with '
                                'the data type of the obj parameter')

        parentnode = self._get_or_create_path(where, createparents)
        return Array(parentnode, name,
                     obj=obj, title=title, byteorder=byteorder)

    createArray = previous_api(create_array)

    def create_carray(self, where, name, atom=None, shape=None, title="",
                      filters=None, chunkshape=None,
                      byteorder=None, createparents=False, obj=None):
        """Create a new chunked array.

        Parameters
        ----------
        where : str or Group
            The parent group from which the new array will hang. It can
            be a path string (for example '/level1/leaf5'), or a Group
            instance (see :ref:`GroupClassDescr`).
        name : str
            The name of the new array
        atom : Atom
            An Atom (see :ref:`AtomClassDescr`) instance representing
            the *type* and *shape* of the atomic objects to be saved.

            .. versionchanged:: 3.0
               The *atom* parameter can be None (default) if *obj* is
               provided.

        shape : tuple
            The shape of the new array.

            .. versionchanged:: 3.0
               The *shape* parameter can be None (default) if *obj* is
               provided.

        title : str, optional
            A description for this node (it sets the TITLE HDF5 attribute
            on disk).
        filters : Filters, optional
            An instance of the Filters class (see :ref:`FiltersClassDescr`)
            that provides information about the desired I/O filters to
            be applied during the life of this object.
        chunkshape : tuple or number or None, optional
            The shape of the data chunk to be read or written in a
            single HDF5 I/O operation.  Filters are applied to those
            chunks of data.  The dimensionality of chunkshape must be
            the same as that of shape.  If None, a sensible value is
            calculated (which is recommended).
        byteorder : str, optional
            The byteorder of the data *on disk*, specified as 'little'
            or 'big'.  If this is not specified, the byteorder is that
            of the given object.
        createparents : bool, optional
            Whether to create the needed groups for the parent path to
            exist (not done by default).
        obj : python object
            The array or scalar to be saved.  Accepted types are NumPy
            arrays and scalars, as well as native Python sequences and
            scalars, provided that values are regular (i.e. they are
            not like ``[[1,2],2]``) and homogeneous (i.e. all the
            elements are of the same type).

            Also, objects that have some of their dimensions equal to 0
            are not supported. Please use an EArray node (see
            :ref:`EArrayClassDescr`) if you want to store an array with
            one of its dimensions equal to 0.

            The *obj* parameter is optional and it can be provided in
            alternative to the *atom* and *shape* parameters.
            If both *obj* and *atom* and/or *shape* are provided they must
            be consistent with each other.

            .. versionadded:: 3.0

        See Also
        --------
        CArray : for more information on chunked arrays

        """

        if obj is not None:
            flavor = flavor_of(obj)
            obj = array_as_internal(obj, flavor)

            if shape is not None and shape != obj.shape:
                raise TypeError('the shape parameter do not match obj.shape')
            else:
                shape = obj.shape

            if atom is not None and atom.dtype != obj.dtype:
                raise TypeError('the atom parameter is not consistent with '
                                'the data type of the obj parameter')
            elif atom is None:
                atom = Atom.from_dtype(obj.dtype)

        parentnode = self._get_or_create_path(where, createparents)
        _checkfilters(filters)
        ptobj = CArray(parentnode, name,
                       atom=atom, shape=shape, title=title, filters=filters,
                       chunkshape=chunkshape, byteorder=byteorder)

        if obj is not None:
            ptobj[...] = obj

        return ptobj

    createCArray = previous_api(create_carray)

    def create_earray(self, where, name, atom=None, shape=None, title="",
                      filters=None, expectedrows=1000,
                      chunkshape=None, byteorder=None,
                      createparents=False, obj=None):
        """Create a new enlargeable array.

        Parameters
        ----------
        where : str or Group
            The parent group from which the new array will hang. It can be a
            path string (for example '/level1/leaf5'), or a Group instance
            (see :ref:`GroupClassDescr`).
        name : str
            The name of the new array
        atom : Atom
            An Atom (see :ref:`AtomClassDescr`) instance representing the
            *type* and *shape* of the atomic objects to be saved.

            .. versionchanged:: 3.0
               The *atom* parameter can be None (default) if *obj* is
               provided.

        shape : tuple
            The shape of the new array.  One (and only one) of the shape
            dimensions *must* be 0.  The dimension being 0 means that the
            resulting EArray object can be extended along it.  Multiple
            enlargeable dimensions are not supported right now.

            .. versionchanged:: 3.0
               The *shape* parameter can be None (default) if *obj* is
               provided.

        title : str, optional
            A description for this node (it sets the TITLE HDF5 attribute on
            disk).
        expectedrows : int, optional
            A user estimate about the number of row elements that will be added
            to the growable dimension in the EArray node.  If not provided, the
            default value is EXPECTED_ROWS_EARRAY (see tables/parameters.py).
            If you plan to create either a much smaller or a much bigger array
            try providing a guess; this will optimize the HDF5 B-Tree creation
            and management process time and the amount of memory used.
        chunkshape : tuple, numeric, or None, optional
            The shape of the data chunk to be read or written in a single HDF5
            I/O operation.  Filters are applied to those chunks of data.  The
            dimensionality of chunkshape must be the same as that of shape
            (beware: no dimension should be 0 this time!).  If None, a sensible
            value is calculated based on the expectedrows parameter (which is
            recommended).
        byteorder : str, optional
            The byteorder of the data *on disk*, specified as 'little' or
            'big'. If this is not specified, the byteorder is that of the
            platform.
        createparents : bool, optional
            Whether to create the needed groups for the parent path to exist
            (not done by default).
        obj : python object
            The array or scalar to be saved.  Accepted types are NumPy
            arrays and scalars, as well as native Python sequences and
            scalars, provided that values are regular (i.e. they are
            not like ``[[1,2],2]``) and homogeneous (i.e. all the
            elements are of the same type).

            The *obj* parameter is optional and it can be provided in
            alternative to the *atom* and *shape* parameters.
            If both *obj* and *atom* and/or *shape* are provided they must
            be consistent with each other.

            .. versionadded:: 3.0

        See Also
        --------
        EArray : for more information on enlargeable arrays

        """

        if obj is not None:
            flavor = flavor_of(obj)
            obj = array_as_internal(obj, flavor)

            earray_shape = (0,) + obj.shape[1:]

            if shape is not None and shape != earray_shape:
                raise TypeError('the shape parameter is not compatible '
                                'with obj.shape.')
            else:
                shape = earray_shape

            if atom is not None and atom.dtype != obj.dtype:
                raise TypeError('the atom parameter is not consistent with '
                                'the data type of the obj parameter')
            elif atom is None:
                atom = Atom.from_dtype(obj.dtype)

        parentnode = self._get_or_create_path(where, createparents)
        _checkfilters(filters)
        ptobj = EArray(parentnode, name,
                       atom=atom, shape=shape, title=title,
                       filters=filters, expectedrows=expectedrows,
                       chunkshape=chunkshape, byteorder=byteorder)

        if obj is not None:
            ptobj.append(obj)

        return ptobj

    createEArray = previous_api(create_earray)

    def create_vlarray(self, where, name, atom=None, title="",
                       filters=None, expectedrows=None,
                       chunkshape=None, byteorder=None,
                       createparents=False, obj=None):
        """Create a new variable-length array.

        Parameters
        ----------
        where : str or Group
            The parent group from which the new array will hang. It can
            be a path string (for example '/level1/leaf5'), or a Group
            instance (see :ref:`GroupClassDescr`).
        name : str
            The name of the new array
        atom : Atom
            An Atom (see :ref:`AtomClassDescr`) instance representing
            the *type* and *shape* of the atomic objects to be saved.

            .. versionchanged:: 3.0
               The *atom* parameter can be None (default) if *obj* is
               provided.

        title : str, optional
            A description for this node (it sets the TITLE HDF5 attribute
            on disk).
        filters : Filters
            An instance of the Filters class (see :ref:`FiltersClassDescr`)
            that provides information about the desired I/O filters to
            be applied during the life of this object.
        expectedrows : int, optional
            A user estimate about the number of row elements that will
            be added to the growable dimension in the `VLArray` node.
            If not provided, the default value is ``EXPECTED_ROWS_VLARRAY``
            (see ``tables/parameters.py``).  If you plan to create either
            a much smaller or a much bigger `VLArray` try providing a guess;
            this will optimize the HDF5 B-Tree creation and management
            process time and the amount of memory used.

            .. versionadded:: 3.0

        chunkshape : int or tuple of int, optional
            The shape of the data chunk to be read or written in a
            single HDF5 I/O operation. Filters are applied to those
            chunks of data. The dimensionality of chunkshape must be 1.
            If None, a sensible value is calculated (which is recommended).
        byteorder : str, optional
            The byteorder of the data *on disk*, specified as 'little' or
            'big'. If this is not specified, the byteorder is that of the
            platform.
        createparents : bool, optional
            Whether to create the needed groups for the parent path to
            exist (not done by default).
        obj : python object
            The array or scalar to be saved.  Accepted types are NumPy
            arrays and scalars, as well as native Python sequences and
            scalars, provided that values are regular (i.e. they are
            not like ``[[1,2],2]``) and homogeneous (i.e. all the
            elements are of the same type).

            The *obj* parameter is optional and it can be provided in
            alternative to the *atom* parameter.
            If both *obj* and *atom* and are provided they must
            be consistent with each other.

            .. versionadded:: 3.0

        See Also
        --------
        VLArray : for more informationon variable-length arrays

        .. versionchanged:: 3.0
           The *expectedsizeinMB* parameter has been replaced by
           *expectedrows*.

        """

        if obj is not None:
            flavor = flavor_of(obj)
            obj = array_as_internal(obj, flavor)

            if atom is not None and atom.dtype != obj.dtype:
                raise TypeError('the atom parameter is not consistent with '
                                'the data type of the obj parameter')
            if atom is None:
                atom = Atom.from_dtype(obj.dtype)
        elif atom is None:
            raise ValueError('atom parameter cannot be None')

        parentnode = self._get_or_create_path(where, createparents)
        _checkfilters(filters)
        ptobj = VLArray(parentnode, name,
                        atom=atom, title=title, filters=filters,
                        expectedrows=expectedrows,
                        chunkshape=chunkshape, byteorder=byteorder)

        if obj is not None:
            ptobj.append(obj)

        return ptobj

    createVLArray = previous_api(create_vlarray)

    def create_hard_link(self, where, name, target, createparents=False):
        """Create a hard link

        Create a hard link to a `target` node with the given `name` in
        `where` location.  `target` can be a node object or a path
        string.  If `createparents` is true, the intermediate groups
        required for reaching `where` are created (the default is not
        doing so).

        The returned node is a regular `Group` or `Leaf` instance.

        """

        targetnode = self.get_node(target)
        parentnode = self._get_or_create_path(where, createparents)
        linkextension._g_create_hard_link(parentnode, name, targetnode)
        # Refresh children names in link's parent node
        parentnode._g_add_children_names()
        # Return the target node
        return self.get_node(parentnode, name)

    createHardLink = previous_api(create_hard_link)

    def create_soft_link(self, where, name, target, createparents=False):
        """
        Create a soft link (aka symbolic link) to a `target` node with
        the given `name` in `where` location.  `target` can be a node
        object or a path string.  If `createparents` is true, the
        intermediate groups required for reaching `where` are created
        (the default is not doing so).

        The returned node is a SoftLink instance.  See the SoftLink class
        (in :ref:`SoftLinkClassDescr`) for more information on soft links.

        """

        if not isinstance(target, str):
            if hasattr(target, '_v_pathname'):   # quacks like a Node
                target = target._v_pathname
            else:
                raise ValueError(
                    "`target` has to be a string or a node object")
        parentnode = self._get_or_create_path(where, createparents)
        slink = SoftLink(parentnode, name, target)
        # Refresh children names in link's parent node
        parentnode._g_add_children_names()
        return slink

    createSoftLink = previous_api(create_soft_link)

    def create_external_link(self, where, name, target, createparents=False):
        """Create an external link.

        Create an external link to a *target* node with the given *name*
        in *where* location.  *target* can be a node object in another
        file or a path string in the form 'file:/path/to/node'.  If
        *createparents* is true, the intermediate groups required for
        reaching *where* are created (the default is not doing so).

        The returned node is an :class:`ExternalLink` instance.

        """

        if not isinstance(target, str):
            if hasattr(target, '_v_pathname'):   # quacks like a Node
                target = target._v_file.filename + ':' + target._v_pathname
            else:
                raise ValueError(
                    "`target` has to be a string or a node object")
        elif target.find(':/') == -1:
            raise ValueError(
                "`target` must expressed as 'file:/path/to/node'")
        parentnode = self._get_or_create_path(where, createparents)
        elink = ExternalLink(parentnode, name, target)
        # Refresh children names in link's parent node
        parentnode._g_add_children_names()
        return elink

    createExternalLink = previous_api(create_external_link)

    # There is another version of _get_node in cython space, but only
    # marginally faster (5% or less, but sometimes slower!) than this one.
    # So I think it is worth to use this one instead (much easier to debug).
    def _get_node(self, nodePath):
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
            node = self._revivenode(nodePath)
            return node

        # The node has not been found in alive or dead nodes.
        # Open it directly from disk.
        node = self.root._g_load_child(nodePath)
        return node

    _getNode = previous_api(_get_node)

    def get_node(self, where, name=None, classname=None):
        """Get the node under where with the given name.

        where can be a Node instance (see :ref:`NodeClassDescr`) or a
        path string leading to a node. If no name is specified, that
        node is returned.

        If a name is specified, this must be a string with the name of
        a node under where.  In this case the where argument can only
        lead to a Group (see :ref:`GroupClassDescr`) instance (else a
        TypeError is raised). The node called name under the group
        where is returned.

        In both cases, if the node to be returned does not exist, a
        NoSuchNodeError is raised. Please note that hidden nodes are
        also considered.

        If the classname argument is specified, it must be the name of
        a class derived from Node. If the node is found but it is not
        an instance of that class, a NoSuchNodeError is also raised.

        """

        self._check_open()

        # For compatibility with old default arguments.
        if name == '':
            name = None

        # Get the parent path (and maybe the node itself).
        if isinstance(where, Node):
            node = where
            node._g_check_open()  # the node object must be open
            nodePath = where._v_pathname
        elif isinstance(where, (basestring, numpy.str_)):
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
            nodePath = join_path(nodePath, name)

        assert node is None or node._v_pathname == nodePath

        # Now we have the definitive node path, let us try to get the node.
        if node is None:
            node = self._get_node(nodePath)

        # Finally, check whether the desired node is an instance
        # of the expected class.
        if classname:
            class_ = get_class_by_name(classname)
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

    getNode = previous_api(get_node)

    def is_visible_node(self, path):
        """Is the node under `path` visible?

        If the node does not exist, a NoSuchNodeError is raised.

        """

        # ``util.isvisiblepath()`` is still recommended for internal use.
        return self.get_node(path)._f_isvisible()

    isVisibleNode = previous_api(is_visible_node)

    def rename_node(self, where, newname, name=None, overwrite=False):
        """Change the name of the node specified by where and name to newname.

        Parameters
        ----------
        where, name
            These arguments work as in
            :meth:`File.get_node`, referencing the node to be acted upon.
        newname : str
            The new name to be assigned to the node (a string).
        overwrite : bool
            Whether to recursively remove a node with the same
            newname if it already exists (not done by default).

        """

        obj = self.get_node(where, name=name)
        obj._f_rename(newname, overwrite)

    renameNode = previous_api(rename_node)

    def move_node(self, where, newparent=None, newname=None, name=None,
                  overwrite=False, createparents=False):
        """Move the node specified by where and name to newparent/newname.

        Parameters
        ----------
        where, name : path
            These arguments work as in
            :meth:`File.get_node`, referencing the node to be acted upon.
        newparent
            The destination group the node will be moved into (a
            path name or a Group instance). If it is
            not specified or None, the current parent
            group is chosen as the new parent.
        newname
            The new name to be assigned to the node in its
            destination (a string). If it is not specified or
            None, the current name is chosen as the
            new name.

        Notes
        -----
        The other arguments work as in :meth:`Node._f_move`.

        """

        obj = self.get_node(where, name=name)
        obj._f_move(newparent, newname, overwrite, createparents)

    moveNode = previous_api(move_node)

    def copy_node(self, where, newparent=None, newname=None, name=None,
                  overwrite=False, recursive=False, createparents=False,
                  **kwargs):
        """Copy the node specified by where and name to newparent/newname.

        Parameters
        ----------
        where : str
            These arguments work as in
            :meth:`File.get_node`, referencing the node to be acted
            upon.
        newparent : str or Group
            The destination group that the node will be copied
            into (a path name or a Group
            instance). If not specified or None, the
            current parent group is chosen as the new parent.
        newname : str
            The name to be assigned to the new copy in its
            destination (a string).  If it is not specified or
            None, the current name is chosen as the
            new name.
        name : str
            These arguments work as in
            :meth:`File.get_node`, referencing the node to be acted
            upon.
        overwrite : bool, optional
            If True, the destination group will be overwritten if it already
            exists.  Defaults to False.
        recursive : bool, optional
            If True, all descendant nodes of srcgroup are recursively copied.
            Defaults to False.
        createparents : bool, optional
            If True, any necessary parents of dstgroup will be created.
            Defaults to False.
        kwargs
           Additional keyword arguments can be used to customize the copying
           process.  See the documentation of :meth:`Group._f_copy`
           for a description of those arguments.

        Returns
        -------
        node : Node
            The newly created copy of the source node (i.e. the destination
            node).  See :meth:`.Node._f_copy` for further details on the
            semantics of copying nodes.

        """

        obj = self.get_node(where, name=name)
        if obj._v_depth == 0 and newparent and not newname:
            npobj = self.get_node(newparent)
            if obj._v_file is not npobj._v_file:
                # Special case for copying file1:/ --> file2:/path
                self.root._f_copy_children(npobj, overwrite=overwrite,
                                           recursive=recursive, **kwargs)
                return npobj
            else:
                raise IOError(
                    "You cannot copy a root group over the same file")
        return obj._f_copy(newparent, newname,
                           overwrite, recursive, createparents, **kwargs)

    copyNode = previous_api(copy_node)

    def remove_node(self, where, name=None, recursive=False):
        """Remove the object node *name* under *where* location.

        Parameters
        ----------
        where, name
            These arguments work as in
            :meth:`File.get_node`, referencing the node to be acted upon.
        recursive : bool
            If not supplied or false, the node will be removed
            only if it has no children; if it does, a
            NodeError will be raised. If supplied
            with a true value, the node and all its descendants will be
            completely removed.

        """

        obj = self.get_node(where, name=name)
        obj._f_remove(recursive)

    removeNode = previous_api(remove_node)

    def get_node_attr(self, where, attrname, name=None):
        """Get a PyTables attribute from the given node.

        Parameters
        ----------
        where, name
            These arguments work as in :meth:`File.get_node`, referencing the
            node to be acted upon.
        attrname
            The name of the attribute to retrieve.  If the named
            attribute does not exist, an AttributeError is raised.

        """

        obj = self.get_node(where, name=name)
        return obj._f_getattr(attrname)

    getNodeAttr = previous_api(get_node_attr)

    def set_node_attr(self, where, attrname, attrvalue, name=None):
        """Set a PyTables attribute for the given node.

        Parameters
        ----------
        where, name
            These arguments work as in
            :meth:`File.get_node`, referencing the node to be acted upon.
        attrname
            The name of the attribute to set.
        attrvalue
            The value of the attribute to set. Any kind of Python
            object (like strings, ints, floats, lists, tuples, dicts,
            small NumPy objects ...) can be stored as an attribute.
            However, if necessary, pickle is automatically used so as
            to serialize objects that you might want to save.
            See the :class:`AttributeSet` class for details.

        Notes
        -----
        If the node already has a large number of attributes, a
        PerformanceWarning is issued.

        """

        obj = self.get_node(where, name=name)
        obj._f_setattr(attrname, attrvalue)

    setNodeAttr = previous_api(set_node_attr)

    def del_node_attr(self, where, attrname, name=None):
        """Delete a PyTables attribute from the given node.

        Parameters
        ----------
        where, name
            These arguments work as in :meth:`File.get_node`, referencing the
            node to be acted upon.
        attrname
            The name of the attribute to delete.  If the named
            attribute does not exist, an AttributeError is raised.

        """

        obj = self.get_node(where, name=name)
        obj._f_delattr(attrname)

    delNodeAttr = previous_api(del_node_attr)

    def copy_node_attrs(self, where, dstnode, name=None):
        """Copy PyTables attributes from one node to another.

        Parameters
        ----------
        where, name
            These arguments work as in :meth:`File.get_node`, referencing the
            node to be acted upon.
        dstnode
            The destination node where the attributes will be copied to. It can
            be a path string or a Node instance (see :ref:`NodeClassDescr`).

        """

        srcobject = self.get_node(where, name=name)
        dstobject = self.get_node(dstnode)
        srcobject._v_attrs._f_copy(dstobject)

    copyNodeAttrs = previous_api(copy_node_attrs)

    def copy_children(self, srcgroup, dstgroup,
                      overwrite=False, recursive=False,
                      createparents=False, **kwargs):
        """Copy the children of a group into another group.

        Parameters
        ----------
        srcgroup : str
            The group to copy from.
        dstgroup : str
            The destination group.
        overwrite : bool, optional
            If True, the destination group will be overwritten if it already
            exists.  Defaults to False.
        recursive : bool, optional
            If True, all descendant nodes of srcgroup are recursively copied.
            Defaults to False.
        createparents : bool, optional
            If True, any necessary parents of dstgroup will be created.
            Defaults to False.
        kwargs : dict
           Additional keyword arguments can be used to customize the copying
           process.  See the documentation of :meth:`Group._f_copy_children`
           for a description of those arguments.

        """

        srcgroup = self.get_node(srcgroup)  # Does the source node exist?
        self._check_group(srcgroup)  # Is it a group?

        srcgroup._f_copy_children(
            dstgroup, overwrite, recursive, createparents, **kwargs)

    copyChildren = previous_api(copy_children)

    def copy_file(self, dstfilename, overwrite=False, **kwargs):
        """Copy the contents of this file to dstfilename.

        Parameters
        ----------
        dstfilename : str
            A path string indicating the name of the destination file. If
            it already exists, the copy will fail with an IOError, unless
            the overwrite argument is true.
        overwrite : bool, optional
            If true, the destination file will be overwritten if it already
            exists.  In this case, the destination file must be closed, or
            errors will occur.  Defaults to False.
        kwargs
            Additional keyword arguments discussed below.

        Notes
        -----
        Additional keyword arguments may be passed to customize the
        copying process. For instance, title and filters may be changed,
        user attributes may be or may not be copied, data may be
        sub-sampled, stats may be collected, etc. Arguments unknown to
        nodes are simply ignored. Check the documentation for copying
        operations of nodes to see which options they support.

        In addition, it recognizes the names of parameters present in
        :file:`tables/parameters.py` as additional keyword arguments.
        See :ref:`parameter_files` for a detailed info on the supported
        parameters.

        Copying a file usually has the beneficial side effect of
        creating a more compact and cleaner version of the original
        file.

        """

        self._check_open()

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
            raise IOError(("file ``%s`` already exists; "
                           "you may want to use the ``overwrite`` "
                           "argument") % dstfilename)

        # Create destination file, overwriting it.
        dstFileh = open_file(
            dstfilename, mode="w", title=title, filters=filters, **kwargs)

        try:
            # Maybe copy the user attributes of the root group.
            if copyuserattrs:
                self.root._v_attrs._f_copy(dstFileh.root)

            # Copy the rest of the hierarchy.
            self.root._f_copy_children(dstFileh.root, recursive=True, **kwargs)
        finally:
            dstFileh.close()

    copyFile = previous_api(copy_file)

    def list_nodes(self, where, classname=None):
        """Return a *list* with children nodes hanging from where.

        This is a list-returning version of :meth:`File.iter_nodes`.

        """

        group = self.get_node(where)  # Does the parent exist?
        self._check_group(group)  # Is it a group?

        return group._f_list_nodes(classname)

    listNodes = previous_api(list_nodes)

    def iter_nodes(self, where, classname=None):
        """Iterate over children nodes hanging from where.

        Parameters
        ----------
        where
            This argument works as in :meth:`File.get_node`, referencing the
            node to be acted upon.
        classname
            If the name of a class derived from
            Node (see :ref:`NodeClassDescr`) is supplied, only instances of
            that class (or subclasses of it) will be returned.

        Notes
        -----
        The returned nodes are alphanumerically sorted by their name.
        This is an iterator version of :meth:`File.list_nodes`.

        """

        group = self.get_node(where)  # Does the parent exist?
        self._check_group(group)  # Is it a group?

        return group._f_iter_nodes(classname)

    iterNodes = previous_api(iter_nodes)

    def __contains__(self, path):
        """Is there a node with that path?

        Returns True if the file has a node with the given path (a
        string), False otherwise.

        """

        try:
            self.get_node(path)
        except NoSuchNodeError:
            return False
        else:
            return True

    def __iter__(self):
        """Recursively iterate over the nodes in the tree.

        This is equivalent to calling :meth:`File.walk_nodes` with no
        arguments.

        Examples
        --------

        ::

            # Recursively list all the nodes in the object tree.
            h5file = tables.open_file('vlarray1.h5')
            print "All nodes in the object tree:"
            for node in h5file:
                print node

        """

        return self.walk_nodes('/')

    def walk_nodes(self, where="/", classname=None):
        """Recursively iterate over nodes hanging from where.

        Parameters
        ----------
        where : str or Group, optional
            If supplied, the iteration starts from (and includes)
            this group. It can be a path string or a
            Group instance (see :ref:`GroupClassDescr`).
        classname
            If the name of a class derived from
            Node (see :ref:`GroupClassDescr`) is supplied, only instances of
            that class (or subclasses of it) will be returned.

        Notes
        -----
        This version iterates over the leaves in the same group in order
        to avoid having a list referencing to them and thus, preventing
        the LRU cache to remove them after their use.

        Examples
        --------

        ::

            # Recursively print all the nodes hanging from '/detector'.
            print "Nodes hanging from group '/detector':"
            for node in h5file.walk_nodes('/detector', classname='EArray'):
                print node

        """

        class_ = get_class_by_name(classname)

        if class_ is Group:  # only groups
            for group in self.walk_groups(where):
                yield group
        elif class_ is Node:  # all nodes
            yield self.get_node(where)
            for group in self.walk_groups(where):
                for leaf in self.iter_nodes(group):
                    yield leaf
        else:  # only nodes of the named type
            for group in self.walk_groups(where):
                for leaf in self.iter_nodes(group, classname):
                    yield leaf

    walkNodes = previous_api(walk_nodes)

    def walk_groups(self, where="/"):
        """Recursively iterate over groups (not leaves) hanging from where.

        The where group itself is listed first (preorder), then each of its
        child groups (following an alphanumerical order) is also traversed,
        following the same procedure.  If where is not supplied, the root
        group is used.

        The where argument can be a path string
        or a Group instance (see :ref:`GroupClassDescr`).

        """

        group = self.get_node(where)  # Does the parent exist?
        self._check_group(group)  # Is it a group?
        return group._f_walk_groups()

    walkGroups = previous_api(walk_groups)

    def _check_open(self):
        """Check the state of the file.

        If the file is closed, a `ClosedFileError` is raised.

        """

        if not self.isopen:
            raise ClosedFileError("the file object is closed")

    _checkOpen = previous_api(_check_open)

    def _iswritable(self):
        """Is this file writable?"""

        return self.mode in ('w', 'a', 'r+')

    _isWritable = previous_api(_iswritable)

    def _check_writable(self):
        """Check whether the file is writable.

        If the file is not writable, a `FileModeError` is raised.

        """

        if not self._iswritable():
            raise FileModeError("the file is not writable")

    _checkWritable = previous_api(_check_writable)

    def _check_group(self, node):
        # `node` must already be a node.
        if not isinstance(node, Group):
            raise TypeError("node ``%s`` is not a group" % (node._v_pathname,))

    _checkGroup = previous_api(_check_group)

    # <Undo/Redo support>
    def is_undo_enabled(self):
        """Is the Undo/Redo mechanism enabled?

        Returns True if the Undo/Redo mechanism has been enabled for this file,
        False otherwise. Please note that this mechanism is persistent, so a
        newly opened PyTables file may already have Undo/Redo support enabled.

        """

        self._check_open()
        return self._undoEnabled

    isUndoEnabled = previous_api(is_undo_enabled)

    def _check_undo_enabled(self):
        if not self._undoEnabled:
            raise UndoRedoError("Undo/Redo feature is currently disabled!")

    _checkUndoEnabled = previous_api(_check_undo_enabled)

    def _create_transaction_group(self):
        tgroup = TransactionGroupG(
            self.root, _trans_group_name,
            "Transaction information container", new=True)
        # The format of the transaction container.
        tgroup._v_attrs._g__setattr('FORMATVERSION', _trans_version)
        return tgroup

    _createTransactionGroup = previous_api(_create_transaction_group)

    def _create_transaction(self, troot, tid):
        return TransactionG(
            troot, _trans_name % tid,
            "Transaction number %d" % tid, new=True)

    _createTransaction = previous_api(_create_transaction)

    def _create_mark(self, trans, mid):
        return MarkG(
            trans, _markName % mid,
            "Mark number %d" % mid, new=True)

    _createMark = previous_api(_create_mark)

    def enable_undo(self, filters=Filters(complevel=1)):
        """Enable the Undo/Redo mechanism.

        This operation prepares the database for undoing and redoing
        modifications in the node hierarchy. This
        allows :meth:`File.mark`, :meth:`File.undo`, :meth:`File.redo` and
        other methods to be called.

        The filters argument, when specified,
        must be an instance of class Filters (see :ref:`FiltersClassDescr`) and
        is meant for setting the compression values for the action log. The
        default is having compression enabled, as the gains in terms of
        space can be considerable. You may want to disable compression if
        you want maximum speed for Undo/Redo operations.

        Calling this method when the Undo/Redo mechanism is already
        enabled raises an UndoRedoError.

        """

        maxundo = self.params['MAX_UNDO_PATH_LENGTH']

        class ActionLog(NotLoggedMixin, Table):
            pass

        class ActionLogDesc(IsDescription):
            opcode = UInt8Col(pos=0)
            arg1 = StringCol(maxundo, pos=1, dflt=b"")
            arg2 = StringCol(maxundo, pos=2, dflt=b"")

        self._check_open()

        # Enabling several times is not allowed to avoid the user having
        # the illusion that a new implicit mark has been created
        # when calling enable_undo for the second time.

        if self.is_undo_enabled():
            raise UndoRedoError("Undo/Redo feature is already enabled!")

        self._markers = {}
        self._seqmarkers = []
        self._nmarks = 0
        self._curtransaction = 0
        self._curmark = -1  # No marks yet

        # Get the Group for keeping user actions
        try:
            tgroup = self.get_node(_trans_group_path)
        except NodeError:
            # The file is going to be changed.
            self._check_writable()

            # A transaction log group does not exist. Create it
            tgroup = self._create_transaction_group()

            # Create a transaction.
            self._trans = self._create_transaction(
                tgroup, self._curtransaction)

            # Create an action log
            self._actionlog = ActionLog(
                tgroup, _action_log_name, ActionLogDesc, "Action log",
                filters=filters)

            # Create an implicit mark
            self._actionlog.append([(_op_to_code["MARK"], str(0), '')])
            self._nmarks += 1
            self._seqmarkers.append(0)  # current action is 0

            # Create a group for mark 0
            self._create_mark(self._trans, 0)
            # Initialize the marker pointer
            self._curmark = int(self._nmarks - 1)
            # Initialize the action pointer
            self._curaction = self._actionlog.nrows - 1
        else:
            # The group seems to exist already
            # Get the default transaction
            self._trans = tgroup._f_get_child(
                _trans_name % self._curtransaction)
            # Open the action log and go to the end of it
            self._actionlog = tgroup.actionlog
            for row in self._actionlog:
                if row["opcode"] == _op_to_code["MARK"]:
                    name = row["arg2"].decode('utf-8')
                    self._markers[name] = self._nmarks
                    self._seqmarkers.append(row.nrow)
                    self._nmarks += 1
            # Get the current mark and current action
            self._curmark = int(self._actionlog.attrs.CURMARK)
            self._curaction = self._actionlog.attrs.CURACTION

        # The Undo/Redo mechanism has been enabled.
        self._undoEnabled = True

    enableUndo = previous_api(enable_undo)

    def disable_undo(self):
        """Disable the Undo/Redo mechanism.

        Disabling the Undo/Redo mechanism leaves the database in the
        current state and forgets past and future database states. This
        makes :meth:`File.mark`, :meth:`File.undo`, :meth:`File.redo` and other
        methods fail with an UndoRedoError.

        Calling this method when the Undo/Redo mechanism is already
        disabled raises an UndoRedoError.

        """

        self._check_open()

        if not self.is_undo_enabled():
            raise UndoRedoError("Undo/Redo feature is already disabled!")

        # The file is going to be changed.
        self._check_writable()

        del self._markers
        del self._seqmarkers
        del self._curmark
        del self._curaction
        del self._curtransaction
        del self._nmarks
        del self._actionlog
        # Recursively delete the transaction group
        tnode = self.get_node(_trans_group_path)
        tnode._g_remove(recursive=1)

        # The Undo/Redo mechanism has been disabled.
        self._undoEnabled = False

    disableUndo = previous_api(disable_undo)

    def mark(self, name=None):
        """Mark the state of the database.

        Creates a mark for the current state of the database. A unique (and
        immutable) identifier for the mark is returned. An optional name (a
        string) can be assigned to the mark. Both the identifier of a mark and
        its name can be used in :meth:`File.undo` and :meth:`File.redo`
        operations. When the name has already been used for another mark,
        an UndoRedoError is raised.

        This method can only be called when the Undo/Redo mechanism has been
        enabled. Otherwise, an UndoRedoError is raised.

        """

        self._check_open()
        self._check_undo_enabled()

        if name is None:
            name = ''
        else:
            if not isinstance(name, str):
                raise TypeError("Only strings are allowed as mark names. "
                                "You passed object: '%s'" % name)
            if name in self._markers:
                raise UndoRedoError("Name '%s' is already used as a marker "
                                    "name. Try another one." % name)

            # The file is going to be changed.
            self._check_writable()

            self._markers[name] = self._curmark + 1

        # Create an explicit mark
        # Insert the mark in the action log
        self._log("MARK", str(self._curmark + 1), name)
        self._curmark += 1
        self._nmarks = self._curmark + 1
        self._seqmarkers.append(self._curaction)
        # Create a group for the current mark
        self._create_mark(self._trans, self._curmark)
        return self._curmark

    def _log(self, action, *args):
        """Log an action.

        The `action` must be an all-uppercase string identifying it.
        Arguments must also be strings.

        This method should be called once the action has been completed.

        This method can only be called when the Undo/Redo mechanism has
        been enabled.  Otherwise, an `UndoRedoError` is raised.

        """

        assert self.is_undo_enabled()

        maxundo = self.params['MAX_UNDO_PATH_LENGTH']
        # Check whether we are at the end of the action log or not
        if self._curaction != self._actionlog.nrows - 1:
            # We are not, so delete the trailing actions
            self._actionlog.remove_rows(self._curaction + 1,
                                        self._actionlog.nrows)
            # Reset the current marker group
            mnode = self.get_node(_markPath % (self._curtransaction,
                                               self._curmark))
            mnode._g_reset()
            # Delete the marker groups with backup objects
            for mark in xrange(self._curmark + 1, self._nmarks):
                mnode = self.get_node(_markPath % (self._curtransaction, mark))
                mnode._g_remove(recursive=1)
            # Update the new number of marks
            self._nmarks = self._curmark + 1
            self._seqmarkers = self._seqmarkers[:self._nmarks]

        if action not in _op_to_code:  # INTERNAL
            raise UndoRedoError("Action ``%s`` not in ``_op_to_code`` "
                                "dictionary: %r" % (action, _op_to_code))

        arg1 = ""
        arg2 = ""
        if len(args) <= 1:
            arg1 = args[0]
        elif len(args) <= 2:
            arg1 = args[0]
            arg2 = args[1]
        else:  # INTERNAL
            raise UndoRedoError("Too many parameters for action log: "
                                "%r").with_traceback(args)
        if (len(arg1) > maxundo
                or len(arg2) > maxundo):  # INTERNAL
            raise UndoRedoError("Parameter arg1 or arg2 is too long: "
                                "(%r, %r)" % (arg1, arg2))
        # print "Logging-->", (action, arg1, arg2)
        self._actionlog.append([(_op_to_code[action],
                                 arg1.encode('utf-8'),
                                 arg2.encode('utf-8'))])
        self._curaction += 1

    def _get_mark_id(self, mark):
        """Get an integer markid from a mark sequence number or name"""

        if isinstance(mark, int):
            markid = mark
        elif isinstance(mark, str):
            if mark not in self._markers:
                lmarkers = sorted(self._markers.iterkeys())
                raise UndoRedoError("The mark that you have specified has not "
                                    "been found in the internal marker list: "
                                    "%r" % lmarkers)
            markid = self._markers[mark]
        else:
            raise TypeError("Parameter mark can only be an integer or a "
                            "string, and you passed a type <%s>" % type(mark))
        # print "markid, self._nmarks:", markid, self._nmarks
        return markid

    _getMarkID = previous_api(_get_mark_id)

    def _get_final_action(self, markid):
        """Get the action to go. It does not touch the self private
        attributes"""

        if markid > self._nmarks - 1:
            # The required mark is beyond the end of the action log
            # The final action is the last row
            return self._actionlog.nrows
        elif markid <= 0:
            # The required mark is the first one
            # return the first row
            return 0

        return self._seqmarkers[markid]

    _getFinalAction = previous_api(_get_final_action)

    def _doundo(self, finalaction, direction):
        """Undo/Redo actions up to final action in the specificed direction"""

        if direction < 0:
            actionlog = \
                self._actionlog[finalaction + 1:self._curaction + 1][::-1]
        else:
            actionlog = self._actionlog[self._curaction:finalaction]

        # Uncomment this for debugging
#         print "curaction, finalaction, direction", \
#               self._curaction, finalaction, direction
        for i in xrange(len(actionlog)):
            if actionlog['opcode'][i] != _op_to_code["MARK"]:
                # undo/redo the action
                if direction > 0:
                    # Uncomment this for debugging
#                     print "redo-->", \
#                           _code_to_op[actionlog['opcode'][i]],\
#                           actionlog['arg1'][i],\
#                           actionlog['arg2'][i]
                    undoredo.redo(self,
                                  # _code_to_op[actionlog['opcode'][i]],
                                  # The next is a workaround for python < 2.5
                                  _code_to_op[int(actionlog['opcode'][i])],
                                  actionlog['arg1'][i].decode('utf8'),
                                  actionlog['arg2'][i].decode('utf8'))
                else:
                    # Uncomment this for debugging
                    # print "undo-->", \
                    #       _code_to_op[actionlog['opcode'][i]],\
                    #       actionlog['arg1'][i].decode('utf8'),\
                    #       actionlog['arg2'][i].decode('utf8')
                    undoredo.undo(self,
                                  # _code_to_op[actionlog['opcode'][i]],
                                  # The next is a workaround for python < 2.5
                                  _code_to_op[int(actionlog['opcode'][i])],
                                  actionlog['arg1'][i].decode('utf8'),
                                  actionlog['arg2'][i].decode('utf8'))
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
        """Go to a past state of the database.

        Returns the database to the state associated with the specified mark.
        Both the identifier of a mark and its name can be used. If the mark is
        omitted, the last created mark is used. If there are no past
        marks, or the specified mark is not older than the current one, an
        UndoRedoError is raised.

        This method can only be called when the Undo/Redo mechanism
        has been enabled. Otherwise, an UndoRedoError
        is raised.

        """

        self._check_open()
        self._check_undo_enabled()

#         print "(pre)UNDO: (curaction, curmark) = (%s,%s)" % \
#               (self._curaction, self._curmark)
        if mark is None:
            markid = self._curmark
            # Correction if we are settled on top of a mark
            opcode = self._actionlog.cols.opcode
            if opcode[self._curaction] == _op_to_code["MARK"]:
                markid -= 1
        else:
            # Get the mark ID number
            markid = self._get_mark_id(mark)
        # Get the final action ID to go
        finalaction = self._get_final_action(markid)
        if finalaction > self._curaction:
            raise UndoRedoError("Mark ``%s`` is newer than the current mark. "
                                "Use `redo()` or `goto()` instead." % (mark,))

        # The file is going to be changed.
        self._check_writable()

        # Try to reach this mark by unwinding actions in the log
        self._doundo(finalaction - 1, -1)
        if self._curaction < self._actionlog.nrows - 1:
            self._curaction += 1
        self._curmark = int(self._actionlog.cols.arg1[self._curaction])
#         print "(post)UNDO: (curaction, curmark) = (%s,%s)" % \
#               (self._curaction, self._curmark)

    def redo(self, mark=None):
        """Go to a future state of the database.

        Returns the database to the state associated with the specified
        mark.  Both the identifier of a mark and its name can be used.
        If the `mark` is omitted, the next created mark is used.  If
        there are no future marks, or the specified mark is not newer
        than the current one, an UndoRedoError is raised.

        This method can only be called when the Undo/Redo mechanism has
        been enabled.  Otherwise, an UndoRedoError is raised.

        """

        self._check_open()
        self._check_undo_enabled()

#         print "(pre)REDO: (curaction, curmark) = (%s, %s)" % \
#               (self._curaction, self._curmark)
        if self._curaction >= self._actionlog.nrows - 1:
            # We are at the end of log, so no action
            return

        if mark is None:
            mark = self._curmark + 1
        elif mark == -1:
            mark = int(self._nmarks)  # Go beyond the mark bounds up to the end
        # Get the mark ID number
        markid = self._get_mark_id(mark)
        finalaction = self._get_final_action(markid)
        if finalaction < self._curaction + 1:
            raise UndoRedoError("Mark ``%s`` is older than the current mark. "
                                "Use `redo()` or `goto()` instead." % (mark,))

        # The file is going to be changed.
        self._check_writable()

        # Get the final action ID to go
        self._curaction += 1

        # Try to reach this mark by redoing the actions in the log
        self._doundo(finalaction, 1)
        # Increment the current mark only if we are not at the end of marks
        if self._curmark < self._nmarks - 1:
            self._curmark += 1
        if self._curaction > self._actionlog.nrows - 1:
            self._curaction = self._actionlog.nrows - 1
#         print "(post)REDO: (curaction, curmark) = (%s,%s)" % \
#               (self._curaction, self._curmark)

    def goto(self, mark):
        """Go to a specific mark of the database.

        Returns the database to the state associated with the specified mark.
        Both the identifier of a mark and its name can be used.

        This method can only be called when the Undo/Redo mechanism has been
        enabled. Otherwise, an UndoRedoError is raised.

        """

        self._check_open()
        self._check_undo_enabled()

        if mark == -1:  # Special case
            mark = self._nmarks  # Go beyond the mark bounds up to the end
        # Get the mark ID number
        markid = self._get_mark_id(mark)
        finalaction = self._get_final_action(markid)
        if finalaction < self._curaction:
            self.undo(mark)
        else:
            self.redo(mark)

    def get_current_mark(self):
        """Get the identifier of the current mark.

        Returns the identifier of the current mark. This can be used
        to know the state of a database after an application crash, or to
        get the identifier of the initial implicit mark after a call
        to :meth:`File.enable_undo`.

        This method can only be called when the Undo/Redo mechanism
        has been enabled. Otherwise, an UndoRedoError
        is raised.

        """

        self._check_open()
        self._check_undo_enabled()
        return self._curmark

    getCurrentMark = previous_api(get_current_mark)

    def _shadow_name(self):
        """Compute and return a shadow name.

        Computes the current shadow name according to the current
        transaction, mark and action.  It returns a tuple with the
        shadow parent node and the name of the shadow in it.

        """

        parent = self.get_node(
            _shadow_parent % (self._curtransaction, self._curmark))
        name = _shadow_name % (self._curaction,)

        return (parent, name)

    _shadowName = previous_api(_shadow_name)

    # </Undo/Redo support>

    def flush(self):
        """Flush all the alive leaves in the object tree."""

        self._check_open()

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
        self._flush_file(0)  # 0 means local scope, 1 global (virtual) scope

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

        if self._undoEnabled and self._iswritable():
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
        self._close_file()
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
        """Return a short string representation of the object tree.

        Examples
        --------

        ::

            >>> f = tables.open_file('data/test.h5')
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
        try:
            date = time.asctime(time.localtime(os.stat(self.filename)[8]))
        except OSError:
            # in-memory file
            date = ""
        astring = self.filename + ' (File) ' + repr(self.title) + '\n'
#         astring += 'root_uep :=' + repr(self.root_uep) + '; '
#         astring += 'format_version := ' + self.format_version + '\n'
#         astring += 'filters :=' + repr(self.filters) + '\n'
        astring += 'Last modif.: ' + repr(date) + '\n'
        astring += 'Object Tree: \n'

        for group in self.walk_groups("/"):
            astring += str(group) + '\n'
            for kind in self._node_kinds[1:]:
                for node in self.list_nodes(group, kind):
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
                  ', root_uep=' + repr(self.root_uep) + \
                  ', filters=' + repr(self.filters) + \
                  ')\n'
        for group in self.walk_groups("/"):
            astring += str(group) + '\n'
            for kind in self._node_kinds[1:]:
                for node in self.list_nodes(group, kind):
                    astring += repr(node) + '\n'
        return astring

    def _refnode(self, node, nodePath):
        """Register `node` as alive and insert references to it."""

        if nodePath != '/':
            # The root group does not participate in alive/dead stuff.
            aliveNodes = self._aliveNodes
            assert nodePath not in aliveNodes, \
                "file already has a node with path ``%s``" % nodePath

            # Add the node to the set of referenced ones.
            aliveNodes[nodePath] = node

    _refNode = previous_api(_refnode)

    def _unrefnode(self, nodePath):
        """Unregister `node` as alive and remove references to it."""

        if nodePath != '/':
            # The root group does not participate in alive/dead stuff.
            aliveNodes = self._aliveNodes
            assert nodePath in aliveNodes, \
                "file does not have a node with path ``%s``" % nodePath

            # Remove the node from the set of referenced ones.
            del aliveNodes[nodePath]

    _unrefNode = previous_api(_unrefnode)

    def _killnode(self, node):
        """Kill the `node`.

        Moves the `node` from the set of alive, referenced nodes to the
        set of dead, unreferenced ones.

        """

        nodePath = node._v_pathname
        assert nodePath in self._aliveNodes, \
            "trying to kill non-alive node ``%s``" % nodePath

        node._g_pre_kill_hook()

        # Remove all references to the node.
        self._unrefnode(nodePath)
        # Save the dead node in the limbo.
        if self._aliveNodes.hasdeadnodes:
            self._deadNodes[nodePath] = node
        else:
            # We have not a cache for dead nodes,
            # so follow the usual deletion procedure.
            node._v__deleting = True
            node._f_close()

    _killNode = previous_api(_killnode)

    def _revivenode(self, nodePath):
        """Revive the node under `nodePath` and return it.

        Moves the node under `nodePath` from the set of dead,
        unreferenced nodes to the set of alive, referenced ones.

        """

        assert nodePath in self._deadNodes, \
            "trying to revive non-dead node ``%s``" % nodePath

        # Take the node out of the limbo.
        node = self._deadNodes.pop(nodePath)
        # Make references to the node.
        self._refnode(node, nodePath)

        node._g_post_revive_hook()

        return node

    _reviveNode = previous_api(_revivenode)

    def _update_node_locations(self, oldPath, newPath):
        """Update location information of nodes under `oldPath`.

        This only affects *already loaded* nodes.
        """

        oldPrefix = oldPath + '/'  # root node can not be renamed, anyway
        oldPrefixLen = len(oldPrefix)

        # Update alive and dead descendents.
        for cache in [self._aliveNodes, self._deadNodes]:
            for nodePath in cache:
                if nodePath.startswith(oldPrefix) and nodePath != oldPrefix:
                    nodeSuffix = nodePath[oldPrefixLen:]
                    newNodePath = join_path(newPath, nodeSuffix)
                    newNodePPath = split_path(newNodePath)[0]
                    descendentNode = self._get_node(nodePath)
                    descendentNode._g_update_location(newNodePPath)

    _updateNodeLocations = previous_api(_update_node_locations)


# If a user hits ^C during a run, it is wise to gracefully close the
# opened files.
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
