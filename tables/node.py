"""
Base class for PyTables nodes
=============================

:Author:   Ivan Vilata i Balaguer
:Contact:  ivan@selidor.net
:Created:  2005-02-11
:License:  BSD
:Revision: $Id$


Classes:

`Node`
    Abstract base class for all PyTables nodes.

Misc variables:

`__docformat__`
    The format of documentation strings in this module.
`__version__`
    Repository version of this file.
"""


import warnings

from tables.registry import classNameDict, classIdDict
from tables.exceptions import \
     ClosedNodeError, NodeError, UndoRedoWarning, PerformanceWarning
from tables.path import joinPath, splitPath, isVisiblePath
from tables.utils import lazyattr
from tables.undoredo import moveToShadow
from tables.attributeset import AttributeSet, NotLoggedAttributeSet



__docformat__ = 'reStructuredText'
"""The format of documentation strings in this module."""

__version__ = '$Revision$'
"""Repository version of this file."""


def _closedrepr(oldmethod):
    """
    Decorate string representation method to handle closed nodes.

    If the node is closed, a string like this is returned::

      <closed MODULE.CLASS at ADDRESS>

    instead of calling `oldmethod` and returning its result.
    """
    def newmethod(self):
        if not self._v_isopen:
            cmod = self.__class__.__module__
            cname = self.__class__.__name__
            addr = hex(id(self))
            return '<closed %s.%s at %s>' % (cmod, cname, addr)
        return oldmethod(self)
    newmethod.__name__ = oldmethod.__name__
    newmethod.__doc__ = oldmethod.__doc__
    return newmethod


class MetaNode(type):

    """
    Node metaclass.

    This metaclass ensures that their instance classes get registered
    into several dictionaries (namely the `tables.utils.classNameDict`
    class name dictionary and the `tables.utils.classIdDict` class
    identifier dictionary).

    It also adds sanity checks to some methods:

      * Check that the node is open when calling string representation
        and provide a default string if so.
    """

    def __new__(class_, name, bases, dict_):
        # Add default behaviour for representing closed nodes.
        for mname in ['__str__', '__repr__']:
            if mname in dict_:
                dict_[mname] = _closedrepr(dict_[mname])

        return type.__new__(class_, name, bases, dict_)


    def __init__(class_, name, bases, dict_):
        super(MetaNode, class_).__init__(name, bases, dict_)

        # Always register into class name dictionary.
        classNameDict[class_.__name__] = class_

        # Register into class identifier dictionary only if the class
        # has an identifier and it is different from its parents'.
        cid = getattr(class_, '_c_classId', None)
        if cid is not None:
            for base in bases:
                pcid = getattr(base, '_c_classId', None)
                if pcid == cid:
                    break
            else:
                classIdDict[cid] = class_



class Node(object):

    """
    Abstract base class for all PyTables nodes.

    This is the base class for *all* nodes in a PyTables hierarchy.
    It is an abstract class, i.e. it may not be directly instantiated;
    however, every node in the hierarchy is an instance of this class.

    A PyTables node is always hosted in a PyTables *file*, under a
    *parent group*, at a certain *depth* in the node hierarchy.  A node
    knows its own *name* in the parent group and its own *path name* in
    the file.

    All the previous information is location-dependent, i.e. it may
    change when moving or renaming a node in the hierarchy.  A node also
    has location-independent information, such as its *HDF5 object
    identifier* and its *attribute set*.

    This class gathers the operations and attributes (both
    location-dependent and independent) which are common to all PyTables
    nodes, whatever their type is.  Nonetheless, due to natural naming
    restrictions, the names of all of these members start with a
    reserved prefix (see the `Group` class).

    Sub-classes with no children (i.e. *leaf nodes*) may define new
    methods, attributes and properties to avoid natural naming
    restrictions.  For instance, ``_v_attrs`` may be shortened to
    ``attrs`` and ``_f_rename`` to ``rename``.  However, the original
    methods and attributes should still be available.

    Public instance variables -- location dependent
    -----------------------------------------------

    _v_depth
        The depth of this node in the tree (an non-negative integer
        value).
    _v_file
        The hosting `File` instance.
    _v_isopen
        Whether this node is open or not.
    _v_name
        The name of this node in its parent group (a string).
    _v_parent
        The parent `Group` instance.
    _v_pathname
        The path of this node in the tree (a string).

    Public instance variables -- location independent
    -------------------------------------------------

    _v_attrs
        The associated `AttributeSet` instance.
    _v_objectID
        A node identifier (may change from run to run).

    Public instance variables -- attribute shorthands
    -------------------------------------------------

    _v_title
        A description of this node.  A shorthand for the ``TITLE``
        attribute.

    Public methods -- hierarchy manipulation
    ----------------------------------------

    _f_close()
        Close this node in the tree.
    _f_copy([newparent][, newname][, overwrite][, recursive][, createparents][, **kwargs])
        Copy this node and return the new one.
    _f_isVisible()
        Is this node visible?
    _f_move([newparent][, newname][, overwrite])
        Move or rename this node.
    _f_remove([recursive])
        Remove this node from the hierarchy.
    _f_rename(newname[, overwrite])
        Rename this node in place.

    Public methods -- attribute handling
    ------------------------------------

    _f_delAttr(name)
        Delete a PyTables attribute from this node.
    _f_getAttr(name)
        Get a PyTables attribute from this node.
    _f_setAttr(name, value)
        Set a PyTables attribute for this node.
    """

    # This makes this class and all derived subclasses be handled by MetaNode.
    __metaclass__ = MetaNode

    # By default, attributes accept Undo/Redo.
    _AttributeSet = AttributeSet


    # <properties>

    # `_v_parent` is accessed via its file to avoid upwards references.
    def _g_getparent(self):
        (parentPath, nodeName) = splitPath(self._v_pathname)
        return self._v_file._getNode(parentPath)

    _v_parent = property(
        _g_getparent, None, None, "The parent `Group` instance.")


    # '_v_attrs' is defined as a lazy read-only attribute.
    # This saves 0.7s/3.8s.
    @lazyattr
    def _v_attrs(self):
        """The associated `AttributeSet` instance."""
        return self._AttributeSet(self)


    # '_v_title' is a direct read-write shorthand for the 'TITLE' attribute
    # with the empty string as a default value.
    def _g_gettitle (self):
        if hasattr(self._v_attrs, 'TITLE'):
            return self._v_attrs.TITLE
        else:
            return ''

    def _g_settitle (self, title):
        self._v_attrs.TITLE = title

    _v_title = property(_g_gettitle, _g_settitle, None,
                        "A description of this node.")

    # </properties>

    # This may be looked up by ``__del__`` when ``__init__`` doesn't get
    # to be called.  See ticket #144 for more info.
    _v_isopen = False
    """The default class attribute for _v_isopen."""

    # The ``_log`` argument is only meant to be used by ``_g_copyAsChild()``
    # to avoid logging the creation of children nodes of a copied sub-tree.
    def __init__(self, parentNode, name, _log=True):
        # Remember to assign these values in the root group constructor
        # as it does not use this method implementation!

        self._v_file = None
        """The hosting `File` instance."""
        self._v_isopen = False
        """Whether this node is open or not."""
        self._v_pathname = None
        """The path of this node in the tree (a string)."""
        self._v_name = None
        """The name of this node in its parent group (a string)."""
        self._v_depth = None
        """The depth of this node in the tree (an non-negative integer value)."""
        self._v_maxTreeDepth = parentNode._v_file.params['MAX_TREE_DEPTH']
        """Maximum tree depth before warning the user."""
        self._v__deleting = False
        """Is the node being deleted?"""

        self._v_objectID = None
        """A node identifier (may change from run to run)."""

        validate = new = self._v_new  # set by subclass constructor

        # Is the parent node a group?  Is it open?
        self._g_checkGroup(parentNode)
        parentNode._g_checkOpen()
        file_ = parentNode._v_file

        # Will the file be able to host a new node?
        if new:
            file_._checkWritable()

        # Bind to the parent node and set location-dependent information.
        if new:
            # Only new nodes need to be referenced.
            # Opened nodes are already known by their parent group.
            parentNode._g_refNode(self, name, validate)
        self._g_setLocation(parentNode, name)

        try:
            # hdf5Extension operations:
            #   Update node attributes.
            self._g_new(parentNode, name, init=True)
            #   Create or open the node and get its object ID.
            if new:
                self._v_objectID = self._g_create()
            else:
                self._v_objectID = self._g_open()

            # The node *has* been created, log that.
            if new and _log and file_.isUndoEnabled():
                self._g_logCreate()

            # This allows extra operations after creating the node.
            self._g_postInitHook()
        except:
            # If anything happens, the node must be closed
            # to undo every possible registration made so far.
            # We do *not* rely on ``__del__()`` doing it later,
            # since it might never be called anyway.
            self._f_close()
            raise


    def _g_logCreate(self):
        self._v_file._log('CREATE', self._v_pathname)


    def __del__(self):
        # Closed `Node` instances can not be killed and revived.
        # Instead, accessing a closed and deleted (from memory, not
        # disk) one yields a *new*, open `Node` instance.  This is
        # because of two reasons:
        #
        # 1. Predictability.  After closing a `Node` and deleting it,
        #    only one thing can happen when accessing it again: a new,
        #    open `Node` instance is returned.  If closed nodes could be
        #    revived, one could get either a closed or an open `Node`.
        #
        # 2. Ease of use.  If the user wants to access a closed node
        #    again, the only condition would be that no references to
        #    the `Node` instance were left.  If closed nodes could be
        #    revived, the user would also need to force the closed
        #    `Node` out of memory, which is not a trivial task.
        #

        if not self._v_isopen:
            return  # the node is already closed or not initialized

        # If we get here, the `Node` is still open.
        file_ = self._v_file
        if self._v_pathname in file_._aliveNodes:
            # If the node is alive, kill it (to save it).
            file_._killNode(self)
        elif file_._aliveNodes.hasdeadnodes:
            # The node is already dead and there are no references to it,
            # so follow the usual deletion procedure.
            # This means closing the (still open) node.
            # `self._v__deleting` is asserted so that the node
            # does not try to unreference itself again from the file.
            self._v__deleting = True
            self._f_close()


    def _g_preKillHook(self):
        """Code to be called before killing the node."""
        pass


    def _g_postReviveHook(self):
        """Code to be called after reviving the node."""
        pass


    def _g_create(self):
        """Create a new HDF5 node and return its object identifier."""
        raise NotImplementedError


    def _g_open(self):
        """Open an existing HDF5 node and return its object identifier."""
        raise NotImplementedError


    def _g_checkOpen(self):
        """
        Check that the node is open.

        If the node is closed, a `ClosedNodeError` is raised.
        """

        if not self._v_isopen:
            raise ClosedNodeError("the node object is closed")
        assert self._v_file.isopen, "found an open node in a closed file"


    def _g_setLocation(self, parentNode, name):
        """
        Set location-dependent attributes.

        Sets the location-dependent attributes of this node to reflect
        that it is placed under the specified `parentNode`, with the
        specified `name`.

        This also triggers the insertion of file references to this
        node.  If the maximum recommended tree depth is exceeded, a
        `PerformanceWarning` is issued.
        """

        file_ = parentNode._v_file
        parentDepth = parentNode._v_depth

        self._v_file = file_
        self._v_isopen = True

        rootUEP = file_.rootUEP
        if name.startswith(rootUEP):
            # This has been called from File._getNode()
            assert parentDepth == 0
            if rootUEP == "/":
                self._v_pathname = name
            else:
                self._v_pathname = name[len(rootUEP):]
            _, self._v_name = splitPath(name)
            self._v_depth = name.count("/") - rootUEP.count("/") + 1
        else:
            # If we enter here is because this has been called elsewhere
            self._v_name = name
            self._v_pathname = joinPath(parentNode._v_pathname, name)
            self._v_depth = parentDepth + 1

        # Check if the node is too deep in the tree.
        if parentDepth >= self._v_maxTreeDepth:
            warnings.warn("""\
node ``%s`` is exceeding the recommended maximum depth (%d);\
be ready to see PyTables asking for *lots* of memory and possibly slow I/O"""
                          % (self._v_pathname, self._v_maxTreeDepth),
                          PerformanceWarning)

        file_._refNode(self, self._v_pathname)


    def _g_updateLocation(self, newParentPath):
        """
        Update location-dependent attributes.

        Updates location data when an ancestor node has changed its
        location in the hierarchy to `newParentPath`.  In fact, this
        method is expected to be called by an ancestor of this node.

        This also triggers the update of file references to this node.
        If the maximum recommended node depth is exceeded, a
        `PerformanceWarning` is issued.  This warning is assured to be
        unique.
        """

        oldPath = self._v_pathname
        newPath = joinPath(newParentPath, self._v_name)
        newDepth = newPath.count('/')

        self._v_pathname = newPath
        self._v_depth = newDepth

        # Check if the node is too deep in the tree.
        if newDepth > self._v_maxTreeDepth:
            warnings.warn("""\
moved descendent node is exceeding the recommended maximum depth (%d);\
be ready to see PyTables asking for *lots* of memory and possibly slow I/O"""
                          % (self._v_maxTreeDepth,), PerformanceWarning)

        file_ = self._v_file
        file_._unrefNode(oldPath)
        file_._refNode(self, newPath)

        # Tell dependent objects about the new location of this node.
        self._g_updateDependent()


    def _g_delLocation(self):
        """
        Clear location-dependent attributes.

        This also triggers the removal of file references to this node.
        """

        file_ = self._v_file
        pathname = self._v_pathname

        self._v_file = None
        self._v_isopen = False
        self._v_pathname = None
        self._v_name = None
        self._v_depth = None

        # If the node object is being deleted,
        # it has already been unreferenced from the file.
        if not self._v__deleting:
            file_._unrefNode(pathname)


    def _g_postInitHook(self):
        """Code to be run after node creation and before creation logging."""
        pass


    def _g_updateDependent(self):
        """
        Update dependent objects after a location change.

        All dependent objects (but not nodes!) referencing this node
        must be updated here.
        """
        if '_v_attrs' in self.__dict__:
            self._v_attrs._g_updateNodeLocation(self)


    def _f_close(self):
        """
        Close this node in the tree.

        This releases all resources held by the node, so it should not
        be used again.  On nodes with data, it may be flushed to disk.

        You should not need to close nodes manually because they are
        automatically opened/closed when they are loaded/evicted from
        the integrated LRU cache.
        """

        # After calling ``_f_close()``, two conditions are met:
        #
        #   1. The node object is detached from the tree.
        #   2. *Every* attribute of the node is removed.
        #
        # Thus, cleanup operations used in ``_f_close()`` in sub-classes
        # must be run *before* calling the method in the superclass.

        if not self._v_isopen:
            return  # the node is already closed

        myDict = self.__dict__

        # Close the associated `AttributeSet`
        # only if it has already been placed in the object's dictionary.
        if '_v_attrs' in myDict:
            self._v_attrs._g_close()

        # Detach the node from the tree if necessary.
        self._g_delLocation()

        # Finally, clear all remaining attributes from the object.
        myDict.clear()

        # Just add a final flag to signal that the node is closed:
        self._v_isopen = False


    def _g_remove(self, recursive, force):
        """
        Remove this node from the hierarchy.

        If the node has children, recursive removal must be stated by
        giving `recursive` a true value; otherwise, a `NodeError` will
        be raised.

        If `force` is set to true, the node will be removed no matter it
        has children or not (useful for deleting hard links).

        It does not log the change.
        """

        # Remove the node from the PyTables hierarchy.
        parent = self._v_parent
        parent._g_unrefNode(self._v_name)
        # Close the node itself.
        self._f_close()
        # hdf5Extension operations:
        #   Remove the node from the HDF5 hierarchy.
        self._g_delete(parent)


    def _f_remove(self, recursive=False, force=False):
        """
        Remove this node from the hierarchy.

        If the node has children, recursive removal must be stated by
        giving `recursive` a true value, or a `NodeError` will be
        raised.

        If the node is a link to a `Group` object, and you are sure that
        you want to delete it, you can do this by setting the `force`
        flag to true.
        """

        self._g_checkOpen()
        file_ = self._v_file
        file_._checkWritable()

        if file_.isUndoEnabled():
            self._g_removeAndLog(recursive, force)
        else:
            self._g_remove(recursive, force)


    def _g_removeAndLog(self, recursive, force):
        file_ = self._v_file
        oldPathname = self._v_pathname
        # Log *before* moving to use the right shadow name.
        file_._log('REMOVE', oldPathname)
        moveToShadow(file_, oldPathname)


    def _g_move(self, newParent, newName):
        """
        Move this node in the hierarchy.

        Moves the node into the given `newParent`, with the given
        `newName`.

        It does not log the change.
        """

        oldParent = self._v_parent
        oldName = self._v_name
        oldPathname = self._v_pathname  # to move the HDF5 node

        # Try to insert the node into the new parent.
        newParent._g_refNode(self, newName)
        # Remove the node from the new parent.
        oldParent._g_unrefNode(oldName)

        # Remove location information for this node.
        self._g_delLocation()
        # Set new location information for this node.
        self._g_setLocation(newParent, newName)

        # hdf5Extension operations:
        #   Update node attributes.
        self._g_new(newParent, self._v_name, init=False)
        #   Move the node.
        #self._v_parent._g_moveNode(oldPathname, self._v_pathname)
        self._v_parent._g_moveNode(oldParent._v_objectID, oldName,
                                   newParent._v_objectID, newName,
                                   oldPathname, self._v_pathname)

        # Tell dependent objects about the new location of this node.
        self._g_updateDependent()


    def _f_rename(self, newname, overwrite=False):
        """
        Rename this node in place.

        Changes the name of a node to `newname` (a string).  If a node
        with the same `newname` already exists and `overwrite` is true,
        recursively remove it before renaming.
        """
        self._f_move(newname=newname, overwrite=overwrite)


    def _f_move( self, newparent=None, newname=None,
                 overwrite=False, createparents=False ):
        """
        Move or rename this node.

        Moves a node into a new parent group, or changes the name of the
        node.  `newparent` can be a `Group` object or a pathname in
        string form.  If it is not specified or ``None`` , the current
        parent group is chosen as the new parent.  `newname` must be a
        string with a new name.  If it is not specified or ``None``, the
        current name is chosen as the new name.  If `createparents` is
        true, the needed groups for the given new parent group path to
        exist will be created.

        Moving a node across databases is not allowed, nor it is moving
        a node *into* itself.  These result in a `NodeError`.  However,
        moving a node *over* itself is allowed and simply does nothing.
        Moving over another existing node is similarly not allowed,
        unless the optional `overwrite` argument is true, in which case
        that node is recursively removed before moving.

        Usually, only the first argument will be used, effectively
        moving the node to a new location without changing its name.
        Using only the second argument is equivalent to renaming the
        node in place.
        """

        self._g_checkOpen()
        file_ = self._v_file
        oldParent = self._v_parent
        oldName = self._v_name

        # Set default arguments.
        if newparent is None and newname is None:
            raise NodeError( "you should specify at least "
                             "a ``newparent`` or a ``newname`` parameter" )
        if newparent is None:
            newparent = oldParent
        if newname is None:
            newname = oldName

        # Get destination location.
        if hasattr(newparent, '_v_file'):  # from node
            newfile = newparent._v_file
            newpath = newparent._v_pathname
        elif hasattr(newparent, 'startswith'):  # from path
            newfile = file_
            newpath = newparent
        else:
            raise TypeError( "new parent is not a node nor a path: %r"
                             % (dstParent,) )

        # Validity checks on arguments.
        # Is it in the same file?
        if newfile is not file_:
            raise NodeError( "nodes can not be moved across databases; "
                             "please make a copy of the node" )

        # The movement always fails if the hosting file can not be modified.
        file_._checkWritable()

        # Moving over itself?
        oldPath = oldParent._v_pathname
        if newpath == oldPath and newname == oldName:
            # This is equivalent to renaming the node to its current name,
            # and it does not change the referenced object,
            # so it is an allowed no-op.
            return

        # Moving into itself?
        self._g_checkNotContains(newpath)

        # Note that the previous checks allow us to go ahead and create
        # the parent groups if `createparents` is true.  `newparent` is
        # used instead of `newpath` to avoid accepting `Node` objects
        # when `createparents` is true.
        newparent = file_._getOrCreatePath(newparent, createparents)
        self._g_checkGroup(newparent)  # Is it a group?

        # Moving over an existing node?
        self._g_maybeRemove(newparent, newname, overwrite)

        # Move the node.
        oldPathname = self._v_pathname
        self._g_move(newparent, newname)

        # Log the change.
        if file_.isUndoEnabled():
            self._g_logMove(oldPathname)


    def _g_logMove(self, oldPathname):
        self._v_file._log('MOVE', oldPathname, self._v_pathname)


    def _g_copy(self, newParent, newName, recursive, _log=True, **kwargs):
        """
        Copy this node and return the new one.

        Creates and returns a copy of the node in the given `newParent`,
        with the given `newName`.  If `recursive` copy is stated, all
        descendents are copied as well.  Additional keyword argumens may
        affect the way that the copy is made.  Unknown arguments must be
        ignored.  On recursive copies, all keyword arguments must be
        passed on to the children invocation of this method.

        If `_log` is false, the change is not logged.  This is *only*
        intended to be used by ``_g_copyAsChild()`` as a means of
        optimising sub-tree copies.
        """
        raise NotImplementedError


    def _g_copyAsChild(self, newParent, **kwargs):
        """
        Copy this node as a child of another group.

        Copies just this node into `newParent`, not recursing children
        nor overwriting nodes nor logging the copy.  This is intended to
        be used when copying whole sub-trees.
        """
        return self._g_copy( newParent, self._v_name,
                             recursive=False, _log=False, **kwargs )


    def _f_copy(self, newparent=None, newname=None,
                overwrite=False, recursive=False, createparents=False,
                **kwargs):
        """
        Copy this node and return the new one.

        Creates and returns a copy of the node, maybe in a different
        place in the hierarchy.  `newparent` can be a `Group` object or
        a pathname in string form.  If it is not specified or ``None``,
        the current parent group is chosen as the new parent.  `newname`
        must be a string with a new name.  If it is not specified or
        ``None``, the current name is chosen as the new name.  If
        `recursive` copy is stated, all descendents are copied as well.
        If `createparents` is true, the needed groups for the given
        new parent group path to exist will be created.

        Copying a node across databases is supported but can not be
        undone.  Copying a node over itself is not allowed, nor it is
        recursively copying a node into itself.  These result in a
        `NodeError`.  Copying over another existing node is similarly
        not allowed, unless the optional `overwrite` argument is true,
        in which case that node is recursively removed before copying.

        Additional keyword arguments may be passed to customize the
        copying process.  For instance, title and filters may be
        changed, user attributes may be or may not be copied, data may
        be subsampled, stats may be collected, etc.  See the
        documentation for the particular node type.

        Using only the first argument is equivalent to copying the node
        to a new location without changing its name.  Using only the
        second argument is equivalent to making a copy of the node in
        the same group.
        """

        self._g_checkOpen()
        srcFile = self._v_file
        srcParent = self._v_parent
        srcName = self._v_name

        dstParent = newparent
        dstName = newname

        # Set default arguments.
        if dstParent is None and dstName is None:
            raise NodeError( "you should specify at least "
                             "a ``newparent`` or a ``newname`` parameter" )
        if dstParent is None:
            dstParent = srcParent
        if dstName is None:
            dstName = srcName

        # Get destination location.
        if hasattr(dstParent, '_v_file'):  # from node
            dstFile = dstParent._v_file
            dstPath = dstParent._v_pathname
        elif hasattr(dstParent, 'startswith'):  # from path
            dstFile = srcFile
            dstPath = dstParent
        else:
            raise TypeError( "new parent is not a node nor a path: %r"
                             % (dstParent,) )

        # Validity checks on arguments.
        if dstFile is srcFile:
            # Copying over itself?
            srcPath = srcParent._v_pathname
            if dstPath == srcPath and dstName == srcName:
                raise NodeError(
                    "source and destination nodes are the same node: ``%s``"
                    % self._v_pathname )

            # Recursively copying into itself?
            if recursive:
                self._g_checkNotContains(dstPath)

        # Note that the previous checks allow us to go ahead and create
        # the parent groups if `createparents` is true.  `dstParent` is
        # used instead of `dstPath` because it may be in other file, and
        # to avoid accepting `Node` objects when `createparents` is
        # true.
        dstParent = srcFile._getOrCreatePath(dstParent, createparents)
        self._g_checkGroup(dstParent)  # Is it a group?

        # Copying to another file with undo enabled?
        dolog = True
        if dstFile is not srcFile and srcFile.isUndoEnabled():
            warnings.warn( "copying across databases can not be undone "
                           "nor redone from this database",
                           UndoRedoWarning )
            dolog = False

        # Copying over an existing node?
        self._g_maybeRemove(dstParent, dstName, overwrite)

        # Copy the node.
        # The constructor of the new node takes care of logging.
        return self._g_copy(dstParent, dstName, recursive, **kwargs)


    def _f_isVisible(self):
        """Is this node visible?"""
        self._g_checkOpen()
        return isVisiblePath(self._v_pathname)


    def _g_checkGroup(self, node):
        # Node must be defined in order to define a Group.
        # However, we need to know Group here.
        # Using classNameDict avoids a circular import.
        if not isinstance(node, classNameDict['Node']):
            raise TypeError("new parent is not a registered node: %s"
                            % node._v_pathname)
        if not isinstance(node, classNameDict['Group']):
            raise TypeError("new parent node ``%s`` is not a group"
                            % node._v_pathname)


    def _g_checkNotContains(self, pathname):
        # The not-a-TARDIS test. ;)
        mypathname = self._v_pathname
        if ( mypathname == '/'  # all nodes fall below the root group
             or pathname == mypathname
             or pathname.startswith(mypathname + '/') ):
            raise NodeError(
                "can not move or recursively copy node ``%s`` into itself"
                % mypathname )


    def _g_maybeRemove(self, parent, name, overwrite):
        if name in parent:
            if not overwrite:
                raise NodeError("""\
destination group ``%s`` already has a node named ``%s``; \
you may want to use the ``overwrite`` argument""" % (parent._v_pathname, name))
            parent._f_getChild(name)._f_remove(True)


    def _g_checkName(self, name):
        """
        Check validity of name for this particular kind of node.

        This is invoked once the standard HDF5 and natural naming checks
        have successfully passed.
        """

        if name.startswith('_i_'):
            # This is reserved for table index groups.
            raise ValueError(
                "node name starts with reserved prefix ``_i_``: %s" % name)


    # <attribute handling>

    def _f_getAttr(self, name):
        """
        Get a PyTables attribute from this node.

        If the named attribute does not exist, an `AttributeError` is
        raised.
        """
        return getattr(self._v_attrs, name)

    def _f_setAttr(self, name, value):
        """
        Set a PyTables attribute for this node.

        If the node already has a large number of attributes, a
        `PerformanceWarning` is issued.
        """
        setattr(self._v_attrs, name, value)

    def _f_delAttr(self, name):
        """
        Delete a PyTables attribute from this node.

        If the named attribute does not exist, an `AttributeError` is
        raised.
        """
        delattr(self._v_attrs, name)

    # </attribute handling>



class NotLoggedMixin:
    # Include this class in your inheritance tree
    # to avoid changes to instances of your class from being logged.

    _AttributeSet = NotLoggedAttributeSet

    def _g_logCreate(self):
        pass

    def _g_logMove(self, oldPathname):
        pass

    def _g_removeAndLog(self, recursive, force):
        self._g_remove(recursive, force)



## Local Variables:
## mode: python
## py-indent-offset: 4
## tab-width: 4
## fill-column: 72
## End:
