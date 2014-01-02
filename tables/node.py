# -*- coding: utf-8 -*-

########################################################################
#
# License: BSD
# Created: 2005-02-11
# Author: Ivan Vilata i Balaguer - ivan@selidor.net
#
# $Id$
#
########################################################################

"""PyTables nodes"""

import warnings

from tables.registry import class_name_dict, class_id_dict
from tables.exceptions import (ClosedNodeError, NodeError, UndoRedoWarning,
                               PerformanceWarning)
from tables.path import join_path, split_path, isvisiblepath
from tables.utils import lazyattr
from tables.undoredo import move_to_shadow
from tables.attributeset import AttributeSet, NotLoggedAttributeSet
from tables._past import previous_api, previous_api_property


__docformat__ = 'reStructuredText'
"""The format of documentation strings in this module."""


def _closedrepr(oldmethod):
    """Decorate string representation method to handle closed nodes.

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
    """Node metaclass.

    This metaclass ensures that their instance classes get registered
    into several dictionaries (namely the `tables.utils.class_name_dict`
    class name dictionary and the `tables.utils.class_id_dict` class
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
        class_name_dict[class_.__name__] = class_

        # Register into class identifier dictionary only if the class
        # has an identifier and it is different from its parents'.
        cid = getattr(class_, '_c_classid', None)
        if cid is not None:
            for base in bases:
                pcid = getattr(base, '_c_classid', None)
                if pcid == cid:
                    break
            else:
                class_id_dict[cid] = class_


class Node(object):
    """Abstract base class for all PyTables nodes.

    This is the base class for *all* nodes in a PyTables hierarchy. It is an
    abstract class, i.e. it may not be directly instantiated; however, every
    node in the hierarchy is an instance of this class.

    A PyTables node is always hosted in a PyTables *file*, under a *parent
    group*, at a certain *depth* in the node hierarchy. A node knows its own
    *name* in the parent group and its own *path name* in the file.

    All the previous information is location-dependent, i.e. it may change when
    moving or renaming a node in the hierarchy. A node also has
    location-independent information, such as its *HDF5 object identifier* and
    its *attribute set*.

    This class gathers the operations and attributes (both location-dependent
    and independent) which are common to all PyTables nodes, whatever their
    type is. Nonetheless, due to natural naming restrictions, the names of all
    of these members start with a reserved prefix (see the Group class
    in :ref:`GroupClassDescr`).

    Sub-classes with no children (e.g. *leaf nodes*) may define new methods,
    attributes and properties to avoid natural naming restrictions. For
    instance, _v_attrs may be shortened to attrs and _f_rename to
    rename. However, the original methods and attributes should still be
    available.

    .. rubric:: Node attributes

    .. attribute:: _v_depth

        The depth of this node in the tree (an non-negative integer value).

    .. attribute:: _v_file

        The hosting File instance (see :ref:`FileClassDescr`).

    .. attribute:: _v_name

        The name of this node in its parent group (a string).

    .. attribute:: _v_pathname

        The path of this node in the tree (a string).

    .. attribute:: _v_objectid

        A node identifier (may change from run to run).

        .. versionchanged:: 3.0
           The *_v_objectID* attribute has been renamed into *_v_object_id*.

    """

    # This makes this class and all derived subclasses be handled by MetaNode.
    __metaclass__ = MetaNode

    # By default, attributes accept Undo/Redo.
    _AttributeSet = AttributeSet

    # <properties>
    # `_v_parent` is accessed via its file to avoid upwards references.
    def _g_getparent(self):
        (parentpath, nodename) = split_path(self._v_pathname)
        return self._v_file._get_node(parentpath)

    _v_parent = property(
        _g_getparent, None, None, ("The parent :class:`Group` instance"))

    # '_v_attrs' is defined as a lazy read-only attribute.
    # This saves 0.7s/3.8s.
    @lazyattr
    def _v_attrs(self):
        """The associated `AttributeSet` instance.

        See Also
        --------
        tables.attributeset.AttributeSet : container for the HDF5 attributes

        """

        return self._AttributeSet(self)

    # '_v_title' is a direct read-write shorthand for the 'TITLE' attribute
    # with the empty string as a default value.
    def _g_gettitle(self):
        if hasattr(self._v_attrs, 'TITLE'):
            return self._v_attrs.TITLE
        else:
            return ''

    def _g_settitle(self, title):
        self._v_attrs.TITLE = title

    _v_title = property(_g_gettitle, _g_settitle, None,
                        ("A description of this node. A shorthand for "
                         "TITLE attribute."))

    # </properties>

    # This may be looked up by ``__del__`` when ``__init__`` doesn't get
    # to be called.  See ticket #144 for more info.
    _v_isopen = False
    """Whehter this node is open or not."""

    _v_objectId = previous_api_property('_v_objectid')
    _v_maxTreeDepth = previous_api_property('_v_maxtreedepth')

    # The ``_log`` argument is only meant to be used by ``_g_copy_as_child()``
    # to avoid logging the creation of children nodes of a copied sub-tree.
    def __init__(self, parentnode, name, _log=True):
        # Remember to assign these values in the root group constructor
        # as it does not use this method implementation!

        self._v_file = None
        """The hosting File instance (see :ref:`FileClassDescr`)."""

        self._v_isopen = False
        """Whether this node is open or not."""

        self._v_pathname = None
        """The path of this node in the tree (a string)."""

        self._v_name = None
        """The name of this node in its parent group (a string)."""

        self._v_depth = None
        """The depth of this node in the tree (an non-negative integer value).
        """

        self._v_maxtreedepth = parentnode._v_file.params['MAX_TREE_DEPTH']
        """Maximum tree depth before warning the user.

        .. versionchanged:: 3.0
           Renamed into *_v_maxtreedepth* from *_v_maxTreeDepth*.

        """

        self._v__deleting = False
        """Is the node being deleted?"""

        self._v_objectid = None
        """A node identifier (may change from run to run).

        .. versionchanged:: 3.0
           The *_v_objectID* attribute has been renamed into *_v_objectid*.

        """

        validate = new = self._v_new  # set by subclass constructor

        # Is the parent node a group?  Is it open?
        self._g_check_group(parentnode)
        parentnode._g_check_open()
        file_ = parentnode._v_file

        # Will the file be able to host a new node?
        if new:
            file_._check_writable()

        # Bind to the parent node and set location-dependent information.
        if new:
            # Only new nodes need to be referenced.
            # Opened nodes are already known by their parent group.
            parentnode._g_refnode(self, name, validate)
        self._g_set_location(parentnode, name)

        try:
            # hdf5extension operations:
            #   Update node attributes.
            self._g_new(parentnode, name, init=True)
            #   Create or open the node and get its object ID.
            if new:
                self._v_objectid = self._g_create()
            else:
                self._v_objectid = self._g_open()

            # The node *has* been created, log that.
            if new and _log and file_.is_undo_enabled():
                self._g_log_create()

            # This allows extra operations after creating the node.
            self._g_post_init_hook()
        except:
            # If anything happens, the node must be closed
            # to undo every possible registration made so far.
            # We do *not* rely on ``__del__()`` doing it later,
            # since it might never be called anyway.
            self._f_close()
            raise

    def _g_log_create(self):
        self._v_file._log('CREATE', self._v_pathname)

    _g_logCreate = previous_api(_g_log_create)

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

        self._v__deleting = True

        # If we get here, the `Node` is still open.
        try:
            node_manager = self._v_file._node_manager
            node_manager.drop_node(self, check_unregistered=False)
        finally:
            # At this point the node can still be open if there is still some
            # alive reference around (e.g. if the __del__ method is called
            # explicitly by the user).
            if self._v_isopen:
                self._v__deleting = True
                self._f_close()

    def _g_pre_kill_hook(self):
        """Code to be called before killing the node."""
        pass

    _g_preKillHook = previous_api(_g_pre_kill_hook)

    def _g_create(self):
        """Create a new HDF5 node and return its object identifier."""
        raise NotImplementedError

    def _g_open(self):
        """Open an existing HDF5 node and return its object identifier."""
        raise NotImplementedError

    def _g_check_open(self):
        """Check that the node is open.

        If the node is closed, a `ClosedNodeError` is raised.

        """

        if not self._v_isopen:
            raise ClosedNodeError("the node object is closed")
        assert self._v_file.isopen, "found an open node in a closed file"

    _g_checkOpen = previous_api(_g_check_open)

    def _g_set_location(self, parentnode, name):
        """Set location-dependent attributes.

        Sets the location-dependent attributes of this node to reflect
        that it is placed under the specified `parentnode`, with the
        specified `name`.

        This also triggers the insertion of file references to this
        node.  If the maximum recommended tree depth is exceeded, a
        `PerformanceWarning` is issued.

        """

        file_ = parentnode._v_file
        parentdepth = parentnode._v_depth

        self._v_file = file_
        self._v_isopen = True

        root_uep = file_.root_uep
        if name.startswith(root_uep):
            # This has been called from File._get_node()
            assert parentdepth == 0
            if root_uep == "/":
                self._v_pathname = name
            else:
                self._v_pathname = name[len(root_uep):]
            _, self._v_name = split_path(name)
            self._v_depth = name.count("/") - root_uep.count("/") + 1
        else:
            # If we enter here is because this has been called elsewhere
            self._v_name = name
            self._v_pathname = join_path(parentnode._v_pathname, name)
            self._v_depth = parentdepth + 1

        # Check if the node is too deep in the tree.
        if parentdepth >= self._v_maxtreedepth:
            warnings.warn("""\
node ``%s`` is exceeding the recommended maximum depth (%d);\
be ready to see PyTables asking for *lots* of memory and possibly slow I/O"""
                          % (self._v_pathname, self._v_maxtreedepth),
                          PerformanceWarning)

        if self._v_pathname != '/':
            file_._node_manager.cache_node(self, self._v_pathname)

    _g_setLocation = previous_api(_g_set_location)

    def _g_update_location(self, newparentpath):
        """Update location-dependent attributes.

        Updates location data when an ancestor node has changed its
        location in the hierarchy to `newparentpath`.  In fact, this
        method is expected to be called by an ancestor of this node.

        This also triggers the update of file references to this node.
        If the maximum recommended node depth is exceeded, a
        `PerformanceWarning` is issued.  This warning is assured to be
        unique.

        """

        oldpath = self._v_pathname
        newpath = join_path(newparentpath, self._v_name)
        newdepth = newpath.count('/')

        self._v_pathname = newpath
        self._v_depth = newdepth

        # Check if the node is too deep in the tree.
        if newdepth > self._v_maxtreedepth:
            warnings.warn("""\
moved descendent node is exceeding the recommended maximum depth (%d);\
be ready to see PyTables asking for *lots* of memory and possibly slow I/O"""
                          % (self._v_maxtreedepth,), PerformanceWarning)

        node_manager = self._v_file._node_manager
        node_manager.rename_node(oldpath, newpath)

        # Tell dependent objects about the new location of this node.
        self._g_update_dependent()

    _g_updateLocation = previous_api(_g_update_location)

    def _g_del_location(self):
        """Clear location-dependent attributes.

        This also triggers the removal of file references to this node.

        """

        node_manager = self._v_file._node_manager
        pathname = self._v_pathname

        if not self._v__deleting:
            node_manager.drop_from_cache(pathname)
            # Note: node_manager.drop_node do not removes the node form the
            # registry if it is still open
            node_manager.registry.pop(pathname, None)

        self._v_file = None
        self._v_isopen = False
        self._v_pathname = None
        self._v_name = None
        self._v_depth = None

    _g_delLocation = previous_api(_g_del_location)

    def _g_post_init_hook(self):
        """Code to be run after node creation and before creation logging."""
        pass

    _g_postInitHook = previous_api(_g_post_init_hook)

    def _g_update_dependent(self):
        """Update dependent objects after a location change.

        All dependent objects (but not nodes!) referencing this node
        must be updated here.

        """

        if '_v_attrs' in self.__dict__:
            self._v_attrs._g_update_node_location(self)

    _g_updateDependent = previous_api(_g_update_dependent)

    def _f_close(self):
        """Close this node in the tree.

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
        self._g_del_location()

        # Finally, clear all remaining attributes from the object.
        myDict.clear()

        # Just add a final flag to signal that the node is closed:
        self._v_isopen = False

    def _g_remove(self, recursive, force):
        """Remove this node from the hierarchy.

        If the node has children, recursive removal must be stated by
        giving `recursive` a true value; otherwise, a `NodeError` will
        be raised.

        If `force` is set to true, the node will be removed no matter it
        has children or not (useful for deleting hard links).

        It does not log the change.

        """

        # Remove the node from the PyTables hierarchy.
        parent = self._v_parent
        parent._g_unrefnode(self._v_name)
        # Close the node itself.
        self._f_close()
        # hdf5extension operations:
        # Remove the node from the HDF5 hierarchy.
        self._g_delete(parent)

    def _f_remove(self, recursive=False, force=False):
        """Remove this node from the hierarchy.

        If the node has children, recursive removal must be stated by giving
        recursive a true value; otherwise, a NodeError will be raised.

        If the node is a link to a Group object, and you are sure that you want
        to delete it, you can do this by setting the force flag to true.

        """

        self._g_check_open()
        file_ = self._v_file
        file_._check_writable()

        if file_.is_undo_enabled():
            self._g_remove_and_log(recursive, force)
        else:
            self._g_remove(recursive, force)

    def _g_remove_and_log(self, recursive, force):
        file_ = self._v_file
        oldpathname = self._v_pathname
        # Log *before* moving to use the right shadow name.
        file_._log('REMOVE', oldpathname)
        move_to_shadow(file_, oldpathname)

    _g_removeAndLog = previous_api(_g_remove_and_log)

    def _g_move(self, newparent, newname):
        """Move this node in the hierarchy.

        Moves the node into the given `newparent`, with the given
        `newname`.

        It does not log the change.

        """

        oldparent = self._v_parent
        oldname = self._v_name
        oldpathname = self._v_pathname  # to move the HDF5 node

        # Try to insert the node into the new parent.
        newparent._g_refnode(self, newname)
        # Remove the node from the new parent.
        oldparent._g_unrefnode(oldname)

        # Remove location information for this node.
        self._g_del_location()
        # Set new location information for this node.
        self._g_set_location(newparent, newname)

        # hdf5extension operations:
        #   Update node attributes.
        self._g_new(newparent, self._v_name, init=False)
        #   Move the node.
        # self._v_parent._g_move_node(oldpathname, self._v_pathname)
        self._v_parent._g_move_node(oldparent._v_objectid, oldname,
                                    newparent._v_objectid, newname,
                                    oldpathname, self._v_pathname)

        # Tell dependent objects about the new location of this node.
        self._g_update_dependent()

    def _f_rename(self, newname, overwrite=False):
        """Rename this node in place.

        Changes the name of a node to *newname* (a string).  If a node with the
        same newname already exists and overwrite is true, recursively remove
        it before renaming.

        """

        self._f_move(newname=newname, overwrite=overwrite)

    def _f_move(self, newparent=None, newname=None,
                overwrite=False, createparents=False):
        """Move or rename this node.

        Moves a node into a new parent group, or changes the name of the
        node. newparent can be a Group object (see :ref:`GroupClassDescr`) or a
        pathname in string form. If it is not specified or None, the current
        parent group is chosen as the new parent.  newname must be a string
        with a new name. If it is not specified or None, the current name is
        chosen as the new name. If createparents is true, the needed groups for
        the given new parent group path to exist will be created.

        Moving a node across databases is not allowed, nor it is moving a node
        *into* itself. These result in a NodeError. However, moving a node
        *over* itself is allowed and simply does nothing. Moving over another
        existing node is similarly not allowed, unless the optional overwrite
        argument is true, in which case that node is recursively removed before
        moving.

        Usually, only the first argument will be used, effectively moving the
        node to a new location without changing its name.  Using only the
        second argument is equivalent to renaming the node in place.

        """

        self._g_check_open()
        file_ = self._v_file
        oldparent = self._v_parent
        oldname = self._v_name

        # Set default arguments.
        if newparent is None and newname is None:
            raise NodeError("you should specify at least "
                            "a ``newparent`` or a ``newname`` parameter")
        if newparent is None:
            newparent = oldparent
        if newname is None:
            newname = oldname

        # Get destination location.
        if hasattr(newparent, '_v_file'):  # from node
            newfile = newparent._v_file
            newpath = newparent._v_pathname
        elif hasattr(newparent, 'startswith'):  # from path
            newfile = file_
            newpath = newparent
        else:
            raise TypeError("new parent is not a node nor a path: %r"
                            % (newparent,))

        # Validity checks on arguments.
        # Is it in the same file?
        if newfile is not file_:
            raise NodeError("nodes can not be moved across databases; "
                            "please make a copy of the node")

        # The movement always fails if the hosting file can not be modified.
        file_._check_writable()

        # Moving over itself?
        oldpath = oldparent._v_pathname
        if newpath == oldpath and newname == oldname:
            # This is equivalent to renaming the node to its current name,
            # and it does not change the referenced object,
            # so it is an allowed no-op.
            return

        # Moving into itself?
        self._g_check_not_contains(newpath)

        # Note that the previous checks allow us to go ahead and create
        # the parent groups if `createparents` is true.  `newparent` is
        # used instead of `newpath` to avoid accepting `Node` objects
        # when `createparents` is true.
        newparent = file_._get_or_create_path(newparent, createparents)
        self._g_check_group(newparent)  # Is it a group?

        # Moving over an existing node?
        self._g_maybe_remove(newparent, newname, overwrite)

        # Move the node.
        oldpathname = self._v_pathname
        self._g_move(newparent, newname)

        # Log the change.
        if file_.is_undo_enabled():
            self._g_log_move(oldpathname)

    def _g_log_move(self, oldpathname):
        self._v_file._log('MOVE', oldpathname, self._v_pathname)

    _g_logMove = previous_api(_g_log_move)

    def _g_copy(self, newparent, newname, recursive, _log=True, **kwargs):
        """Copy this node and return the new one.

        Creates and returns a copy of the node in the given `newparent`,
        with the given `newname`.  If `recursive` copy is stated, all
        descendents are copied as well.  Additional keyword argumens may
        affect the way that the copy is made.  Unknown arguments must be
        ignored.  On recursive copies, all keyword arguments must be
        passed on to the children invocation of this method.

        If `_log` is false, the change is not logged.  This is *only*
        intended to be used by ``_g_copy_as_child()`` as a means of
        optimising sub-tree copies.

        """

        raise NotImplementedError

    def _g_copy_as_child(self, newparent, **kwargs):
        """Copy this node as a child of another group.

        Copies just this node into `newparent`, not recursing children
        nor overwriting nodes nor logging the copy.  This is intended to
        be used when copying whole sub-trees.

        """

        return self._g_copy(newparent, self._v_name,
                            recursive=False, _log=False, **kwargs)

    _g_copyAsChild = previous_api(_g_copy_as_child)

    def _f_copy(self, newparent=None, newname=None,
                overwrite=False, recursive=False, createparents=False,
                **kwargs):
        """Copy this node and return the new node.

        Creates and returns a copy of the node, maybe in a different place in
        the hierarchy. newparent can be a Group object (see
        :ref:`GroupClassDescr`) or a pathname in string form. If it is not
        specified or None, the current parent group is chosen as the new
        parent.  newname must be a string with a new name. If it is not
        specified or None, the current name is chosen as the new name. If
        recursive copy is stated, all descendants are copied as well. If
        createparents is true, the needed groups for the given new parent group
        path to exist will be created.

        Copying a node across databases is supported but can not be
        undone. Copying a node over itself is not allowed, nor it is
        recursively copying a node into itself. These result in a
        NodeError. Copying over another existing node is similarly not allowed,
        unless the optional overwrite argument is true, in which case that node
        is recursively removed before copying.

        Additional keyword arguments may be passed to customize the copying
        process. For instance, title and filters may be changed, user
        attributes may be or may not be copied, data may be sub-sampled, stats
        may be collected, etc. See the documentation for the particular node
        type.

        Using only the first argument is equivalent to copying the node to a
        new location without changing its name. Using only the second argument
        is equivalent to making a copy of the node in the same group.

        """

        self._g_check_open()
        srcfile = self._v_file
        srcparent = self._v_parent
        srcname = self._v_name

        dstparent = newparent
        dstname = newname

        # Set default arguments.
        if dstparent is None and dstname is None:
            raise NodeError("you should specify at least "
                            "a ``newparent`` or a ``newname`` parameter")
        if dstparent is None:
            dstparent = srcparent
        if dstname is None:
            dstname = srcname

        # Get destination location.
        if hasattr(dstparent, '_v_file'):  # from node
            dstfile = dstparent._v_file
            dstpath = dstparent._v_pathname
        elif hasattr(dstparent, 'startswith'):  # from path
            dstfile = srcfile
            dstpath = dstparent
        else:
            raise TypeError("new parent is not a node nor a path: %r"
                            % (dstparent,))

        # Validity checks on arguments.
        if dstfile is srcfile:
            # Copying over itself?
            srcpath = srcparent._v_pathname
            if dstpath == srcpath and dstname == srcname:
                raise NodeError(
                    "source and destination nodes are the same node: ``%s``"
                    % self._v_pathname)

            # Recursively copying into itself?
            if recursive:
                self._g_check_not_contains(dstpath)

        # Note that the previous checks allow us to go ahead and create
        # the parent groups if `createparents` is true.  `dstParent` is
        # used instead of `dstPath` because it may be in other file, and
        # to avoid accepting `Node` objects when `createparents` is
        # true.
        dstparent = srcfile._get_or_create_path(dstparent, createparents)
        self._g_check_group(dstparent)  # Is it a group?

        # Copying to another file with undo enabled?
        if dstfile is not srcfile and srcfile.is_undo_enabled():
            warnings.warn("copying across databases can not be undone "
                          "nor redone from this database",
                          UndoRedoWarning)

        # Copying over an existing node?
        self._g_maybe_remove(dstparent, dstname, overwrite)

        # Copy the node.
        # The constructor of the new node takes care of logging.
        return self._g_copy(dstparent, dstname, recursive, **kwargs)

    def _f_isvisible(self):
        """Is this node visible?"""

        self._g_check_open()
        return isvisiblepath(self._v_pathname)

    _f_isVisible = previous_api(_f_isvisible)

    def _g_check_group(self, node):
        # Node must be defined in order to define a Group.
        # However, we need to know Group here.
        # Using class_name_dict avoids a circular import.
        if not isinstance(node, class_name_dict['Node']):
            raise TypeError("new parent is not a registered node: %s"
                            % node._v_pathname)
        if not isinstance(node, class_name_dict['Group']):
            raise TypeError("new parent node ``%s`` is not a group"
                            % node._v_pathname)

    _g_checkGroup = previous_api(_g_check_group)

    def _g_check_not_contains(self, pathname):
        # The not-a-TARDIS test. ;)
        mypathname = self._v_pathname
        if (mypathname == '/'  # all nodes fall below the root group
           or pathname == mypathname
           or pathname.startswith(mypathname + '/')):
            raise NodeError("can not move or recursively copy node ``%s`` "
                            "into itself" % mypathname)

    _g_checkNotContains = previous_api(_g_check_not_contains)

    def _g_maybe_remove(self, parent, name, overwrite):
        if name in parent:
            if not overwrite:
                raise NodeError("""\
destination group ``%s`` already has a node named ``%s``; \
you may want to use the ``overwrite`` argument""" % (parent._v_pathname, name))
            parent._f_get_child(name)._f_remove(True)

    _g_maybeRemove = previous_api(_g_maybe_remove)

    def _g_check_name(self, name):
        """Check validity of name for this particular kind of node.

        This is invoked once the standard HDF5 and natural naming checks
        have successfully passed.

        """

        if name.startswith('_i_'):
            # This is reserved for table index groups.
            raise ValueError(
                "node name starts with reserved prefix ``_i_``: %s" % name)

    _g_checkName = previous_api(_g_check_name)

    # <attribute handling>
    def _f_getattr(self, name):
        """Get a PyTables attribute from this node.

        If the named attribute does not exist, an AttributeError is raised.

        """

        return getattr(self._v_attrs, name)

    _f_getAttr = previous_api(_f_getattr)

    def _f_setattr(self, name, value):
        """Set a PyTables attribute for this node.

        If the node already has a large number of attributes, a
        PerformanceWarning is issued.

        """

        setattr(self._v_attrs, name, value)

    _f_setAttr = previous_api(_f_setattr)

    def _f_delattr(self, name):
        """Delete a PyTables attribute from this node.

        If the named attribute does not exist, an AttributeError is raised.

        """

        delattr(self._v_attrs, name)

    _f_delAttr = previous_api(_f_delattr)

    # </attribute handling>


class NotLoggedMixin:
    # Include this class in your inheritance tree
    # to avoid changes to instances of your class from being logged.

    _AttributeSet = NotLoggedAttributeSet

    def _g_log_create(self):
        pass

    _g_logCreate = previous_api(_g_log_create)

    def _g_log_move(self, oldpathname):
        pass

    _g_logMove = previous_api(_g_log_move)

    def _g_remove_and_log(self, recursive, force):
        self._g_remove(recursive, force)

    _g_removeAndLog = previous_api(_g_remove_and_log)


## Local Variables:
## mode: python
## py-indent-offset: 4
## tab-width: 4
## fill-column: 72
## End:
