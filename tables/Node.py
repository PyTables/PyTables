########################################################################
#
#       License: BSD
#       Created: February 11, 2005
#       Author:  Ivan Vilata - reverse:com.carabos@ivilata
#
#       $Source$
#       $Id$
#
########################################################################

"""
Base class for PyTables nodes.

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

from tables.constants import MAX_TREE_DEPTH
from tables.registry import classNameDict, classIdDict
from tables.exceptions import NodeError, UndoRedoWarning, PerformanceWarning
from tables.undoredo import moveToShadow
from tables.AttributeSet import AttributeSet



__docformat__ = 'reStructuredText'
"""The format of documentation strings in this module."""

__version__ = '$Revision$'
"""Repository version of this file."""



class MetaNode(type):

    """
    Node metaclass.

    This metaclass ensures that their instance classes get registered
    into several dictionaries (namely the `tables.utils.classNameDict`
    class name dictionary and the `tables.utils.classIdDict` class
    identifier dictionary).
    """

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
    the file.  When using a translation map (see the `File` class), its
    *HDF5 name* might differ from its PyTables name.

    All the previous information is location-dependent, i.e. it may
    change when moving or renaming a node in the hierarchy.  A node also
    has location-independent information, such as its *HDF5 object
    identifier* and its *attribute set*.

    This class gathers the operations and attributes (both
    location-dependent and independent) which are common to all PyTables
    nodes, whatever their type is.  Nonetheless, due to natural naming
    restrictions, the names of all of these members start with a
    reserved prefix (see the `Group` class).

    Sub-classes with no children (i.e. leaf nodes) may define new
    methods, attributes and properties to avoid natural naming
    restrictions.  For instance, ``_v_attrs`` may be shortened to
    ``attrs`` and ``_f_rename`` to ``rename``.  However, the original
    methods and attributes should still be available.

    Instance variables (location dependent):

    _v_file
        The hosting `File` instance.
    _v_parent
        The parent `Group` instance.
    _v_depth
        The depth of this node in the tree (an non-negative integer
        value).
    _v_name
        The name of this node in its parent group (a string).
    _v_hdf5name
        The name of this node in the hosting HDF5 file (a string).
    _v_pathname
        The path name of this node in the tree (a string).
    _v_rootgroup
        The root group instance.  This is deprecated; please use
        ``node._v_file.root``.

    Instance variables (location independent):

    _v_objectID
        The identifier of this node in the hosting HDF5 file.
    _v_attrs
        The associated `AttributeSet` instance.

    Instance variables (attribute shorthands):

    _v_title
        A description of this node.  A shorthand for the ``TITLE``
        attribute.

    Public methods (hierarchy manipulation):

    _f_close()
        Close this node in the tree.
    _f_remove([recursive])
        Remove this node from the hierarchy.
    _f_rename(newname)
        Rename this node in place.
    _f_move([newparent][, newname][, overwrite])
        Move or rename this node.
    _f_copy([newparent][, newname][, overwrite][, recursive][, **kwargs])
        Copy this node and return the new one.

    Public methods (attribute handling):

    _f_getAttr(name)
        Get a PyTables attribute from this node.
    _f_setAttr(name, value)
        Set a PyTables attribute for this node.
    _f_delAttr(name)
        Delete a PyTables attribute from this node.
    """

    # This makes this class and all derived subclasses be handled by MetaNode.
    __metaclass__ = MetaNode


    # <undo-redo support>
    _c_canUndoCreate = False  # Can creation/copying be undone and redone?
    _c_canUndoRemove = False  # Can removal be undone and redone?
    _c_canUndoMove   = False  # Can movement/renaming be undone and redone?
    # </undo-redo support>


    # <properties>

    # '_v_rootgroup' is deprecated in favour of 'node._v_file.root'.
    def _g_getrootgroup(self):
        warnings.warn(
            "``node._v_rootgroup`` is deprecated; please use ``node._v_file.root``",
            DeprecationWarning)
        return self._v_file.root

    _v_rootgroup = property(
        _g_getrootgroup, None, None, "The root group instance.")


    # '_v_attrs' is defined as a lazy read-only attribute.
    # This saves 0.7s/3.8s (at least for leaf nodes).
    def _g_getattrs(self):
        mydict = self.__dict__
        if '_v_attrs' in mydict:
            return mydict['_v_attrs']
        else:
            mydict['_v_attrs'] = attrs = AttributeSet(self)
            return attrs

    _v_attrs = property(_g_getattrs, None, None,
                        "The associated `AttributeSet` instance.")


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


    def _g_setLocation(self, parent, ptname, h5name=None):
        """
        Set location-dependent attributes.

        Sets the location-dependent attributes of this node to reflect
        that it is placed under the specified `parent` node, with the
        specified PyTables and HDF5 names (`ptname` and `h5name`,
        respectively).  If the HDF5 name is ``None``, it is found using
        the translation map from the parent's file.

        This also triggers the insertion of file references to this
        node.  If the maximum recommended node depth is exceeded, a
        `PerformanceWarning` is issued.
        """

        file_ = parent._v_file
        parentDepth = parent._v_depth

        mydict = self.__dict__
        mydict['_v_file'] = file_

        if h5name is None:
            h5name = file_.trMap.get(ptname, ptname)
        mydict['_v_name'] = ptname
        mydict['_v_hdf5name'] = h5name

        mydict['_v_parent'] = parent
        mydict['_v_pathname'] = parent._g_join(ptname)
        mydict['_v_depth'] = parentDepth + 1

        # Check if the node is too deep in the tree.
        if parentDepth >= MAX_TREE_DEPTH:
            warnings.warn("""\
node ``%s`` is exceeding the recommended maximum depth (%d);\
be ready to see PyTables asking for *lots* of memory and possibly slow I/O"""
                          % (self._v_pathname, MAX_TREE_DEPTH),
                          PerformanceWarning)

        file_._refNode(self, self._v_pathname)


    def _g_updateLocation(self):
        """
        Update location-dependent attributes.

        Updates location data when an ancestor node has changed its
        location in the hierarchy.  In fact, this method is expected to
        be called by an ancestor of this node.

        This also triggers the update of file references to this node.
        If the maximum recommended node depth is exceeded, a
        `PerformanceWarning` is issued.  This warning is assured to be
        unique.
        """

        file_ = self._v_file
        parent = self._v_parent
        parentDepth = parent._v_depth
        oldPathname = self._v_pathname
        newPathname = parent._g_join(self._v_name)

        mydict = self.__dict__
        mydict['_v_pathname'] = newPathname
        mydict['_v_depth'] = parentDepth + 1

        # Check if the node is too deep in the tree.
        if parentDepth >= MAX_TREE_DEPTH:
            warnings.warn("""\
moved descendent node is exceeding the recommended maximum depth (%d);\
be ready to see PyTables asking for *lots* of memory and possibly slow I/O"""
                          % (MAX_TREE_DEPTH,), PerformanceWarning)

        file_._unrefNode(oldPathname)
        file_._refNode(self, newPathname)


    def _g_delLocation(self):
        """
        Clear location-dependent attributes.

        This also triggers the removal of file references to this node.
        """

        file_ = self._v_file
        pathname = self._v_pathname

        mydict = self.__dict__
        mydict['_v_file'] = None

        mydict['_v_name'] = None
        mydict['_v_hdf5name'] = None

        mydict['_v_parent'] = None
        mydict['_v_pathname'] = None
        mydict['_v_depth'] = None

        file_._unrefNode(pathname)


    def _f_close(self):
        """
        Close this node in the tree.

        This makes the node inaccessible from the object tree.  The
        closing operation is *not* recursive, i.e. closing a group does
        not close its children.  On nodes with data, it may flush it to
        disk.
        """
        raise NotImplementedError


    def _g_remove(self, recursive):
        """
        Remove this node from the hierarchy.

        If the node has children, recursive removal must be stated by
        giving `recursive` a true value; otherwise, a `NodeError` will
        be raised.

        It does not log the change.
        """

        raise NotImplementedError


    def _f_remove(self, recursive=False):
        """
        Remove this node from the hierarchy.

        If the node has children, recursive removal must be stated by
        giving `recursive` a true value, or a `NodeError` will be
        raised.
        """

        file_ = self._v_file

        if file_.isUndoEnabled():
            if self._c_canUndoMove:
                oldPathname = self._v_pathname
                # Log *before* moving to use the right shadow name.
                file_._log('REMOVE', oldPathname)
                moveToShadow(file_, oldPathname)
            else:
                warnings.warn(
                    "removal can not be undone nor redone for this node",
                    UndoRedoWarning)
                self._g_remove(recursive)
        else:
            self._g_remove(recursive)


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
        self._g_new(newParent, self._v_hdf5name)
        #   Move the node.
        self._v_parent._g_moveNode(oldPathname, self._v_pathname)
        #   Update attribute set attributes.
        self._v_attrs._g_new(self)


    def _f_rename(self, newname):
        """
        Rename this node in place.

        Changes the name of a node to `newname` (a string).
        """
        self._f_move(newname = newname)


    def _f_move(self, newparent=None, newname=None, overwrite=False):
        """
        Move or rename this node.

        Moves a node into a new parent group, or changes the name of the
        node.  `newparent` can be a `Group` object or a pathname in
        string form.  If it is not specified or ``None` , the current
        parent group is chosen as the new parent.  `newname` must be a
        string with a new name.  If it is not specified or ``None``, the
        current name is chosen as the new name.

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

        file_ = self._v_file
        oldParent = self._v_parent
        oldName = self._v_name

        # Set default arguments.
        if newparent is None and newname is None:
            raise NodeError("""\
you should specify at least a ``newparent`` or a ``newname`` parameter""")
        if newparent is None:
            newparent = oldParent
        if newname is None:
            newname = oldName

        # Validity checks on arguments.
        newparent = file_.getNode(newparent)  # Does the new parent exist?
        self._g_checkGroup(newparent)  # Is it a group?

        if newparent._v_file is not file_:  # Is it in the same file?
            raise NodeError("""\
nodes can not be moved across databases; please make a copy of the node""")

        # Moving over itself?
        if (newparent is oldParent) and (newname == oldName):
            # This is equivalent to renaming the node to its current name,
            # and it does not change the referenced object,
            # so it is an allowed no-op.
            return

        self._g_checkNotContains(newparent)  # Moving into itself?
        self._g_maybeRemove(  # Moving over an existing node?
            newparent, newname, overwrite)

        undoEnabled = file_.isUndoEnabled()
        canUndoMove = self._c_canUndoMove
        if undoEnabled and not canUndoMove:
            warnings.warn(
                "movement can not be undone nor redone for this node",
                UndoRedoWarning)

        # Move the node.
        oldPathname = self._v_pathname
        self._g_move(newparent, newname)
        newPathname = self._v_pathname

        # Log the change.
        if undoEnabled and canUndoMove:
            file_._log('MOVE', oldPathname, newPathname)


    def _g_copy(self, newParent, newName, recursive, **kwargs):
        """
        Copy this node and return the new one.

        Creates and returns a copy of the node in the given `newParent`,
        with the given `newName`.  If `recursive` copy is stated, all
        descendents are copied as well.  Additional keyword argumens may
        affect the way that the copy is made.  Unknown arguments must be
        ignored.  On recursive copies, all keyword arguments must be
        passed on to the children invocation of this method.

        It does not log the change.
        """
        raise NotImplementedError


    def _f_copy(self, newparent=None, newname=None,
                overwrite=False, recursive=False, **kwargs):
        """
        Copy this node and return the new one.

        Creates and returns a copy of the node, maybe in a different
        place in the hierarchy.  `newparent` can be a `Group` object or
        a pathname in string form.  If it is not specified or ``None``,
        the current parent group is chosen as the new parent.  `newname`
        must be a string with a new name.  If it is not specified or
        ``None``, the current name is chosen as the new name.  If
        `recursive` copy is stated, all descendents are copied as well.

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

        file_ = self._v_file
        oldParent = self._v_parent
        oldName = self._v_name

        # Set default arguments.
        if newparent is None and newname is None:
            raise NodeError("""\
you should specify at least a ``newparent`` or a ``newname`` parameter""")
        if newparent is None:
            newparent = oldParent
        if newname is None:
            newname = oldName

        # Validity checks on arguments.
        newparent = file_.getNode(newparent)  # Does the new parent exist?
        self._g_checkGroup(newparent)  # Is it a group?

        dolog = True  # Is it in the same file?
        if newparent._v_file is not file_ and file_.isUndoEnabled():
            warnings.warn("""\
copying across databases can not be undone nor redone from this database""",
                          UndoRedoWarning)
            dolog = False

        # Copying over itself?
        if (newparent is oldParent) and (newname == oldName):
            raise NodeError(
                "source and destination nodes are the same node: ``%s``"
                % (self._v_pathname,))

        if recursive:
            self._g_checkNotContains(newparent)  # Copying into itself?
        self._g_maybeRemove(  # Copying over an existing node?
            newparent, newname, overwrite)

        undoEnabled = file_.isUndoEnabled()
        canUndoCreate = self._c_canUndoCreate
        if undoEnabled and not canUndoCreate:
            warnings.warn(
                "copying can not be undone nor redone for this node",
                UndoRedoWarning)

        # Copy the node.
        newNode = self._g_copy(newparent, newname, recursive, **kwargs)

        # Log the change.
        if dolog and undoEnabled and canUndoCreate:
            file_._log('CREATE', newNode._v_pathname)

        return newNode


    def _g_checkGroup(self, node):
        # Node must be defined in order to define a Group.
        # However, we need to know Group here.
        # Using classNameDict avoids a circular import.
        if not isinstance(node, classNameDict['Group']):
            raise TypeError(
                "new parent node is not a group: %r" % (node,))


    def _g_checkNotContains(self, node):
        # The not-a-TARDIS test. ;)
        if node is self or node._g_isDescendentOf(self):
            raise NodeError(
                "can not move or recursively copy node ``%s`` into itself"
                % (self._v_pathname,))


    def _g_maybeRemove(self, parent, name, overwrite):
        if name in parent:
            if not overwrite:
                raise NodeError("""\
destination group ``%s`` already has a node named ``%s``; \
you may want to use the ``overwrite`` argument""" % (parent._v_pathname, name))
            getattr(parent, name)._f_remove(True)


    def _g_isDescendentOf(self, group):
        file_ = self._v_file

        # If the nodes are in different files its is not worth
        # climibing up the hierarchy.
        if file_ is not group._v_file:
            return False

        parent = self._v_parent
        while parent is not file_:
            if parent is group:
                return True
            parent = parent._v_parent
        return False


    def _g_putUnder(self, parent, name):
        """
        Put node in hierarchy.

        Puts this node (whether newly created or already existing in an
        HDF5 file) in the hierarchy, under the specified `parent` group,
        with the specified `name`.  Please note that the given `name`
        has a different meaning depending on when the node was created.

        If the node has been created in this session, the provided
        `name` is assumed to be its PyTables name, and its HDF5 name is
        found using the translation map from the parent's file.  Else,
        the provided `name` is assumed to be its HDF5 name, and its
        PyTables name is found using the translation map from the
        parent's file.
        """

        # All this will eventually end up in the node constructor.

        new = self._v_new

        # Find out the PyTables and HDF5 names.
        trMap = parent._v_file.trMap
        if new:
            # New node: get HDF5 name from PyTables name.
            ptname = name  # always the provided one
            h5name = trMap.get(name, name)
        else:
            # Opened node: get PyTables name from HDF5 name.
            h5name = name  # always the provided one
            ptname = name
            # This code might seem inefficient but it will be rarely used.
            for (ptname_, h5name_) in trMap.iteritems():
                if h5name_ == h5name:
                    ptname = ptname_
                    break

        parent._g_refNode(self, ptname, new)
        self._g_setLocation(parent, ptname, h5name)

        # hdf5Extension operations:
        #   Update node attributes.
        self._g_new(parent, h5name)


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



## Local Variables:
## mode: python
## py-indent-offset: 4
## tab-width: 4
## fill-column: 72
## End:
