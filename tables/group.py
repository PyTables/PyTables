########################################################################
#
#       License: BSD
#       Created: September 4, 2002
#       Author:  Francesc Alted - faltet@pytables.com
#
#       $Id$
#
########################################################################

"""Here is defined the Group class.

See Group class docstring for more info.

Classes:

    Group
    RootGroup
    TransactionGroupG
    TransactionG
    MarkG

Functions:


Misc variables:

    __version__

"""

import warnings
import weakref

import tables.misc.proxydict
from tables import hdf5Extension
from tables import utilsExtension
from tables.registry import classIdDict
from tables.exceptions import (
    NodeError, NoSuchNodeError, NaturalNameWarning, PerformanceWarning,
    HDF5ExtError)
from tables.filters import Filters
from tables.registry import getClassByName
from tables.path import checkNameValidity, joinPath, isVisibleName
from tables.node import Node, NotLoggedMixin
from tables.leaf import Leaf
from tables.unimplemented import UnImplemented, Unknown
from tables.attributeset import AttributeSet

from tables.link import Link, SoftLink
try:
    from tables.link import ExternalLink
except ImportError:
    are_extlinks_available = False
else:
    are_extlinks_available = True


__version__ = "$Revision$"

obversion = "1.0"



class _ChildrenDict(tables.misc.proxydict.ProxyDict):
    def _getValueFromContainer(self, container, key):
        return container._f_getChild(key)



class Group(hdf5Extension.Group, Node):
    """
    Basic PyTables grouping structure.

    Instances of this class are grouping structures containing *child*
    instances of zero or more groups or leaves, together with supporting
    metadata.  Each group has exactly one *parent* group.

    Working with groups and leaves is similar in many ways to working
    with directories and files, respectively, in a Unix filesystem.  As
    with Unix directories and files, objects in the object tree are
    often described by giving their full (or absolute) path names.  This
    full path can be specified either as a string (like in
    '/group1/group2') or as a complete object path written in *natural
    naming* schema (like in ``file.root.group1.group2``).

    A collateral effect of the *natural naming* schema is that the names
    of members in the ``Group`` class and its instances must be
    carefully chosen to avoid colliding with existing children node
    names.  For this reason and to avoid polluting the children
    namespace all members in a ``Group`` start with some reserved
    prefix, like ``_f_`` (for public methods), ``_g_`` (for private
    ones), ``_v_`` (for instance variables) or ``_c_`` (for class
    variables). Any attempt to create a new child node whose name starts
    with one of these prefixes will raise a ``ValueError`` exception.

    Another effect of natural naming is that children named after Python
    keywords or having names not valid as Python identifiers (e.g.
    ``class``, ``$a`` or ``44``) can not be accessed using the
    ``node.child`` syntax.  You will be forced to use
    ``node._f_getChild(child)`` to access them (which is recommended for
    programmatic accesses).

    You will also need to use ``_f_getChild()`` to access an existing
    child node if you set a Python attribute in the ``Group`` with the
    same name as that node (you will get a `NaturalNameWarning` when
    doing this).

    Public instance variables
    -------------------------

    The following instance variables are provided in addition to those
    in `Node`:

    _v_children
        Dictionary with all nodes hanging from this group.
    _v_filters
        Default filter properties for child nodes.

        You can (and are encouraged to) use this property to get, set
        and delete the ``FILTERS`` HDF5 attribute of the group, which
        stores a `Filters` instance.  When the group has no such
        attribute, a default `Filters` instance is used.

    _v_groups
        Dictionary with all groups hanging from this group.
    _v_hidden
        Dictionary with all hidden nodes hanging from this group.
    _v_leaves
        Dictionary with all leaves hanging from this group.
    _v_links
        Dictionary with all links hanging from this group.
    _v_nchildren
        The number of children hanging from this group.
    _v_unknown
        Dictionary with all unknown nodes hanging from this group.

    Public methods
    --------------

    .. admonition:: Caveat

       The following methods are documented for completeness, and they
       can be used without any problem.  However, you should use the
       high-level counterpart methods in the `File` class, because they
       are most used in documentation and examples, and are a bit more
       powerful than those exposed here.

    The following methods are provided in addition to those in `Node`:

    * _f_close()
    * _f_copy([newparent][, newname][, overwrite][, recursive][, createparents][, **kwargs])
    * _f_copyChildren(dstgroup[, overwrite][, recursive][, createparents][, **kwargs])
    * _f_getChild(childname)
    * _f_iterNodes([classname])
    * _f_listNodes([classname])
    * _f_walkGroups()
    * _f_walkNodes([classname][, recursive])

    Special methods
    ---------------

    Following are described the methods that automatically trigger
    actions when a ``Group`` instance is accessed in a special way.

    This class defines the ``__setattr__``, ``__getattr__`` and
    ``__delattr__`` methods, and they set, get and delete *ordinary
    Python attributes* as normally intended.  In addition to that,
    ``__getattr__`` allows getting *child nodes* by their name for the
    sake of easy interaction on the command line, as long as there is no
    Python attribute with the same name.  Groups also allow the
    interactive completion (when using ``readline``) of the names of
    child nodes. For instance::

        nchild = group._v_nchildren  # get a Python attribute

        # Add a Table child called 'table' under 'group'.
        h5file.createTable(group, 'table', myDescription)

        table = group.table          # get the table child instance
        group.table = 'foo'          # set a Python attribute
        # (PyTables warns you here about using the name of a child node.)
        foo = group.table            # get a Python attribute
        del group.table              # delete a Python attribute
        table = group.table          # get the table child instance again

    * __contains__(name)
    * __delattr__(name)
    * __getattr__(name)
    * __iter__()
    * __repr__()
    * __setattr__(name, value)
    * __str__()
    """

    # Class identifier.
    _c_classId = 'GROUP'

    # Children containers that should be loaded only in a lazy way.
    # These are documented in the ``Group._g_addChildrenNames`` method.
    _c_lazy_children_attrs = (
        '__members__', '_v_children', '_v_groups', '_v_leaves',
        '_v_links', '_v_unknown', '_v_hidden')

    # <properties>

    # `_v_nchildren` is a direct read-only shorthand
    # for the number of *visible* children in a group.
    def _g_getnchildren(self):
        return len(self._v_children)

    _v_nchildren = property(_g_getnchildren, None, None,
                            "The number of children hanging from this group.")


    # `_v_filters` is a direct read-write shorthand for the ``FILTERS``
    # attribute with the default `Filters` instance as a default value.
    def _g_getfilters(self):
        filters = getattr(self._v_attrs, 'FILTERS', None)
        if filters is None:
            filters = Filters()
        return filters

    def _g_setfilters(self, value):
        if not isinstance(value, Filters):
            raise TypeError(
                "value is not an instance of `Filters`: %r" % (value,))
        self._v_attrs.FILTERS = value

    def _g_delfilters(self):
        del self._v_attrs.FILTERS

    _v_filters = property(
        _g_getfilters, _g_setfilters, _g_delfilters,
        """
        Default filter properties for child nodes.

        You can (and are encouraged to) use this property to get, set
        and delete the ``FILTERS`` HDF5 attribute of the group, which
        stores a `Filters` instance.  When the group has no such
        attribute, a default `Filters` instance is used.
        """ )

    # </properties>


    def __init__(self, parentNode, name,
                 title="", new=False, filters=None,
                 _log=True):
        """Create the basic structures to keep group information.

        title -- The title for this group
        new -- If this group is new or has to be read from disk
        filters -- A Filters instance

        """

        # Remember to assign these values in the root group constructor
        # if it does not use this one!

        # First, set attributes belonging to group objects.

        self._v_version = obversion
        """The object version of this group."""

        self._v_new = new
        """Is this the first time the node has been created?"""
        self._v_new_title = title
        """New title for this node."""
        self._v_new_filters = filters
        """New default filter properties for child nodes."""
        self._v_maxGroupWidth = parentNode._v_file.params['MAX_GROUP_WIDTH']
        """Maximum number of children on each group before warning the user."""

        # Finally, set up this object as a node.
        super(Group, self).__init__(parentNode, name, _log)


    def _g_postInitHook(self):
        if self._v_new:
            if self._v_file.params['PYTABLES_SYS_ATTRS']:
                # Save some attributes for the new group on disk.
                setAttr = self._v_attrs._g__setattr
                # Set the title, class and version attributes.
                setAttr('TITLE', self._v_new_title)
                setAttr('CLASS', self._c_classId)
                setAttr('VERSION', self._v_version)

                # Set the default filter properties.
                newFilters = self._v_new_filters
                if newFilters is None:
                    # If no filters have been passed in the constructor,
                    # inherit them from the parent group, but only if they
                    # have been inherited or explicitly set.
                    newFilters = getattr(
                        self._v_parent._v_attrs, 'FILTERS', None)
                if newFilters is not None:
                    setAttr('FILTERS', newFilters)
        else:
            # If the file has PyTables format, get the VERSION attr
            if 'VERSION' in self._v_attrs._v_attrnamessys:
                self._v_version = self._v_attrs.VERSION
            else:
                self._v_version = "0.0 (unknown)"
            # We don't need to get more attributes from disk,
            # since the most important ones are defined as properties.


    def __del__(self):
        if (self._v_isopen and
            self._v_pathname in self._v_file._aliveNodes and
            '_v_children' in self.__dict__):
            # The group is going to be killed.  Rebuild weak references
            # (that Python cancelled just before calling this method) so
            # that they are still usable if the object is revived later.
            selfRef = weakref.ref(self)
            self._v_children.containerRef = selfRef
            self._v_groups.containerRef = selfRef
            self._v_leaves.containerRef = selfRef
            self._v_links.containerRef = selfRef
            self._v_unknown.containerRef = selfRef
            self._v_hidden.containerRef = selfRef

        super(Group, self).__del__()


    def _g_getChildGroupClass(self, childName):
        """
        Get the class of a not-yet-loaded group child.

        `childName` must be the name of a *group* child.
        """

        childCID = self._g_getGChildAttr(childName, 'CLASS')

        if childCID in classIdDict:
            return classIdDict[childCID]  # look up group class
        else:
            return Group  # default group class


    def _g_getChildLeafClass(self, childName, warn=True):
        """
        Get the class of a not-yet-loaded leaf child.

        `childName` must be the name of a *leaf* child.  If the child
        belongs to an unknown kind of leaf, or if its kind can not be
        guessed, `UnImplemented` will be returned and a warning will be
        issued if `warn` is true.
        """

        if self._v_file.params['PYTABLES_SYS_ATTRS']:
            childCID = self._g_getLChildAttr(childName, 'CLASS')
        else:
            childCID = None

        if childCID in classIdDict:
            return classIdDict[childCID]  # look up leaf class
        else:
            # Unknown or no ``CLASS`` attribute, try a guess.
            childCID2 = utilsExtension.whichClass(
                self._v_objectID, childName)
            if childCID2 == 'UNSUPPORTED':
                if warn:
                    if childCID is None:
                        warnings.warn(
                            "leaf ``%s`` is of an unsupported type; "
                            "it will become an ``UnImplemented`` node"
                            % self._g_join(childName))
                    else:
                        warnings.warn(
                            "leaf ``%s`` has an unknown class ID ``%s``; "
                            "it will become an ``UnImplemented`` node"""
                            % (self._g_join(childName), childCID))
                return UnImplemented
            assert childCID2 in classIdDict
            return classIdDict[childCID2]  # look up leaf class


    def _g_addChildrenNames(self):
        """
        Add children names to this group taking into account their
        visibility and kind.
        """

        myDict = self.__dict__

        # The names of the lazy attributes
        myDict['__members__'] = members = []
        """The names of visible children nodes for readline-style completion."""
        myDict['_v_children'] = children = _ChildrenDict(self)
        """The number of children hanging from this group."""
        myDict['_v_groups'] = groups = _ChildrenDict(self)
        """Dictionary with all groups hanging from this group."""
        myDict['_v_leaves'] = leaves = _ChildrenDict(self)
        """Dictionary with all leaves hanging from this group."""
        myDict['_v_links'] = links = _ChildrenDict(self)
        """Dictionary with all links hanging from this group."""
        myDict['_v_unknown'] = unknown = _ChildrenDict(self)
        """Dictionary with all unknown nodes hanging from this group."""
        myDict['_v_hidden'] = hidden = _ChildrenDict(self)
        """Dictionary with all hidden nodes hanging from this group."""

        # Get the names of *all* child groups and leaves.
        (groupNames, leafNames, linkNames, unknownNames) = \
                     self._g_listGroup(self._v_parent)

        # Separate groups into visible groups and hidden nodes,
        # and leaves into visible leaves and hidden nodes.
        for (childNames, childDict) in (
            (groupNames, groups),
            (leafNames, leaves),
            (linkNames, links),
            (unknownNames, unknown)):

            for childName in childNames:
                # See whether the name implies that the node is hidden.
                # (Assigned values are entirely irrelevant.)
                if isVisibleName(childName):
                    # Visible node.
                    members.insert(0, childName)
                    children[childName] = None
                    childDict[childName] = None
                else:
                    # Hidden node.
                    hidden[childName] = None


    def _g_checkHasChild(self, name):
        """Check whether 'name' is a children of 'self' and return its type. """
        # Get the HDF5 name matching the PyTables name.
        node_type = self._g_get_objinfo(name)
        if node_type == "NoSuchNode":
            raise NoSuchNodeError(
                "group ``%s`` does not have a child named ``%s``"
                % (self._v_pathname, name))
        return node_type


    def __iter__(self):
        """
        Iterate over child nodes hanging directly from the group.

        This iterator is *not* recursive.  Example of use::

            # Non-recursively list all the nodes hanging from '/detector'
            print \"Nodes in '/detector' group:\"
            for node in h5file.root.detector:
                print node
        """
        return self._f_iterNodes()


    def __contains__(self, name):
        """
        Is there a child with that `name`?

        Returns a true value if the group has a child node (visible or
        hidden) with the given `name` (a string), false otherwise.
        """
        self._g_checkOpen()
        try:
            self._g_checkHasChild(name)
        except NoSuchNodeError:
            return False
        return True


    def _f_walkNodes(self, classname=None):
        """
        Iterate over descendent nodes.

        This method recursively walks *self* top to bottom (preorder),
        iterating over child groups in alphanumerical order, and
        yielding nodes.  If `classname` is supplied, only instances of
        the named class are yielded.

        If `classname` is 'Group', it behaves like
        `Group._f_walkGroups()`, yielding only groups.  If you don't
        want a recursive behavior, use `Group._f_iterNodes()` instead.

        Example of use::

            # Recursively print all the arrays hanging from '/'
            print \"Arrays in the object tree '/':\"
            for array in h5file.root._f_walkNodes('Array'):
                print array
        """

        self._g_checkOpen()

        # For compatibility with old default arguments.
        if classname == '':
            classname = None

        if classname == "Group":
            # Recursive algorithm
            for group in self._f_walkGroups():
                yield group
        else:
            for group in self._f_walkGroups():
                for leaf in group._f_iterNodes(classname):
                    yield leaf


    def _g_join(self, name):
        """Helper method to correctly concatenate a name child object
        with the pathname of this group."""

        if name == "/":
            # This case can happen when doing copies
            return self._v_pathname
        return joinPath(self._v_pathname, name)


    def _g_widthWarning(self):
        """Issue a `PerformanceWarning` on too many children."""

        warnings.warn("""\
group ``%s`` is exceeding the recommended maximum number of children (%d); \
be ready to see PyTables asking for *lots* of memory and possibly slow I/O."""
                      % (self._v_pathname, self._v_maxGroupWidth),
                      PerformanceWarning)


    def _g_refNode(self, childNode, childName, validate=True):
        """
        Insert references to a `childNode` via a `childName`.

        Checks that the `childName` is valid and does not exist, then
        creates references to the given `childNode` by that `childName`.
        The validation of the name can be omitted by setting `validate`
        to a false value (this may be useful for adding already existing
        nodes to the tree).
        """

        # Check for name validity.
        if validate:
            checkNameValidity(childName)
            childNode._g_checkName(childName)

        # Check if there is already a child with the same name.
        # This can be triggered because of the user
        # (via node construction or renaming/movement).
        # Links are not checked here because they are copied and referenced
        # using ``File.getNode`` so they already exist in `self`.
        if (not isinstance(childNode, Link)) and childName in self:
            raise NodeError(
                "group ``%s`` already has a child node named ``%s``"
                % (self._v_pathname, childName))

        # Show a warning if there is an object attribute with that name.
        if childName in self.__dict__:
            warnings.warn(
                "group ``%s`` already has an attribute named ``%s``; "
                "you will not be able to use natural naming "
                "to access the child node"
                % (self._v_pathname, childName), NaturalNameWarning)

        # Check group width limits.
        if len(self._v_children) + len(self._v_hidden) >= self._v_maxGroupWidth:
            self._g_widthWarning()

        # Update members information.
        # Insert references to the new child.
        # (Assigned values are entirely irrelevant.)
        if isVisibleName(childName):
            # Visible node.
            self.__members__.insert(0, childName)  # enable completion
            self._v_children[childName] = None  # insert node
            if isinstance(childNode, Unknown):
                self._v_unknown[childName] = None
            elif isinstance(childNode, Link):
                self._v_links[childName] = None
            elif isinstance(childNode, Leaf):
                self._v_leaves[childName] = None
            elif isinstance(childNode, Group):
                self._v_groups[childName] = None
        else:
            # Hidden node.
            self._v_hidden[childName] = None  # insert node


    def _g_unrefNode(self, childName):
        """
        Remove references to a node.

        Removes all references to the named node.
        """

        # This can *not* be triggered because of the user.
        assert childName in self, \
               ("group ``%s`` does not have a child node named ``%s``"
                % (self._v_pathname, childName))

        # Update members information, if needed
        if '_v_children' in self.__dict__:
            if childName in self._v_children:
                # Visible node.
                members = self.__members__
                memberIndex = members.index(childName)
                del members[memberIndex]  # disables completion

                del self._v_children[childName]  # remove node
                self._v_unknown.pop(childName, None)
                self._v_links.pop(childName, None)
                self._v_leaves.pop(childName, None)
                self._v_groups.pop(childName, None)
            else:
                # Hidden node.
                del self._v_hidden[childName]  # remove node


    def _g_move(self, newParent, newName):
        # Move the node to the new location.
        oldPath = self._v_pathname
        super(Group, self)._g_move(newParent, newName)
        newPath = self._v_pathname

        # Update location information in children.  This node shouldn't
        # be affected since it has already been relocated.
        self._v_file._updateNodeLocations(oldPath, newPath)


    def _g_copy(self, newParent, newName, recursive, _log=True, **kwargs):
        # Compute default arguments.
        title = kwargs.get('title', self._v_title)
        filters = kwargs.get('filters', None)
        stats = kwargs.get('stats', None)

        # Fix arguments with explicit None values for backwards compatibility.
        if title is None:  title = self._v_title
        # If no filters have been passed to the call, copy them from the
        # source group, but only if inherited or explicitly set.
        if filters is None:
            filters = getattr(self._v_attrs, 'FILTERS', None)

        # Create a copy of the object.
        newNode = Group(newParent, newName,
                        title, new=True, filters=filters, _log=_log)

        # Copy user attributes if needed.
        if kwargs.get('copyuserattrs', True):
            self._v_attrs._g_copy(newNode._v_attrs, copyClass=True)

        # Update statistics if needed.
        if stats is not None:
            stats['groups'] += 1

        if recursive:
            # Copy child nodes if a recursive copy was requested.
            # Some arguments should *not* be passed to children copy ops.
            kwargs = kwargs.copy()
            kwargs.pop('title', None)
            self._g_copyChildren(newNode, **kwargs)

        return newNode


    def _g_copyChildren(self, newParent, **kwargs):
        """Copy child nodes.

        Copies all nodes descending from this one into the specified
        `newParent`.  If the new parent has a child node with the same
        name as one of the nodes in this group, the copy fails with a
        `NodeError`, maybe resulting in a partial copy.  Nothing is
        logged.
        """
        # Recursive version of children copy.
        ##for srcChild in self._v_children.itervalues():
        ##    srcChild._g_copyAsChild(newParent, **kwargs)

        # Non-recursive version of children copy.
        parentStack = [(self, newParent)]  # [(source, destination), ...]
        while parentStack:
            (srcParent, dstParent) = parentStack.pop()
            for srcChild in srcParent._v_children.itervalues():
                dstChild = srcChild._g_copyAsChild(dstParent, **kwargs)
                if isinstance(srcChild, Group):
                    parentStack.append((srcChild, dstChild))


    def _f_getChild(self, childname):
        """
        Get the child called `childname` of this group.

        If the child exists (be it visible or not), it is returned.
        Else, a `NoSuchNodeError` is raised.

        Using this method is recommended over ``getattr()`` when doing
        programmatic accesses to children if the `childname` is unknown
        beforehand or when its name is not a valid Python identifier.
        """

        self._g_checkOpen()

        self._g_checkHasChild(childname)

        childPath = joinPath(self._v_pathname, childname)
        return self._v_file._getNode(childPath)


    def _f_listNodes(self, classname=None):
        """
        Return a *list* with children nodes.

        This is a list-returning version of `Group._f_iterNodes()`.
        """
        return list(self._f_iterNodes(classname))


    def _f_iterNodes(self, classname=None):
        """
        Iterate over children nodes.

        Child nodes are yielded alphanumerically sorted by node name.
        If the name of a class derived from `Node` is supplied in the
        `classname` parameter, only instances of that class (or
        subclasses of it) will be returned.

        This is an iterator version of `Group._f_listNodes()`.
        """

        self._g_checkOpen()

        if not classname:
            # Returns all the children alphanumerically sorted
            names = self._v_children.keys()
            names.sort()
            for name in names:
                yield self._v_children[name]
        elif classname == 'Group':
            # Returns all the groups alphanumerically sorted
            names = self._v_groups.keys()
            names.sort()
            for name in names:
                yield self._v_groups[name]
        elif classname == 'Leaf':
            # Returns all the leaves alphanumerically sorted
            names = self._v_leaves.keys()
            names.sort()
            for name in names:
                yield self._v_leaves[name]
        elif classname == 'Link':
            # Returns all the links alphanumerically sorted
            names = self._v_links.keys()
            names.sort()
            for name in names:
                yield self._v_links[name]
        elif classname == 'IndexArray':
            raise TypeError(
                "listing ``IndexArray`` nodes is not allowed")
        else:
            class_ = getClassByName(classname)

            children = self._v_children
            childNames = children.keys()
            childNames.sort()

            for childName in childNames:
                childNode = children[childName]
                if isinstance(childNode, class_):
                    yield childNode


    def _f_walkGroups(self):
        """
        Recursively iterate over descendent groups (not leaves).

        This method starts by yielding *self*, and then it goes on to
        recursively iterate over all child groups in alphanumerical
        order, top to bottom (preorder), following the same procedure.
        """

        self._g_checkOpen()

        stack = [self]
        yield self
        # Iterate over the descendants
        while stack:
            objgroup=stack.pop()
            groupnames = objgroup._v_groups.keys()
            # Sort the groups before delivering. This uses the groups names
            # for groups in tree (in order to sort() can classify them).
            groupnames.sort()
            for groupname in groupnames:
                stack.append(objgroup._v_groups[groupname])
                yield objgroup._v_groups[groupname]


    def __delattr__(self, name):
        """
        Delete a Python attribute called `name`.

        This method deletes an *ordinary Python attribute* from the
        object.  It does *not* remove children nodes from this group;
        for that, use `File.removeNode()` or `Node._f_remove()`.  It
        does *neither* delete a PyTables node attribute; for that, use
        `File.delNodeAttr()`, `Node._f_delAttr()` or `Node._v_attrs`.

        If there is an attribute and a child node with the same `name`,
        the child node will be made accessible again via natural naming.
        """
        try:
            super(Group, self).__delattr__(name)  # nothing particular
        except AttributeError, ae:
            hint = " (use ``node._f_remove()`` if you want to remove a node)"
            raise ae.__class__(str(ae) + hint)


    def __getattr__(self, name):
        """
        Get a Python attribute or child node called `name`.

        If the object has a Python attribute called `name`, its value is
        returned.  Else, if the node has a child node called `name`, it
        is returned.  Else, an ``AttributeError`` is raised.
        """
        # That is true since a `NoSuchNodeError` is an `AttributeError`.
        myDict = self.__dict__
        if name in myDict:
            return myDict[name]
        elif name in self._c_lazy_children_attrs:
            self._g_addChildrenNames()
            return myDict[name]
        return self._f_getChild(name)


    def __setattr__(self, name, value):
        """
        Set a Python attribute called `name` with the given `value`.

        This method stores an *ordinary Python attribute* in the object.
        It does *not* store new children nodes under this group; for
        that, use the ``File.create*()`` methods (see the `File` class).
        It does *neither* store a PyTables node attribute; for that, use
        `File.setNodeAttr()`, `Node._f_setAttr()` or `Node._v_attrs`.

        If there is already a child node with the same `name`, a
        `NaturalNameWarning` will be issued and the child node will not
        be accessible via natural naming nor ``getattr()``.  It will
        still be available via `File.getNode()`, `Group._f_getChild()`
        and children dictionaries in the group (if visible).
        """

        # Show a warning if there is an child node with that name.
        #
        # ..note::
        #
        #   Using ``if name in self:`` is not right since that would
        #   require ``_v_children`` and ``_v_hidden`` to be already set
        #   when the very first attribute assignments are made.
        #   Moreover, this warning is only concerned about clashes with
        #   names used in natural naming, i.e. those in ``__members__``.
        #
        # ..note::
        #
        #   The check ``'__members__' in myDict`` allows attribute
        #   assignment to happen before calling `Group.__init__()`, by
        #   avoiding to look into the still not assigned ``__members__``
        #   attribute.  This allows subclasses to set up some attributes
        #   and then call the constructor of the superclass.  If the
        #   check above is disabled, that results in Python entering an
        #   endless loop on exit!

        myDict = self.__dict__
        if '__members__' in myDict and name in self.__members__:
            warnings.warn(
                "group ``%s`` already has a child node named ``%s``; "
                "you will not be able to use natural naming "
                "to access the child node"
                % (self._v_pathname, name), NaturalNameWarning)

        super(Group, self).__setattr__(name, value)


    def _f_flush(self):
        """ Flush this Group """
        self._g_checkOpen()
        self._g_flushGroup()


    def _g_closeDescendents(self):
        """
        Close all the *loaded* descendent nodes of this group.
        """

        def closeNodes(prefix, nodePaths, getNode):
            for nodePath in nodePaths:
                if nodePath.startswith(prefix):
                    try:
                        node = getNode(nodePath)
                        # Avoid descendent nodes to also iterate over
                        # their descendents, which are already to be
                        # closed by this loop.
                        if hasattr(node, '_f_getChild'):
                            node._g_close()
                        else:
                            node._f_close()
                        del node
                    except KeyError:
                        pass

        prefix = self._v_pathname + '/'
        if prefix == '//':
            prefix = '/'

        # Close all loaded nodes.
        aliveNodes = self._v_file._aliveNodes
        deadNodes = self._v_file._deadNodes
        reviveNode = self._v_file._reviveNode
        # First, close the alive nodes and delete them
        # so they are not placed in the limbo again.
        # These two steps ensure tables are closed *before* their indices.
        closeNodes(prefix,
                   [path for path in aliveNodes
                    if '/_i_' not in path],  # not indices
                   lambda path: aliveNodes[path])
        # Close everything else (i.e. indices)
        closeNodes(prefix,
                   [path for path in aliveNodes],
                   lambda path: aliveNodes[path])

        # Next, revive the dead nodes, close and delete them
        # so they are not placed in the limbo again.
        # These two steps ensure tables are closed *before* their indices.
        closeNodes(prefix,
                   [path for path in deadNodes
                    if '/_i_' not in path],  # not indices
                   lambda path: reviveNode(path))
        # Close everything else (i.e. indices)
        closeNodes(prefix,
                   [path for path in deadNodes],
                   lambda path: reviveNode(path))


    def _g_close(self):
        """Close this (open) group."""
        # hdf5Extension operations:
        #   Close HDF5 group.
        self._g_closeGroup()

        # Close myself as a node.
        super(Group, self)._f_close()


    def _f_close(self):
        """Close this group and all its descendents.

        This method has the behavior described in `Node._f_close()`.  It
        should be noted that this operation closes all the nodes
        descending from this group.

        You should not need to close nodes manually because they are
        automatically opened/closed when they are loaded/evicted from
        the integrated LRU cache.
        """

        # If the group is already closed, return immediately
        if not self._v_isopen:
            return

        # First, close all the descendents of this group, unless a) the
        # group is being deleted (evicted from LRU cache) or b) the node
        # is being closed during an aborted creation, in which cases
        # this is not an explicit close issued by the user.
        if not (self._v__deleting or self._v_objectID is None):
            self._g_closeDescendents()

        # When all the descendents have been closed, close this group.
        # This is done at the end because some nodes may still need to
        # be loaded during the closing process; thus this node must be
        # open until the very end.
        self._g_close()


    def _g_remove(self, recursive=False, force=False):
        """Remove (recursively if needed) the Group.

        This version correctly handles both visible and hidden nodes.
        """
        if self._v_nchildren > 0:
            if not (recursive or force):
                raise NodeError("group ``%s`` has child nodes; "
                                "please set `recursive` or `force` to true "
                                "to remove it"
                                % (self._v_pathname,))

            # First close all the descendents hanging from this group,
            # so that it is not possible to use a node that no longer exists.
            self._g_closeDescendents()

        # Remove the node itself from the hierarchy.
        super(Group, self)._g_remove(recursive, force)


    def _f_copy(self, newparent=None, newname=None,
                overwrite=False, recursive=False, createparents=False,
                **kwargs):
        """
        Copy this node and return the new one.

        This method has the behavior described in `Node._f_copy()`.  In
        addition, it recognizes the following keyword arguments:

        `title`
            The new title for the destination.  If omitted or ``None``,
            the original title is used.  This only applies to the
            topmost node in recursive copies.
        `filters`
            Specifying this parameter overrides the original filter
            properties in the source node.  If specified, it must be an
            instance of the `Filters` class.  The default is to copy the
            filter properties from the source node.
        `copyuserattrs`
            You can prevent the user attributes from being copied by
            setting this parameter to ``False``.  The default is to copy
            them.
        `stats`
            This argument may be used to collect statistics on the copy
            process.  When used, it should be a dictionary whith keys
            ``'groups'``, ``'leaves'``, ``'links'`` and ``'bytes'``
            having a numeric value.  Their values will be incremented to
            reflect the number of groups, leaves and bytes,
            respectively, that have been copied during the operation.
        """
        return super(Group, self)._f_copy(
            newparent, newname,
            overwrite, recursive, createparents, **kwargs)


    def _f_copyChildren(self, dstgroup, overwrite=False, recursive=False,
                        createparents=False, **kwargs):
        """
        Copy the children of this group into another group.

        Children hanging directly from this group are copied into
        `dstgroup`, which can be a `Group` object or its pathname in
        string form.  If `createparents` is true, the needed groups for
        the given destination group path to exist will be created.

        The operation will fail with a `NodeError` if there is a child
        node in the destination group with the same name as one of the
        copied children from this one, unless `overwrite` is true; in
        this case, the former child node is recursively removed before
        copying the later.

        By default, nodes descending from children groups of this node
        are not copied.  If the `recursive` argument is true, all
        descendant nodes of this node are recursively copied.

        Additional keyword arguments may be passed to customize the
        copying process.  For instance, title and filters may be
        changed, user attributes may be or may not be copied, data may
        be subsampled, stats may be collected, etc.  Arguments unknown
        to nodes are simply ignored.  Check the documentation for
        copying operations of nodes to see which options they support.
        """

        self._g_checkOpen()

        # `dstgroup` is used instead of its path to avoid accepting
        # `Node` objects when `createparents` is true.  Also, note that
        # there is no risk of creating parent nodes and failing later
        # because of destination nodes already existing.
        dstParent = self._v_file._getOrCreatePath(dstgroup, createparents)
        self._g_checkGroup(dstParent)  # Is it a group?

        if not overwrite:
            # Abort as early as possible when destination nodes exist
            # and overwriting is not enabled.
            for childName in self._v_children:
                if childName in dstParent:
                    raise NodeError(
                        "destination group ``%s`` already has "
                        "a node named ``%s``; "
                        "you may want to use the ``overwrite`` argument"""
                        % (dstParent._v_pathname, childName) )

        for child in self._v_children.itervalues():
            child._f_copy(dstParent, None, overwrite, recursive, **kwargs)


    def __str__(self):
        """
        Return a short string representation of the group.

        Example of use::

            >>> f=tables.openFile('data/test.h5')
            >>> print f.root.group0
            /group0 (Group) 'First Group'
        """

        pathname = self._v_pathname
        classname = self.__class__.__name__
        title = self._v_title
        return "%s (%s) %r" % (pathname, classname, title)


    def __repr__(self):
        """
        Return a detailed string representation of the group.

        Example of use::

            >>> f = tables.openFile('data/test.h5')
            >>> f.root.group0
            /group0 (Group) 'First Group'
              children := ['tuple1' (Table), 'group1' (Group)]
        """

        rep = [ '%r (%s)' %  \
                (childname, child.__class__.__name__)
                for (childname, child) in self._v_children.items() ]
        childlist = '[%s]' % (', '.join(rep))

        return "%s\n  children := %s" % \
               (str(self), childlist)



# Special definition for group root
class RootGroup(Group):
    def __init__(self, ptFile, name, title, new, filters):
        myDict = self.__dict__

        # Set group attributes.
        self._v_version = obversion
        self._v_new = new
        if new:
            self._v_new_title = title
            self._v_new_filters = filters
        else:
            self._v_new_title = None
            self._v_new_filters = None

        # Set node attributes.
        self._v_file = ptFile
        self._v_isopen = True  # root is always open
        self._v_pathname = '/'
        self._v_name = '/'
        self._v_depth = 0
        self._v_maxGroupWidth = ptFile.params['MAX_GROUP_WIDTH']
        self._v__deleting = False
        self._v_objectID = None  # later

        # Only the root node has the file as a parent.
        # Bypass __setattr__ to avoid the ``Node._v_parent`` property.
        myDict['_v_parent'] = ptFile
        ptFile._refNode(self, '/')

        # hdf5Extension operations (do before setting an AttributeSet):
        #   Update node attributes.
        self._g_new(ptFile, name, init=True)
        #   Open the node and get its object ID.
        self._v_objectID = self._g_open()

        # Set disk attributes and read children names.
        #
        # This *must* be postponed because this method needs the root node
        # to be created and bound to ``File.root``.
        # This is an exception to the rule, handled by ``File.__init()__``.
        #
        ##self._g_postInitHook()


    def _g_loadChild(self, childName):
        """
        Load a child node from disk.

        The child node `childName` is loaded from disk and an adequate
        `Node` object is created and returned.  If there is no such
        child, a `NoSuchNodeError` is raised.
        """

        if self._v_file.rootUEP != "/":
            childName = joinPath(self._v_file.rootUEP, childName)
        # Is the node a group or a leaf?
        node_type = self._g_checkHasChild(childName)

        # Nodes that HDF5 report as H5G_UNKNOWN
        if node_type == 'Unknown':
            return Unknown(self, childName)

        # Guess the PyTables class suited to the node,
        # build a PyTables node and return it.
        if node_type == "Group":
            if self._v_file.params['PYTABLES_SYS_ATTRS']:
                childClass = self._g_getChildGroupClass(childName)
            else:
                # Default is a Group class
                childClass = Group
            return childClass(self, childName, new=False)
        elif node_type == "Leaf":
            childClass = self._g_getChildLeafClass(childName, warn=True)
            # Building a leaf may still fail because of unsupported types
            # and other causes.
            ###return childClass(self, childName)  # uncomment for debugging
            try:
                return childClass(self, childName)
            except Exception, exc:  #XXX
                warnings.warn(
                    "problems loading leaf ``%s``::\n\n"
                    "  %s\n\n"
                    "The leaf will become an ``UnImplemented`` node."
                    % (self._g_join(childName), exc))
                # If not, associate an UnImplemented object to it
                return UnImplemented(self, childName)
        elif node_type == "SoftLink":
            return SoftLink(self, childName)
        elif node_type == "ExternalLink":
            return ExternalLink(self, childName)
        else:
            return UnImplemented(self, childName)


    def _f_rename(self, newname):
        raise NodeError("the root node can not be renamed")

    def _f_move(self, newparent=None, newname=None, createparents=False):
        raise NodeError("the root node can not be moved")

    def _f_remove(self, recursive = False):
        raise NodeError("the root node can not be removed")



class TransactionGroupG(NotLoggedMixin, Group):
    _c_classId = 'TRANSGROUP'

    def _g_widthWarning(self):
        warnings.warn("""\
the number of transactions is exceeding the recommended maximum (%d);\
be ready to see PyTables asking for *lots* of memory and possibly slow I/O"""
                      % (self._v_maxGroupWidth,), PerformanceWarning)


class TransactionG(NotLoggedMixin, Group):
    _c_classId = 'TRANSG'

    def _g_widthWarning(self):
        warnings.warn("""\
transaction ``%s`` is exceeding the recommended maximum number of marks (%d);\
be ready to see PyTables asking for *lots* of memory and possibly slow I/O"""
                      % (self._v_pathname, self._v_maxGroupWidth),
                      PerformanceWarning)


class MarkG(NotLoggedMixin, Group):
    # Class identifier.
    _c_classId = 'MARKG'

    import re
    _c_shadowNameRE = re.compile(r'^a[0-9]+$')

    def _g_widthWarning(self):
        warnings.warn("""\
mark ``%s`` is exceeding the recommended maximum action storage (%d nodes);\
be ready to see PyTables asking for *lots* of memory and possibly slow I/O"""
                      % (self._v_pathname, self._v_maxGroupWidth),
                      PerformanceWarning)

    def _g_reset(self):
        """
        Empty action storage (nodes and attributes).

        This method empties all action storage kept in this node: nodes
        and attributes.
        """

        # Remove action storage nodes.
        for child in self._v_children.values():
            child._g_remove(True, True)

        # Remove action storage attributes.
        attrs = self._v_attrs
        shname = self._c_shadowNameRE
        for attrname in attrs._v_attrnamesuser[:]:
            if shname.match(attrname):
                attrs._g__delattr(attrname)



## Local Variables:
## mode: python
## py-indent-offset: 4
## tab-width: 4
## fill-column: 72
## End:
