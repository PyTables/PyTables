########################################################################
#
#       License: BSD
#       Created: September 4, 2002
#       Author:  Francesc Altet - faltet@carabos.com
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

import tables.hdf5Extension as hdf5Extension
import tables.utilsExtension as utilsExtension
import tables.proxydict
from tables.constants import MAX_GROUP_WIDTH
from tables.registry import classIdDict
from tables.exceptions import \
     NodeError, NoSuchNodeError, NaturalNameWarning, PerformanceWarning
from tables.utils import checkNameValidity, joinPath, isVisibleName, \
     getClassByName
from tables.Node import Node, NotLoggedMixin
from tables.Leaf import Leaf, Filters
from tables.UnImplemented import UnImplemented, OldIndexArray
from tables.AttributeSet import AttributeSet



__version__ = "$Revision$"

obversion = "1.0"



class _ChildrenDict(tables.proxydict.ProxyDict):
    def _getValueFromContainer(self, container, key):
        return container._f_getChild(key)



class Group(hdf5Extension.Group, Node):
    """This is the python counterpart of a group in the HDF5 structure.

    It provides methods to set properties based on information
    extracted from the HDF5 files and to walk throughout the
    tree. Every group has parents and children, which are all Group
    instances, except for the root group whose parent is a File
    instance.

    In Group instances, a special feature called "natural naming" is
    used, i.e. the instance attributes that represent HDF5 groups are
    the same as the names of the children. This offers the user a very
    convenient way to access nodes in tree by simply naming all the
    path from the root group.

    For this reason and in order to not pollute the children
    namespace, it is explicitely forbidden to assign "normal"
    attributes to Group instances, and the only ones allowed must
    start by "_c_" (for class variables), "_f_" (for methods), "_g_"
    (for private methods) or "_v_" (for instance variables) prefixes.

    Instance variables (in addition to those in `Node`):

    _v_nchildren
        The number of children hanging from this group.
    _v_children
        Dictionary with all nodes hanging from this group.
    _v_groups
        Dictionary with all groups hanging from this group.
    _v_leaves
        Dictionary with all leaves hanging from this group.
    _v_hidden
        Dictionary with all hidden nodes hanging from this group.
    _v_filters
        Default filter properties for child nodes --see `Filters`.  A
        shorthand for ``FILTERS`` attribute.

    Public methods (in addition to those in `Node`):

    * __setattr__(name, value)
    * __getattr__(name)
    * __delattr__(name)
    * __iter__()
    * __contains__(name)
    * _f_getChild(childname)
    * _f_listNodes(classname)
    * _f_walkGroups()
    * _f_walkNodes(classname, recursive)
    * _f_copyChildren(dstgroup[, overwrite][, recursive][, **kwargs])
    """

    # Class identifier.
    _c_classId = 'GROUP'


    # <properties>

    # `_v_nchildren` is a direct read-only shorthand
    # for the number of *visible* children in a group.
    def _g_getnchildren(self):
        return len(self._v_children)

    _v_nchildren = property(_g_getnchildren, None, None,
                            "The number of children hanging from this group.")


    # `_v_filters` is a direct read-write shorthand
    # for the ``FILTERS`` attribute
    # with the default `Filters` instance as a default value.
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

    _v_filters = property(_g_getfilters, _g_setfilters, _g_delfilters,
                          "Default filter properties for child nodes.")

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

        # Adding the names of visible children nodes here
        # allows readline-style completion to work on them
        # although they are actually not attributes of this object.
        # This must be the *very first* assignment and it must bypass
        # ``__setattr()__`` to let the later work from this moment on.
        self.__dict__['__members__'] = []  # 1st one, bypass __setattr__

        self._v_version = obversion
        """The object version of this group."""

        self._v_new = new
        """Is this the first time the node has been created?"""
        self._v_new_title = title
        """New title for this node."""
        self._v_new_filters = filters
        """New default filter properties for child nodes."""

        self._v_children = _ChildrenDict(self)
        """The number of children hanging from this group."""
        self._v_groups = _ChildrenDict(self)
        """Dictionary with all groups hanging from this group."""
        self._v_leaves = _ChildrenDict(self)
        """Dictionary with all leaves hanging from this group."""
        self._v_hidden = _ChildrenDict(self)  # only place for hidden children
        """Dictionary with all hidden nodes hanging from this group."""

        # Finally, set up this object as a node.
        super(Group, self).__init__(parentNode, name, _log)


    def _g_postInitHook(self):
        if self._v_new:
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
                # inherit them from the parent group.
                filters = self._v_parent._v_filters
            else:
                filters = newFilters
            setAttr('FILTERS', filters)
        else:
            # We don't need to get attributes from disk,
            # since the most important ones are defined as properties.
            # However, we *do* need to get the names of children nodes.
            self._g_addChildrenNames()


    def __del__(self):
        if self._v_isopen and self._v_pathname in self._v_file._aliveNodes:
            # The group is going to be killed.  Rebuild weak references
            # (that Python cancelled just before calling this method) so
            # that they are still usable if the object is revived later.
            selfRef = weakref.ref(self)
            self._v_children.containerRef = selfRef
            self._v_groups.containerRef = selfRef
            self._v_leaves.containerRef = selfRef
            self._v_hidden.containerRef = selfRef

        super(Group, self).__del__()


    def _g_getChildGroupClass(self, childName, warn=True):
        """
        Get the class of a not-yet-loaded group child.

        `childName` must be the name of a *group* child.  If the child
        belongs to an unknown kind of group, or if it lacks a ``CLASS``
        attribute, `Group` will be returned and a warning will be issued
        if `warn` is true and the node belongs to a PyTables file.
        """

        childH5Name = self._v_file._h5NameFromPTName(childName)
        childCID = self._g_getGChildAttr(childH5Name, 'CLASS')

        if childCID in classIdDict:
            return classIdDict[childCID]  # look up group class
        else:
            if warn and self._v_file._isPTFile:
                # All kinds of groups in a PyTables file should have
                # a known ``CLASS`` attribute value.
                warnings.warn(
                    "group ``%s`` has an unknown class ID ``%s``; "
                    "it will become a standard ``Group`` node"
                    % (self._g_join(childName), childCID))
            return Group  # default group class


    def _g_getChildLeafClass(self, childName, warn=True):
        """
        Get the class of a not-yet-loaded leaf child.

        `childName` must be the name of a *leaf* child.  If the child
        belongs to an unknown kind of leaf, or if its kind can not be
        guessed, `UnImplemented` will be returned and a warning will be
        issued if `warn` is true.
        """

        childH5Name = self._v_file._h5NameFromPTName(childName)
        childCID = self._g_getLChildAttr(childH5Name, 'CLASS')

        if childCID in classIdDict:
            return classIdDict[childCID]  # look up leaf class
        elif childCID is None:
            # No ``CLASS`` attribute, try a guess.
            childCID = utilsExtension.whichClass(self._v_objectID, childH5Name)
            if childCID == 'UNSUPPORTED':
                if warn:
                    warnings.warn("leaf ``%s`` is of an unsupported type; "
                                  "it will become an ``UnImplemented`` node"
                                  % self._g_join(childName))
                return UnImplemented
            assert childCID in classIdDict
            return classIdDict[childCID]  # look up leaf class
        else:
            if warn:
                warnings.warn("leaf ``%s`` has an unknown class ID ``%s``; "
                              "it will become an ``UnImplemented`` node"""
                              % (self._g_join(childName), childCID))
            return UnImplemented  # default leaf class


    def _g_addChildrenNames(self):
        """
        Add children names to this group taking into account their
        visibility and kind.
        """

        # Get the names of *all* child groups and leaves.
        (groupNames, leafNames) = self._g_listGroup()

        # (Cache some objects.)
        ptNameFromH5Name = self._v_file._ptNameFromH5Name
        members = self.__members__
        children = self._v_children
        hidden = self._v_hidden

        # Separate groups into visible groups and hidden nodes,
        # and leaves into visible leaves and hidden nodes.
        for (childNames, childDict) in (
            (groupNames, self._v_groups),
            (leafNames,  self._v_leaves)):

            for childName in childNames:
                # Get the PyTables name matching this HDF5 name.
                childName = ptNameFromH5Name(childName)

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


    def _g_checkHasChild(self, childName):
        """
        Check that the group has a child called `childName`.

        If it does not, a `NoSuchNodeError` is raised.
        """

        if childName not in self:
            raise NoSuchNodeError(
                "group ``%s`` does not have a child named ``%s``"
                % (self._v_pathname, childName))


    def _g_loadChild(self, childName):
        """
        Load a child node from disk.

        The child node `childName` is loaded from disk and an adequate
        `Node` object is created and returned.  If there is no such
        child, a `NoSuchNodeError` is raised.
        """

        self._g_checkHasChild(childName)

        # Get the HDF5 name matching the PyTables name.
        childH5Name = self._v_file._h5NameFromPTName(childName)

        # Is the node a group or a leaf?
        if childName in self._v_groups:
            childIsGroup = True
        elif childName in self._v_leaves:
            childIsGroup = False
        else:
            # Worst case: hidden nodes are not separated into groups and
            # leaves: we need to list children to get the kind of node.
            # This is less efficient, so do we only do it if unavoidable.
            assert childName in self._v_hidden
            (groupNames, leafNames) = self._g_listGroup()
            assert childH5Name in groupNames or childH5Name in leafNames
            childIsGroup = childH5Name in groupNames

        # Guess the PyTables class suited to the node,
        # build a PyTables node and return it.
        if childIsGroup:
            childClass = self._g_getChildGroupClass(childName, warn=True)
            return childClass(self, childName, new=False)
        else:
            childClass = self._g_getChildLeafClass(childName, warn=True)
            # Building a leaf may still fail because of unsupported types
            # and other causes.
            return childClass(self, childName)  # uncomment for debugging
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


    def __iter__(self, classname=None, recursive=0):
        """Iterate over the children on self"""
        return self._f_walkNodes(classname, recursive)


    def __contains__(self, name):
        """
        Is there a child with that `name`?

        Returns ``True`` if the group has a child node (visible or
        hidden) with the given `name` (a string), ``False`` otherwise.
        """
        self._g_checkOpen()
        return name in self._v_children or name in self._v_hidden


    def _f_walkNodes(self, classname=None, recursive=True):
        """Iterate over the nodes of self

        If "classname" is supplied, only instances of this class
        are returned. If "recursive" is false, only children
        hanging immediately after the group are returned. If
        true, a recursion over all the groups hanging from it is
        performed. """

        self._g_checkOpen()

        # For compatibility with old default arguments.
        if classname == '':
            classname = None

        if not recursive:
            # Non-recursive algorithm
            for leaf in self._f_listNodes(classname):
                yield leaf
        else:
            if classname == "Group":
                # Recursive algorithm
                for group in self._f_walkGroups():
                    yield group
            else:
                for group in self._f_walkGroups():
                    for leaf in group._f_listNodes(classname):
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
                      % (self._v_pathname, MAX_GROUP_WIDTH),
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
        if childName in self:
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
        if len(self._v_children) + len(self._v_hidden) >= MAX_GROUP_WIDTH:
            self._g_widthWarning()

        # Insert references to the new child.
        # (Assigned values are entirely irrelevant.)
        if isVisibleName(childName):
            # Visible node.
            self.__members__.insert(0, childName)  # enable completion

            self._v_children[childName] = None  # insert node
            if isinstance(childNode, Leaf):
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

        if childName in self._v_children:
            # Visible node.
            members = self.__members__
            memberIndex = members.index(childName)
            del members[memberIndex]  # disables completion

            del self._v_children[childName]  # remove node
            self._v_leaves.pop(childName, None)
            self._v_groups.pop(childName, None)
        else:
            # Hidden node.
            del self._v_hidden[childName]  # remove node


    def _g_updateLocation(self, newParentPath):
        # Update location of self.
        oldPath = self._v_pathname
        super(Group, self)._g_updateLocation(newParentPath)
        newPath = self._v_pathname
        # Update location information in children.
        self._g_updateChildrenLocation(oldPath, newPath)


    def _g_move(self, newParent, newName):
        # Move the node to the new location.
        oldPath = self._v_pathname
        super(Group, self)._g_move(newParent, newName)
        newPath = self._v_pathname

        # Update location information in children.
        self._g_updateChildrenLocation(oldPath, newPath)


    def _g_updateChildrenLocation(self, oldPath, newPath):
        # Update location information of *already loaded* children.
        file_ = self._v_file
        oldPathSlash = oldPath + '/'  # root node can not be renamed, anyway

        # Update alive descendents.
        # XXX What happens if the _aliveNodes dictionary changes during the
        # next loop (have in mind that it is recursive)?
        # F. Altet 2006-08-06
        # Answer: no problem, python will complain ;-) F. Altet 2006-09-27
        for nodePath in file_._aliveNodes:
            if nodePath.startswith(oldPathSlash):
                descendentNode = file_._getNode(nodePath)
                descendentNode._g_updateLocation(newPath)

        # Update dead descendents.
        # XXX What happens if the _deadNodes dictionary changes during the
        # next loop (have in mind that it is recursive)?
        # F. Altet 2006-08-06
        for nodePath in file_._deadNodes:
            if nodePath.startswith(oldPathSlash):
                descendentNode = file_._getNode(nodePath)
                descendentNode._g_updateLocation(newPath)


    def _g_copy(self, newParent, newName, recursive, _log=True, **kwargs):
        # Compute default arguments.
        title = kwargs.get('title', self._v_title)
        filters = kwargs.get('filters', self._v_filters)
        stats = kwargs.get('stats', None)

        # Fix arguments with explicit None values for backwards compatibility.
        if title is None:  title = self._v_title
        if filters is None:  filters = self._v_filters

        # Create a copy of the object.
        newNode = Group(newParent, newName,
                        title, new=True, filters=filters, _log=_log)

        # Copy user attributes if needed.
        if kwargs.get('copyuserattrs', True):
            self._v_attrs._g_copy(newNode._v_attrs)

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
        """

        self._g_checkOpen()

        childName = childname
        self._g_checkHasChild(childName)

        childPath = joinPath(self._v_pathname, childName)
        return self._v_file._getNode(childPath)


    def _f_listNodes(self, classname=None):
        """
        Return a list with children nodes.

        The list is alphanumerically sorted by node name.  If the name
        of a class derived from `Node` is supplied in the `classname`
        parameter, only instances of that class (or subclasses of it)
        will be returned.  `IndexArray` objects are not allowed to be
        listed.
        """

        self._g_checkOpen()

        if not classname:
            # Returns all the children alphanumerically sorted
            names = self._v_children.keys()
            names.sort()
            return [ self._v_children[name] for name in names ]
        elif classname == 'Group':
            # Returns all the groups alphanumerically sorted
            names = self._v_groups.keys()
            names.sort()
            return [ self._v_groups[name] for name in names ]
        elif classname == 'Leaf':
            # Returns all the leaves alphanumerically sorted
            names = self._v_leaves.keys()
            names.sort()
            return [ self._v_leaves[name] for name in names ]
        elif classname == 'IndexArray':
            raise TypeError(
                "listing ``IndexArray`` nodes is not allowed")
        else:
            class_ = getClassByName(classname)

            children = self._v_children
            childNames = children.keys()
            childNames.sort()

            nodelist = []
            for childName in childNames:
                childNode = children[childName]
                if isinstance(childNode, class_):
                    nodelist.append(childNode)

            return nodelist


    def _f_iterNodes(self, classname=None):
        """
        Return an iterator yielding children nodes.

        The list is alphanumerically sorted by node name.  If the name
        of a class derived from `Node` is supplied in the `classname`
        parameter, only instances of that class (or subclasses of it)
        will be returned.  `IndexArray` objects are not allowed to be
        listed.

        This is an iterator version of Group._f_listNodes()
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
        """Iterate over the Groups (not Leaves) hanging from self.

        The groups are returned ordered from top to bottom, and
        alphanumerically sorted when in the same level.

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
        for that, use `File.removeNode()` or `Group._f_remove()`.  It
        does *neither* delete a PyTables node attribute; for that, use
        `File.delNodeAttr()`, `Node._f_delAttr()` or `Node._v_attrs`.

        If there were an attribute and a child node with the same
        `name`, the child node will be made accessible again via natural
        naming.
        """
        try:
            super(Group, self).__delattr__(name)  # nothing particular
        except AttributeError, ae:
            hint = " (use ``node._f_remove()`` if you want to remove a node)"
            raise ae.__class__(str(ae) + hint)


    def __getattr__(self, name):
        """
        Get a Python attribute or child node called `name`.

        If the object has a Python attribute called `name`, itis value
        returned.  Else, if the node has a child node called `name`, it
        is returned.  Else, an `AttributeError` is raised.
        """
        # That is true since a `NoSuchNodeError` is an `AttributeError`.

        myDict = self.__dict__
        if name in myDict:
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
        be accessible via natural naming nor `getattr()`.  It will still
        be available via `File.getNode()`, `Group._f_getChild()` and
        children dictionaries in the group (if visible).
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

        myDict[name] = value


    def _f_flush(self):
        """ Flush this Group """
        self._g_checkOpen()
        self._g_flushGroup()

    def _g_closeNodes(self):
        """Recursively close all nodes in `self` and their descendents.

        This version correctly handles both visible and hidden nodes.
        """

        stack = [self]
        # Iterate over the descendants
        while stack:
            objgroup=stack.pop()
            stack.extend(objgroup._v_groups.values())
            # Collect any hidden group
            for node in objgroup._v_hidden.values():
                if isinstance(node, Group):
                    stack.append(node)
                else:
                    # If it is not a group, close it
                    node._f_close()
            # Close the visible leaves
            for leaf in objgroup._v_leaves.values():
                leaf._f_close()
            # Close the current group only if it is not myself to avoid
            # recursivity in case of calling from '/' group
            if objgroup is not self:
                objgroup._f_close()

    def _f_close(self):
        """
        Close this node in the tree.

        This method has the behavior described in `Node._f_close()`.  It
        should be noted that this operation disables access to nodes
        descending from this group.  Therefore, if you want to
        explicitly close them, you will need to walk the nodes hanging
        from this group *before* closing it.
        """

        if not self._v_isopen:
            return  # the node is already closed

        # hdf5Extension operations:
        #   Close HDF5 group.
        self._g_closeGroup()

        # Clear group object attributes.
        self.__dict__['__members__'] = []  # 1st one, bypass __setattr__

        # Close myself as a node.
        super(Group, self)._f_close()


    def _g_remove(self, recursive=False):
        """Remove (recursively if needed) the Group.

        This version correctly handles both visible and hidden nodes.
        """
        if self._v_nchildren > 0:
            if not recursive:
                raise NodeError("group ``%s`` has child nodes; "
                                "please state recursive removal to remove it"
                                % (self._v_pathname,))

            # First close all the descendents hanging from this group,
            # so that it is not possible to use a node that no longer exists.
            # We let the ``File`` instance close the nodes
            # since it knows which of them are loaded and which not.
            self._v_file._closeDescendentsOf(self)

        # Remove the node itself from the hierarchy.
        super(Group, self)._g_remove(recursive)


    def _f_copy(self, newparent=None, newname=None,
                overwrite=False, recursive=False, **kwargs):
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
            ``'groups'``, ``'leaves'`` and ``'bytes'`` having a numeric
            value.  Their values will be incremented to reflect the
            number of groups, leaves and bytes, respectively, that have
            been copied during the operation.
        """
        return super(Group, self)._f_copy(
            newparent, newname, overwrite, recursive, **kwargs)


    def _f_copyChildren(self, dstgroup, overwrite=False, recursive=False,
                        **kwargs):
        """
        Copy the children of this group into another group.

        Children hanging directly from this group are copied into
        `dstgroup`, which can be a `Group` object or its pathname in
        string form.

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

        dstParent = self._v_file.getNode(dstgroup)  # Does new parent exist?
        self._g_checkGroup(dstParent)  # Is it a group?

        if not overwrite:
            # Abort as early as possible when destination nodes exist
            # and overwriting is not enabled.
            for childName in self._v_children:
                if childName in dstParent:
                    raise NodeError("""\
destination group ``%s`` already has a node named ``%s``; \
you may want to use the ``overwrite`` argument"""
                                    % (dstParent._v_pathname, childName))

        for child in self._v_children.itervalues():
            child._f_copy(dstParent, None, overwrite, recursive, **kwargs)


    def __str__(self):
        """The string representation for this object."""

        if not self._v_isopen:
            return repr(self)

        # Get the associated filename
        filename = self._v_file.filename
        # The pathname
        pathname = self._v_pathname
        # Get this class name
        classname = self.__class__.__name__
        # The title
        title = self._v_title
        return "%s (%s) %r" % (pathname, classname, title)


    def __repr__(self):
        """A detailed string representation for this object."""

        if not self._v_isopen:
            return "<closed Group>"

        rep = [ '%r (%s)' %  \
                (childname, child.__class__.__name__)
                for (childname, child) in self._v_children.items() ]
        childlist = '[%s]' % (', '.join(rep))

        return "%s\n  children := %s" % \
               (str(self), childlist)



# Special definition for group root
class RootGroup(Group):
    def __init__(self, ptFile, h5name, title, new, filters):
        # Set group attributes.
        self.__dict__['__members__'] = []   # 1st one, bypass __setattr__

        self._v_version = obversion
        self._v_new = new
        if new:
            self._v_new_title = title
            self._v_new_filters = filters
        else:
            self._v_new_title = None
            self._v_new_filters = None

        self._v_children = _ChildrenDict(self)
        self._v_groups = _ChildrenDict(self)
        self._v_leaves = _ChildrenDict(self)
        self._v_hidden = _ChildrenDict(self)

        # Set node attributes.
        self._v_file = ptFile
        self._v_isopen = True  # root is always open
        self._v_pathname = '/'  # Can it be h5name? I don't think so.
        self._v_name = '/'
        self._v_hdf5name = h5name
        self._v_depth = 0
        self._v__deleting = False
        self._v_objectID = None  # later

        self._v_parent = ptFile  # only the root node has the file as a parent
        ptFile._refNode(self, '/')

        # hdf5Extension operations (do before setting an AttributeSet):
        #   Update node attributes.
        self._g_new(ptFile, h5name, init=True)
        #   Open the node and get its object ID.
        self._v_objectID = self._g_open()

        # Set disk attributes and read children names.
        #
        # This *must* be postponed because this method needs the root node
        # to be created and bound to ``File.root``.
        # This is an exception to the rule, handled by ``File.__init()__``.
        #
        ##self._g_postInitHook()


    def _f_rename(self, newname):
        raise NodeError("the root node can not be renamed")

    def _f_move(self, newparent = None, newname = None):
        raise NodeError("the root node can not be moved")

    def _f_remove(self, recursive = False):
        raise NodeError("the root node can not be removed")



class IndexesDescG(NotLoggedMixin, Group):
    _c_classId = 'DINDEX'

    def _g_widthWarning(self):
        warnings.warn("""\
the number of indexed columns on a single description group is exceeding
the recommended maximum (%d); be ready to see PyTables asking for *lots*
of memory and possibly slow I/O"""
                      % (MAX_GROUP_WIDTH,), PerformanceWarning)


class IndexesTableG(NotLoggedMixin, Group):
    _c_classId = 'TINDEX'

    def _g_widthWarning(self):
        warnings.warn("""\
the number of indexed columns on a single table is exceeding the \
recommended maximum (%d); be ready to see PyTables asking for *lots* \
of memory and possibly slow I/O"""
                      % (MAX_GROUP_WIDTH,), PerformanceWarning)

    def _g_checkName(self, name):
        if not name.startswith('_i_'):
            raise ValueError(
                "names of index groups must start with ``_i_``: %s" % name)


class IndexesColumnBackCompatG(NotLoggedMixin, Group):
    """This is meant to hidden indexes of pre-PyTables 1.0 files."""
    _c_classId = 'INDEX'



class TransactionGroupG(NotLoggedMixin, Group):
    _c_classId = 'TRANSGROUP'

    def _g_widthWarning(self):
        warnings.warn("""\
the number of transactions is exceeding the recommended maximum (%d);\
be ready to see PyTables asking for *lots* of memory and possibly slow I/O"""
                      % (MAX_GROUP_WIDTH,), PerformanceWarning)



class TransactionG(NotLoggedMixin, Group):
    _c_classId = 'TRANSG'

    def _g_widthWarning(self):
        warnings.warn("""\
transaction ``%s`` is exceeding the recommended maximum number of marks (%d);\
be ready to see PyTables asking for *lots* of memory and possibly slow I/O"""
                      % (self._v_pathname, MAX_GROUP_WIDTH),
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
                      % (self._v_pathname, MAX_GROUP_WIDTH),
                      PerformanceWarning)


    def _g_reset(self):
        """
        Empty action storage (nodes and attributes).

        This method empties all action storage kept in this node: nodes
        and attributes.
        """

        # Remove action storage nodes.
        for child in self._v_children.values():
            child._g_remove(True)

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
