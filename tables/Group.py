########################################################################
#
#       License: BSD
#       Created: September 4, 2002
#       Author:  Francesc Altet - faltet@carabos.com
#
#       $Source: /home/ivan/_/programari/pytables/svn/cvs/pytables/pytables/tables/Group.py,v $
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

import sys
import warnings

import tables.hdf5Extension as hdf5Extension
from tables.constants import MAX_GROUP_WIDTH
from tables.registry import classNameDict, classIdDict
from tables.exceptions import NodeError, NoSuchNodeError, PerformanceWarning
from tables.utils import checkNameValidity, joinPath
from tables.Node import Node
from tables.Leaf import Leaf, Filters
from tables.UnImplemented import UnImplemented
from tables.AttributeSet import AttributeSet



__version__ = "$Revision: 1.86 $"

obversion = "1.0"



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
    _v_filters
        Default filter properties for child nodes --see `Filters`.  A
        shorthand for ``FILTERS`` attribute.

    Public methods (in addition to those in `Node`):

    * __delattr__(name)
    * __getattr__(name)
    * __setattr__(name, object)
    * __iter__()
    * __contains__(name)
    * _f_listNodes(classname)
    * _f_walkGroups()
    * _f_walkNodes(classname, recursive)
    * _f_copyChildren(where[, recursive][, filters][, copyuserattrs]
                      [, start][, stop ][, step][, overwrite][, stats])
    """

    # Class identifier.
    _c_classId = 'GROUP'


    # <undo-redo support>
    _c_canUndoCreate = True  # Can creation/copying be undone and redone?
    _c_canUndoRemove = True  # Can removal be undone and redone?
    _c_canUndoMove   = True  # Can movement/renaming be undone and redone?
    # </undo-redo support>


    # <properties>

    # '_v_filters' is a direct read-write shorthand for the 'FILTERS' attribute
    # with the default Filters instance as a default value.
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


    def __init__(self, title = "", new = 1, filters=None):
        """Create the basic structures to keep group information.

        title -- The title for this group
        new -- If this group is new or has to be read from disk
        filters -- A Filters instance

        """
        self.__dict__["_v_new"] = new
        self.__dict__["_v_new_title"] = title
        self.__dict__["_v_new_filters"] = filters
        self.__dict__["_v_groups"] = {}
        self.__dict__["_v_leaves"] = {}
        self.__dict__["_v_children"] = {}
        self.__dict__["_v_nchildren"] = 0
        self.__dict__["_v_indices"] = []
        return

    def __iter__(self, classname=None, recursive=0):
        """Iterate over the children on self"""

        return self._f_walkNodes(classname, recursive)

    def __contains__(self, name):
        """
        Is there a child with that `name`?

        Returns ``True`` if the group has a child node with the given
        `name` (a string), ``False`` otherwise.
        """
        return name in self._v_children

    def _f_walkNodes(self, classname=None, recursive=False):
        """Iterate over the nodes of self

        If "classname" is supplied, only instances of this class
        are returned. If "recursive" is false, only children
        hanging immediately after the group are returned. If
        true, a recursion over all the groups hanging from it is
        performed. """

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

    # This iterative version of _g_openFile is due to John Nielsen
    def _g_openFile(self):
        """Recusively read an HDF5 file and generate its object tree."""

        stack=[self]
        while stack:
            objgroup=stack.pop()
            pgroupId=objgroup._v_parent._v_objectID
            locId=objgroup._v_objectID
            (groups, leaves)=self._g_listGroup(pgroupId, locId,
                                               objgroup._v_hdf5name)
            for name in groups:
                classId = objgroup._g_getGChildAttr(name, 'CLASS')

                if classId == 'INDEX':
                    # Index groups are not included in the object tree,
                    # but their names are appended to _v_indices.
                    objgroup._v_indices.append(name)
                    continue

                groupClass = Group  # default group class
                if classId in classIdDict:
                    groupClass = classIdDict[classId]
                elif self._v_file._isPTFile:
                    # Warn only in PyTables files, where 'CLASS' has meaning.
                    warnings.warn("""\
group ``%s`` has an unknown class ID ``%s``; \
it will become a standard ``Group`` node"""
                                  % (objgroup._g_join(name), classId))
                new_objgroup = groupClass(new = False)
                new_objgroup._g_putUnder(objgroup, name)
                stack.append(new_objgroup)

            for name in leaves:
                leafClass = objgroup._g_getLeafClass(objgroup, name)
                objleaf = leafClass()
                # Try if object can be loaded
                try:
                    objleaf._g_putUnder(objgroup, name)
                except:  #XXX
                    (typerr, value, traceback) = sys.exc_info()
                    warnings.warn("""\
problems loading leaf ``%s``: %s; \
it will become an ``UnImplemented`` node""" % (objgroup._g_join(name), value))
                    # If not, associate an UnImplemented object to it
                    objleaf = UnImplemented()
                    objleaf._g_putUnder(objgroup, name)

    def _g_getLeafClass(self, parent, name):
        """Return a proper Leaf class depending on the object to be opened."""

        if self._v_file._isPTFile:
            # We can call this only if we are certain than file has
            # the attribute CLASS
            classId = self._v_attrs._g_getChildSysAttr(name, "CLASS")
        else:
            classId = self._v_attrs._g_getChildAttr(name, "CLASS")
        if classId is None:
            # No CLASS attribute, try a guess
            classId = hdf5Extension.whichClass(self._v_objectID, name)
            if classId == "UNSUPPORTED":
                warnings.warn("""\
leaf ``%s`` is of an unsupported type; \
it will become an ``UnImplemented`` node""" % (parent._g_join(name),))
                return UnImplemented
        if classId in classIdDict:
            return classIdDict[classId]
        else:
            warnings.warn("""\
leaf ``%s`` has an unknown class ID ``%s``; \
it will become an ``UnImplemented`` node""" % (parent._g_join(name), classId))
            return UnImplemented

    def _g_join(self, name):
        """Helper method to correctly concatenate a name child object
        with the pathname of this group."""

        if name == "/":
            # This case can happen when doing copies
            return self._v_pathname
        return joinPath(self._v_pathname, name)


    def _g_checkWidth(self):
        """
        Check for width performance limits.

        Checks for performance limitations related to group width.  If
        one of those limitations is met in the group, a
        `PerformanceWarning` is issued.
        """

        # Check if there are too many children in this group.
        if self._v_nchildren >= MAX_GROUP_WIDTH:
            warnings.warn("""\
group ``%s`` is exceeding the recommended maximum number of children (%d);\
be ready to see PyTables asking for *lots* of memory and possibly slow I/O"""
                          % (self._v_pathname, MAX_GROUP_WIDTH),
                          PerformanceWarning)


    def _g_refNode(self, node, name, validate=True):
        """
        Insert references to a `node` via a `name`.

        Checks that the `name` is valid and does not exist, then creates
        references to the given `node` by that `name`.  The validation
        of the name can be omitted by setting `validate` to a false
        value (this may be useful for adding already existing nodes to
        the tree).
        """

        # Check for name validity.
        if validate:
            checkNameValidity(name)

        # Check if there is already a child with the same name.
        if name in self._v_children:
            raise NodeError(
                "group ``%s`` already has a child node named ``%s``"
                % (self._v_pathname, name))

        # Check group width limits.
        self._g_checkWidth()

        # Insert references to the new child.
        self._v_children[name] = node
        if isinstance(node, Leaf):
            self._v_leaves[name] = node
        if isinstance(node, Group):
            self._v_groups[name] = node

        mydict = self.__dict__
        mydict[name] = node
        mydict['_v_nchildren'] += 1


    def _g_unrefNode(self, name):
        """
        Remove references to a node.

        Removes all references to the named node.
        """

        del self._v_children[name]
        if name in self._v_leaves:
            del self._v_leaves[name]
        if name in self._v_groups:
            del self._v_groups[name]

        mydict = self.__dict__
        del mydict[name]
        mydict['_v_nchildren'] -= 1


    def _g_putUnder(self, parent, name):
        # All this will eventually end up in the node constructor.

        super(Group, self)._g_putUnder(parent, name)
        # Update class variables
        if self._v_new:
            self._g_create()
        else:
            self._g_open()
        # Attach the AttributeSet attribute
        # This doesn't become a property because it takes more time!
        self.__dict__["_v_attrs"] = AttributeSet(self)
        if self._v_new:
            setAttr = self._v_attrs._g__setattr
            # Set the title, class and version attribute
            setAttr('TITLE', self._v_new_title)
            setAttr('CLASS', self._c_classId)
            setAttr('VERSION', obversion)
            # Set the filters object
            if self._v_new_filters is None:
                # If not filters has been passed in the constructor,
                filters = self._v_parent._v_filters
            else:
                filters = self._v_new_filters
            setAttr('FILTERS', filters)
        else:
            # We don't need to get attributes on disk. The most importants
            # are defined as properties
            pass


    # Define some attrs as a property.
    # In the case of groups, it is faster to not define the _v_attrs property
    # I don't know exactly why. This should be further investigated.
#     def _get_attrs (self):
#         return AttributeSet(self)
#     # attrs can't be set or deleted by the user
#     _v_attrs = property(_get_attrs, None, None, "Attrs of this object")


    def _g_move(self, newParent, newName):
        oldPathname = self._v_pathname

        # Move the node to the new location.
        super(Group, self)._g_move(newParent, newName)

        # Update location information in children.
        myself = True
        for group in self._f_walkGroups():
            if myself:
                # Do not change location information again for this group.
                myself = False
            else:
                # Update location information for the descendent group.
                group._g_updateLocation()
            # Update location information for its child leaves.
            for leaf in group._f_listNodes('Leaf'):
                leaf._g_updateLocation()


    def _g_copy(self, newParent, newName, recursive, **kwargs):
        # Compute default arguments.
        title = kwargs.get('title', self._v_title)
        filters = kwargs.get('filters', self._v_filters)
        stats = kwargs.get('stats', None)

        # Fix arguments with explicit None values for backwards compatibility.
        if title is None:  title = self._v_title
        if filters is None:  filters = self._v_filters

        # Create a copy of the object.
        newNode = self._v_file.createGroup(
            newParent, newName, title, filters, _log = False)

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
        ##for (srcChildName, srcChild) in self._v_children.iteritems():
        ##    srcChild._g_copy(newParent, srcChildName, True, **kwargs)

        # Non-recursive version of children copy.
        parentStack = [(self, newParent)]  # [(source, destination), ...]
        while parentStack:
            (srcParent, dstParent) = parentStack.pop()
            for (srcChildName, srcChild) in srcParent._v_children.iteritems():
                dstChild = srcChild._g_copy(
                    dstParent, srcChildName, False, **kwargs)
                if isinstance(srcChild, Group):
                    parentStack.append((srcChild, dstChild))


    def _g_open(self):
        """Call the openGroup method in super class to open the existing
        group on disk. """

        # All this will eventually end up in the node constructor.

        # Call the superclass method to open the existing group
        self.__dict__["_v_objectID"] = self._g_openGroup()

    def _g_create(self):
        """Call the createGroup method in super class to create the group on
        disk. Also set attributes for this group. """

        # All this will eventually end up in the node constructor.

        # Call the superclass method to create a new group
        self.__dict__["_v_objectID"] = \
                     self._g_createGroup()


    def _f_listNodes(self, classname=None):
        """
        Return a list with children nodes.

        The list is alphanumerically sorted by node name.  If the name
        of a class derived from `Node` is supplied in the `classname`
        parameter, only instances of that class (or subclasses of it)
        will be returned.  `IndexArray` objects are not allowed to be
        listed.
        """

        # For compatibility with old default arguments.
        if classname == '':
            classname = None

        if classname is None:
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
            if classname not in classNameDict:
                raise TypeError(
                    "there is no registered node class named ``%s``"
                    % (classname,))
            class_ = classNameDict[classname]

            children = self._v_children
            childNames = children.keys()
            childNames.sort()

            nodelist = []
            for childName in childNames:
                childNode = children[childName]
                if isinstance(childNode, class_):
                    nodelist.append(childNode)

            return nodelist


    def _f_walkGroups(self):
        """Iterate over the Groups (not Leaves) hanging from self.

        The groups are returned ordered from top to bottom, and
        alphanumerically sorted when in the same level.

        """

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
        Remove a child from the hierarchy.

        Removal via this method is *never* recursive because of the
        potential damage it may cause when used inadvertently.  If a
        recursive behavior is needed, use the ``_f_remove()`` method of
        the child node.  See `Node._f_remove()`.
        """

        if name not in self._v_children:
            raise NoSuchNodeError(
                "group ``%s`` does not have a child named ``%s``"
                % (self._v_pathname, name))

        try:
            node = self._v_children[name]
            node._f_remove()
        except NodeError:
            # This error message is clearer than the original one
            # for this operation.
            raise NodeError("""\
child group ``%s`` has child nodes; \
please use ``child._f_remove(True)`` to remove it"""
                            % (node._v_pathname,))

    def __getattr__(self, name):
        """Get the object named "name" hanging from me."""

        if not self._v_file.isopen:
            raise RuntimeError, "You are trying to access to a closed file handler. Giving up!."

        if name in self._v_children:
            return self._v_children[name]
        else:
            raise NoSuchNodeError(
                "group ``%s`` does not have a child named ``%s``"
                % (self._v_pathname, name))

    def __setattr__(self, name, value):
        """Attach new nodes to the tree.

        name -- The name of the new node
        value -- The new node object

        If "name" group already exists in "self", raise the NodeError
        exception. A ValueError is raised when the "name" starts
        by a reserved prefix is not a valid Python identifier.
        A TypeError is raised when "value" is not a PyTables node
        (a Group or a Leaf).

        """

        # This method will be eventually assimilated into Node.__init__.

        # Check if the object is a PyTables node.
        # Only PyTables nodes can be assigned into a Group.
        if not isinstance(value, Node):
            raise TypeError("assigned object is not a node: %r" % (value,))

        # Check for name validity
        if self._v_new or not self._v_file._isPTFile:
            # Check names only for new objects or objects coming from
            # non-pytables files
            checkNameValidity(name)

        # Put value object with name "name" in object tree
        value._g_putUnder(self, name)

    def _f_flush(self):
        """ Flush this Group """
        self._g_flushGroup()


    def _f_close(self):
        """
        Close this node in the tree.

        This method has the behavior described in `Node._f_close()`.  It
        should be noted that this operation disables access to nodes
        descending from this group.  Therefore, if you want to
        explicitly close them, you will need to walk the nodes hanging
        from this group *before* closing it.
        """

        self._g_closeGroup()
        # Delete the back references in Group
        if self._v_name <> "/":
            self._v_parent._g_unrefNode(self._v_name)
        ##################################
        #self._v_children.clear()
        ##################################
        # Delete back references
        self._g_delLocation()
        # Detach the AttributeSet instance
        self._v_attrs._f_close()
        del self.__dict__["_v_attrs"]
        # Delete the filters instance
        if self.__dict__.has_key("_v_filters"):
            del self.__dict__["_v_filters"]


    def _g_remove(self, recursive = False):
        if self._v_nchildren > 0:
            if recursive:
                # First close all the children hanging from this group
                for group in self._f_walkGroups():
                    for leaf in group._f_listNodes('Leaf'):
                        # Delete the back references in Leaf
                        leaf._f_close()
                    # Close this group
                    group._f_close()
                # Finally, remove this group
                self._g_deleteGroup()
            else:
                raise NodeError("""\
group ``%s`` has child nodes; \
please state recursive removal to remove it"""
                                % (self._v_pathname,))
        else:
            # This group has no children, so we can delete it
            # without any other measure
            self._f_close()
            self._g_deleteGroup()


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


    def _f_copyChildren(self, where, recursive = False,
                        filters = None, copyuserattrs = True,
                        start = 0, stop = None, step = 1,
                        overwrite = False, stats = None):
        """(Recursively) Copy the children of a group into another location

        'where' is the destination group.  If should exist or a
        NodeError will be raised. It can be specified as a string or
        as a Group instance. 'recursive' specifies whether the copy
        should recurse into subgroups or not. The default is not
        recurse. Specifying a 'filters' parameter overrides the
        original filter properties in source nodes. You can prevent
        the user attributes from being copied by setting
        'copyuserattrs' to a false value; the default is copy
        them. 'start', 'stop' and 'step' specify the range of rows
        in leaves to be copied; the default is to copy all the
        rows. 'overwrite' means whether the possible existing children
        hanging from 'where' and having the same names than children
        in this group should be overwritten or not.

        The optional keyword argument 'stats' may be used to collect
        statistics on the copy process.  When used, it should be a
        dictionary whith keys 'groups', 'leaves' and 'bytes' having a
        numeric value.  Their values will be incremented to reflect
        the number of groups, leaves and bytes, respectively, that
        have been copied in the operation.
        """

        dstParent = self._v_file.getNode(where)  # Does the new parent exist?
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
            child._f_copy(
                dstParent, None, overwrite, recursive, stats = stats,
                filters = filters, copyuserattrs = copyuserattrs,
                start = start, stop = stop, step = step)


    def __str__(self):
        """The string representation for this object."""
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

        rep = [ '%r (%s)' %  \
                (childname, child.__class__.__name__)
                for (childname, child) in self._v_children.items() ]
        childlist = '[%s]' % (', '.join(rep))

        return "%s\n  children := %s" % \
               (str(self), childlist)


# Special definition for group root
class RootGroup(Group):
    def __init__(self, file, ptname, h5name, new = True):
        super(RootGroup, self).__init__(new = new)

        mydict = self.__dict__

        # Hidden children are only stored here.
        mydict['_v_hidden'] = {}

        # Explicitly set location-dependent attributes.
        # Calling _g_setLocation is not needed for the root group.
        mydict["_v_file"] = file
        mydict["_v_name"] = ptname
        mydict["_v_hdf5name"] = h5name
        mydict["_v_parent"] = file
        mydict["_v_pathname"] = ptname   # Can be h5name? I don't think so
        mydict["_v_depth"] = 0   # root is level 0
        file._refNode(self, self._v_pathname)

        # hdf5Extension operations (do before setting an AttributeSet):
        #   Update node attributes.
        self._g_new(file, h5name)
        #   Get HDF5 identifier.
        mydict["_v_objectID"] = self._g_openGroup()

        mydict["_v_attrs"] = AttributeSet(self)
        if new:
            # Set the title
            mydict["_v_title"] = file.title
            # Set the filters instance
            mydict["_v_filters"] = file.filters
            # Save the RootGroup attributes on disk
            setAttr = self._v_attrs._g__setattr
            setAttr('TITLE', file.title)
            setAttr('CLASS', self._c_classId)
            setAttr('VERSION', obversion)
            setAttr('FILTERS', file.filters)
        else:
            # Get the title for the rootGroup group
            if hasattr(self, "TITLE"):
                mydict["_v_title"] = attrsRoot.TITLE
            else:
                mydict["_v_title"] = ""
            # Get all the groups recursively (build the object tree)
            self._g_openFile()


    def _g_closeNodes(self, nodedict):
        """Recursively close all nodes in `nodedict` and their descendents."""

        for node in nodedict.values():
            if isinstance(node, Group):
                # Descend into children (including the very `node`).
                for group in node._f_walkGroups():
                    for leaf in group._f_listNodes(classname='Leaf'):
                        leaf._f_close()
                    group._f_close()
            else:
                node._f_close()


    def _f_close(self):
        # First, close visible nodes.
        self._g_closeNodes(self._v_children)
        # Then, close hidden nodes.
        self._g_closeNodes(self._v_hidden)
        # Finally, close myself.
        super(RootGroup, self)._f_close()


    def _f_rename(self, newname):
        raise NodeError("the root node can not be renamed")

    def _f_move(self, newparent = None, newname = None):
        raise NodeError("the root node can not be moved")

    def _f_remove(self, recursive = False):
        raise NodeError("the root node can not be removed")


    def _g_checkWidth(self):
        # The root node also has hidden children.
        if self._v_nchildren + len(self._v_hidden) >= MAX_GROUP_WIDTH:
            warnings.warn("""\
group ``/`` is exceeding the recommended maximum number of children (%d);\
be ready to see PyTables asking for *lots* of memory and possibly slow I/O"""
                          % (MAX_GROUP_WIDTH,), PerformanceWarning)


    def _g_refNode(self, node, name, validate=True):
        if not name.startswith('_p_'):
            # Visible child.
            super(RootGroup, self)._g_refNode(node, name, validate)
            return

        # Hidden child.
        # Ummm, this is utterly redundant with Group._g_refNode...
        if validate:
            checkNameValidity(name)
        if name in self._v_hidden:
            raise NodeError(
                "group ``%s`` already has a hidden child node named ``%s``"
                % (self._v_pathname, name))
        self._g_checkWidth()
        self._v_hidden[name] = node
        self.__dict__[name] = node


    def _g_unrefNode(self, name):
        if not name.startswith('_p_'):
            # Visible child.
            super(RootGroup, self)._g_unrefNode(name)
            return

        # Hidden child.
        del self._v_hidden[name]
        del self.__dict__[name]


    def __getattr__(self, name):
        try:
            return super(RootGroup, self).__getattr__(name)
        except NoSuchNodeError:
            if name not in self._v_hidden:
                raise  # Use the same error for hidden nodes.
            return self._v_hidden[name]

    # There is no need to redefine __setattr__ as long as _g_refNode is used
    # (as it should be).

    def __contains__(self, name):
        """
        Is there a child with that name?

        Returns ``True`` if the group has a child node with the given
        `name` (be it visible or hidden, a string), ``False`` otherwise.
        """
        if super(RootGroup, self).__contains__(name):
            return True
        return name in self._v_hidden


    def _g_setLocation(self, parent, name):
        # The root group does not need to get location-dependent information.
        pass



class TransactionGroupG(Group):
    _c_classId = 'TRANSGROUP'

    def _g_checkWidth(self):
        if self._v_nchildren >= MAX_GROUP_WIDTH:
            warnings.warn("""\
the number of transactions is exceeding the recommended maximum (%d);\
be ready to see PyTables asking for *lots* of memory and possibly slow I/O"""
                          % (MAX_GROUP_WIDTH,), PerformanceWarning)



class TransactionG(Group):
    _c_classId = 'TRANSG'

    def _g_checkWidth(self):
        if self._v_nchildren >= MAX_GROUP_WIDTH:
            warnings.warn("""\
transaction ``%s`` is exceeding the recommended maximum number of marks (%d);\
be ready to see PyTables asking for *lots* of memory and possibly slow I/O"""
                          % (self._v_pathname, MAX_GROUP_WIDTH),
                          PerformanceWarning)



class MarkG(Group):
    # Class identifier.
    _c_classId = 'MARKG'


    import re
    _c_shadowNameRE = re.compile(r'^a[0-9]+$')


    def _g_checkWidth(self):
        if self._v_nchildren >= MAX_GROUP_WIDTH:
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
