import hdf5Extension
from Table import Table

class Group(hdf5Extension.Group):
    """This is the python counterpart of a group in the HDF5
    structure. It provides methods to set properties based on
    information extracted from the HDF5 files and to walk throughout
    the tree. Every group has parents and childrens, which are all
    Group instances, except for the root group whose parent is a File
    instance."""

    def __init__(self):
        """Create the basic structures to keep group information."""
        
        self.__dict__["_v_" + "childs"] = []
        self.__dict__["_v_" + "groups"] = []
        self.__dict__["_v_" + "leaves"] = []
        self.__dict__["_v_" + "objgroups"] = {}
        self.__dict__["_v_" + "objleaves"] = {}
        self.__dict__["_v_" + "objchilds"] = {}

    def _f_walkHDFile(self):
        """Recusively reads an HDF5 file and generates a tree object
        which will become the python replica of the file hierarchy."""
        
        #print "Group name ==> ", self._v_name
        pgroupId =  self._v_parent._v_groupId
        (groups, leaves) = self._f_listGroup(pgroupId, self._v_name)
        #print "Self ==>", self
        #print "  Groups ==> ", groups
        #print "  Leaves ==> ", leaves
        for name in groups:
            objgroup = Group()
            # Set some attributes to caracterize and open this object
            objgroup._f_updateFields(self, name, create = 0)
            #print "Getting the", objgroup._v_pathname, "group"
            # Call walkHDFile recursively over the group's tree
            objgroup._f_walkHDFile()
        for name in leaves:
            objgroup = Table(self, name, self._v_rootgroup)
            # Set some attributes to caracterize and open this object
            objgroup._f_putObjectInTree(create = 0)
        #print "Group name (end)==> ", self._v_name

    def _f_listObjects(self):
        """Returns groups and final nodes hanging
        from self as a 2-tuple of dictionaries."""
        
        groups = self._v_objgroups
        leaves = self._v_objleaves
        return (groups, leaves)

    def _f_getObject(self, where, rootgroup = None):
        """Get the object hanging from "where". "where" can be a path
        string, or a group instance. If "where" is not a string or a
        Group instance, it raises a TypeError exception. If object
        doesn't exist, it raises a LookupError exception. """
        
        if type(where) == type(str()):
            # This is a string pathname. Get the object ...
            if rootgroup == None:
                assert hasattr(self, "_v_rootgroup"), \
                       "_f_getObject: no way to found the rootgroup!."
                rootgroup = self._v_rootgroup
            object = rootgroup._f_getObjectFromPath(where)
            if not object:
                # We didn't find the pathname in the object tree.
                # This should be signaled as an error!.
                raise LookupError, \
                      "\"%s\" pathname not found in HDF5 group tree." % \
                      (where)
        elif isinstance(where, Group) or isinstance(where, Table):
            # This is the parent group object
            object = where
        else:
            raise TypeError, "Wrong \'where\' parameter type (%s)." % \
                  (type(where))
        return object
        
    def _f_getObjectFromPath(self, pathname):
        """Get the object hanging from "where". "where" must be a path
        string."""
        
        if not pathname.startswith("/"):
            raise LookupError, \
                  "\"%s\" pathname must absolute so must start with /." % \
                  (pathname)
        else:
            found = self._f_getObjectFromPathRecursive(pathname)
            if found:
                return found
            else:
                raise LookupError, \
                      "\"%s\" pathname not found in HDF5 group tree." % \
                      (pathname)

    def _f_getObjectFromPathRecursive(self, pathname):
        """Get the object hanging from "where". "where" must be a path
        string."""
        
        #print "Looking for \"%s\" in \"%s\"" % (pathname, self._v_pathname)
        found = 0
        if pathname == self._v_pathname:
            # I'm the group we are looking for. Return it and finish.
            return self
        else:
            for groupobj in self._v_objgroups.itervalues():
                #print "Looking for", pathname, "in", groupobj._v_pathname
                found = groupobj._f_getObjectFromPathRecursive(pathname)
                #print "found ==> ", found
                if found: break
            for leaveobj in self._v_objleaves.itervalues():
                #print "Looking for", pathname, "in", leaveobj._v_pathname
                if leaveobj._v_pathname == pathname:
                    found = leaveobj
                    break
        return found

    def _f_join(self, name):
        """Helper method to correctly concatenate a name child object
        with the pathname of this group."""
        
        if self._v_pathname == "/":
            return "/" + name
        else:
            return self._v_pathname + "/" + name
        
    def _f_setproperties(self, name, value):
        """Set some properties for general objects in the tree."""
        
        self._v_childs.append(name)
        self._v_objchilds[name] = self
        # New attributes for the new Group instance
        newattr = value.__dict__
        newattr["_v_" + "rootgroup"] = self._v_rootgroup
        newattr["_v_" + "parent"] = self
        newattr["_v_" + "name"] = name
        newattr["_v_" + "pathname"] = self._f_join(name)

    def _f_updateFields(self, pgroup, name, create):
        """Set attributes for a freshly created Group instance."""
        
        # New attributes for the new Group instance
        pgroup._f_setproperties(name, self)
        # Update this instance attributes
        pgroup._v_groups.append(name)
        pgroup._v_objgroups[name] = self
        if create:
            # Call the h5.Group._f_createGroup method to create a new group
            self.__dict__["_v_" + "groupId"] = \
                                self._f_createGroup(pgroup._v_groupId, name)
        else:
            self.__dict__["_v_" + "groupId"] = \
                                  self._f_openGroup(pgroup._v_groupId, name)

    def _f_newGroup(self, where, name, rootgroup):
        """Create a new Group with name "name" in "where"
        location. If "name" group already exists in "where", raise the
        NameError exception."""
        
        pgroup = self._f_getObject(where, rootgroup)
        if name not in pgroup._v_groups:
            self._f_updateFields(pgroup, name, create = 1)
        else:
            raise NameError, "\"%s\" group already has a child named %s" % \
                  (str(self._v_pathname), name)

    def __setattr__(self, name, value):
        """This is another way to create Group objects."""
        
        value._f_newGroup(self, name, self._v_rootgroup)
        
    def __delattr__(self, name):
        """In the future, this should delete objects both in memory
        and in the file."""
        
        if name in self._v_leaves:
            print "Add code to delete", name, "attribute"
            #self._v_leaves.remove(name)
        else:
            raise AttributeError, "%s instance has no attribute %s" % \
                  (str(self.__class__), name)

    def __getattr__(self, name):
        """Get the object named "name" hanging from me."""
        
        #print "Getting the", name, "attribute in Group", self
        if name in self._v_groups:
            return self._v_objgroups[name]
        elif name in self._v_leaves:
            return self._v_objleaves[name]
        else:
            #print "Name ==>", self
            raise AttributeError, "\"%s\" group has not a %s attribute" % \
                  (self._v_pathname, name)

        
