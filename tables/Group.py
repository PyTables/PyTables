import hdf5Extension
from Table import Table

class Group(hdf5Extension.Group):

    def __init__(self):
        self.__dict__["_v_" + "childs"] = []
        self.__dict__["_v_" + "groups"] = []
        self.__dict__["_v_" + "leaves"] = []
        self.__dict__["_v_" + "objgroups"] = {}
        self.__dict__["_v_" + "objleaves"] = {}
        self.__dict__["_v_" + "objchilds"] = {}

    def _f_walkHDFile(self):
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
        "Returns dicts for groups and final nodes hanging from self as dicts"
        groups = self._v_objgroups
        leaves = self._v_objleaves
        return (groups, leaves)

    def _f_getObject(self, where, rootgroup = None):
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
        elif type(where) == type(self):
            # This is the parent group object
            object = where
        else:
            raise TypeError, "Wrong \'where\' parameter type (%s)." % \
                  (type(where))
        return object
        
    def _f_getObjectFromPath(self, pathname):
        #print "Looking for \"%s\" in \"%s\"" % (pathname, self._v_pathname)
        found = 0
        if pathname == self._v_pathname:
            # I'm the group we are looking for. Return it and finish.
            return self
        else:
            for groupobj in self._v_objgroups.itervalues():
                #print "Looking for", pathname, "in", groupobj._v_pathname
                found = groupobj._f_getObjectFromPath(pathname)
                #print "found ==> ", found
                if found: break
            for leaveobj in self._v_objleaves.itervalues():
                #print "Looking for", pathname, "in", leaveobj._v_pathname
                if leaveobj._v_pathname == pathname:
                    found = leaveobj
                    break
        return found

    def _f_join(self, name):
        if self._v_pathname == "/":
            return "/" + name
        else:
            return self._v_pathname + "/" + name
        
    def _f_setproperties(self, name, value):
        "Set some properties for general objects in the tree"
        self._v_childs.append(name)
        self._v_objchilds[name] = self
        # New attributes for the new Group instance
        newattr = value.__dict__
        newattr["_v_" + "rootgroup"] = self._v_rootgroup
        newattr["_v_" + "parent"] = self
        newattr["_v_" + "name"] = name
        newattr["_v_" + "pathname"] = self._f_join(name)

    def _f_updateFields(self, pgroup, name, create):
        "Set attributes for the new Group instance"
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
        pgroup = self._f_getObject(where, rootgroup)
        if name not in pgroup._v_groups:
            self._f_updateFields(pgroup, name, create = 1)
        else:
            #print "Object %s exists!. Doing nothing." % self._f_join(name)
            raise ValueError, "\"%s\" group already has child %s" % \
                  (str(self._v_pathname), name)

    def __setattr__(self, name, value):
        # Alternative way to create Group objects
        # group.name = Group()
        value._f_newGroup(self, name, self._v_rootgroup)
        
    def __delattr__(self, name):
        if name in self._v_leaves:
            print "Add code to delete", name, "attribute"
            #self._v_leaves.remove(name)
        else:
            raise AttributeError, "%s instance has no attribute %s" % \
                  (str(self.__class__), name)

    def __getattr__(self, name):
        #print "Getting the", name, "attribute in Group", self
        if name in self._v_groups:
            return self._v_objgroups[name]
        elif name in self._v_leaves:
            return self._v_objleaves[name]
        else:
            #print "Name ==>", self
            raise AttributeError, "\"%s\" group has not a %s attribute" % \
                  (self._v_pathname, name)

        
