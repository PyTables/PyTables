from __future__ import generators
import sys

import hdf5Extension
from Group import Group
from Table import Table
from IsRecord import IsRecord

class File(hdf5Extension.File):
    """This class hosts the most part of PyTables API. It is in charge
    of open, flush and close the HDF5 files. In addition it provides
    accessors to functionality present in Group and Table module."""
    
    def __init__(self, name, mode="r"):
        """Open an HDF5 file. The supported access modes are: "r"
        means read-only; no data can be modified. "w" means write; a
        new file is created, an existing file with the same name is
        deleted. "a" means append (in analogy with serial files); an
        existing file is opened for reading and writing, and if the
        file does not exist it is created. "r+" is similar to "a", but
        the file must already exist."""

        self.name = name
        #print "Opening the %s HDF5 file ...." % self.name
        self._v_mode = mode
        self._v_groupId = self.getFileId()
        return

    def getRootGroup(self):
        """Returns a Group instance which will act as the root group
        in the hierarchical tree. If file is opened in "r", "r+" or
        "a" mode, and the file already exists, this method dynamically
        builds a python object tree emulating the structure present on
        file.  """
          
        # root is a standard attribute
        self.root = rootgroup = Group()
        # Create new attributes for the root Group instance
        newattr = rootgroup.__dict__
        newattr["_v_rootgroup"] = rootgroup  # For compatibility with Group
        newattr["_v_" + "groupId"] = self._v_groupId
        newattr["_v_" + "parent"] = self
        newattr["_v_" + "name"] = "/"
        newattr["_v_" + "pathname"] = "/"
        if (self._v_mode == "r" or
            self._v_mode == "r+" or
            self._v_mode == "a"):
            rootgroup._f_walkHDFile()
        return rootgroup

    def newTable(self, where, name, *args, **kwargs):
        """Returns a new Table instance with name "name" in "where"
        location.  "where" parameter can be a path string, or another
        group instance.  Other optional parameters are: "tableTitle"
        which set a TITLE attribute on the HDF5 table entity.
        "compress" is a boolean option and specifies if data
        compression will be enabled or not. "expectedrows" is an user
        estimate about the number of records that will be on
        table. This parameter is used to set important internal
        parameters, as buffer size or HDF5 chunk size. If not
        provided, the default value is appropiate to tables until 1 MB
        in size. If you plan to save bigger tables by providing a
        guess to PyTables will optimize the HDF5 B-Tree creation and
        management process time and memory used."""
        
        object = Table(where, name, self.root)
        object.newTable(*args, **kwargs)
        return object

    def newGroup(self, where, name):
        """Returns a new Group instance with name "name" in "where"
	  location. "where" parameter can be a path (for example
	  "/Particles/TParticle1" string, or another Group
	  instance."""

        object = Group()
        object._f_newGroup(where, name, self.root)
        return object

    def getNode(self, where):
        """Returns the object node (group or leave) in "where"
        location. "where" can be a path string, group instance or
        table instance."""
        
        return self.root._f_getObject(where)
        
    def getGroup(self, where):
        """Returns the object node (group or leave) in "where"
        location. "where" can be a path string or group instance. If
        where doesn't point to a Group, a ValueError error is
        raised."""
        
        group = self.root._f_getObject(where)
        if isinstance(group, Group):
            return table
        else:
            raise ValueError, \
                  "%s parameter should be a path to a Group or Group instance." % group
        
    def getTable(self, where):
        """Returns the object node (group or leave) in "where"
        location. "where" can be a path string or table instance.  If
        where doesn't point to a Group, a ValueError error is
        raised."""
        
        table = self.root._f_getObject(where)
        if isinstance(table, Table):
            return table
        else:
            raise ValueError, \
                  "%s parameter should be a path to a Table or Table instance." % table
        
    def listNodes(self, where):
        """Returns all the object nodes (groups or tables) hanging
        from "where". "where" can be a path string or group
        instance."""
        
        group = self.getNode(where)
        return group._v_objchilds.iteritems()

    def listGroups(self, where):
        """Returns all the groups hanging from "where". "where" can be
        a path string or group instance."""
        
        group = self.getNode(where)
        return group._v_objgroups.iteritems()

    def listLeaves(self, where):
        """Returns all the Leaves (Tables at the moment) hanging
        from "where". "where" can be a path string or group
        instance."""
        
        group = self.getNode(where)
        return group._v_objleaves.iteritems()
        
    def walkGroups(self, where):
        """Recursively obtains groups (not leaves) hanging from
        "where"."""
        
        group = self.getNode(where)
        # Returns this group
        yield(group._v_name, group)
        groups = group._v_objgroups.items()
        #print "Groups: %s" % (groups)
        for (groupname, groupobj) in groups:
            # This syntax is semantically incorrect
            #yield self.walkGroups(groupobj)
            # Use this!
            for x in self.walkGroups(groupobj):
                # Iterate over groupobj
                yield x

    def getRecordObject(self, table):
        """Returns the record object associated with the "table".
        "table" can be a path string or table instance."""
        
        table = self.getTable(table)
        return table.record
    
    def appendRecord(self, table, record):
        """ Append the "record" object to the "table" output
        buffer. "table" can be a path string or table instance."""
        
        table = self.getTable(table)
        return table.appendRecord(record)

    def readRecords(self, table):
        """Generator thats return a Record instance from a "table"
        object each time it is called. "table" can be a path string or
        table instance."""
        
        table = self.getTable(table)
        return table.readAsRecords()
        
    def flushTable(self, table):
        """Flush the table object to disk. "table" can be a path
        string or table instance."""
        
        table = self.getTable(table)
        return table.flush()
        
    def flush(self):
        """Flush all the objects on all the HDF5 objects tree."""
        # Iterate over the group tree
        for (groupname, groupobj) in self.walkGroups(self.root):
            # Iterate over leaves in group
            for (leafname, leafobject) in self.listLeaves(groupobj):
                # Call the generic flush for every leaf object
                leafobject.flush()
                
    def close(self):
        """Flush all the objects in HDF5 file and close the file."""
        # Flush all the buffers
        self.flush()
        #print "Closing the %s HDF5 file ...." % self.name
        self.closeFile()
        # Add code to recursively delete the object tree
        # (or it's enough with deleting the root group object?)
        #del self.root

