from __future__ import generators
import sys

import hdf5Extension
from Group import Group
from Table import Table
from IsRecord import IsRecord

class File(hdf5Extension.File):
    
    def __init__(self, filename, mode):
        self.filename = filename
        #print "Opening the %s HDF5 file ...." % self.filename
        self._v_mode = mode
        self._v_groupId = self.getFileId()
        return

    # Instance factory
    def getRootGroup(self):
        self.rootgroup = rootgroup = Group()
        # Create new attributes for the root Group instance
        newattr = rootgroup.__dict__
        newattr["_v_" + "rootgroup"] = rootgroup
        newattr["_v_" + "groupId"] = self._v_groupId
        newattr["_v_" + "parent"] = self
        newattr["_v_" + "root"] = 1 # Signals that this is the root group
        newattr["_v_" + "name"] = "/"
        newattr["_v_" + "pathname"] = "/"
        if (self._v_mode == "r" or
            self._v_mode == "r+" or
            self._v_mode == "a"):
            rootgroup._f_walkHDFile()
        return rootgroup

    def newTable(self, where, name, *args, **kwargs):
        object = Table(where, name, self.rootgroup)
        object.newTable(*args, **kwargs)
        return object

    def newGroup(self, where, name):
        object = Group()
        object._f_newGroup(where, name, self.rootgroup)
        return object

    def getNode(self, pathname):
        return self.rootgroup._f_getObject(pathname)
        
    def listNodes(self, pathname):
        group = self.getNode(pathname)
        return group._v_objchilds.iteritems()

    def listGroups(self, pathname):
        group = self.getNode(pathname)
        return group._v_objgroups.iteritems()

    def listLeaves(self, pathname):
        group = self.getNode(pathname)
        return group._v_objleaves.iteritems()
        
    def walkGroups(self, pathname):
        "Recursively obtains groups hanging from pathname"
        group = self.getNode(pathname)
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

    def flush(self):
        "Flush all the objects on this tree"
        # Iterate over the group tree
        for (groupname, groupobj) in self.walkGroups(self.rootgroup):
            # Iterate over leaves in group
            for (leafname, leafobject) in self.listLeaves(groupobj):
                # Call the generic flush for every leaf object
                leafobject.flush()
                
    def close(self):
        "Flush all the objects and close the file"
        # Flush all the buffers
        self.flush()
        #print "Closing the %s HDF5 file ...." % self.filename
        self.closeFile()
        # Add code to recursively delete the object tree
        # (or it's enough with deleting the root group object?)
        #del self.rootgroup

# This class is accessible only for the examples
class Record(IsRecord):
    """ A record has several columns. Represent the here as class
    variables, whose values are their types. The IsRecord
    class will take care the user won't add any new variables and
    that their type is correct.  """
    
    var1 = '4s'
    var2 = 'i'
    var3 = 'd'

def test4(file):
    totalrows = 10
    fast = 1
    title = "Table Title"
    # Create an instance of HDF5 Table
    fileh = File(filename = file, mode = "w")
    # print "FileId ==> ", a.getFileId()
    group = fileh.getRootGroup()
    pathname = "/"
    for j in range(5):
        # Create a table
        #print "Creating a table on -->", group._f_join('tuple'+str(j))
        #c = fileh.newTable(pathname, 'tuple'+str(j),
        c = fileh.newTable(group, 'tuple'+str(j),
                           varnames = ('var1', 'var2', 'var3'),
                           fmt = '3sid', tableTitle = title,
                           compress = 0, expectedrows = totalrows)

        # We can get the leaves objects in this way...
        c = getattr(group, "tuple"+str(j))
        # Fill the table
        for i in xrange(totalrows):
            c.commitBuffered(str(i), i * j, 12.1e10)

        # Flush the table (that will be done automatically when
        # File.close will be recursive)
        c.flush()
        
        # Example of tuple selection
        b = [ tupla[1] for tupla in c.readAllTable() if tupla[1] < 20 ]
        print "Total selected records ==> ", len(b)
        
        # Create a new group
        # Two different ways to do that
        #group2 = group.group = Group()
        #setattr(group, "group" + str(j), Group())
        #group2 = getattr(group, "group"+str(j))
        group2 = fileh.newGroup(pathname, 'group'+str(j))
        #group2 = fileh.newGroup(group, 'group'+str(j))
        

        # Iterate over d
        group = group2
        #group = getattr(group, "group" + str(j))

        if pathname == "/":
            pathname = "/" + "group"+str(j)
        else:
            pathname += "/" + "group"+str(j)
    
    # Close the file (eventually destroy the extended type)
    fileh.close()
    #raise SystemExit

def test5(file):
    # Create an instance of HDF5 Table
    fileh = File(filename = file, mode = "a")
    # print "FileId ==> ", a.getFileId()
    rootgroup = fileh.getRootGroup()
    for (groupname, groupobj) in fileh.walkGroups(rootgroup):
        #print "GroupName ==>", groupname
        print "Group pathname:", groupobj._v_pathname
        #print "Subgroups:", [ name for (name, ob) in fileh.listGroups(groupobj) ]
        #print "Leaves:", [ name for (name, ob) in fileh.listLeaves(groupobj) ]
        for (name, obj) in fileh.listLeaves(groupobj):
            print "Nrecords of table", obj._v_pathname, ":", obj.nrecords
        
    # Close the file (eventually destroy the extended type)
    fileh.close()

def test6(file, totalrows):
    # Create an instance of HDF5 Table
    fileh = File(filename = file, mode = "w")
    # print "FileId ==> ", a.getFileId()
    group = fileh.getRootGroup()
    # Create a table
    c = fileh.newTable(group, 'tuple1',
                       varnames = ('var1', 'var2', 'var3'),
                       fmt = '3sid', tableTitle = "Table Title",
                       compress = 0, expectedrows = totalrows)

    # Fill the table
    for i in xrange(totalrows):
        c.commitBuffered(str(i), i, 12.1e10)
        
    # Close the file (eventually destroy the extended type)
    fileh.close()

def test7(file, totalrows):
    # Create an instance of HDF5 Table
    fileh = File(filename = file, mode = "w")
    # print "FileId ==> ", a.getFileId()
    group = fileh.getRootGroup()
    # Create a table
    c = fileh.newTable(group, 'tuple1', Record(),
                       tableTitle = "This is the Table Title",
                       compress = 0, expectedrows = totalrows)

    # Print some info
    print "Varnames ==>", c.varnames
    print "Table Format ==>", c._v_fmt
    #d = c.getRecord()      # Get the record object
    d = c.record            # another way
    fmt = d.__fmt__  # This is useful for the fastest way to inject records
    print "Record size ==", struct.calcsize("<"+fmt)
    # Fill the table
    for i in xrange(totalrows):
        # All of this versions are supported
        # In the comments are times for saving 1e5 records
        # The next way should be considered the best when balancing
        # safety and speed. This should be documented as the best way.
        d.var1 = str(i)
        d.var2 = i
        d.var3 = 12.1e10
        #c.commitBuffered(d)      # This injects the Record values      # 6.7s
        # This oter way is elegant, but a bit inefficient
        #c.commitBuffered( d( var1=str(i), var2 = i, var3 = 12.1e10 )) # 9.0s
        # This is not very useful
        #c.commitBuffered(d())   # This injects the defaults           # 6.5s
        # This one is dangerous, but very quick
        #c.commitBuffered(d(str(i), i, 12.1e10))                       # 4.8s
        # This very quick and simple
        #c.commitBuffered((str(i), i, 12.1e10))                        # 4.4s
        # This is explicitly forbidden
        #c.commitBuffered(d(str(i), i, var3 = 12.1e10))
        # This is the quickest, but dangerous and also the ugliest
        #c.commitBuffered(struct.pack(fmt, str(i), i, 12.1e10))        # 4.0s
        # Just a speed test. This is the maximum speed achivable without calls
        # to struct.pack (for a "4sld" format)
        #c.commitBuffered(" " * 16)        # 3.0s
        
    # Close the file (eventually destroy the extended type)
    fileh.close()

def test8(file):
    # Create an instance of HDF5 Table
    fileh = File(filename = file, mode = "r")
    # print "FileId ==> ", a.getFileId()
    root = fileh.getRootGroup()
    c = root.tuple0
    print "Number of records of c ==>", c.nrecords
    
    # Example of tuple selection
    #e = [ t[1] for t in c.readAsTuples() if t[1] < 20 ]
    # Record method
    e = [ p.var2 for p in c.readAsRecords() if p.var2 < 20 ]
    
    print "Total selected records ==> ", len(e)
    print "Last record ==>", p
    fileh.close()

def test9(file, totalrows, fast):
    title = "This is the table title"
    # Create an instance of HDF5 Table
    fileh = File(filename = file, mode = "w")
    group = fileh.getRootGroup()
    for j in range(3):
        # Create a table
        table = fileh.newTable(group, 'tuple'+str(j), Record(),
                           tableTitle = title, compress = 0,
                           expectedrows = totalrows)
        # Get the record object associated with the new table
        d = table.record 
        # Fill the table
        for i in xrange(totalrows):
            if fast:
                table.appendValues(str(i), i * j, 12.1e10)
            else:
                d.var1 = str(i)
                d.var2 = i * j
                d.var3 = 12.1e10
                table.appendRecord(d)      # This injects the Record values
                #table.appendRecord(d())     # The same, but slower
                
        # Create a new group
        group2 = fileh.newGroup(group, 'group'+str(j))
        # Iterate over this new group (group2)
        group = group2
    
    # Close the file (eventually destroy the extended type)
    fileh.close()

def test10(filename, fast):
    # Open the HDF5 file in read-only mode
    fileh = File(filename = filename, mode = "r")
    rootgroup = fileh.getRootGroup()
    for (groupname, groupobj) in fileh.walkGroups(rootgroup):
        #print "Group pathname:", groupobj._v_pathname
        for (name, table) in fileh.listLeaves(groupobj):
            #print "Table title for", table._v_pathname, ":", table.tableTitle
            print "Nrecords in", table._v_pathname, ":", table.nrecords

            if fast:
                # Example of tuple selection (fast version)
                e = [ t[1] for t in table.readAsTuples() if t[1] < 20 ]
            else:
                # Record method (slow, but convenient)
                e = [ p.var2 for p in table.readAsRecords() if p.var2 < 20 ]
                # print "Last record ==>", p
    
            print "Total selected records ==> ", len(e)
        
    # Close the file (eventually destroy the extended type)
    fileh.close()

def test11(filename, addedrows, fast):
    """ Test for adding rows: it doesn't work! """
    # Open the HDF5 file in read-only mode
    fileh = File(filename = filename, mode = "a")
    rootgroup = fileh.getRootGroup()
    for (groupname, groupobj) in fileh.walkGroups(rootgroup):
        #print "Group pathname:", groupobj._v_pathname
        for (name, table) in fileh.listLeaves(groupobj):
            #print "Table title for", table._v_pathname, ":", table.tableTitle
            print "Nrecords in old", table._v_pathname, ":", table.nrecords

            # Get the record object associated with the new table
            d = table.record 
            #print "Record Format ==>", d._v_fmt
            #print "Table Format ==>", table._v_fmt
            # Fill the table
            for i in xrange(addedrows):
                if fast:
                    table.appendValues(str(i), i, 12.1e10)
                else:
                    d.var1 = str(i)
                    d.var2 = i
                    d.var3 = 12.1e10
                    table.appendRecord(d)      # This injects the Record values
            # Flush buffers to disk (may be commented out, but it shouldn't)
            table.flush()   
                            
            if fast:
                # Example of tuple selection (fast version)
                e = [ t[1] for t in table.readAsTuples() if t[1] < 20 ]
            else:
                # Record method (slow, but convenient)
                e = [ p.var2 for p in table.readAsRecords() if p.var2 < 20 ]
                print "Last record ==>", p
    
            print "Total selected records in new table ==> ", len(e)
        
    # Close the file (eventually destroy the extended type)
    fileh.close()

# Add code to test here
if __name__=="__main__":
    # Create the file
    #test4(sys.argv[1])
    # Read and process the file
    #test5(sys.argv[1])
    #test6(sys.argv[1], int(sys.argv[2]))
    #test7(sys.argv[1], int(sys.argv[2]))
    #test8(sys.argv[1])
    fast = 0
    if len(sys.argv) > 3:
        fast = int(sys.argv[3])
    test9(sys.argv[1], int(sys.argv[2]), fast)
    test10(sys.argv[1], fast)
    test11(sys.argv[1], int(sys.argv[2]) * 2, fast)
