from __future__ import generators
import struct
import types
import re
import string
import hdf5Extension
from IsRecord import IsRecord, metaIsRecord

class Table(hdf5Extension.Table):

    def __init__(self, where, name, rootgroup):
        # Initialize the superclass
        self._v_name = name
        if type(where) == type(str()):
            # This is the parent group pathname. Get the object ...
            objgroup = rootgroup._f_getObjectFromPath(where)
            if objgroup:
                self._v_parent = objgroup
            else:
                # We didn't find the pathname in the object tree.
                # This should be signaled as an error!.
                raise LookupError, \
                      "\"%s\" pathname not found in the HDF5 group tree." % \
                      (where)

        elif type(where) == type(rootgroup):
            # This is the parent group object
            self._v_parent = where
        else:
            raise TypeError, "where parameter type (%s) is inappropriate." % \
                  (type(where))

    def getRecord(self):
        """ Return the recordObject object """
        return self.record

    def newTable(self, recordObject, tableTitle,
                 compress = 1, expectedrows = 10000):
        self.varnames = tuple(recordObject.__slots__)
        self._v_fmt = recordObject._v_fmt
        self.record = recordObject   # Record points to the recordObject
        self.tableTitle = tableTitle
        self._v_compress = compress
        self.calcBufferSize(expectedrows, compress)
        # Create an instance attribute for each variable
        # In the future this should be changed to H5Array objects
        for var in self.varnames:
            #self.__dict__[var] = None
            setattr(self, var, None)
        # Initialize the number of records
        self.nrecords = 0
        # Create the group
        self._f_putObjectInTree(create = 1)

    def _f_putObjectInTree(self, create):
        pgroup = self._v_parent
        # Update this instance attributes
        pgroup._v_leaves.append(self._v_name)
        pgroup._v_objleaves[self._v_name] = self
        # New attributes for the new Table instance
        pgroup._f_setproperties(self._v_name, self)
        self._v_groupId = pgroup._v_groupId
        if create:
            # Call the _h5.Table method to create the table on disk
            self.createTable(self.varnames, self._v_fmt,
                             self.tableTitle, self._v_compress,
                             self._v_rowsize, self._v_chunksize)
            
        else:
            # Open the table
            self.openTable()
            #print "Opening table ==> (%s)" % self._v_pathname
            (self.nrecords, self.varnames, self._v_fmt) = self.getTableInfo()
            # print "Format for this existing table ==>", self._v_fmt
            # We still have to code how to get this attributes
            self.tableTitle = self.getTableTitle()
            #print "Table Title ==>", self.tableTitle
            # This one is probably not necessary to set it, but...
            self._v_compress = 0  # This means, we don't know if compression
                                   # is active or not. May be save this info
                                   # in a table attribute?
            # Compute buffer size
            self.calcBufferSize(self.nrecords, self._v_compress)
            # Get the variable types
            self.types = re.findall(r'(\d*\w)', self._v_fmt)
            # Build a dictionary with the types as values and varnames as keys
            recDict = {}
            i = 0
            for varname in self.varnames:
                recDict[varname] = self.types[i]
                i += 1
            # Append this entry to indicate the alignment!
            recDict['_v_align'] = self._v_fmt[0]
            # Create an instance record to host the record fields
            recordObject = metaIsRecord("",(),recDict)() # () is important!
            self.record = recordObject   # This points to the recordObject
        
    def calcBufferSize(self, expectedrows, compress):
        fmt = self._v_fmt
        rowsize = struct.calcsize(fmt)
        self._v_rowsize = rowsize
        self.spacebuffer = " " * rowsize
        # List to collect binary tuples
        self._v_packedtuples = []
        self._v_recunsaved = 0
        if fmt[0] not in "@=<>!":
            rowsizeinfile = struct.calcsize("=" + fmt)
        else:
            rowsizeinfile = rowsize
        #print "Creating the table in file ==> ", self.file
        #print "Row size ==> ", rowsize
        #print "Row size in file ==> ", rowsizeinfile
        expectedfsizeinKb = (expectedrows * rowsizeinfile) / 1024
        #print "Expected data rows ==> ", expectedrows
        #print "Expected data set (no compress) ==> ", expectedfsizeinKb, "KB"

        # Some code to compute appropiate values for chunksize & buffersize
        # chunksize:  The chunksize for the HDF5 library
        # buffersize: The Table internal buffer size
        #
        # Reasoning: HDF5 takes the data in bunches of chunksize length
        # to write the on disk. A BTree in memory is used to map structures
        # on disk. The more chunks that are allocated for a dataset the
        # larger the B-tree. Large B-trees take memory and causes file
        # storage overhead as well as more disk I/O and higher contention
        # for the meta data cache.
        # You have to balance between memory and I/O overhead (small B-trees)
        # and time to access to data (big B-trees).
        #
        # The tuning of the chunksize & buffersize parameters affects the
        # performance and the memory size consumed. This is based on numerical
        # experiments on a Intel (Athlon 900MHz) arquitecture and, as always,
        # your mileage may vary.
        
        if expectedfsizeinKb <= 100:
            # Values for files less than 100 KB of size
            buffersize = 5000
            chunksize = 1024
        elif (expectedfsizeinKb > 100 and
            expectedfsizeinKb <= 1000):
            # Values for files less than 1 MB of size
            buffersize = 20000
            chunksize = 2048
        elif (expectedfsizeinKb > 1000 and
              expectedfsizeinKb <= 20 * 1000):
            # Values for sizes between 1 MB and 20 MB
            buffersize = 40000
            chunksize = 4096
        elif (expectedfsizeinKb > 20 * 1000 and
              expectedfsizeinKb <= 200 * 1000):
            # Values for sizes between 20 MB and 200 MB
            buffersize = 50000
            chunksize = 8192
        else:  # Greater than 200 MB
            # This values gives an increment of memory of 50 MB for a table
            # size of 2.2 GB. I think this increment should be attributed to
            # the BTree created to save the table data.
            # If we increment this values more than that, the HDF5 takes
            # considerably more CPU. If you don't want to spend 50 MB
            # (or more, depending on the final table size) to
            # the BTree, and want to save files bigger than 2 GB,
            # try to increment this values, but be ready for a quite big
            # overhead needed to traverse the BTree.
            buffersize = 60000
            chunksize = 16384
        # Correction for compression. Double the chunksize
        # to improve compression level
        if compress:
            chunksize *= 2
        # Max Tuples to fill the buffer
        self._v_maxTuples = buffersize / rowsize
        self._v_chunksize = chunksize
        #print "Buffersize ==> ", buffersize
        #print "Chunksize  ==> ", chunksize
        #print "MaxTuples ==> ", self._v_maxTuples

    def saveBufferedRecords(self):
        "Save buffered table records"
        #print "Flusing nrecords ==> ", self._v_recunsaved
        self.append_records("".join(self._v_packedtuples), self._v_recunsaved)
        self.nrecords += self._v_recunsaved
        # Reset the buffer and the tuple counter
        self._v_packedtuples = []
        self._v_recunsaved = 0
        
    #def commitBuffered(self, recordObject):
    def appendValues(self, *values):
        """ Append the (alphanumerically ordered) values parameters in the
        output buffer as if they are a record."""
        # By using list of strings rather than increasing each time a
        # monolithic string buffer, we obtain a performance improvement
        # between 20% and 25% in time. In exchange, we consume slightly more
        # memory.
        self._v_packedtuples.append(struct.pack(self._v_fmt, *values))
        #self._v_packedtuples.append(self.spacebuffer) # for speed test only
        self._v_recunsaved += 1
        #print "Values without saving -->", self._v_recunsaved 
        if self._v_recunsaved  == self._v_maxTuples:
            self.saveBufferedRecords()

    def appendRecord(self, recordObject):
        """ Append the record object in the output buffer."""
        # I don't know how to check that record is IsRecord instance in
        # the case we create the object programatically.
        # Anyway, hasattr() should do the job!
        ##if isinstance(recordObject, metaIsRecord):
        if hasattr(recordObject, "_f_pack"):
            # We pack with _f_pack2 because we don't need to pass parameters
            # and is a bit faster that _f_pack
            self._v_packedtuples.append(recordObject._f_pack2())
        elif isinstance(recordObject, types.StringType):
            self._v_packedtuples.append(recordObject)
        else:
            print type(recordObject)
            raise RuntimeError, \
                  "commit parameter is neither a string or IsRecord instance."
        self._v_recunsaved += 1
        if self._v_recunsaved  == self._v_maxTuples:
          self.saveBufferedRecords()

    def readAsTuples(self):
        """  Generator to return a tuple with a table record in each cycle.
        This method is twice faster than readAsRecords, but it yields
        the records as tables, instead of full-object records.
        """
        # Create a buffer for the readout
        nrecordsinbuf = self._v_maxTuples
        rowsz = self._v_rowsize
        buffer = " " * rowsz * self._v_maxTuples
        # Iterate over the table
        for i in xrange(0, self.nrecords, nrecordsinbuf):
            recout = self.read_records(i, nrecordsinbuf, buffer)
            for j in xrange(recout):
                tupla = struct.unpack(self._v_fmt, buffer[j*rowsz:(j+1)*rowsz])
                #print "tupla %d ==> %s" % (i + j, tupla)
                yield tupla
        
    def readAsRecords(self):
        """  Generator to return a IsRecord instance with a table record
        in each cycle. This method is slower than readAsTuples, but in
        exchange, it return full-fledged instance records.
        """
        # Create a buffer for the readout
        nrecordsinbuf = self._v_maxTuples
        rowsz = self._v_rowsize
        buffer = " " * rowsz * self._v_maxTuples
        vars = self.record   # get the pointer to the Record object
        # Iterate over the table
        for i in xrange(0, self.nrecords, nrecordsinbuf):
            recout = self.read_records(i, nrecordsinbuf, buffer)
            for j in xrange(recout):
                vars._f_unpack(buffer[j*rowsz:(j+1)*rowsz])
                yield vars
                #print "tupla %d ==> %s" % (i + j, tupla)
        
    def flush(self):
        "Save whatever remaining records in buffer"
        if self._v_recunsaved > 0:
          #print "Flushing the table ==>", self._v_pathname
          self.saveBufferedRecords()

    def close(self):
        """ Close the table. """
        print "Flushing the HDF5 table ...."
        self.flush()
        self.closeTable()
        
