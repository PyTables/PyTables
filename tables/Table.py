########################################################################
#
#       License: BSD
#       Created: September 4, 2002
#       Author:  Francesc Alted - falted@openlc.org
#
#       $Source: /home/ivan/_/programari/pytables/svn/cvs/pytables/pytables/tables/Table.py,v $
#       $Id: Table.py,v 1.7 2002/11/13 20:35:35 falted Exp $
#
########################################################################

"""Here is defined the Table class.

See Table class docstring for more info.

Classes:

    Table

Functions:


Misc variables:

    __version__


"""

__version__ = "$Revision: 1.7 $"

from __future__ import generators
import struct
import types
import re
import string
import hdf5Extension
from Leaf import Leaf
from IsRecord import IsRecord, metaIsRecord


class Table(Leaf, hdf5Extension.Table):
    """Represent a table in the object tree.

    It provides methods to create new tables or open existing ones, as
    well as methods to write/read data and metadata to/from table
    objects over the file.

    Data can be written or read both as records or as tuples. Records
    are recommended because they are more intuitive and less error
    prone although they are slow. Using tuples (or value sequences) is
    faster, but the user must be very careful because when passing the
    sequence of values, they have to be in the correct order
    (alphanumerically ordered by field names). If not, unexpected
    results can appear (most probably ValueError exceptions will be
    raised).

    Methods:

        appendAsRecord(RecordObject)
        appendAsTuple(tupleValues)
        appendAsValues(*values)
        readAsRecords()
        readAsTuples()
        flush()  # This can be moved to Leaf
        close()  # This can be moved to Leaf

    Instance variables:

        name -- the node name
        title -- the title for this node  # This can be moved to Leaf
        record -- the record object for this table
        nrows -- the number of rows in this table
        varnames -- the field names for the table
        vartypes -- the typecodes for the table fields

    """

    def __init__(self, RecordObject = None, title = "",
                 compress = 3, expectedrows = 10000):
        """Create an instance Table.

        Keyword arguments:

        RecordObject -- The IsRecord instance. If None, the table
            metadata is read from disk, else, it's taken from previous
            parameters.

        title -- Sets a TITLE attribute on the HDF5 table entity.

        compress -- Specifies a compress level for data. The allowed
            range is 0-9. A value of 0 disables compression. The
            default is compression level 3, that balances between
            compression effort and CPU consumption.

        expectedrows -- An user estimate about the number of rows
            that will be on table. If not provided, the default value
            is appropiate for tables until 1 MB in size (more or less,
            depending on the record size). If you plan to save bigger
            tables try providing a guess; this will optimize the HDF5
            B-Tree creation and management process time and memory
            used.

        """
        # Initialize this object in case is a new Table
        if RecordObject:
            self.record = RecordObject   # Record points to the RecordObject
            self.title = title
            self._v_compress = compress
            self._v_expectedrows = expectedrows
            # Flag that tells if this table is new or has to be read from disk
            self._v_new = 1
        else:
            self._v_new = 0

    def create(self):
        """Create a new table on disk."""
        
        # Compute some important parameters for createTable
        self.varnames = tuple(self.record.__slots__)
        self._v_fmt = self.record._v_fmt
        self._calcBufferSize(self._v_expectedrows)
        # Create the table on disk
        self.createTable(self.varnames, self._v_fmt, self.title,
                         self._v_compress, self._v_rowsize, self._v_chunksize)
        # Initialize the number of rows
        self.nrows = 0
        # Initialize the shape attribute
        self.shape = (len(self.varnames), self.nrows)
        # Get the variable types
        self.vartypes = re.findall(r'(\d*\w)', self._v_fmt)
                         
    def open(self):
        """Opens a table from disk and read the metadata on it.

        Creates an user Record on the flight to easy the access to the
        actual data.

        """
        # Open the table
        self.openTable()
        #print "Opening table ==> (%s)" % self._v_pathname
        (self.nrows, self.varnames, self._v_fmt) = self.getTableInfo()
        # print "Format for this existing table ==>", self._v_fmt
        # We still have to code how to get this attributes
        self.title = self.getTitle()
        #print "Table Title ==>", self.title
        # This one is probably not necessary to set it, but...
        self._v_compress = 0  # This means, we don't know if compression
                              # is active or not. May be save this info
                              # in a table attribute?
        # Compute buffer size
        self._calcBufferSize(self.nrows)
        # Update the shape attribute
        self.shape = (len(self.varnames), self.nrows)
        # Get the variable types
        self.vartypes = re.findall(r'(\d*\w)', self._v_fmt)
        # Build a dictionary with the types as values and varnames as keys
        recordDict = {}
        i = 0
        for varname in self.varnames:
            recordDict[varname] = self.vartypes[i]
            i += 1
        # Append this entry to indicate the alignment!
        recordDict['_v_align'] = self._v_fmt[0]
        # Create an instance record to host the record fields
        RecordObject = metaIsRecord("", (), recordDict)()
        self.record = RecordObject   # This points to the RecordObject
        
    def _calcBufferSize(self, expectedrows):
        """Calculate the buffer size and the HDF5 chunk size.

        The logic to do that is based purely in experiments playing
        with different buffer sizes, chunksize and compression
        flag. It is obvious that using big buffers optimize the I/O
        speed when dealing with tables. This might (should) be further
        optimized doing more experiments.

        """
        fmt = self._v_fmt
        compress = self._v_compress
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

    def _saveBufferedRows(self):
        """Save buffered table rows."""
        #print "Flusing nrows ==> ", self._v_recunsaved
        self.append_records("".join(self._v_packedtuples), self._v_recunsaved)
        self.nrows += self._v_recunsaved
        # Reset the buffer and the tuple counter
        self._v_packedtuples = []
        self._v_recunsaved = 0
        # Set the shape attribute
        self.shape = (len(self.varnames), self.nrows)
        
    def appendAsRecord(self, RecordObject):
        """Append the "RecordObject" to the output buffer.

        "RecordObject" has to be a IsRecord descendant instance.

        """
        # We should add a test unit case to test that this try works
        # well
        # Test if RecordObject is really a descendant of IsRecord
        if hasattr(RecordObject, "_f_pack"):
            # We pack with _f_pack2 because we don't need to pass parameters
            # and is a bit faster that _f_pack
            self._v_packedtuples.append(RecordObject._f_pack2())
        else:
            raise ValueError, \
                  "arg 1 with type %s, is not a IsRecord descendant." % \
                  type(RecordObject)

        self._v_recunsaved += 1
        if self._v_recunsaved  == self._v_maxTuples:
            self._saveBufferedRows()

    def appendAsTuple(self, tupleValues):
        """Append the "tupleValues" tuple to the table output buffer.
        
        "tupleValues" is a tuple that has values for all the user
        record fields. The user has to provide them in the order
        determined by alphanumerically sorting the record name
        fields. This method is faster (and unsafer, because requires
        user to introduce the values in correct order!) than
        appendAsRecord method.

        """
        self._v_packedtuples.append(struct.pack(self._v_fmt, *tupleValues))
        #self._v_packedtuples.append(self.spacebuffer) # for speed test only
        self._v_recunsaved += 1
        if self._v_recunsaved  == self._v_maxTuples:
            self._saveBufferedRows()

    def appendAsValues(self, *values):
        """ Append the "values" parameters to the table output buffer.
        
        "values" is a serie of parameters that provides values for all
        the user record fields. The user has to provide them in the
        order determined by alphanumerically sorting the record
        fields. This method is faster (and unsafer, because requires
        user to introduce the values in correct order) than
        appendAsRecord method.

        """
        self._v_packedtuples.append(struct.pack(self._v_fmt, *values))
        #self._v_packedtuples.append(self.spacebuffer) # for speed test only
        self._v_recunsaved += 1
        if self._v_recunsaved  == self._v_maxTuples:
            self._saveBufferedRows()

    def readAsRecords(self):
        """Return an iterator yielding record instances built from rows

        This method is a generator, i.e. it keeps track on the last
        record returned so that next time it is invoked it returns the
        next available record. It is slower than readAsTuples but in
        exchange, it returns full-fledged instance records.

        """
        # Create a buffer for the readout
        nrowsinbuf = self._v_maxTuples
        rowsz = self._v_rowsize
        buffer = " " * rowsz * self._v_maxTuples
        vars = self.record   # get the pointer to the Record object
        # Iterate over the table
        for i in xrange(0, self.nrows, nrowsinbuf):
            recout = self.read_records(i, nrowsinbuf, buffer)
            for j in xrange(recout):
                vars._f_unpack(buffer[j*rowsz:(j+1)*rowsz])
                yield vars
                #print "tupla %d ==> %s" % (i + j, tupla)
        
    def readAsTuples(self):
        """Returns an iterator yielding tuples built from rows
        
        This method is a generator, i.e. it keeps track on the last
        record returned so that next time it is invoked it returns the
        next available record. This method is twice as faster than
        readAsRecords, but it yields the rows as (alphanumerically
        orderd) tuples, instead of full-fledged instance records.

        """
        # Create a buffer for the readout
        nrowsinbuf = self._v_maxTuples
        rowsz = self._v_rowsize
        buffer = " " * rowsz * self._v_maxTuples
        # Iterate over the table
        for i in xrange(0, self.nrows, nrowsinbuf):
            recout = self.read_records(i, nrowsinbuf, buffer)
            for j in xrange(recout):
                tupla = struct.unpack(self._v_fmt, buffer[j*rowsz:(j+1)*rowsz])
                yield tupla
                
    def flush(self):
        """Flush the table buffers."""
        if self._v_recunsaved > 0:
          self._saveBufferedRows()

    def close(self):
        """Flush the table buffers and close the HDF5 dataset."""
        self.flush()
        self.closeTable()

