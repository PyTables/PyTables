########################################################################
#
#       License: BSD
#       Created: September 4, 2002
#       Author:  Francesc Alted - falted@openlc.org
#
#       $Source: /home/ivan/_/programari/pytables/svn/cvs/pytables/pytables/tables/Table.py,v $
#       $Id: Table.py,v 1.26 2003/02/24 15:57:49 falted Exp $
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

__version__ = "$Revision: 1.26 $"

from __future__ import generators
import sys
import struct
import types
import re
import copy
import string
from numarray import *
import chararray
import recarray
import recarray2         # Private version of recarray for PyTables
import hdf5Extension
from Leaf import Leaf
from IsRecord import IsRecord, metaIsRecord, Col, fromstructfmt

byteorderDict={"=": sys.byteorder,
               "@": sys.byteorder,
               '<': 'little',
               '>': 'big'}

revbyteorderDict={'little': '<',
                  'big': '>'}

class Row:
    """Row Class

    This class hosts accessors to a recarray row.
    
    """

    def __init__(self, input):

        self.__dict__["_array"] = input
        self.__dict__["_fields"] = input._fields
        self.__dict__["_row"] = 0

    def __call__(self, row):
        """ set the row for this record object """
        
        if row < self._array.shape[0]:
            self.__dict__["_row"] = row
            return self
        else:
            return None

    def __getattr__(self, fieldName):
        """ get the field data of the record"""

        # In case that the value is an array, the user should be responsible to
        # copy it if he wants to keep it.
        try:
            #value = self._fields[fieldName][self._row]
            # If we use this line:
            #return self._fields[fieldName][self._row]
            # we get weird endless memory leaks.
            # The next one works just perfect. But, why?
            return self.__dict__["_fields"][fieldName][self.__dict__['_row']]
            #return -1
            #return self._array.field(fieldName)[self._row]
        except:
            (type, value, traceback) = sys.exc_info()
            raise AttributeError, "Error accessing \"%s\" attr.\n %s" % \
                  (fieldName, "Error was: \"%s: %s\"" % (type,value))

        if isinstance(value, num.NumArray):
            return copy.deepcopy(value)
        else:
             return value

    def __setattr__(self, fieldName, value):
        """ set the field data of the record"""

        #self._fields[fieldName][self._row] = value
        self.__dict__["_fields"][fieldName][self.__dict__['_row']] = value
        #self._array.field(fieldName)[self._row] = value

    def __str__(self):
        """ represent the record as an string """
        
        outlist = []
        for name in self._array._names:
            outlist.append(`self._fields[name][self._row]`)
            #outlist.append(`self._array.field(name)[self._row]`)
        return "(" + ", ".join(outlist) + ")"

    def _all(self):
        """ represent the record as a list """
        
        outlist = []
        for name in self._fields:
            outlist.append(self._fields[name][self._row])
            #outlist.append(self._array.field(name)[self._row])
        return outlist

    # Moved out of scope
    def _f_del__(self):
        print "Deleting Row object"
        pass

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

        append(RecordObject)
        fetchall()
        flush()  # This can be moved to Leaf
        close()  # This can be moved to Leaf

    Instance variables:

        name -- the node name
        title -- the title for this node  # This can be moved to Leaf
        description -- the metaobject for this table (can be a dictionary)
        record -- A pointer to the current record object
        nrows -- the number of rows in this table
        colnames -- the field names for the table
        coltypes -- the type class for the table fields

    """

    def __init__(self, description = None, title = "",
                 compress = 0, expectedrows = 10000):
        """Create an instance Table.

        Keyword arguments:

        description -- The IsRecord instance. If None, the table metadata
            is read from disk, else, it's taken from previous
            parameters. It can be a dictionary where the keys are the
            field names, and the values the type definitions. And it
            can be also a RecArray object (from recarray module).

        title -- Sets a TITLE attribute on the HDF5 table entity.

        compress -- Specifies a compress level for data. The allowed
            range is 0-9. A value of 0 disables compression. The
            default is 0 (no compression).

        expectedrows -- An user estimate about the number of rows
            that will be on table. If not provided, the default value
            is appropiate for tables until 1 MB in size (more or less,
            depending on the record size). If you plan to save bigger
            tables try providing a guess; this will optimize the HDF5
            B-Tree creation and management process time and memory
            used.

        """

        # Common variables
        self.title = title
        self._v_compress = compress
        self._v_expectedrows = expectedrows
        # Initialize the number of rows to a default
        self.nrows = 0
        
        # Initialize this object in case is a new Table
        if isinstance(description, types.DictType):
            # Dictionary case
            self.description = metaIsRecord("", (), description)()
            # Flag that tells if this table is new or has to be read from disk
            self._v_new = 1
        elif isinstance(description, recarray.RecArray):
            # RecArray object case
            self.newRecArray(description)
            # Provide a better guess for the expected number of rows
            # But beware with the small recarray lengths!
            # Commented out until a better approach is found
            #if self._v_expectedrows == expectedrows:
            #    self._v_expectedrows = self.nrows
            # Flag that tells if this table is new or has to be read from disk
            self._v_new = 1
        elif description:
            # IsRecord subclass case
            self.description = description
            # Flag that tells if this table is new or has to be read from disk
            self._v_new = 1
        else:
            self._v_new = 0

    def newBuffer(self, init=0):
        """Create a new recarray buffer for I/O purposes"""

        recarr = recarray2.array(None, formats=self.description._v_recarrfmt,
                                shape=(self._v_maxTuples,),
                                names = self.colnames)
        # Initialize the recarray with the defaults in description
        if init:
            for field in self.description.__slots__:
                recarr._fields[field][:] = self.description.__dflts__[field]
        #self.arrlist = []
        #for col in self.varnames:
        #    self.arrlist.append(self.arrdict[col])

        return recarr

    def newRecArray(self, recarr):
        """Save a recarray to disk, and map it as a Table object

        This method is aware of byteswapped and non-contiguous recarrays
        """

        # Check if recarray is discontigous:
        if not recarr.iscontiguous():
            # Make a copy to ensure that it is contiguous
            # We always should make a copy because I think that
            # HDF5 does not support strided buffers, but just offsets
            # between fields
            recarr = recarr.copy()
        # Initialize the number of rows
        self.nrows = len(recarr)
        # If self._v_recarray exists, and has data, it would be marked as
        # the initial buffer
        if self.nrows > 0:
            self._v_recarray = recarr
        self.colnames = recarr._names
        fields = {}
        for i in range(len(self.colnames)):
            fields[self.colnames[i]] = Col(recarr._fmt[i],
                                           recarr._repeats[i],
                                           pos=i)  # Position matters!
        # Set the byteorder
        self._v_byteorder = recarr._byteorder
        # Append this entry to indicate the alignment!
        fields['_v_align'] = revbyteorderDict[recarr._byteorder]
        # Create an instance description to host the record fields
        self.description = metaIsRecord("", (), fields)()
        # The rest of the info is automatically added when self.create()
        # is called

    def create(self):
        """Create a new table on disk."""

        # Compute some important parameters for createTable
        self.colnames = tuple(self.description.__slots__)
        self._v_fmt = self.description._v_fmt
        self._calcBufferSize(self._v_expectedrows)
        # Create the table on disk
        self.createTable(self.title)
        # Initialize the shape attribute
        self.shape = (self.nrows,)
        # Get the column types
        self.coltypes = self.description._v_formats
        # Extract the shapes for columns
        self.colshapes = self.description._v_shapes
        # Compute the byte order
        self._v_byteorder = byteorderDict[self._v_fmt[0]]
        # Create the arrays for buffering
        self._v_buffer = self.newBuffer()
        self.row = hdf5Extension.Row(self._v_buffer)
        #self.row = Row(self._v_buffer)
                         
    def open(self):
        """Opens a table from disk and read the metadata on it.

        Creates an user description on the flight to easy the access to
        the actual data.

        """
        # Get table info
        (self.nrows, self.colnames, self._v_fmt) = self.getTableInfo()
        self.title = self.getAttrStr("TITLE")
        # This one is probably not necessary to set it, but...
        self._v_compress = 0  # This means, we don't know if compression
                              # is active or not. May be save this info
                              # in a table attribute?
        # Compute buffer size
        self._calcBufferSize(self.nrows)
        # Update the shape attribute
        self.shape = (self.nrows,)
        # Get the variable types
        lengthtypes = re.findall(r'(\d*\w)', self._v_fmt)
        # Build a dictionary with the types as values and colnames as keys
        fields = {}
        for i in range(len(self.colnames)):
            try:
                length = int(lengthtypes[i][:-1])
            except:
                length = 1
            vartype = fromstructfmt[lengthtypes[i][-1]]
            fields[self.colnames[i]] = Col(vartype, length, pos = i)

        # Append this entry to indicate the alignment!
        fields['_v_align'] = self._v_fmt[0]
        self._v_byteorder = byteorderDict[self._v_fmt[0]]
        # Create an instance description to host the record fields
        self.description = metaIsRecord("", (), fields)()
        # Extract the coltypes
        self.coltypes = self.description._v_formats
        # Extract the shapes for columns
        self.colshapes = self.description._v_shapes
        # Create the arrays for buffering
        self._v_buffer = self.newBuffer(init=0)
        #self.row = self._v_buffer._row
        self.row = hdf5Extension.Row(self._v_buffer)
        
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
        #rowsize = self.description._v_record.itemsize()
        self._v_rowsize = rowsize
        # Counter for the binary tuples
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
        #print "Buffersize, MaxTuples ==>", buffersize, self._v_maxTuples
        self._v_chunksize = chunksize

    def _saveBufferedRows(self):
        """Save buffered table rows."""
        # The next two lines are very similar in performance!
        #self.append_records0(str(recarr._data), self._v_recunsaved)
        self.append_records(self._v_buffer, self._v_recunsaved)
        self.nrows += self._v_recunsaved
        # Reset the buffer and the tuple counter
        self._v_recunsaved = 0
        # Set the shape attribute (the self.nrows may be less than the maximum)
        self.shape = (self.nrows,)
        
    def append(self, row):
        """Append the "row" object to the output buffer.

        "row" has to be a recarray2.Row object 

        """
        self._v_recunsaved += 1
        #row.__dict__["_row"] = self._v_recunsaved
        #row.setRow(self._v_recunsaved)
        row.incRow()
        if self._v_recunsaved  == self._v_maxTuples:
            self._saveBufferedRows()
            # Reset the recarray row counter
            #row.__dict__["_row"] = 0
            row.setRow(0)

    def fetchall(self):
        """Return an iterator yielding record instances built from rows

        This method is a generator, i.e. it keeps track on the last
        record returned so that next time it is invoked it returns the
        next available record. It is slower than readAsTuples but in
        exchange, it returns full-fledged instance records.

        """
        # Create a buffer for the readout
        nrowsinbuf = self._v_maxTuples
        buffer = self._v_buffer  # Get a recarray as buffer
        row = self.row   # get the pointer to the Row object
        #rowdict = row.__dict__
        #self.nrow = 0
        for i in xrange(0, self.nrows, nrowsinbuf):
            recout = self.read_records(i, nrowsinbuf, buffer)
            if self._v_byteorder <> sys.byteorder:
                buffer.byteswap()
            row.setNBuf(i)
            row.setRow(-1)
            for j in xrange(recout):
                #rowdict["_row"] = j
                #self.nrow = i + j  # This line is faster
                #self.nrow += 1
                yield row()
                #row.incRow()
        
    def getRows(self, start, stop, step = 1):
        # Create a recarray for the readout
        if stop > self.nrows:
            stop = self.nrows
        nrows = ((stop - start - 1) // step) + 1
        # Create the resulting recarray
        result = recarray.array(None, formats=self.description._v_recarrfmt,
                                shape=(nrows,),
                                names = self.colnames)
        # Setup a buffer for the readout
        nrowsinbuf = self._v_maxTuples   # Shortcut
        #nrowsinbuf = 3   # Small value is useful when debugging
        buffer = self._v_buffer  # Get a recarray as buffer
        nrowsread = start
        startr = 0
        gap = 0
        nextelement = start
        # This a efficient, although somewhat complicated algorithm
        # so as to get a selection of a table using an extended range
        # May be is it possible to simplify it??
        for i in xrange(start, stop, nrowsinbuf):
            if nextelement >= nrowsread + nrowsinbuf:
                nrowsread += nrowsinbuf
                continue
            startb = gap
            stopb = stop - nrowsread
            if stopb > nrowsinbuf:
                stopb = nrowsinbuf
            stopr = startr + ((stopb-startb-1)//step) + 1
            if stopr > nrows:
                break
            #print "startb, stopb, startr, stopr",\
            #      startb, stopb, startr, stopr
            recout = self.read_records(i, nrowsinbuf, buffer)
            result[startr:stopr] = buffer[startb:stopb:step]
            nrowsread += nrowsinbuf
            startr = stopr
            if step < nrowsinbuf:
                gap = (stopb - startb) % step
            else:
                gap = step % nrowsinbuf
                nextelement = startb + step 

        # Get the appropriate rows according to step
        # Also, make a copy in order to get a contiguous recarray

        # Explicitely delete the last reference to buffer
        del buffer
        result._byteorder = self._v_byteorder
        return result
    
    def __getitem__(self, slice):

        if isinstance(slice, types.IntType):
            step = 1
            start = slice
            if start < 0:
                start = self.nrows + start
            stop = start + 1
        else:
            start = slice.start
            if start is None:
                start = 0
            elif start < 0:
                start = self.nrows + start
            stop = slice.stop
            if stop is None:
                stop = self.nrows
            elif stop < 0 :
                stop = self.nrows + stop

            step = slice.step
            if step is None:
                step = 1
        return self.getRows(start, stop, step)

    def flush(self):
        """Flush the table buffers."""
        if self._v_recunsaved > 0:
          self._saveBufferedRows()

    def close(self):
        """Flush the table buffers and close the HDF5 dataset."""
        self.flush()
        # Delete the reference to Row in _v_buffer recarray!
        # This allows to delete both the Row and RecArray objects
        # because Row has back-references to RecArray
        # No longer needed as Row has been moved to Table module (here)
        #if hasattr(self,'_v_buffer') and hasattr(self._v_buffer, "_row"):
        #    del self._v_buffer._row

    # Moved out of scope
    def _f_del__(self):
        """Delete some objects"""
        print "Deleting Table object", self._v_name
        pass

    def __repr__(self):
        """This provides column metainfo in addition to standard __str__"""

        header = str(self)
        byteorder = self._v_byteorder
        columns = ["Number of columns: %s\n  Column metainfo:" % \
                   len(self.colnames)]
        columns += ['%s := (%s, %s)' % (self.colnames[i],
                                       repr(self.coltypes[i]),
                                       self.colshapes[i])
                    for i in range(len(self.colnames))]
        columns = "\n    ".join(columns)
        
        return "%s\n  Byteorder: %s\n  %s" % \
               (header, byteorder, columns)
