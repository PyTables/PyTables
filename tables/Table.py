########################################################################
#
#       License: BSD
#       Created: September 4, 2002
#       Author:  Francesc Alted - falted@openlc.org
#
#       $Source: /home/ivan/_/programari/pytables/svn/cvs/pytables/pytables/tables/Table.py,v $
#       $Id: Table.py,v 1.50 2003/06/19 11:14:35 falted Exp $
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

__version__ = "$Revision: 1.50 $"

from __future__ import generators
import sys
import struct
import types
import re
import copy
import string
import warnings
import numarray
import numarray.strings as strings
import numarray.records as records
#import recarray2         # Private version of records for PyTables
import hdf5Extension
from Leaf import Leaf
from IsDescription import IsDescription, metaIsDescription, Col, fromstructfmt

try:
    import Numeric
    Numeric_imported = 1
except:
    Numeric_imported = 0


byteorderDict={"=": sys.byteorder,
               "@": sys.byteorder,
               '<': 'little',
               '>': 'big'}

revbyteorderDict={'little': '<',
                  'big': '>'}


class Table(Leaf, hdf5Extension.Table, object):
    """Represent a table in the object tree.

    It provides methods to create new tables or open existing ones, as
    well as to write/read data to/from table objects over the
    file. A method is also provided to iterate over the rows without
    loading the entire table or column in memory.

    Data can be written or read both as Row() instances or as numarray
    (NumArray or RecArray) objects.
    
    Methods:
    
      Common to all leaves:
        close()
        flush()
        getAttr(attrname)
        rename(newname)
        remove()
        setAttr(attrname, attrvalue)
        
      Specific of Table:
        iterrows()
        read([start] [, stop] [, step] [, field [, flavor]])

    Instance variables:
    
      Common to all leaves:
        name -- the leaf node name
        hdf5name -- the HDF5 leaf node name
        title -- the leaf title
        shape -- the leaf shape
        byteorder -- the byteorder of the leaf
        
      Specific of Table:
        description -- the metaobject describing this table
        row -- a reference to the Row object associated with this table
        nrows -- the number of rows in this table
        rowsize -- the size, in bytes, of each row
        colnames -- the field names for the table (list)
        coltypes -- the type class for the table fields (dictionary)
        colshapes -- the shapes for the table fields (dictionary)

    """

    def __init__(self, description = None, title = "",
                 compress = 0, complib="zlib", expectedrows = 10000):
        """Create an instance Table.

        Keyword arguments:

        description -- A IsDescription subclass or a dictionary where
            the keys are the field names, and the values the type
            definitions. And it can be also a RecArray object (from
            recarray module). If None, the table metadata is read from
            disk, else, it's taken from previous parameters.

        title -- Sets a TITLE attribute on the HDF5 table entity.

        compress -- Specifies a compress level for data. The allowed
            range is 0-9. A value of 0 disables compression. The
            default is 0 (no compression).

        complib -- Specifies the compression library to be used. Right
            now, "zlib", "lzo" and "ucl" values are supported.

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
            self.description = metaIsDescription("", (), description)()
            # Flag that tells if this table is new or has to be read from disk
            self._v_new = 1
        elif isinstance(description, records.RecArray):
            # RecArray object case
            self._newRecArray(description)
            # Provide a better guess for the expected number of rows
            # But beware with the small recarray lengths!
            # Commented out until a better approach is found
            #if self._v_expectedrows == expectedrows:
            #    self._v_expectedrows = self.nrows
            # Flag that tells if this table is new or has to be read from disk
            self._v_new = 1
        elif (type(description) == type(IsDescription) and
              issubclass(description, IsDescription)):
            # IsDescription subclass case
            self.description = description()
            # Flag that tells if this table is new or has to be read from disk
            self._v_new = 1
        elif description is None:
            self._v_new = 0
        else:
            raise ValueError, \
"""description parameter is not one of the supported types:
  IsDescription subclass, dictionary or RecArray."""

	if self._v_new:
	    if hdf5Extension.isLibAvailable(complib)[0]:
		self._v_complib = complib
	    else:
		warnings.warn( \
"""You are asking for the %s compression library, but this is not installed locally.
  Defaulting to zlib instead!.""" %(complib))
                self._v_complib = "zlib"   # Should always exists

    def _newBuffer(self, init=1):
        """Create a new recarray buffer for I/O purposes"""

        #recarr = recarray2.array(None, formats=self.description._v_recarrfmt,
        recarr = records.array(None, formats=self.description._v_recarrfmt,
                               shape=(self._v_maxTuples,),
                               names = self.colnames)
        # Initialize the recarray with the defaults in description
        if init:
            for field in self.description.__slots__:
                recarr._fields[field][:] = self.description.__dflts__[field]

        return recarr

    def _newRecArray(self, recarr):
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
            # Special case for strings (from numarray 0.6 on)
            if isinstance(recarr._fmt[i], records.Char):
                fields[self.colnames[i]] = Col(recarr._fmt[i],
                                               recarr._sizes[i],
                                               pos=i)  # Position matters!
            else:
                fields[self.colnames[i]] = Col(recarr._fmt[i],
                                               recarr._repeats[i],
                                               pos=i)  # Position matters!
        # Set the byteorder
        self.byteorder = recarr._byteorder
        # Append this entry to indicate the alignment!
        fields['_v_align'] = revbyteorderDict[recarr._byteorder]
        # Create an instance description to host the record fields
        self.description = metaIsDescription("", (), fields)()
        # The rest of the info is automatically added when self.create()
        # is called

    def _create(self):
        """Create a new table on disk."""

        # Compute some important parameters for createTable
        self.colnames = tuple(self.description.__slots__)
        self._v_fmt = self.description._v_fmt
        #print "self._v_fmt (create)-->", self._v_fmt
        self._calcBufferSize(self._v_expectedrows)
        # Create the table on disk
        self._createTable(self.title, self._v_complib)
        # Initialize the shape attribute
        self.shape = (self.nrows,)
        # Get the column types
        self.coltypes = self.description.__types__
        # Extract the shapes for columns
        self.colshapes = self.description._v_shapes
        # Compute the byte order
        self.byteorder = byteorderDict[self._v_fmt[0]]
        # Create the arrays for buffering
        self._v_buffer = self._newBuffer(init=1)
        # A copy of the original initialised recarray (useful when writing)
        self._v_buffercpy = self._newBuffer(init=1)
        self.row = hdf5Extension.Row(self._v_buffer, self)
                         
    def _open(self):
        """Opens a table from disk and read the metadata on it.

        Creates an user description on the flight to easy the access to
        the actual data.

        """
        # Get table info
        (self.nrows, self.colnames, self._v_fmt) = self._getTableInfo()
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
        #print "self._v_fmt (open)-->", self._v_fmt
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
        self.byteorder = byteorderDict[self._v_fmt[0]]
        # Create an instance description to host the record fields
        self.description = metaIsDescription("", (), fields)()
        # Extract the coltypes
        self.coltypes = self.description.__types__
        # Extract the shapes for columns
        self.colshapes = self.description._v_shapes
        # Create the arrays for buffering
        self._v_buffer = self._newBuffer(init=0)
        self.row = hdf5Extension.Row(self._v_buffer, self)
        
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
        # Protection against row sizes too large (HDF5 refuse to work
        # with row sizes larger than 10 KB or so).
        if rowsize > 8192:
            raise RuntimeError, \
        """Row size too large. Maximum size is 8192 bytes, and you are asking
        for a row size of %s bytes.""" % (rowsize)
            
        self.rowsize = rowsize
        bufmultfactor = 1000 * 10
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
            buffersize = 5 * bufmultfactor
            chunksize = 1024
        elif (expectedfsizeinKb > 100 and
            expectedfsizeinKb <= 1000):
            # Values for files less than 1 MB of size
            buffersize = 20 * bufmultfactor
            chunksize = 2048
        elif (expectedfsizeinKb > 1000 and
              expectedfsizeinKb <= 20 * 1000):
            # Values for sizes between 1 MB and 20 MB
            buffersize = 40  * bufmultfactor
            chunksize = 4096
        elif (expectedfsizeinKb > 20 * 1000 and
              expectedfsizeinKb <= 200 * 1000):
            # Values for sizes between 20 MB and 200 MB
            buffersize = 50 * bufmultfactor
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
            buffersize = 60 * bufmultfactor
            chunksize = 16384
        # Correction for compression.
        if compress:
            chunksize = 1024   # This seems optimal for compression
            pass

        # Max Tuples to fill the buffer
        self._v_maxTuples = buffersize // rowsize
        # Safeguard against row sizes being extremely large
        # I think this is not necessary because of the protection against
        # too large row sizes, but just in case.
        if self._v_maxTuples == 0:
            self._v_maxTuples = 1
        # A new correction for avoid too many calls to HDF5 I/O calls
        # But this does not apport advantages rather the contrary,
        # the memory comsumption grows, and performance is worse.
        #if expectedrows//self._v_maxTuples > 50:
        #    buffersize *= 4
        #    self._v_maxTuples = buffersize // rowsize
        self._v_chunksize = chunksize

    def _saveBufferedRows(self):
        """Save buffered table rows."""
        # Save the records on disk
        self._append_records(self._v_buffer, self.row._getUnsavedNRows())
        # Get a fresh copy of the default values
        # This copy seems to make the writing with compression a 5%
        # faster than if the copy is not made. Why??
        if hasattr(self, "_v_buffercpy"):
            self._v_buffer[:] = self._v_buffercpy[:]

        # Update the number of saved rows in this buffer
        self.nrows += self.row._getUnsavedNRows()
        # Reset the buffer unsaved counter and the buffer read row counter
        self.row._setUnsavedNRows(0)
        # Set the shape attribute (the self.nrows may be less than the maximum)
        self.shape = (self.nrows,)
        
    def _fetchall(self):
        """Iterate over all the rows

        This method is a generator, i.e. it keeps track on the last
        record returned so that next time it is invoked it returns the
        next available record.

        """
        # Create a buffer for the readout
        nrowsinbuf = self._v_maxTuples
        buffer = self._v_buffer
        self._open_read(buffer)  # Open the table for reading
        row = self.row   # get the pointer to the Row object
        row._initLoop(0, self.nrows, 1, nrowsinbuf)
        for i in xrange(0, self.nrows, nrowsinbuf):
            recout = self._read_records(i, nrowsinbuf)
            #recout = nrowsinbuf
            if self.byteorder <> sys.byteorder:
                buffer.byteswap()
            # Set the buffer counter (case for step=1)
            row._setBaseRow(i, 0)
            for j in xrange(recout):
                yield row()
                
        self._close_read()  # Close the table
        
    def _fetchrange(self, start, stop, step):
        """Iterate over a range of rows"""
        row = self.row   # get the pointer to the Row object
        nrowsinbuf = self._v_maxTuples   # Shortcut
        buffer = self._v_buffer  # Shortcut to the buffer
        self._open_read(buffer)  # Open the table for reading
        # Some start values for the main loop
        nrowsread = start
        startb = 0
        nextelement = start
        row._initLoop(start, stop, step, nrowsinbuf)
        for i in xrange(start, stop, nrowsinbuf):
            # Skip this iteration if there is no interesting information
            if ((nextelement >= nrowsread + nrowsinbuf) or 
                (startb >= stop - nrowsread)):
                nrowsread += nrowsinbuf
                continue
            # Compute the end for this iteration
            stopb = stop - nrowsread
            if stopb > nrowsinbuf:
                stopb = nrowsinbuf
            # Read a chunk
            nrowsread += self._read_records(i, nrowsinbuf)
            if self.byteorder <> sys.byteorder:
                buffer.byteswap()
            # Set the buffer counter
            row._setBaseRow(i, startb)
            # Loop over the values for this buffer
            for j in xrange(startb, stopb, step):
                yield row._getRow()
            # Compute some indexes for the next iteration
            startb = (j+step) % nrowsinbuf
            nextelement += step

        self._close_read()  # Close the table

    def _processRange(self, start=None, stop=None, step=None):
        
        if (not (start is None)) and ((stop is None) and (step is None)):
            step = 1
            if start < 0:
                start = self.nrows + start
            stop = start + 1
        else:
            if start is None:
                start = 0
            elif start < 0:
                start = self.nrows + start

            if stop is None:
                stop = self.nrows
            elif stop <= 0 :
                stop = self.nrows + stop
            elif stop > self.nrows:
                stop = self.nrows

            if step is None:
                step = 1
            elif step <= 0:
                raise ValueError, \
                      "Zero or negative step values are not allowed!"
        return (start, stop, step)
    
    def iterrows(self, start=None, stop=None, step=None):
        """Iterator over all the rows, or a range"""
        
        (start, stop, step) = self._processRange(start, stop, step)
        if (start == 0) and ((stop == self.nrows) and (step == 1)):
            return self._fetchall()
        else:
            return self._fetchrange(start, stop, step)

    def _readAllFields(self, start=None, stop=None, step=None):
        """Read a range of rows and return a RecArray"""

        (start, stop, step) = self._processRange(start, stop, step)
        # Create a recarray for the readout
        if start >= stop:
            return records.array(None, formats=self.description._v_recarrfmt,
                                  shape=(0,),
                                  names = self.colnames)
        nrows = ((stop - start - 1) // step) + 1
        # Create the resulting recarray
        result = records.array(None, formats=self.description._v_recarrfmt,
                                shape=(nrows,),
                                names = self.colnames)
        # Setup a buffer for the readout
        nrowsinbuf = self._v_maxTuples   # Shortcut
        #nrowsinbuf = 3   # Small value is useful when debugging
        buffer = self._v_buffer  # Get a recarray as buffer
        self._open_read(buffer)  # Open the table for reading
        nrowsread = start
        startr = 0
        startb = 0
        nextelement = start
        for i in xrange(start, stop, nrowsinbuf):
            if ((nextelement >= nrowsread + nrowsinbuf) or
                (startb >= stop - nrowsread)):
                nrowsread += nrowsinbuf
                continue
            # Compute the end for this iteration
            stopb = stop - nrowsread
            if stopb > nrowsinbuf:
                stopb = nrowsinbuf
            stopr = startr + ((stopb-startb-1)//step) + 1
            # Read a chunk
            nrowsread += self._read_records(i, nrowsinbuf)
            # Assign the correct part to result
            result[startr:stopr] = buffer[startb:stopb:step]
            # Compute some indexes for the next iteration
            startr = stopr
            j = range(startb, stopb, step)[-1]
            startb = (j+step) % nrowsinbuf
            nextelement += step

        self._close_read()  # Close the table

        # Set the byteorder properly
        result._byteorder = self.byteorder
        return result

    def _readCol(self, start=None, stop=None, step=None, field=None):
        """Read a range of rows and return an in-memory object.
        """
        
        for fieldTable in self.colnames:
            if fieldTable == field:
                typeField = self.coltypes[field]
                lengthField = self.colshapes[field][0]
                break
        else:
            raise LookupError, \
                  """The column name '%s' not found in table {%s}""" % \
                  (field, self)
            
        (start, stop, step) = self._processRange(start, stop, step)
        # Return a rank-0 array if start > stop
        if start >= stop:
            if isinstance(typeField, records.Char):
                return strings.array(shape=(0,), itemsize = 0)
            else:
                return numarray.array(shape=(0,), type=typeField)
                
        nrows = ((stop - start - 1) // step) + 1
        # Create the resulting recarray
        if isinstance(typeField, records.Char):
            result = strings.array(shape=(nrows,), itemsize=lengthField)
        else:
            if lengthField > 1:
                result = numarray.array(shape=(nrows, lengthField),
                                        type=typeField)
            else:
                result = numarray.array(shape=(nrows, ), type=typeField)
        # Setup a buffer for the readout
        nrowsinbuf = self._v_maxTuples   # Shortcut
        buffer = self._v_buffer  # Get a recarray as buffer
        self._open_read(buffer)  # Open the table for reading
        nrowsread = start
        startr = 0
        startb = 0
        nextelement = start
        for i in xrange(start, stop, nrowsinbuf):
            if ((nextelement >= nrowsread + nrowsinbuf) or
                (startb >= stop - nrowsread)):
                nrowsread += nrowsinbuf
                continue
            # Compute the end for this iteration
            stopb = stop - nrowsread
            if stopb > nrowsinbuf:
                stopb = nrowsinbuf
            stopr = startr + ((stopb-startb-1)//step) + 1
            # Read a chunk
            nrowsread += self._read_records(i, nrowsinbuf)
            #nrowsread += nrowsinbuf
            # Assign the correct part to result
            # The bottleneck is in this assignment. Hope that the numarray
            # people might improve this in the short future
            result[startr:stopr] = buffer._fields[field][startb:stopb:step]
            # Compute some indexes for the next iteration
            startr = stopr
            j = range(startb, stopb, step)[-1]
            startb = (j+step) % nrowsinbuf
            nextelement += step

        self._close_read()  # Close the table

        # Set the byteorder properly
        result._byteorder = self.byteorder
        return result

    def read(self, start=None, stop=None, step=None, field=None, flavor=None):
        """Read a range of rows and return an in-memory object.

        If "start", "stop", or "step" parameters are supplied, a row
        range is selected. If "field" is specified, only this "field"
        is returned as a NumArray object. If "field" is not supplied
        all the fields are selected and a RecArray is returned.  If
        both "field" and "flavor" are provided, an additional
        conversion to an object of this flavor is made. "flavor" must
        have any of the next values: "Numeric", "Tuple" or "List".

        """
        if field == None:
            return self._readAllFields(start, stop, step)
        elif flavor == None:
            return self._readCol(start, stop, step, field)
        else:
            arr = self._readCol(start, stop, step, field)
            # Convert to Numeric, tuple or list if needed
            if flavor == "Numeric":
                if Numeric_imported:
                    # This works for both numeric and chararrays
                    # arr=Numeric.array(arr, typecode=arr.typecode())
                    # The next is 10 times faster (for tolist(),
                    # we should check for tostring()!)
                    if arr.__class__.__name__ == "CharArray":
                        arrstr = arr.tostring()
                        shape = list(arr.shape)
                        shape.append(arr.itemsize())
                        arr=Numeric.reshape(Numeric.array(arrstr), shape)
                    else:
                        # tolist() method creates a list with a sane byteorder
                        if arr.shape <> ():
                            arr=Numeric.array(arr.tolist(),
                                              typecode=arr.typecode())
                        else:
                            # This works for rank-0 arrays
                            # (but is slower for big arrays)
                            arr=Numeric.array(arr, typecode=arr.typecode())
                        
                else:
                    # Warn the user
                    warnings.warn( \
"""You are asking for a Numeric object, but Numeric is not installed locally.
  Returning a numarray object instead!.""")
            elif flavor == "Tuple":
                arr = tuple(arr.tolist())
            elif flavor == "List":
                arr = arr.tolist()
            else:
                raise ValueError, \
"""You are asking for an unsupported flavor (%s). Supported values are:
"Numeric", "Tuple" and "List".""" % (flavor)

        return arr
            
    # This version of _readCol does not work well. Perhaps a bug in the
    # H5TB_read_fields_name entry?
    def _readCol2(self, start=None, stop=None, step=None, field=None):
        """Read a column from a table in a row range"""

        for fieldTable in self.colnames:
            if fieldTable == field:
                typeField = self.coltypes[field]
                lengthField = self.colshapes[field][0]
                break
        else:
            raise LookupError, \
                  """The column name '%s' not found in table {%s}""" % \
                  (field, self)
            
        (start, stop, step) = self._processRange(start, stop, step)
        # Return a rank-0 array if start > stop
        if start >= stop:
            if isinstance(typeField, records.Char):
                return strings.array(shape=(0,), itemsize = 0)
            else:
                return numarray.array(shape=(0,), type=typeField)
                
        nrows = ((stop - start - 1) // step) + 1
        # Create the resulting recarray
        if isinstance(typeField, records.Char):
            result = strings.array(shape=(nrows,), itemsize=lengthField)
        else:
            if lengthField > 1:
                result = numarray.array(shape=(nrows, lengthField),
                                        type=typeField)
            else:
                result = numarray.array(shape=(nrows, ), type=typeField)
        # Setup a buffer for the readout
        nrowsinbuf = self._v_maxTuples   # Shortcut
        #buffer = self._v_buffer  # Get a recarray as buffer
        # Create the buffer array
        typesize = lengthField
        if isinstance(typeField, records.Char):
            buffer = strings.array(shape=(nrowsinbuf,), itemsize=lengthField)
        else:
            if lengthField > 1:
                buffer = numarray.array(shape=(nrowsinbuf, lengthField),
                                        type=typeField)
            else:
                buffer = numarray.array(shape=(nrowsinbuf, ), type=typeField)
            typesize *= buffer._type.bytes
        self._open_read(buffer)  # Open the table for reading
        nrowsread = start
        startr = 0
        startb = 0
        nextelement = start
        for i in xrange(start, stop, nrowsinbuf):
            if ((nextelement >= nrowsread + nrowsinbuf) or
                (startb >= stop - nrowsread)):
                nrowsread += nrowsinbuf
                continue
            # Compute the end for this iteration
            stopb = stop - nrowsread
            if stopb > nrowsinbuf:
                stopb = nrowsinbuf
            stopr = startr + ((stopb-startb-1)//step) + 1
            # Read a chunk
            #nrowsread += self._read_records(i, nrowsinbuf, buffer)
            nrowsread += self._read_field_name(field, i, nrowsinbuf)
            #nrowsread += nrowsinbuf
            # Assign the correct part to result
            # The bottleneck is in this assignment. Hope that the numarray
            # people might improve this in the short future
            #result[startr:stopr] = buffer._fields[field][startb:stopb:step]
            result[startr:stopr] = buffer[startb:stopb:step]
            # Compute some indexes for the next iteration
            startr = stopr
            j = range(startb, stopb, step)[-1]
            startb = (j+step) % nrowsinbuf
            nextelement += step

        self._close_read()  # Close the table

        # Set the byteorder properly
        result._byteorder = self.byteorder
        return result

    # Moved out of scope
    def _g_getitem__(self, slice):

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
        return self.read(start, stop, step)

    def flush(self):
        """Flush the table buffers."""
        #if self._v_recunsaved > 0:
        if hasattr(self, 'row') and self.row._getUnsavedNRows() > 0:
          self._saveBufferedRows()

    # Moved out of scope
    def _g_del__(self):
        """Delete some objects"""
        print "Deleting Table object", self._v_name
        pass

    def __repr__(self):
        """This provides column metainfo in addition to standard __str__"""

        rep = [ '%r: Col(\'%r\', %r)' %  \
                (k, self.coltypes[k], self.colshapes[k])
                for k in self.colnames ]
        columns = '{\n    %s }' % (',\n    '.join(rep))
        
        return "%s\n  description := %s\n  byteorder = %s" % \
               (str(self), columns, self.byteorder)
               
