########################################################################
#
#       License: BSD
#       Created: September 4, 2002
#       Author:  Francesc Alted - falted@openlc.org
#
#       $Source: /home/ivan/_/programari/pytables/svn/cvs/pytables/pytables/tables/Table.py,v $
#       $Id: Table.py,v 1.72 2003/09/12 17:14:33 falted Exp $
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

__version__ = "$Revision: 1.72 $"

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
import hdf5Extension
from Leaf import Leaf
from IsDescription import IsDescription, Description, metaIsDescription, \
     Col, StringCol, fromstructfmt

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
        removeRows(start, stop)

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
            #self.description = metaIsDescription("", (), description)()
            self.description = Description(description)
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
            descr = description()
            self.description = Description(descr.columns)
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
"""You are asking for the %s compression library, but it is not available.
  Defaulting to zlib instead!.""" %(complib))
                self._v_complib = "zlib"   # Should always exists

    def _newBuffer(self, init=1):
        """Create a new recarray buffer for I/O purposes"""

        recarr = records.array(None, formats=self.description._v_recarrfmt,
                               shape=(self._v_maxTuples,),
                               names = self.colnames)
        # Initialize the recarray with the defaults in description
        recarr._fields = recarr._get_fields()
        if init:
            #for field in self.description.__slots__:
            for field in self.colnames:
                #print "__dflts__-->", self.description.__dflts__.keys()
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
            colname = self.colnames[i]
            # Special case for strings (from numarray 0.6 on)
            if isinstance(recarr._fmt[i], records.Char):
                fields[colname] =  StringCol(length=recarr._itemsizes[i],
                                             dflt=None,
                                             shape=recarr._repeats[i],
                                             pos=i)
            else:
                fields[colname] = Col(dtype=recarr._fmt[i],
                                      shape=recarr._repeats[i],
                                      pos=i)  # Position matters
        # Set the byteorder
        self.byteorder = recarr._byteorder
        # Append this entry to indicate the alignment!
        fields['_v_align'] = revbyteorderDict[recarr._byteorder]
        # Create an instance description to host the record fields
        #self.description = metaIsDescription("", (), fields)()
        self.description = Description(fields)
        # The rest of the info is automatically added when self.create()
        # is called

    def _create(self):
        """Create a new table on disk."""

        # Compute some important parameters for createTable
        #self.colnames = tuple(self.description.__slots__)
        self.colnames = tuple(self.description.__names__)
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
        self.colitemsizes = self.description._v_itemsizes
        # Compute the byte order
        self.byteorder = byteorderDict[self._v_fmt[0]]
        # Create the arrays for buffering
        #self._v_buffer = self._newBuffer(init=1)
        # A copy of the original initialised recarray (useful when writing)
        #self._v_buffercpy = self._newBuffer(init=1)
        #self.row = hdf5Extension.Row(self._v_buffer, self)
        self.row = hdf5Extension.Row(self)
                         
    def _open(self):
        """Opens a table from disk and read the metadata on it.

        Creates an user description on the flight to easy the access to
        the actual data.

        """
        # Get table info
        (self.nrows, self.colnames, self._v_fmt) = self._getTableInfo()
        #print "self._v_fmt (open)-->", self._v_fmt
        # This one is probably not necessary to set it, but...
        self._v_compress = 0  # This means, we don't know if compression
                              # is active or not. Maybe it is worth to save
                              # this info in a table attribute?

        # Get the byteorder
        byteorder = self._v_fmt[0]
        # Remove the byteorder
        self._v_fmt = self._v_fmt[1:]
        self.byteorder = byteorderDict[byteorder]
        # Create a recarray with no data
        headerRA = records.array(None, formats=self._v_fmt)
        rowsize = headerRA._itemsize
        # Get the column types
        coltypes = [str(f) for f in headerRA._fmt]
        # Get the column shapes
        colshapes = headerRA._repeats
        # Build a dictionary with the types as values and colnames as keys
        fields = {}
        for i in range(len(self.colnames)):
            if coltypes[i] == "CharType":
                itemsize = headerRA._itemsizes[i]
                fields[self.colnames[i]] = StringCol(length = itemsize,
                                                     shape = colshapes[i],
                                                     pos = i)
            else:
                fields[self.colnames[i]] = Col(dtype = coltypes[i],
                                               shape = colshapes[i],
                                               pos = i)
        # Set the alignment!
        fields['_v_align'] = byteorder
        # Create an instance description to host the record fields
        # The next line makes memory leaks to appear!
        #self.description = metaIsDescription("", (), fields)()
        self.description = Description(fields)
        
        # Extract the coltypes, shapes and itemsizes
        self.coltypes = self.description.__types__
        self.colshapes = self.description._v_shapes
        self.colitemsizes = self.description._v_itemsizes
        # Compute buffer size
        self._calcBufferSize(self.nrows)
        # Update the shape attribute
        self.shape = (self.nrows,)
        # Associate a Row object to table
        self.row = hdf5Extension.Row(self)
        
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
        # Create a helper RecArray with no data
        if hasattr(self, "description"):
            headerRA = records.array(None, formats=self.description._v_recarrfmt)
            rowsize = headerRA._itemsize
        else:
            rowsize = struct.calcsize(fmt)

        # Protection against row sizes too large (HDF5 refuse to work
        # with row sizes larger than 10 KB or so).
        if rowsize > 8192:
            raise RuntimeError, \
        """Row size too large. Maximum size is 8192 bytes, and you are asking
        for a row size of %s bytes.""" % (rowsize)
            
        self.rowsize = rowsize
        # A bigger buffer makes the writing faster and reading slower (!)
        #bufmultfactor = 1000 * 10
        # A smaller buffer also makes the tests to not take too much memory
        # We choose the smaller one
        # In addition, with the new iterator in the Row class, this seems to
        # be the best choice in terms of performance!
        bufmultfactor = int(1000 * 1.0)
        # Counter for the binary tuples
        self._v_recunsaved = 0
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
        # Rational: HDF5 takes the data in bunches of chunksize length
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
            # These values gives an increment of memory of 50 MB for a table
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

    def _processRange(self, start=None, stop=None, step=None):
        
        assert (type(start) in
                [types.NoneType, types.IntType, types.LongType]), \
            "Non valid start parameter: %s" % start
        
        assert (type(stop) in
                [types.NoneType, types.IntType, types.LongType]), \
            "Non valid stop parameter: %s" % stop
        
        assert (type(step) in
                [types.NoneType, types.IntType, types.LongType]), \
            "Non valid step parameter: %s" % step
        
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
        """Iterator over all the rows or a range"""

        return self.__call__(start, stop, step)

    def __call__(self, start=None, stop=None, step=None):
        """Iterate over all the rows or a range.
        
        It returns the same iterator than
        Table.iterrows(start, stop, step).
        It is, therefore, a shorter way to call it.
        """
        
        (start, stop, step) = self._processRange(start, stop, step)

        return self.row(start, stop, step)
        
    def __iter__(self):
        """Iterate over all the rows."""

        return self.__call__()

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
        
        if field and not field in self.colnames:
            raise LookupError, \
                  """The column name '%s' not found in table {%s}""" % \
                  (field, self)
        
        (start, stop, step) = self._processRange(start, stop, step)
        
        if flavor == None:
            return self._read(start, stop, step, field)
        else:
            arr = self._read(start, stop, step, field)
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

    def _read(self, start, stop, step, field=None):
        """Read a range of rows and return an in-memory object.
        """

        if field:
            typeField = self.coltypes[field]
        # Return a rank-0 array if start > stop
        if start >= stop:
            if field == None:
                return records.array(None,
                                     formats=self.description._v_recarrfmt,
                                     shape=(0,),
                                     names = self.colnames)
            elif isinstance(typeField, records.Char):
                return strings.array(shape=(0,), itemsize = 0)
            else:
                return numarray.array(shape=(0,), type=typeField)
                
        nrows = ((stop - start - 1) // step) + 1
        # Compute the shape of the resulting column object
        if field:
            shape = self.colshapes[field]
            itemsize = self.colitemsizes[field]
            if type(shape) in [types.IntType, types.LongType]:
                if shape == 1:
                    shape = (nrows,)
                else:
                    shape = (nrows, shape)
            else:
                shape2 = [nrows]
                shape2.extend(shape)
                shape = tuple(shape2)

            # Create the resulting recarray
            if isinstance(typeField, records.Char):
                # String-column case
                result = strings.array(shape=shape, itemsize=itemsize)
            elif field:
                # Non-string column case
                result = numarray.array(shape=shape, type=typeField)
        else:
            # Recarray case
            result = records.array(None, formats=self.description._v_recarrfmt,
                                   shape=(nrows,),
                                   names = self.colnames)

        # Call the routine to fill-up the resulting array
        if step == 1 and not field and 0:
            # This optimization works three times faster than
            # the row._fillCol method (up to 160 MB/s in a pentium IV @ 2GHz)
            self._open_read(result)
            #print "Start, stop -->", start, stop
            self._read_records(start, stop-start)
            self._close_read()  # Close the table
        elif step == 1 and field and 0:
            # This optimization in Pyrex works, but the call to row._fillCol
            # is almost always faster (!!), so disable it.
            print "Start, stop, field -->", start, stop, field
            # Both versions seems to work well!
            # Column name version
            #self._read_field_name(result, start, stop, field)
            # Column index version
            field_index = -1
            for i in range(len(self.colnames)):
                if self.colnames[i] == field:
                    field_index = i
                    break
            print "col index:", field_index
            self._read_field_index(result, start, stop, field_index)
        else:
            self.row._fillCol(result, start, stop, step, field)
        # Set the byteorder properly
        result._byteorder = self.byteorder
        return result
    
    def __getitem__(self, key):
        """Returns a table row, table slice or table column.

        It takes different actions depending on the type of the "key"
        parameter:

        If "key"is an integer, the corresponding table row is returned
        as a RecArray.Record object. If "key" is a slice, the row
        slice determined by key is returned as a RecArray object.
        Finally, if "key" is a string, it is interpreted as a column
        name in the table, and, if it exists, it is read and returned
        as a NumArray or CharArray object (whatever is appropriate).

"""

        if isinstance(key, types.IntType):
            return self.read(key, key+1, 1)[0]
        elif isinstance(key, types.SliceType):
            return self.read(key.start, key.stop, key.step)
        elif isinstance(key, types.StringType):
            return self.read(field=key)
        else:
            raise ValueError, "Non-valid __getitem__ parameter %s" % \
                  key

    # This addtion has to be thought more carefully because of two things
    # 1.- The colnames has to be valid python identifiers, and that
    #     restriction has still to be added.
    # 2.- The access to local variables in Table is slowed down, because
    #     __getattr__ is always called
    # 3.- The most important, a colname cannot be the same of a standard
    #     Table attribute, because, if so, this attribute can't be reached.
#     def __getattr__(self, colname):
#         """Get the table column object named "colname"."""

#         return self._readCol(field=colname)

    def removeRows(self, start=None, stop=None):
        """Remove a range of rows.

        If only "start" is supplied, this row is to be deleted.
        If "start" and "stop" parameters are supplied, a row
        range is selected to be removed.

        """

        # If "stop" is not provided, select the index pointed by start only
        if stop is None:
            stop = start + 1
        # Check for correct values of start and stop    
        (start, stop, step) = self._processRange(start, stop, 1)
        nrows = stop - start
        nrows = self._remove_row(start, nrows)
        self.nrows -= nrows    # discount the removed rows from the total
        return nrows

    def flush(self):
        """Flush the table buffers."""
        #if self._v_recunsaved > 0:
        if hasattr(self, 'row') and self.row._getUnsavedNRows() > 0:
          self._saveBufferedRows()
        # Close a possible opened table for append:
        self._close_append()
        # Clean the Row instance
        # In some situations, this maybe undefined (When?)
        if hasattr(self, "row"):
            # If not call row._cleanup()
            # the memory consumption seems to increase instead of decrease (!)
            # However, the reading/write speed seems to improve a bit (!!)
            # I choose to favor speed vs memory consumption
            self.row._cleanup()
            pass

    def close(self):
        """Flush the buffers and close this object on tree"""
        Leaf.close(self)
        # We must delete the row object, as this make a back reference to Table
        # In some situations, this maybe undefined
        if hasattr(self, "row"):
            del self.row
        self.description._close()
        # Free the description class!
        del self.description

        # After the objects are disconnected, destroy the
        # object dictionary using the brute force ;-)
        # This should help to the garbage collector
        self.__dict__.clear()        

    def __repr__(self):
        """This provides column metainfo in addition to standard __str__"""

        return "%s\n  description := %r\n  byteorder := %s" % \
               (str(self), self.description, self.byteorder)
               
