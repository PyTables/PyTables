########################################################################
#
#       License: BSD
#       Created: September 4, 2002
#       Author:  Francesc Alted - falted@pytables.org
#
#       $Source: /home/ivan/_/programari/pytables/svn/cvs/pytables/pytables/tables/Table.py,v $
#       $Id: Table.py,v 1.123 2004/08/06 16:34:36 falted Exp $
#
########################################################################

"""Here is defined the Table class.

See Table class docstring for more info.

Classes:

    Table
    Cols
    Column

Functions:


Misc variables:

    __version__


"""

__version__ = "$Revision: 1.123 $"

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
from utils import calcBufferSize, processRange, processRangeRead
import Group
from Leaf import Leaf, Filters
from Index import Index, IndexProps
from IsDescription import IsDescription, Description, metaIsDescription, \
     Col, StringCol, fromstructfmt
from VLArray import Atom, StringAtom

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

    Data can be written or read both as Row instances or as numarray
    (NumArray or RecArray) objects.
    
    Methods:

        append(rows)
        __call__(start, stop, step)
        __iter__()
        iterrows(start, stop, step)
        read([start] [, stop] [, step] [, field [, flavor]])
        removeRows(start, stop)
        where(condition)
        whereInRange(condition [, start] [, stop] [, step])
        getWhereList(condition [, flavor])
        iterWhereList(sequence)
        reIndex()
        reIndexDirty()

    Instance variables:

        description -- the metaobject describing this table
        row -- a reference to the Row object associated with this table
        nrows -- the number of rows in this table
        rowsize -- the size, in bytes, of each row
        indexed -- whether or not some column in Table is indexed
        indexprops -- The properties of an indexed Table. Exists only
            if the Table is indexed
        cols -- accessor to the columns using a natural name schema
        colnames -- the field names for the table (list)
        coltypes -- the type class for the table fields (dictionary)
        colshapes -- the shapes for the table fields (dictionary)
        colindexed -- whether the table fields are indexed (dictionary)

    """

    def __init__(self, description = None, title = "",
                 filters = None, expectedrows = 10000):
        """Create an instance Table.

        Keyword arguments:

        description -- A IsDescription subclass or a dictionary where
            the keys are the field names, and the values the type
            definitions. And it can be also a RecArray object (from
            recarray module). If None, the table metadata is read from
            disk, else, it's taken from previous parameters.

        title -- Sets a TITLE attribute on the HDF5 table entity.

        filters -- An instance of the Filters class that provides
            information about the desired I/O filters to be applied
            during the life of this object.

        expectedrows -- An user estimate about the number of rows
            that will be on table. If not provided, the default value
            is appropiate for tables until 1 MB in size (more or less,
            depending on the record size). If you plan to save bigger
            tables try providing a guess; this will optimize the HDF5
            B-Tree creation and management process time and memory
            used.

        """

        # Common variables
        self._v_new_title = title
        self._v_new_filters = filters
        self._v_expectedrows = expectedrows
        # Initialize the number of rows to a default
        self.nrows = 0
        # Initialize the possible cuts in columns
        self.ops = []
        self.opsValues = []
        # Initialize this object in case is a new Table
        if isinstance(description, types.DictType):
            # Dictionary case
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

    def _newBuffer(self, init=1):
        """Create a new recarray buffer for I/O purposes"""

        recarr = records.array(None, formats=self.description._v_recarrfmt,
                               shape=(self._v_maxTuples,),
                               names = self.colnames)
        # Initialize the recarray with the defaults in description
        recarr._fields = recarr._get_fields()
        if init:
            for field in self.colnames:
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
        for i in xrange(len(self.colnames)):
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
        self.description = Description(fields)
        # The rest of the info is automatically added when self.create()
        # is called

    def _create(self):
        """Create a new table on disk."""

        # Compute some important parameters for createTable
        self.colnames = tuple(self.description.__names__)
        self._v_fmt = self.description._v_fmt
        # Create the table on disk
        self._createTable(self._v_new_title, self.filters.complib)
        # Initialize the shape attribute
        self.shape = (self.nrows,)
        # Get the column types
        self.coltypes = self.description.__types__
        # Extract the shapes for columns
        self.colshapes = self.description._v_shapes
        self.colitemsizes = self.description._v_itemsizes
        # Compute the byte order
        self.byteorder = byteorderDict[self._v_fmt[0]]
        # Create the Row object helper
        self.row = hdf5Extension.Row(self)
        # Get if a column is indexed or not in creation time
        colobjects = self.description._v_ColObjects
        self.colindexed = {}
        self.indexed = 0  # Specifies that some column is indexed in Table
        for colname in self.colnames:
            if colobjects[colname].indexed:
                self.colindexed[colname] = 1
                self.indexed = 1
            else:
                self.colindexed[colname] = 0
        if self.indexed:
            # Check whether we want automatic indexing after an append or not
            # The default is yes
            if hasattr(self.description, "_v_indexprops"):
                self.indexprops = self.description._v_indexprops
            else:
                # if user has not defined properties, assign the default
                self.indexprops = IndexProps()
            # Save AUTOMATIC_INDEX and REINDEX flags as attributes
            self.attrs.AUTOMATIC_INDEX = self.indexprops.auto
            self.attrs.REINDEX = self.indexprops.reindex
            self._indexedrows = 0
            self._unsaved_indexedrows = 0
        # Create a cols accessor
        self.cols = Cols(self)

    def _open(self):
        """Opens a table from disk and read the metadata on it.

        Creates an user description on the flight to easy the access to
        the actual data.

        """
        # Get table info
        (self.nrows, self.colnames, self.rowsize, itemsizes, colshapes,
         coltypes, self._v_fmt) = self._getTableInfo()
        # Get the byteorder
        byteorder = self._v_fmt[0]
        # Remove the byteorder
        self._v_fmt = self._v_fmt[1:]
        self.byteorder = byteorderDict[byteorder]
        coltypes = [str(records.numfmt[type]) for type in coltypes]
        # Build a dictionary with the types as values and colnames as keys
        fields = {}
        for i in xrange(len(self.colnames)):
            if coltypes[i] == "CharType":
                itemsize = itemsizes[i]
                fields[self.colnames[i]] = StringCol(length = itemsize,
                                                     shape = colshapes[i],
                                                     pos = i)
            else:
                fields[self.colnames[i]] = Col(dtype = coltypes[i],
                                               shape = colshapes[i],
                                               pos = i)
        # Set the alignment!
        fields['_v_align'] = byteorder
        if self._v_file._isPTFile:
            # Checking of validity names for fields is not necessary
            # when opening a PyTables file
            fields['__check_validity'] = 0
        # Create an instance description to host the record fields
        self.description = Description(fields)
        
        # Extract the coltypes, shapes and itemsizes
        self.coltypes = self.description.__types__
        self.colshapes = self.description._v_shapes
        self.colitemsizes = self.description._v_itemsizes
        # Compute buffer size
        (self._v_maxTuples, self._v_chunksize) = \
              calcBufferSize(self.rowsize, self.nrows,
                             self.filters.complevel)
        # Update the shape attribute
        self.shape = (self.nrows,)
        # Associate a Row object to table
        self.row = hdf5Extension.Row(self)
        # Create a cols accessor
        self.cols = Cols(self)
        # Check whether the columns are indexed or not
        self.colindexed = {}
        self.indexed = 0
        for colname in self.colnames:
            iname = "_i_"+self.name+"_"+colname
            if iname in self._v_parent._v_indices:
                self.colindexed[colname] = 1
                indexobj = getattr(self.cols, colname).index
                self.indexed = 1
            else:
                self.colindexed[colname] = 0
        # Get the automatic_indexing attribute
        if self.indexed:
            automatic_index = self.attrs.AUTOMATIC_INDEX
            reindex = self.attrs.REINDEX
            self.indexprops=IndexProps(auto=automatic_index, reindex=reindex)
            self._indexedrows = indexobj.nelements
            self._unsaved_indexedrows = self.nrows - self._indexedrows

    def _saveBufferedRows(self):
        """Save buffered table rows"""
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
        if self.indexed and self.indexprops.auto:
            start = self._indexedrows
            nrows = self._unsaved_indexedrows
            for (colname, colindexed) in self.colindexed.iteritems():
                if colindexed:
                    indexcol = getattr(self.cols, colname)
                    rowsadded = indexcol.addRowsToIndex(start, nrows)
            self._unsaved_indexedrows -= rowsadded
            self._indexedrows += rowsadded
        return

    def where(self, condition=None):
        """Iterator that selects values fulfilling the 'condition' param.
        
        condition can be used to specify selections along a column in the
        form:

        condition=(0<table.cols.col1<0.3)

        If the column to which the condition is applied is indexed,
        the index will be used so as to accelerate the search.
        
        """

        assert isinstance(condition, Column), \
"Wrong condition parameter type. Only Column instances are suported."

        self.whereColname = condition.name   # Flag for Row.__iter__
        if condition.indexed and not condition.dirty:
            # Get the coordinates to lookup
            ncoords = condition.index._getLookupRange(condition)
            return self.row(coords=None, ncoords=ncoords)
        # Fall back to in-kernel selection method
        (start, stop, step) = processRangeRead(self.nrows, None, None, None)
        return self.row(start, stop, step, coords=None, ncoords=0)

    def whereInRange(self, condition, start=None, stop=None, step=None):
        """Iterator that selects values fulfilling the 'condition' param.
        
        'condition' can be used to specify selections along a column
        in the form:

        condition=(0<table.cols.col1<0.3)

        This method differs from Table.where in that it accepts a
        range of rows to do the search for values. However, it will
        use the in-kernel search method, i.e. it won't take advantage
        of a possible indexed column, while Table.where do.
        
        """

        assert isinstance(condition, Column), \
"Wrong condition parameter type. Only Column instances are suported."

        self.whereColname = condition.name   # Flag for Row.__iter__
        (start, stop, step) = processRangeRead(self.nrows, start, stop, step)
        if start < stop:
            # we don't want to use the indexed search method because
            # it doesn't support index range searches
            self._dontuseindex = 1
            #self.colindexed[condition.name] = 0
            # call row with coords=None and ncoords=0
            # (in-kernel selection method)
            return self.row(start, stop, step, coords=None, ncoords=0)
        # Fall-back action is to return an empty RecArray
        return records.array(None,
                             formats=self.description._v_recarrfmt,
                             shape=(0,),
                             names = self.colnames)
        
    def getWhereList(self, condition=None, flavor="List"):
        """Get the row coordinates that fulfill the 'condition' param

        'condition' can be used to specify selections along a column
        in the form:

        condition=(0<table.cols.col1<0.3)

        'flavor' is the desired type of the returned list. It can take
        the 'List', 'Tuple' or 'NumArray' values.

        """

        assert isinstance(condition, Column), \
"Wrong condition parameter type. Only Column instances are suported."
        assert flavor in ["NumArray", "List", "Tuple"], \
"Wrong condition parameter type. Only Column instances are suported."
        # Take advantage of indexation, if present
        if condition.indexed == 1:
            # get the number of coords and set-up internal variables
            ncoords = condition.index._getLookupRange(condition)
            # create buffers for indices
            condition.index.indices._initIndexSlice(ncoords)
            # get the coordinates that passes the selection cuts
            coords = condition.index.getCoords(0, ncoords)
            # Remove buffers for indices
            condition.index.indices._destroyIndexSlice()
            # get the remaining rows from the table
            start = condition.index.nelements
            remainCoords = [p.nrow() for p in \
                            self.whereInRange(condition, start, self.nrows, 1)]
            nremain = len(remainCoords)
            # append the new values to the existing ones
            coords.resize(ncoords+nremain)
            coords[ncoords:] = remainCoords
        else:
            coords = [p.nrow() for p in self.where(condition)]
            coords = numarray.array(coords, type=numarray.Int64)
        # re-initialize internal selection values
        self.ops = []
        self.opsValues = []
        self.whereColname = None
        # do some conversion (if needed)
        if flavor == "List":
            coords = coords.tolist()
        elif flavor == "Tuple":
            coords = tuple(coords.tolist())
        return coords

    def iterWhereList(self, sequence=None):
        """Iterate over a list of row coordinates."""
        
        assert hasattr(sequence, "__getitem__"), \
"Wrong 'sequence' parameter type. Only sequences are suported."
        coords = numarray.array(sequence, type=numarray.Int64) 
        return self.row(coords=coords, ncoords=0)
        
    def __call__(self, start=None, stop=None, step=None):
        """Iterate over all the rows or a range.
        
        It returns the same iterator than Table.iterrows(start, stop,
        step).  It is, therefore, a shorter way to call it.
        
        """
        (start, stop, step) = processRangeRead(self.nrows, start, stop, step)
        if start < stop:
            return self.row(start, stop, step, coords=None, ncoords=0)
        # Fall-back action is to return an empty RecArray
        return records.array(None,
                             formats=self.description._v_recarrfmt,
                             shape=(0,),
                             names = self.colnames)
        
    def __iter__(self):
        """Iterate over all the rows."""

        return self.__call__()

    def iterrows(self, start=None, stop=None, step=None):
        """Iterator over all the rows or a range

        Specifying a negative value of step is not supported yet.

        """

        return self.__call__(start, stop, step)

    def read(self, start=None, stop=None, step=None,
             field=None, flavor="numarray", coords = None):
        """Read a range of rows and return an in-memory object.

        If "start", "stop", or "step" parameters are supplied, a row
        range is selected. If "field" is specified, only this "field"
        is returned as a NumArray object. If "field" is not supplied
        all the fields are selected and a RecArray is returned.  If
        both "field" and "flavor" are provided, an additional
        conversion to an object of this flavor is made. "flavor" must
        have any of the next values: "numarray", "Numeric", "Tuple" or
        "List".

        If coords is specified, only the indices in coords that are in
        the range of (start, stop) are returned. If coords is
        specified, step only can be assigned to be 1, otherwise an
        error is issued.
        
        """
        
        if field and not field in self.colnames:
            raise LookupError, \
                  """The column name '%s' not found in table {%s}""" % \
                  (field, self)

        (start, stop, step) = processRangeRead(self.nrows, start, stop, step)

        if coords is not None and len(coords) and step != 1:
            raise NotImplementedError, \
                  """You can't pass a step different from 1 when a coords
                  parameter is specified."""
        
        if flavor == None:
            flavor = "numarray"
            
        if flavor == "numarray":
            return self._read(start, stop, step, field, coords)
        else:
            arr = self._read(start, stop, step, field, coords)
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
                        if str(arr.type()) == "Bool":
                            # Typecode boolean does not exist on Numeric
                            typecode = "1"
                        else:
                            typecode = arr.typecode()                        
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
                # Fixes bug #972534
                arr = tuple(self.tolist(arr))
            elif flavor == "List":
                # Fixes bug #972534
                arr = self.tolist(arr)
            else:
                raise ValueError, \
"""You are asking for an unsupported flavor (%s). Supported values are:
"Numeric", "Tuple" and "List".""" % (flavor)

        return arr

    def tolist(self, arr):
        """Converts a RecArray or Record to a list of rows"""
        outlist = []
        if isinstance(arr, records.Record):
            for i in xrange(arr.array._nfields):
                outlist.append(arr.array.field(i)[arr.row])
            outlist = tuple(outlist)  # return a tuple for records
        elif isinstance(arr, records.RecArray):
            for j in xrange(arr.nelements()):
                tmplist = []
                for i in xrange(arr._nfields):
                    tmplist.append(arr.field(i)[j])
                outlist.append(tuple(tmplist))
        # Fixes bug #991715
        else:
            # Other objects are passed "as is"
            outlist = list(arr)   
        return outlist

    def _read(self, start, stop, step, field=None, coords=None):
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

        if isinstance(coords, numarray.NumArray):
            # I should test for stop and start values as well
            nrows = len(coords)
        else:    
            # (stop-start)//step  is not enough
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
        if step == 1 and not field:
            # This optimization works three times faster than
            # the row._fillCol method (up to 170 MB/s on a pentium IV @ 2GHz)
            self._open_read(result)
            if isinstance(coords, numarray.NumArray):
                if len(coords) > 0:
                    self._read_elements(0, coords)
            else:
                self._read_records(start, stop-start)
            self._close_read()  # Close the table
        elif field and 1:
            # This optimization works in Pyrex, but the call to row._fillCol
            # is almost always faster (!!), so disable it.
            # Update: for step>50, this seems to work always faster than
            # row._fillCol
            # The H5Sselect_elements is faster than H5Sselect_hyperslab
            # for all values of the stride
            # Both versions seems to work well!
            # Column name version
            self._read_field_name(result, start, stop, step, field)
            # Column index version
#             field_index = -1
#             for i in xrange(len(self.colnames)):
#                 if self.colnames[i] == field:
#                     field_index = i
#                     break
#             print "col index:", field_index
#             self._read_field_index(result, start, stop, field_index)
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
            # Index out of range protection
            if key >= self.nrows:
                raise IndexError, "Index out of range"
            if key < 0:
                # To support negative values
                key += self.nrows
            (start, stop, step) = processRange(self.nrows, key, key+1, 1)
            #return self._read(start, stop, step, None, None)[0]
            # For the scalar case, convert the Record and return it as a tuple
            # Fixes bug #972534
            return self.tolist(self._read(start, stop, step, None, None)[0])
        elif isinstance(key, types.SliceType):
            (start, stop, step) = processRange(self.nrows,
                                               key.start, key.stop, key.step)
            return self._read(start, stop, step, None, None)
        elif isinstance(key, types.StringType):
            return self.read(field=key)
        else:
            raise ValueError, "Non-valid index or slice: %s" % key

    def append(self, rows=None):
        """Append a series of rows to the end of the table

        rows can be either a recarray or a structure that is able to
        be converted to a recarray compliant with the table format.

        """

        if rows == None:
            return
        # Try to convert the object into a recarray
        try:
            recarray = records.array(rows,
                                     formats=self.description._v_recarrfmt,
                                     names=self.colnames)
        except:
            (type, value, traceback) = sys.exc_info()
            raise ValueError, \
"rows parameter cannot be converted into a recarray object compliant with table '%s'. The error was: <%s>" % (str(self), value)
        lenrows = recarray.shape[0]
        self._open_append(recarray)
        self._append_records(recarray, lenrows)
        self._close_append()
        # Update the number of saved rows
        self.nrows += lenrows
        # Set the shape attribute (the self.nrows may be less than the maximum)
        self.shape = (self.nrows,)
        # Save indexedrows
        if self.indexed and self.indexprops.auto:
            # Update the number of unsaved indexed rows
            self._unsaved_indexedrows += lenrows
            self.addRowsToIndex()
        return

    def addRowsToIndex(self):
        "Add remaining rows to index"
        if self.indexed:
            # Update the number of unsaved indexed rows
            start = self._indexedrows
            nrows = self._unsaved_indexedrows
            for (colname, colindexed) in self.colindexed.iteritems():
                if colindexed:
                    indexcol = getattr(self.cols, colname)
                    rowsadded = indexcol.addRowsToIndex(start, nrows)
            self._unsaved_indexedrows -= rowsadded
            self._indexedrows += rowsadded
        return

    def removeRows(self, start=None, stop=None):
        """Remove a range of rows.

        If only "start" is supplied, this row is to be deleted.
        If "start" and "stop" parameters are supplied, a row
        range is selected to be removed.

        """

        (start, stop, step) = processRangeRead(self.nrows, start, stop, 1)
        nrows = stop - start
        nrows = self._remove_row(start, nrows)
        self.nrows -= nrows    # discount the removed rows from the total
        # removeRows is a invalidating index operation
        if self.indexed:
            if self.indexprops.reindex:
                self._indexedrows = self.reIndex()
                self._unsaved_indexedrows = self.nrows - self._indexedrows
            else:
                # Mark all the indexes as dirty
                for (colname, colindexed) in self.colindexed.iteritems():
                    if colindexed:
                        indexcol = getattr(self.cols, colname)
                        indexcol.dirty = 1
        return nrows

    def reIndex(self):
        """Recompute the existing indexes in table"""
        for (colname, colindexed) in self.colindexed.iteritems():
            if colindexed:
                indexcol = getattr(self.cols, colname)
                indexedrows = indexcol.reIndex()
        return indexedrows

    def reIndexDirty(self):
        """Recompute the existing indexes in table if they are dirty"""
        for (colname, colindexed) in self.colindexed.iteritems():
            if colindexed:
                indexcol = getattr(self.cols, colname)
                indexedrows = indexcol.reIndexDirty()
        return indexedrows

    def _g_copyRows(self, object, start, stop, step):
        "Copy rows from self to object"
        (start, stop, step) = processRangeRead(self.nrows, start, stop, step)
        nrowsinbuf = self._v_maxTuples
        recarray = self._newBuffer(init=0)
        object._open_append(recarray)
        nrowsdest = object.nrows
        for start2 in xrange(start, stop, step*nrowsinbuf):
            # Save the records on disk
            stop2 = start2+step*nrowsinbuf
            if stop2 > stop:
                stop2 = stop
            #object.append(self[start2:stop2:step])
            # Optimized version (it saves some conversions)
            nrows = ((stop2 - start2 - 1) // step) + 1    
            self.row._fillCol(recarray, start2, stop2, step, None)
            object._append_records(recarray, nrows)
            nrowsdest += nrows
        object._close_append()
        # Update the number of saved rows in this buffer
        object.nrows = nrowsdest
        # Set the shape attribute (the self.nrows may be less than the maximum)
        object.shape = (nrowsdest,)
        return

    # This is an optimized version of copy
    def _g_copy(self, group, name, start, stop, step, title, filters):
        "Private part of Leaf.copy() for each kind of leaf"
        # Build the new Table object
        object = Table(self.description._v_ColObjects, title=title,
                       filters=filters,
                       expectedrows=self.nrows)
        setattr(group, name, object)
        # Now, fill the new table with values from the old one
        self._g_copyRows(object, start, stop, step)
        nbytes=self.nrows*self.rowsize
        return (object, nbytes)

    # No optimized version
#     def _g_copy_orig(self, group, name, start, stop, step, title, filters):
#         "Private part of Leaf.copy() for each kind of leaf"
#         # Build the new Table object
#         object = Table(self.description._v_ColObjects, title=title,
#                        filters=filters,
#                        expectedrows=self.nrows)
#         setattr(group, name, object)
#         # Now, fill the new table with values from the old one
#         nrowsinbuf = self._v_maxTuples
#         for start2 in xrange(start, stop, step*nrowsinbuf):
#             # Save the records on disk
#             stop2 = start2+step*nrowsinbuf
#             if stop2 > stop:
#                 stop2 = stop
#             object.append(self[start2:stop2:step])
#         return object

    def flush(self):
        """Flush the table buffers."""
        if hasattr(self, 'row') and self.row._getUnsavedNRows() > 0:
          self._saveBufferedRows()
          # Flush the data to disk
          self._flush(self._v_parent, self._v_hdf5name)
        # Close a possible opened table for append
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
        if hasattr(self, "cols"):
            self.cols._f_close()
            self.cols = None
        # Close the Table
        Leaf.close(self)
        # We must delete the row object, as this make a back reference to Table
        # In some situations, this maybe undefined
        if hasattr(self, "row"):
            del self.row
        self.description._close()
        # Free the description class!
        del self.description
        if hasattr(self, "indexprops"):
            del self.indexprops

        # After the objects are disconnected, destroy the
        # object dictionary using the brute force ;-)
        # This should help to the garbage collector
        self.__dict__.clear()        

    def __repr__(self):
        """This provides column metainfo in addition to standard __str__"""

        return "%s\n  description := %r\n  byteorder := %s" % \
               (str(self), self.description, self.byteorder)
               

class Cols(object):
    """This is a container for columns in a table

    It provides methods to get Column objects that gives access to the
    data in the column.

    Like with Group instances and AttributeSet instances, the natural
    naming is used, i.e. you can access the columns on a table like if
    they were normal Cols attributes.
    
    Instance variables:

        _v_table -- The parent table instance
        _v_colnames -- List with all column names

    Methods:
    
        __getitem__(colname)
        
    """

    def __init__(self, table):
        """Create the container to keep the column information.

        table -- The parent table
        
        """
        self.__dict__["_v_table"] = table
        self.__dict__["_v_colnames"] = table.colnames
        # Put the column in the local dictionary
        for name in table.colnames:
            self.__dict__[name] = Column(table, name)

    def __len__(self):
        return len(self._v_colnames)

    def __getitem__(self, name):
        """Get the column named "name" as an item."""

        if not isinstance(name, types.StringType):
            raise TypeError, \
"Only strings are allowed as keys of a Cols instance. You passed object: %s" % name
        # If attribute does not exist, return None
        if not name in self._v_colnames:
            raise AttributeError, \
"Column name '%s' does not exist in table:\n'%s'" % (name, str(self._v_table))

        return self.__dict__[name]

    def _f_close(self):
        # First, close the columns (ie possible indices open)
        for col in self._v_colnames:
            self[col].close()
            # Delete the reference to column
            del self.__dict__[col]
        # delete back references:
        self._v_table == None
        self._v_colnames = None
        # Delete all the columns references
        #self.__dict__.clear()

    def __str__(self):
        """The string representation for this object."""
        # The pathname
        pathname = self._v_table._v_pathname
        # Get this class name
        classname = self.__class__.__name__
        # The number of columns
        ncols = len(self._v_colnames)
        return "%s.cols (%s), %s columns" % (pathname, classname, ncols)

    def __repr__(self):
        """A detailed string representation for this object."""

        out = str(self) + "\n"
        for name in self._v_colnames:
            # Get this class name
            classname = getattr(self, name).__class__.__name__
            # The shape for this column
            shape = self._v_table.colshapes[name]
            # The type
            tcol = self._v_table.coltypes[name]
            if shape == 1:
                shape = (1,)
            out += "  %s (%s%s, %s)" % (name, classname, shape, tcol) + "\n"
        return out

               
class Column(object):
    """This is an accessor for the actual data in a table column

    Instance variables:

        table -- the parent table instance
        name -- the name of the associated column
        type -- the type of column
        nrows -- the number of rows of the column
        index -- the Index object
        indexed -- Whether the column is indexed or not
        dirty -- whether the index is dirty or not

    Methods:
    
        __getitem__(key)
        createIndex()
        addRowsToIndex()
        reIndex()
        reIndexDirty()
        closeIndexes()
        
    """

    def __init__(self, table, name):
        """Create the container to keep the column information.

        table -- The parent table instance
        name -- The name of the column that is associated with this object
        
        """
        self.table = table
        self.name = name
        self.type = table.coltypes[name]
        self.nrows = table.nrows
        # Check whether an index exists or not
        iname = "_i_"+table.name+"_"+name
        self.index = None
        self.indexed = 0
        self.dirty = 0
        if iname in table._v_parent._v_indices:
            self.index = Index(where=self, name=iname,
                               expectedrows=table._v_expectedrows)
            self.indexed = 1
        elif hasattr(table, "colindexed") and table.colindexed[name]:
            # The user wants to indexate this column,
            # but it doesn't exists yet. Create it without a warning.
            self.createIndex(warn=0)

    def __len__(self):
        return self.table.nrows

    def __getitem__(self, key):
        """Returns a column element or slice

        It takes different actions depending on the type of the
        "key" parameter:

        If "key" is an integer, the corresponding element in the
        column is returned as a NumArray/CharArray, or a scalar
        object, depending on its shape. If "key" is a slice, the row
        slice determined by this slice is returned as a NumArray or
        CharArray object (whatever is appropriate).

        """
        
        if isinstance(key, types.IntType):
            if key < 0:
                # To support negative values
                key += self.table.nrows
            (start, stop, step) = processRange(self.table.nrows, key, key+1, 1)
            return self.table._read(start, stop, step, self.name, None)[0]
        elif isinstance(key, types.SliceType):
            (start, stop, step) = processRange(self.table.nrows, key.start,
                                               key.stop, key.step)
            return self.table._read(start, stop, step, self.name, None)
        else:
            raise TypeError, "'%s' key type is not valid in this context" % \
                  (key)

    def __lt__(self, other):
        self.table.ops.append(1)
        self.table.opsValues.append(other)
        return self

    def __le__(self, other):
        self.table.ops.append(2)
        self.table.opsValues.append(other)
        return self

    def __gt__(self, other):
        self.table.ops.append(3)
        self.table.opsValues.append(other)
        return self

    def __ge__(self, other):
        self.table.ops.append(4)
        self.table.opsValues.append(other)
        return self

    def __eq__(self, other):
        self.table.ops.append(5)
        self.table.opsValues.append(other)
        return self

    def __ne__(self, other):
        self.table.ops.append(6)
        self.table.opsValues.append(other)
        return self

    def createIndex(self, filters=None, warn=1, testmode=0):
        """Create an index for this column"""
        assert self.table.colshapes[self.name] == 1, \
               "Only scalar columns can be indexed."
        # Create the atom
        atomtype = self.table.coltypes[self.name]
        if str(atomtype) == "CharType":
            atom = StringAtom(shape=self.table.colshapes[self.name],
                              length=self.table.colitemsizes[self.name])
        else:
            atom = Atom(dtype=atomtype,
                        shape=self.table.colshapes[self.name])
        # Compose the name
        name = "_i_"+self.table.name+"_"+self.name
        # The filters for indexes are not inherited anymore. 2004-08-04
        if filters is None and hasattr(self.table, "indexprops"):
            filters = self.table.indexprops.filters
        # Create the index itself
        self.index = Index(atom, self, name,
                           "Index for "+self.table.name+":"+self.name,
                           filters=filters,
                           expectedrows=self.table._v_expectedrows,
                           testmode=testmode)
        self.indexed = 1
        self.dirty = 0
        self.table.colindexed[self.name] = 1
        # Feed the index with values
        nelemslice = self.index.sorted.nelemslice
        if self.table.nrows < self.index.sorted.nelemslice:
            if warn:
                # print "Debug: Not enough info for indexing"
                warnings.warn( \
"Not enough rows for indexing. You need at least %s rows and you provided %s." % (self.index.sorted.nelemslice, self.table.nrows))
            return 0
        return self.addRowsToIndex(0, self.table.nrows)

    def addRowsToIndex(self, start, nrows):
        """Add more elements to the existing index """
        nelemslice = self.index.nelemslice
        indexedrows = 0
        for i in xrange(start, nrows-nelemslice+1, nelemslice):
            arr = self[i:i+nelemslice]
            self.index.append(arr)
            indexedrows += nelemslice
        return indexedrows
        
    def reIndex(self):
        """Recompute the existing index"""
        if self.indexed:
            # Delete the existing Index
            self.index._f_remove()
            # Create a new Index without warnings
            return self.createIndex(warn=1)
        else:
            return 0  # The column is not intended for indexing
 
    def reIndexDirty(self):
        """Recompute the existing index only if it is dirty"""
        if self.indexed and self.dirty:
            # Delete the existing Index
            self.index._f_remove()
            # Create a new Index withoutr warnings
            return self.createIndex(warn=0)
        else:
            # The column is not intended for indexing or is not dirty
            return 0  
 
    def closeIndexes(self):
        """Close any open index of this column"""
        if self.index:
            self.index._f_close()
            self.index = None

    def close(self):
        """Close this column"""
        # Close indexes
        self.closeIndexes()
        # delete some back references
        self.table = None
        self.type = None

    def __str__(self):
        """The string representation for this object."""
        # The pathname
        pathname = self.table._v_pathname
        # Get this class name
        classname = self.__class__.__name__
        # The shape for this column
        shape = self.table.colshapes[self.name]
        if shape == 1:
            shape = (1,)
        # The type
        tcol = self.table.coltypes[self.name]
        return "%s.cols.%s (%s%s, %s)" % (pathname, self.name,
                                          classname, shape, tcol)

    def __repr__(self):
        """A detailed string representation for this object."""
        return str(self)
               
