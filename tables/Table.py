########################################################################
#
#       License: BSD
#       Created: September 4, 2002
#       Author:  Francesc Altet - faltet@carabos.com
#
#       $Source: /cvsroot/pytables/pytables/tables/Table.py,v $
#       $Id$
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

import sys
import warnings

import numarray
import numarray.strings as strings
import numarray.records as records

try:
    import Numeric
    Numeric_imported = True
except ImportError:
    Numeric_imported = False

import tables.hdf5Extension as hdf5Extension
from tables.utils import calcBufferSize, processRange, processRangeRead
from tables.Leaf import Leaf
from tables.Index import Index, IndexProps
from IsDescription import IsDescription, Description, Col, StringCol
from VLArray import Atom, StringAtom



__version__ = "$Revision: 1.140 $"


# Map Numarray record codes to Numarray types.
# This is extended with additional dataypes used by PyTables.
codeToNAType = records.numfmt.copy()
codeToNAType['t4'] = 'Time32'  # 32 bit integer time value
codeToNAType['t8'] = 'Time64'  # 64 bit real time value


byteorderDict={"=": sys.byteorder,
               "@": sys.byteorder,
               '<': 'little',
               '>': 'big'}

revbyteorderDict={'little': '<',
                  'big': '>'}

class Table(hdf5Extension.Table, Leaf):
    """Represent a table in the object tree.

    It provides methods to create new tables or open existing ones, as
    well as to write/read data to/from table objects over the
    file. A method is also provided to iterate over the rows without
    loading the entire table or column in memory.

    Data can be written or read both as Row instances or as numarray
    (NumArray or RecArray) objects.
    
    Methods:

        __getitem__(key)
        __iter__()
        __setitem__(key, value)
        append(rows)
        flushRowsToIndex()
        iterrows(start, stop, step)
        itersequence(sequence)
        modifyRows(start, rows)
        modifyColumns(start, columns, names)
        read([start] [, stop] [, step] [, field [, flavor]])
        reIndex()
        reIndexDirty()
        removeRows(start, stop)
        removeIndex(column)
        where(condition [, start] [, stop] [, step])
        whereIndexed(condition [, start] [, stop] [, step])
        whereInRange(condition [, start] [, stop] [, step])
        getWhereList(condition [, flavor])

    Instance variables:

        description -- the metaobject describing this table
        row -- a reference to the Row object associated with this table
        nrows -- the number of rows in this table
        rowsize -- the size, in bytes, of each row
        cols -- accessor to the columns using a natural name schema
        colnames -- the field names for the table (list)
        coltypes -- the type class for the table fields (dictionary)
        colstypes -- the string type for the table fields (dictionary)
        colshapes -- the shapes for the table fields (dictionary)
        colindexed -- whether the table fields are indexed (dictionary)
        indexed -- whether or not some field in Table is indexed
        indexprops -- properties of an indexed Table. Exists only
            if the Table is indexed

    """

    # Class identifier.
    _c_classId = 'TABLE'


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
        self.opsColnames = []
        # Initialize this object in case is a new Table
        if isinstance(description, dict):
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

    def _getTime64ColNames(self):
        """Returns a list containing 'Time64' column names."""

        # This should be generalised into some infrastructure to support
        # other kinds of columns to be converted.
        # ivilata(2004-12-21)

        return [
            cname for cname in self.colnames
            if self.colstypes[cname] == 'Time64']

    def _create(self):
        """Create a new table on disk."""

        # All this will eventually end up in the node constructor.

        # Compute some important parameters for createTable
        self.colnames = tuple(self.description.__names__)
        self._v_fmt = self.description._v_fmt
        # Create the table on disk
        self._createTable(self._v_new_title, self.filters.complib)
        # Initialize the shape attribute
        self.shape = (self.nrows,)
        # Get the column types and string types
        self.coltypes = self.description.__types__
        self.colstypes = self.description.__stypes__
        # Extract the shapes for columns
        self.colshapes = self.description._v_shapes
        self.colitemsizes = self.description._v_itemsizes
        # Compute the byte order
        self.byteorder = byteorderDict[self._v_fmt[0]]
        # Find Time64 column names. (This should be generalised.)
        self._time64colnames = self._getTime64ColNames()
        # Create the Row object helper
        self.row = hdf5Extension.Row(self)

        self.colindexed = {}  # Is the key column indexed?
        self.indexed = 0      # Are there any indexed columns?
        colobjects = self.description._v_ColObjects
        for (colname, colobj) in colobjects.iteritems():
            colindexed = colobj.indexed
            self.colindexed[colname] = colindexed
            if colindexed:
                self.indexed = 1 # True

        if self.indexed:
            # Check whether we want automatic indexing after an append or not
            # If the user has not defined properties, assign the default
            self.indexprops = getattr(
                self.description, '_v_indexprops', IndexProps())
            self._indexedrows = 0
            self._unsaved_indexedrows = 0
            # Save AUTOMATIC_INDEX and REINDEX flags as attributes
            setAttr = self._v_attrs._g__setattr
            setAttr('AUTOMATIC_INDEX', self.indexprops.auto)
            setAttr('REINDEX', self.indexprops.reindex)
        # Create a cols accessor
        self.cols = Cols(self)

    def _open(self):
        """Opens a table from disk and read the metadata on it.

        Creates an user description on the flight to easy the access to
        the actual data.

        """

        # All this will eventually end up in the node constructor.

        # Get table info
        (self.nrows, self.colnames, self.rowsize, itemsizes, colshapes,
         colstypes, self._v_fmt) = self._getTableInfo()
        # Get the byteorder
        byteorder = self._v_fmt[0]
        # Remove the byteorder
        self._v_fmt = self._v_fmt[1:]
        # The expectedrows would be the actual number
        self._v_expectedrows = self.nrows
        self.byteorder = byteorderDict[byteorder]
        colstypes = [str(codeToNAType[type]) for type in colstypes]

        fields = {}           # Maps column names to Col objects.
        self.colindexed = {}  # Is the specified column indexed?
        self.indexed = 0      # Are there any indexed columns?
        indexcname = None     # Column name of some indexed column.
        for i in xrange(len(self.colnames)):
            colname = self.colnames[i]
            colshape = colshapes[i]
            colstype = colstypes[i]

            # Is this column indexed?
            indexname = '_i_%s_%s' % (self.name, colname)
            indexed = indexname in self._v_parent._v_indices
            self.colindexed[colname] = indexed
            if indexed:
                self.indexed = 1 # True
                indexcname = colname

            if colstype == 'CharType':
                itemsize = itemsizes[i]
                colobj = StringCol(length = itemsize, shape = colshape,
                                   pos = i, indexed = indexed)
            else:
                colobj = Col(dtype = colstype, shape = colshape,
                             pos = i, indexed = indexed)
            fields[colname] = colobj

        # Set the alignment!
        fields['_v_align'] = byteorder
        if self._v_file._isPTFile:
            # Checking validity names for fields is not necessary
            # when opening a PyTables file
            fields['__check_validity__'] = 0
        # Create an instance description to host the record fields
        self.description = Description(fields)
        
        # Extract the coltypes, colstypes, shapes and itemsizes
        self.coltypes = self.description.__types__
        self.colstypes = self.description.__stypes__
        self.colshapes = self.description._v_shapes
        self.colitemsizes = self.description._v_itemsizes
        # Find Time64 column names. (This should be generalised.)
        self._time64colnames = self._getTime64ColNames()
        # Compute buffer size
        (self._v_maxTuples, self._v_chunksize) = \
              calcBufferSize(self.rowsize, self.nrows)
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
        if self.indexed:
            autoindex = getattr(self.attrs, 'AUTOMATIC_INDEX', None)
            reindex = getattr(self.attrs, 'REINDEX', None)
            indexobj = getattr(self.cols, indexcname).index

            self.indexprops = IndexProps(auto=autoindex, reindex=reindex,
                                         filters=indexobj.filters)
            self._indexedrows = indexobj.nelements
            self._unsaved_indexedrows = self.nrows - self._indexedrows


    def where(self, condition=None, start=None, stop=None, step=None):
        """Iterator that selects values fulfilling the 'condition' param.
        
        condition can be used to specify selections along a column in the
        form:

        condition=(0<table.cols.col1<0.3)

        If the column to which the condition is applied is indexed,
        the index will be used in order to accelerate the
        search. Else, the in-kernel iterator will be choosed instead.
        
        """

        if not isinstance(condition, Column):
            raise TypeError("""\
Wrong 'condition' parameter type. Only Column instances are suported.""")

        if condition.index and not condition.dirty:
            # Call the indexed version method
            return self.whereIndexed(condition, start, stop, step)
        # Fall back to in-kernel selection method
        return self.whereInRange(condition, start, stop, step)

    def whereIndexed(self, condition, start=None, stop=None, step=None):
        """Iterator that selects values fulfilling the 'condition' param.
        
        condition can be used to specify selections along a column in the
        form:

        condition=(0<table.cols.col1<0.3)

        This method is only intended to be used for indexed columns.
        """

        if not isinstance(condition, Column):
            raise TypeError("""\
Wrong 'condition' parameter type. Only Column instances are suported.""")
        if condition.index is None:
            raise ValueError("""\
This method is intended only for indexed columns.""")
        if condition.dirty:
            raise ValueError("""\
This method is intended only for indexed columns, but this column has a dirty index. Try re-indexing it in order to put the index in a sane state.""")

        self.whereColname = condition.name   # Flag for Row.__iter__
        # Get the coordinates to lookup
        ncoords = condition.index.getLookupRange(condition)
        if ncoords > 0:
            # Call the indexed version of Row iterator (coords=None,ncoords>=0)
            (start, stop, step) = processRangeRead(self.nrows, start, stop,
                                                   step)
            return self.row(start, stop, step, coords=None, ncoords=ncoords)
        else:
            # Fall-back action is to return an empty iterator
            self.ops = []
            self.opsValues = []
            self.opsColnames = []
            self.whereColname = None
            return iter([])

    def readIndexed(self, condition):
        """Returns a RecArray fulfilling the 'condition' param.
        
        condition can be used to specify selections along a column in the
        form:

        condition=(0<table.cols.col1<0.3)

        This method is only intended to be used for indexed columns.
        """

        assert isinstance(condition, Column), \
"Wrong condition parameter type. Only Column instances are suported."
        assert condition.index is not None, \
               "This method is intended only for indexed columns"
        assert condition.dirty == 0, \
               "This method is intended only for indexed columns, but this column has a dirty index. Try re-indexing it in order to put the index in a sane state. "

        self.whereColname = condition.name   # Flag for Row.__iter__
        # Get the coordinates to lookup
        nrecords = condition.index.getLookupRange(condition)
        recarr = records.array(None,
                               formats=self.description._v_recarrfmt,
                               shape=(nrecords,),
                               names = self.colnames)
        if nrecords > 0:
            # Read the contents of a selection in a recarray
            condition.index.indices._initIndexSlice(nrecords)
            coords = condition.index.getCoords(0, nrecords)
            recout = self._read_elements_ra(recarr, coords)
            if self.byteorder <> sys.byteorder:
                recarr._byteswap()
            condition.index.indices._destroyIndexSlice()
        # Delete indexation caches
        self.ops = []
        self.opsValues = []
        self.opsColnames = []
        self.whereColname = None
        return recarr

    def readIndexed(self, condition):
        """Returns a RecArray fulfilling the 'condition' param.
        
        condition can be used to specify selections along a column in the
        form:

        condition=(0<table.cols.col1<0.3)

        This method is only intended to be used for indexed columns.
        """

        assert isinstance(condition, Column), \
"Wrong condition parameter type. Only Column instances are suported."
        assert condition.index is not None, \
               "This method is intended only for indexed columns"
        assert condition.dirty == 0, \
               "This method is intended only for indexed columns, but this column has a dirty index. Try re-indexing it in order to put the index in a sane state. "

        self.whereColname = condition.name   # Flag for Row.__iter__
        # Get the coordinates to lookup
        nrecords = condition.index.getLookupRange(condition)
        recarr = records.array(None,
                               formats=self.description._v_recarrfmt,
                               shape=(nrecords,),
                               names = self.colnames)
        if nrecords > 0:
            # Read the contents of a selection in a recarray
            condition.index.indices._initIndexSlice(nrecords)
            coords = condition.index.getCoords(0, nrecords)
            recout = self._read_elements_ra(recarr, coords)
            if self.byteorder <> sys.byteorder:
                recarr._byteswap()
            condition.index.indices._destroyIndexSlice()
        # Delete indexation caches
        self.ops = []
        self.opsValues = []
        self.opsColnames = []
        self.whereColname = None
        return recarr

    def whereInRange(self, condition, start=None, stop=None, step=None):
        """Iterator that selects values fulfilling the 'condition' param.
        
        'condition' can be used to specify selections along a column
        in the form:

        condition=(0<table.cols.col1<0.3)

        This method will use the in-kernel search method, i.e. it
        won't take advantage of a possible indexed column.
        
        """

        if not isinstance(condition, Column):
            raise TypeError("""\
Wrong 'condition' parameter type. Only Column instances are suported.""")

        self.whereColname = condition.name   # Flag for Row.__iter__
        (start, stop, step) = processRangeRead(self.nrows, start, stop, step)
        if start < stop:
            # call row with coords=None and ncoords=-1 (in-kernel selection)
            return self.row(start, stop, step, coords=None, ncoords=-1)
        # Fall-back action is to return an empty RecArray
        return iter([])
        
    def getWhereList(self, condition, flavor="List"):
        """Get the row coordinates that fulfill the 'condition' param

        'condition' can be used to specify selections along a column
        in the form:

        condition=(0<table.cols.col1<0.3)

        'flavor' is the desired type of the returned list. It can take
        the 'List', 'Tuple' or 'NumArray' values.

        """

        if not isinstance(condition, Column):
            raise TypeError("""\
Wrong 'condition' parameter type. Only Column instances are suported.""")

        supportedFlavors = ['NumArray', 'List', 'Tuple']
        if flavor not in supportedFlavors:
            raise ValueError("""\
Specified 'flavor' value is not one of %s.""" % (supportedFlavors,))

        # Take advantage of indexation, if present
        if condition.index is not None:
            # get the number of coords and set-up internal variables
            ncoords = condition.index.getLookupRange(condition)
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
        self.opsColnames = []
        self.whereColname = None
        # do some conversion (if needed)
        if flavor == "List":
            coords = coords.tolist()
        elif flavor == "Tuple":
            coords = tuple(coords.tolist())
        return coords

    def itersequence(self, sequence=None, sort=1):
        """Iterate over a list of row coordinates.

        sort means that sequence will be sorted so that I/O would
        perform better. If your sequence is already sorted or you
        don't want to sort it, put this parameter to 0. The default is
        to sort the sequence.
        """

        if not hasattr(sequence, '__getitem__'):
            raise TypeError("""\
Wrong 'sequence' parameter type. Only sequences are suported.""")

        coords = numarray.array(sequence, type=numarray.Int64)
        # That would allow the retrieving on a sequential order
        # so, given better speed in the general situation
        if sort:
            coords.sort()
        return self.row(coords=coords, ncoords=-1)
        
    def iterrows(self, start=None, stop=None, step=None):
        """Iterate over all the rows or a range.
        
        Specifying a negative value of step is not supported yet.
        
        """
        (start, stop, step) = processRangeRead(self.nrows, start, stop, step)
        if start < stop:
            return self.row(start, stop, step, coords=None, ncoords=-1)
        # Fall-back action is to return an empty iterator
        return iter([])
        
    def __iter__(self):
        """Iterate over all the rows."""

        return self.iterrows()

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
                            shape = arr.shape
                            arr=Numeric.fromstring(arr._data,
                                                   typecode=arr.typecode())
                            arr.shape = shape
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
            if type(shape) in (int,long):
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
            else:
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
                    self._read_elements(result, coords)
            else:
                self._read_records(result, start, stop-start)
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

        If 'key' is an integer, the corresponding table row is
        returned as a tuple object. If 'key' is a slice, the row slice
        determined by key is returned as a RecArray object.  Finally,
        if 'key' is a string, it is interpreted as a column name in
        the table, and, if it exists, it is read and returned as a
        NumArray or CharArray object (whatever is appropriate).

"""

        if type(key) in (int,long):
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
        elif isinstance(key, slice):
            (start, stop, step) = processRange(self.nrows,
                                               key.start, key.stop, key.step)
            return self._read(start, stop, step, None, None)
        elif isinstance(key, str):
            return self.read(field=key)
        else:
            raise ValueError, "Non-valid index or slice: %s" % str(key)

    def __setitem__(self, key, value):
        """Sets a table row or table slice.

        It takes different actions depending on the type of the 'key'
        parameter:

        If 'key' is an integer, the corresponding table row is set to
        'value' (List or Tuple). If 'key' is a slice, the row slice
        determined by key is set to value (a RecArray or list of
        rows).

        """

        if self._v_file.mode == 'r':
            raise IOError("""\
Attempt to write over a file opened in read-only mode.""")

        if type(key) in (int,long):
            # Index out of range protection
            if key >= self.nrows:
                raise IndexError, "Index out of range"
            if key < 0:
                # To support negative values
                key += self.nrows
            return self.modifyRows(key, key+1, 1, [value])
        elif isinstance(key, slice):
            (start, stop, step) = processRange(self.nrows,
                                               key.start, key.stop, key.step)
            return self.modifyRows(start, stop, step, value)
        else:
            raise ValueError, "Non-valid index or slice: %s" % key

    def append(self, rows=None):
        """Append a series of rows to the end of the table

        rows can be either a recarray or a structure that is able to
        be converted to a recarray compliant with the table format.

        Returns the number of rows appended.

        It raises an 'ValueError' in case the rows parameter could not
        be converted to an object compliant with table description.

        """

        if self._v_file.mode == 'r':
            raise IOError("""\
Attempt to write over a file opened in read-only mode.""")

        if rows is None:
            return 0
        # Try to convert the object into a recarray
        try:
            # This always makes a copy of the original,
            # so the resulting object is safe to in-place conversion.
            recarray = records.array(rows,
                                     formats=self.description._v_recarrfmt,
                                     names=self.colnames)
        except:  #XXX
            (typerr, value, traceback) = sys.exc_info()
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
        if self.indexed:
            # Update the number of unsaved indexed rows
            self._unsaved_indexedrows += lenrows
            if self.indexprops.auto:
                self.flushRowsToIndex(lastrow=0)
        return lenrows

    def _saveBufferedRows(self):
        """Save buffered table rows"""
        # Save the records on disk
        # Data is copied below to the buffer,
        # so the operation is safe to in-place conversion.
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
            self.flushRowsToIndex(lastrow=0)
        return

    def modifyRows(self, start=None, stop=None, step=1, rows=None):
        """Modify a series of rows in the slice [start:stop:step]

        rows can be either a recarray or a structure that is able to
        be converted to a recarray compliant with the table format.

        Returns the number of modified rows.

        It raises an 'ValueError' in case the rows parameter could not
        be converted to an object compliant with table description.

        It raises an 'IndexError' in case the modification will exceed
        the length of the table.

        """

        if rows is None:      # Nothing to be done
            return
        if start is None:
            start = 0

        if start < 0:
            raise ValueError("'start' must have a positive value.")
        if step < 1:
            raise ValueError("'step' must have a value greater or equal than 1.")
        if stop is None:
            # compute the stop value. start + len(rows)*step does not work
            stop = start + (len(rows)-1)*step + 1
        
        (start, stop, step) = processRange(self.nrows, start, stop, step)
        if stop > self.nrows:
            raise IndexError, \
"This modification will exceed the length of the table. Giving up."
        # Compute the number of rows to read. (stop-start)/step does not work
        nrows = ((stop - start - 1) / step) + 1
        if len(rows) < nrows:
            raise ValueError, \
           "The value has not enough elements to fill-in the specified range"
        # Try to convert the object into a recarray
        try:
            # This always makes a copy of the original,
            # so the resulting object is safe to in-place conversion.
            recarray = records.array(rows,
                                     formats=self.description._v_recarrfmt,
                                     names=self.colnames)
            # records.array does not seem to change the names
            # attibute in case rows is a recarray.
            # Change it manually and report this
            # 2004-08-08
            recarray._names = self.colnames
        except:  #XXX
            (typerr, value, traceback) = sys.exc_info()
            raise ValueError, \
"rows parameter cannot be converted into a recarray object compliant with table format '%s'. The error was: <%s>" % (str(self.description._v_recarrfmt), value)
        lenrows = len(recarray)
        if start + lenrows > self.nrows:
            raise IndexError, \
"This modification will exceed the length of the table. Giving up."
        self._modify_records(start, stop, step, recarray)
        # Redo the index if needed
        if self.indexed:
            # Mark all the indexes as dirty
            for (colname, colindexed) in self.colindexed.iteritems():
                if colindexed:
                    indexcol = getattr(self.cols, colname)
                    indexcol.dirty = 1
            if self.indexprops.reindex:
                self._indexedrows = self.reIndex()
                self._unsaved_indexedrows = self.nrows - self._indexedrows
        return lenrows

    def modifyColumns(self, start=None, stop=None, step=1,
                      columns=None, names=None):
        """Modify a series of columns in the row slice [start:stop:step]

        columns can be either a recarray or a list of arrays (the
        columns) that is able to be converted to a recarray compliant
        with the specified colnames subset of the table format.

        names specifies the column names of the table to be modified.

        Returns the number of modified rows.

        It raises an 'ValueError' in case the columns parameter could
        not be converted to an object compliant with table
        description.

        It raises an 'IndexError' in case the modification will exceed
        the length of the table.

        """

        if type(names) not in (list, tuple):
            raise TypeError("""\
The 'names' parameter must be a list of strings.""")

        if columns is None:      # Nothing to be done
            return 0
        if start is None:
            start = 0

        if start < 0:
            raise ValueError("'start' must have a positive value.")
        if step < 1:
            raise ValueError("'step' must have a value greater or equal than 1.")
        # Get the column formats to be modified:
        formats = []
        colnames = list(self.colnames)
        for colname in names:
            if colname in colnames:
                pos = colnames.index(colname)
                formats.append(self.description._v_recarrfmt[pos])
            else:
                raise KeyError, \
                      "Column '%s' does not exists on table"
        # Try to convert the object columns into a recarray
        try:
            if isinstance(columns, records.RecArray):
                # This always makes a copy of the original,
                # so the resulting object is safe to in-place conversion.
                recarray = records.array(columns, formats=formats,
                                         names=names)
                # records.array does not seem to change the names
                # attibute in case rows is a recarray.
                # Change it manually and report this
                # 2004-08-08
                recarray._names = names
                # I don't know why I should do that here
                recarray._fields = recarray._get_fields()  # Refresh the cache
            else:
                # This always makes a copy of the original,
                # so the resulting object is safe to in-place conversion.
                recarray = records.fromarrays(columns, formats=formats,
                                              names=names)
        except:  #XXX
            (typerr, value, traceback) = sys.exc_info()
            raise ValueError, \
"columns parameter cannot be converted into a recarray object compliant with table '%s'. The error was: <%s>" % (str(self), value)

        if stop is None:
            # compute the stop value. start + len(rows)*step does not work
            stop = start + (len(recarray)-1)*step + 1
        (start, stop, step) = processRange(self.nrows, start, stop, step)
        if stop > self.nrows:
            raise IndexError, \
"This modification will exceed the length of the table. Giving up."
        # Compute the number of rows to read. (stop-start)/step does not work
        nrows = ((stop - start - 1) / step) + 1
        if len(recarray) < nrows:
            raise ValueError, \
           "The value has not enough elements to fill-in the specified range"
        # Now, read the original values:
        mod_recarr = self.read(start, stop, step)
        mod_recarr._fields = mod_recarr._get_fields()  # Refresh the cache
        # Modify the appropriate columns in the original recarray
        for name in names:
            mod_recarr._fields[name][:] = recarray._fields[name]
        # save this modified rows in table
        self._modify_records(start, stop, step, mod_recarr)
        # Redo the index if needed
        if self.indexed:
            # First, mark the modified indexes as dirty
            for (colname, colindexed) in self.colindexed.iteritems():
                if colindexed and colname in names:
                    col = getattr(self.cols, colname)
                    col.dirty = 1
            # Then, reindex if needed
            if self.indexprops.reindex:
                self._indexedrows = self.reIndex()
                self._unsaved_indexedrows = self.nrows - self._indexedrows
        return nrows

    def flushRowsToIndex(self, lastrow=1):
        "Add remaining rows in buffers to non-dirty indexes"
        rowsadded = 0
        if self.indexed:
            # Update the number of unsaved indexed rows
            start = self._indexedrows
            nrows = self._unsaved_indexedrows
            for (colname, colindexed) in self.colindexed.iteritems():
                if colindexed:
                    col = getattr(self.cols, colname)
                    if nrows > 0 and not col.dirty:
                        rowsadded = col._addRowsToIndex(start, nrows, lastrow)
            self._unsaved_indexedrows -= rowsadded
            self._indexedrows += rowsadded
        return rowsadded

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
        self.shape = (self.nrows,)    # update to the new shape
        # removeRows is a invalidating index operation
        if self.indexed:
            if self.indexprops.reindex:
                self._indexedrows = self.reIndex()
                self._unsaved_indexedrows = self.nrows - self._indexedrows
            else:
                # Mark all the indexes as dirty
                for (colname, colindexed) in self.colindexed.iteritems():
                    if colindexed:
                        col = getattr(self.cols, colname)
                        col.dirty = 1
                        
        return nrows

    def removeIndex(self, index=None):
        "Remove the index associated with the specified column"
        if not isinstance(index, Index):
            raise TypeError("""\
Wrong 'index' parameter type. Only Index instances are accepted.""")
        index.column.removeIndex()

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
            # The recarray is created anew,
            # so the operation is safe to in-place conversion.
            object._append_records(recarray, nrows)
            nrowsdest += nrows
        object._close_append()
        # Update the number of saved rows in this buffer
        object.nrows = nrowsdest
        # Set the shape attribute (the self.nrows may be less than the maximum)
        object.shape = (nrowsdest,)
        return

    # This is an optimized version of copy
    def _g_copyWithStats(self, group, name, start, stop, step, title, filters):
        "Private part of Leaf.copy() for each kind of leaf"
        # Build the new Table object
        description = self.description._v_ColObjects
        # Checking validity names for fields in destination is not necessary
        description['__check_validity__'] = 0
        # Add a possible IndexProps property to that
        if hasattr(self, "indexprops"):
            description["_v_indexprops"] = self.indexprops
        object = self._v_file.createTable(
            group, name, description, title=title, filters=filters,
            expectedrows=self.nrows, _log = False)
        # Now, fill the new table with values from the old one
        self._g_copyRows(object, start, stop, step)
        nbytes=self.nrows*self.rowsize
        if object.indexed:
            object._indexedrows = 0
            object._unsaved_indexedrows = object.nrows
            if object.indexprops.auto:
                object.flushRowsToIndex(lastrow=1)
        return (object, nbytes)

    def flush(self):
        """Flush the table buffers."""
        # Flush any unsaved row
        if hasattr(self, 'row'):
            if self.row._getUnsavedNRows() > 0:
                self._saveBufferedRows()
            # Flush the data to disk
            super(Table, self).flush()
        if hasattr(self, "indexed") and self.indexed and self.indexprops.auto:
            # Flush any unindexed row
            rowsadded = self.flushRowsToIndex(lastrow=1)
            if rowsadded > 0 and self._indexedrows <> self.nrows:
                raise RuntimeError , "Internal error: the number of indexed rows (%s) and rows in table (%s) must be equal!. Please, report this to the author." % (self._indexedrows, self.nrows)
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

    def _f_close(self, flush = True):
        # Close the Table
        super(Table, self)._f_close(flush)
        if hasattr(self, "cols"):
            self.cols._f_close()
            self.cols = None
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

        if self.indexed:
            return \
"""%s
  description := %r
  indexprops := %r
  byteorder := %s""" % \
        (str(self), self.description, self.indexprops, self.byteorder)
        else:
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
        __len__()
        
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

        if not isinstance(name, str):
            raise TypeError, \
"Only strings are allowed as keys of a Cols instance. You passed object: %s" % name
        # If attribute does not exist, return None
        if not name in self._v_colnames:
            raise KeyError, \
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
        self.__dict__.clear()

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
        index -- the Index object (None if doesn't exists)
        dirty -- whether the index is dirty or not (property)

    Methods:
        __getitem__(key)
        __setitem__(key, value)
        createIndex()
        reIndex()
        reIndexDirty()
        removeIndex()
        closeIndex()
    """

    def __init__(self, table, name):
        """Create the container to keep the column information.

        table -- The parent table instance
        name -- The name of the column that is associated with this object
        """
        self.table = table
        self.name = name
        self.type = table.coltypes[name]
        # Check whether an index exists or not
        indexname = '_i_%s_%s' % (table.name, name)
        self.index = None
        if indexname in table._v_parent._v_indices:
            self.index = Index(where=self, name=indexname,
                               expectedrows=table._v_expectedrows)
        elif hasattr(table, "colindexed") and table.colindexed[name]:
            # The user wants to indexate this column,
            # but it doesn't exists yet. Create it without a warning.
            self.createIndex(warn=0)

    # Define dirty as a property
    def _get_dirty(self):
        if self.index:
            dirty = self.index._v_attrs._g_getAttr("DIRTY")
            if dirty is None:
                return 0
            else:
                return dirty
        else:
            # The index does not exist, so it can't be dirty
            return 0
    
    def _set_dirty(self, dirty):
        # Only set the index column as dirty if it exists
        if self.index:
            self.index._v_attrs._g_setAttr("DIRTY", dirty)
            self.index.lrri[-1] = 0
            self.index.nelementsLR = 0
            self.index.nelements = 0
            #self.table._indexedrows = 0
            #self.table._unsaved_indexedrows = self.table.nrows
        
    # Define a property.  The 'delete this attribute'
    # method is defined as None, so the attribute can't be deleted.
    dirty = property(_get_dirty, _set_dirty, None, "Column dirtyness")

    def __len__(self):
        return self.table.nrows

    def __getitem__(self, key):
        """Returns a column element or slice

        It takes different actions depending on the type of the 'key'
        parameter:

        If 'key' is an integer, the corresponding element in the
        column is returned as a NumArray/CharArray, or a scalar
        object, depending on its shape. If 'key' is a slice, the row
        slice determined by this slice is returned as a NumArray or
        CharArray object (whatever is appropriate).

        """

        if type(key) in (int,long):
            # Index out of range protection
            if key >= self.table.nrows:
                raise IndexError, "Index out of range"
            if key < 0:
                # To support negative values
                key += self.table.nrows
            (start, stop, step) = processRange(self.table.nrows, key, key+1, 1)
            return self.table._read(start, stop, step, self.name, None)[0]
        elif isinstance(key, slice):
            (start, stop, step) = processRange(self.table.nrows, key.start,
                                               key.stop, key.step)
            return self.table._read(start, stop, step, self.name, None)
        else:
            raise TypeError, "'%s' key type is not valid in this context" % \
                  (key)

    def __setitem__(self, key, value):
        """Sets a column element or slice.

        It takes different actions depending on the type of the 'key'
        parameter:

        If 'key' is an integer, the corresponding element in the
        column is set to 'value' (scalar or NumArray/CharArray,
        depending on column's shape). If 'key' is a slice, the row
        slice determined by 'key' is set to 'value' (a
        NumArray/CharArray or list of elements).

        """

        if self.table._v_file.mode == 'r':
            raise IOError("""\
Attempt to write over a file opened in read-only mode.""")

        if type(key) in (int,long):
            # Index out of range protection
            if key >= self.table.nrows:
                raise IndexError, "Index out of range"
            if key < 0:
                # To support negative values
                key += self.table.nrows
            return self.table.modifyColumns(key, key+1, 1,
                                            [[value]], names=[self.name])
        elif isinstance(key, slice):
            (start, stop, step) = processRange(self.table.nrows,
                                               key.start, key.stop, key.step)
            return self.table.modifyColumns(start, stop, step,
                                            [value], names=[self.name])
        else:
            raise ValueError, "Non-valid index or slice: %s" % key

    def _addComparison(self, noper, other):
        self.table.ops.append(noper)
        self.table.opsValues.append(other)
        self.table.opsColnames.append(self.name)
#         if (len(self.table.ops) > 1 and
#             (self.table.ops[-1] < 10 and self.table.ops[-2] < 10)):
#             # To deal with 'number < col < number' style comparisons
#             # add a logical and
#             self._addLogical(10)  # 10 is __and__

    def __lt__(self, other):
        self._addComparison(1, other)
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

    def _addLogical(self, noper):
        self.table.ops.append(noper)
        self.table.opsValues.append(None)
        self.table.opsColnames.append(None)
        
    def __and__(self, other):
        self._addLogical(10)
        return self

    def __or__(self, other):
        self._addLogical(11)
        return self

    def __xor__(self, other):
        self._addLogical(12)
        return self

    def createIndex(self, warn=1, testmode=0):
        """Create an index for this column"""
        if self.table.colshapes[self.name] != 1:
            raise ValueError("Only scalar columns can be indexed.")
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
        if hasattr(self.table, "indexprops"):
            filters = self.table.indexprops.filters
        else:
            filters = None  # Get the defaults
        # Create the index itself
        if self.index:
            raise ValueError, \
"%s for column '%s' already exists. If you want to re-create it, please, try with reIndex() method better" % (str(self.index), str(self.name))
        self.index = Index(atom, self, name,
                           "Index for "+self.table._v_pathname+".cols."+self.name,
                           filters=filters,
                           expectedrows=self.table._v_expectedrows,
                           testmode=testmode)
        self.dirty = 0
        self.table.colindexed[self.name] = 1
        # Feed the index with values
        return self._addRowsToIndex(0, self.table.nrows, lastrow=1)

    def _addRowsToIndex(self, start, nrows, lastrow):
        """Add more elements to the existing index """
        nelemslice = self.index.nelemslice
        #assert self.table.nrows >= self.index.sorted.nelemslice
        indexedrows = 0
        for i in xrange(start, start+nrows-nelemslice+1, nelemslice):
            self.index.append(self[i:i+nelemslice])
            indexedrows += nelemslice
        # index the remaining rows
        nremain = nrows - indexedrows
        if nremain > 0 and lastrow:
            self.index.appendLastRow(self[indexedrows:nrows], self.table.nrows)
            indexedrows += nremain
        return indexedrows
        
    def reIndex(self):
        """Recompute the existing index"""
        if self.index is not None:
            # Delete the existing Index
            self.index._g_remove()
            self.index = None
            # Create a new Index without warnings
            return self.createIndex(warn=0)
        else:
            return 0  # The column is not intended for indexing
 
    def reIndexDirty(self):
        """Recompute the existing index only if it is dirty"""
        if self.index is not None and self.dirty:
            # Delete the existing Index
            self.index._g_remove()
            # Create a new Index without warnings
            return self.createIndex(warn=0)
        else:
            # The column is not intended for indexing or is not dirty
            return 0  
 
    def removeIndex(self):
        """Delete the associated column's index"""
        # delete some references
        if self.index:
            self.index._g_remove()
            self.index = None
            self.table.colindexed[self.name] = 0
        else:
            return  # Do nothing

    def closeIndex(self):
        """Close the index of this column"""
        if self.index:
            self.index._f_close()
            self.index = None

    def close(self):
        """Close this column"""
        # Close indexes
        self.closeIndex()
        # delete some back references
        self.table = None
        self.type = None
        # After the objects are disconnected, destroy the
        # object dictionary using the brute force ;-)
        # This should help to the garbage collector
        self.__dict__.clear()        

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
        return "%s.cols.%s (%s%s, %s, idx=%s)" % \
               (pathname, self.name, classname, shape, tcol, self.index)

    def __repr__(self):
        """A detailed string representation for this object."""
        return str(self)
               
