########################################################################
#
#       License: BSD
#       Created: September 4, 2002
#       Author:  Francesc Altet - faltet@carabos.com
#
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

import warnings
import re
from time import time

import numarray
import numarray.numarraycore
import numarray.records as records

try:
    import Numeric
    Numeric_imported = True
except ImportError:
    Numeric_imported = False

try:
    import numpy
    numpy_imported = True
except ImportError:
    numpy_imported = False

import tables.nestedrecords as nestedrecords
from tables.nriterators import flattenNames
import tables.TableExtension as TableExtension
from tables.utils import calcBufferSize, processRange, processRangeRead, \
     joinPath, convertNAToNumeric, convertNAToNumPy, fromnumpy, tonumpy, is_idx
from tables.Leaf import Leaf
from tables.Index import Index, IndexProps, split_index_condXXX
from tables.IsDescription import \
     IsDescription, Description, Col, StringCol, EnumCol
from tables.Atom import Atom, StringAtom
from tables.Group import IndexesTableG, IndexesDescG
from tables.exceptions import NodeError, HDF5ExtError, PerformanceWarning
from tables.constants import MAX_COLUMNS, EXPECTED_ROWS_TABLE

__version__ = "$Revision$"


# 2.2: Added support for complex types. Introduced in version 0.9.
# 2.2.1: Added suport for time types.
# 2.3: Changed the indexes naming schema.
# 2.4: Changed indexes naming schema (again).
# 2.5: Added the FIELD_%d_FILL attributes.
# 2.6: Added the FLAVOR attribute (optional).
obversion = "2.6"  # The Table VERSION number

# The supported flavors for Table object
supportedFlavors = ['numarray', 'numpy', 'numeric', 'python']

# Map Numarray record codes to Numarray types.
# This is extended with additional dataypes used by PyTables.
codeToNAType = records.numfmt.copy()
codeToNAType['t4'] = 'Time32'  # 32 bit integer time value
codeToNAType['t8'] = 'Time64'  # 64 bit real time value
codeToNAType['e'] = 'Enum'  # enumerated value, must also find base type


# Paths and names for hidden nodes related with indexes.
_indexName   = '_i_%s'  # %s -> encoded table path

# Compile a regular expression for expressions like '(2,2)Int8'
prog = re.compile(r'([\(\),\d\s]*)([A-Za-z]+[0-9]*)')

def _getEncodedTableName(tablename):
    return _indexName % tablename

def _getIndexTableName(parent, tablename):
    return joinPath(parent._v_pathname, _getEncodedTableName(tablename))

def _getIndexColName(parent, tablename, colname):
    return joinPath(_getIndexTableName(parent, tablename), colname)


class Table(TableExtension.Table, Leaf):
    """Represent a table in the object tree.

    It provides methods to create new tables or open existing ones, as
    well as to write/read data to/from table objects over the
    file. A method is also provided to iterate over the rows without
    loading the entire table or column in memory.

    Data can be written or read both as Row instances, numarray
    (NumArray or RecArray) objects or NestedRecArray objects.

    Methods:

        __getitem__(key)
        __iter__()
        __setitem__(key, value)
        append(rows)
        col(name)
        flushRowsToIndex()
        iterrows(start, stop, step)
        itersequence(sequence)
        modifyRows(start, rows)
        modifyColumn(columns, names, [start] [, stop] [, step])
        modifyColumns(columns, names, [start] [, stop] [, step])
        read([start] [, stop] [, step] [, field [, flavor]])
        reIndex()
        reIndexDirty()
        removeRows(start [, stop])
        removeIndex(column)
        where(condition [, start] [, stop] [, step])
        whereAppend(dstTable, condition [, start] [, stop] [, step])
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
        indexprops -- properties of an indexed Table

    """

    # Class identifier.
    _c_classId = 'TABLE'


    # <properties>

    # `row` is defined as a read-only attribute.

    def _g_getrow(self):
        return TableExtension.Row(self)

    row = property(_g_getrow, None, None, "The associated `Row` instance.")

    # Some read-only shorthands.

    def _g_getshape(self):
        return (self.nrows,)
    shape = property(_g_getshape, None, None, "The shape of this table.")

    def _g_getrowsize(self):
        return self.description._v_totalsize
    rowsize = property(_g_getrowsize, None, None,
                       "The size in bytes of each row in the table.")

    def _g_getbyteorder(self):
        return self.description._v_byteorder
    byteorder = property(_g_getbyteorder, None, None,
                         "The endianness of data in memory "
                         "('big', 'little' or 'non-relevant').")

    def _g_getflavor(self):
        # Check if there is some "FLAVOR" attribute (remember, it is optional)
        return getattr(self._v_attrs, "FLAVOR", "numarray")
    def _g_setflavor(self, value):
        if value not in ["numarray", "numpy"]:
            raise ValueError, \
"""The supported flavors for a table are: "numarray" and "numpy" and you passed %s.
""" % (value)
        setattr(self._v_attrs, "FLAVOR", value)
    flavor = property(_g_getflavor, _g_setflavor, None,
                      "The flavor that will be used when returning read data.")

    def _g_getemptyarray(self, type, isize=None):
        # Acts as a cache for empty arrays
        if type in self._emptyArrayCache.keys():
            return self._emptyArrayCache[type]
        else:
            if type != "CharType":
                self._emptyArrayCache[type] = arr = numarray.array(shape=0, type=type)
            else:
                self._emptyArrayCache[type] = arr = strings.array([], itemsize=isize)
            return arr

    # List here the lazy attributes.
    def _g_getrbuffer(self):
        mydict = self.__dict__
        if '_v_rbuffer' in mydict:
            return mydict['_v_rbuffer']
        else:
            mydict['_v_rbuffer'] = rbuffer = self._newBuffer(init=0)
            return rbuffer

    _v_rbuffer = property(_g_getrbuffer, None, None,
                          "A buffer for reading.")

    def _g_getwbuffer(self):
        mydict = self.__dict__
        if '_v_wbuffer' in mydict:
            return mydict['_v_wbuffer']
        else:
            mydict['_v_wbuffer'] = wbuffer = self._newBuffer(init=1)
            mydict['_v_wbuffercpy'] = wbuffer.copy()
            return wbuffer

    _v_wbuffer = property(_g_getwbuffer, None, None,
                          "*The* buffer for writing.")

    # </properties>


    def __init__(self, parentNode, name,
                 description=None, title="", filters=None,
                 expectedrows=EXPECTED_ROWS_TABLE,
                 _log=True):
        """Create an instance Table.

        Keyword arguments:

        description -- A IsDescription subclass or a dictionary where
            the keys are the field names, and the values the type
            definitions. And it can be also a RecArray, NestedRecArray
            or heterogenous numpy object. If None, the table
            metadata is read from disk, else, it's taken from previous
            parameters.

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

        self._v_new = new = description is not None
        """Is this the first time the node has been created?"""
        self._v_new_title = title
        """New title for this node."""
        self._v_new_filters = filters
        """New filter properties for this node."""

        self._v_recarray = None
        """A record array to be stored in the table."""
        self._v_expectedrows = expectedrows
        """The expected number of rows to be stored in the table."""
        self.nrows = 0
        """Current number of rows in the table."""
        self._unsaved_nrows = 0
        """Number of rows in buffers but still not in disk."""
        self.description = None
        """A `Description` instance describing the structure of the table."""
        self._time64colnames = []
        """The names of ``Time64`` columns."""
        self._strcolnames = []
        """The names of ``CharType`` columns."""
        self._colenums = {}
        """Maps the name of an enumerated column to its ``Enum`` instance."""
        self._v_maxTuples = None
        """The number of rows that fit in the table buffer."""
        self._v_chunksize = None
        """The HDF5 chunk size."""

        self.indexed = False
        """Does this table have any indexed columns?"""
        self.indexprops = None
        """
        Index properties for this table (an `IndexProps` instance).
        ``None`` if the table is not indexed.
        """
        self._indexedrows = 0
        """Number of rows indexed in disk."""
        self._unsaved_indexedrows = 0
        """Number of rows indexed in memory but still not in disk."""

        self.colnames = ()
        """
        A tuple containing the (possibly nested) names of the columns in
        the table.
        """
        self.coltypes = {}
        """Maps the name of a column to its data type."""
        self.colstypes = {}
        """Maps the name of a column to its data string type."""
        self.colshapes = {}
        """Maps the name of a column to it shape."""
        self.colitemsizes = {}
        """Maps the name of a column to the size of its base items."""
        self.coldflts = {}
        """Maps the name of a column to its default value."""
        self.colindexed = {}
        """Is the column which name is used as a key indexed? (dictionary)"""

        # Initialize the possible cuts in columns.
        self.ops = []
        """
        Current row selection operators (1=lt, 2=le, 3=gt, 4=ge, 5=eq,
        6=ne, 10=and, 11=or, 12=xor).
        """
        self.opsValues = []
        """Right-side operands (constant values) for the row selection."""
        self.opsColnames = []
        """Left-side operands (column names) for the row selection."""
        self.whereColname = None
        """The name of the column where the selection condition is applied."""

        self.whereCondition = None  ##XXX
        """Condition string and variable map for selection of values."""
        self.whereIndex = None  ##XXX
        """Path of the indexed column to be used in an indexed search."""
        self._splittedCondCache = {}  ##XXX
        """Cache of already splitted conditions."""
        # It is manipulated by ``split_index_condXXX()``.
        self._emptyArrayCache = {}  ##XXX
        """Cache of empty arrays."""

        self.cols = None
        """
        A `Cols` instance that serves as an accessor to `Column` objects.
        """

        # Private variable to keep the value of flavor until it the table
        # would be created.
        self._flavor = None

        # Initialize this object in case is a new Table
        if isinstance(description, dict):
            # Dictionary case
            self.description = Description(description)
        elif isinstance(description, records.RecArray):
            # RecArray object case
            self._flavor = "numarray"
            self._newRecArray(description)
            # Provide a better guess for the expected number of rows
            # But beware with the small recarray lengths!
            # Commented out until a better approach is found
            #if self._v_expectedrows == expectedrows:
            #    self._v_expectedrows = self.nrows
        elif numpy_imported and isinstance(description, numpy.ndarray):
            # NumPy object case
            self._flavor = "numpy"
            nra = fromnumpy(description)
            self._newRecArray(nra)
        elif (type(description) == type(IsDescription) and
              issubclass(description, IsDescription)):
            # IsDescription subclass case
            descr = description()
            self.description = Description(descr.columns)
        elif isinstance(description, Description):
            # It is a Description instance already
            self.description = description
        elif description is not None:
            raise TypeError(
                "the ``description`` argument is not of a supported type: "
                "``IsDescription`` subclass, ``Description`` instance, "
                "dictionary, ``RecArray`` or ``NestedRecArray`` instance""")

        super(Table, self).__init__(parentNode, name, new, filters, _log)


    def _g_postInitHook(self):
        # We are putting here the index-related issues
        # as well as filling general info for table
        # This is needed because we need first the index objects created

        # Create a cols accessor.
        self.cols = Cols(self, self.description)

        if self._v_new:
            return
        # The following code is only for opened tables.

        # Do the indexes group exist?
        indexesGroupPath = _getIndexTableName(self._v_parent, self._v_name)
        igroup = indexesGroupPath in self._v_file
        for colobj in self.description._v_walk(type="Col"):
            colname = colobj._v_pathname
            # Is this column indexed?
            if igroup:
                indexname = _getIndexColName(
                    self._v_parent, self._v_name, colname)
                indexed = indexname in self._v_file
                self.colindexed[colname] = indexed
                if indexed:
                    indexobj = self.cols._f_col(colname).index
            else:
                indexed = False
                self.colindexed[colname] = False
            if indexed:
                self.indexed = True

        # Create an index properties object.
        # It does not matter to which column 'indexobj' belongs,
        # since their respective index objects share
        # the same filters and number of elements.
        autoindex = getattr(self.attrs, 'AUTOMATIC_INDEX', None)
        reindex = getattr(self.attrs, 'REINDEX', None)
        if self.indexed:
            filters = indexobj.filters
        else:
            filters = getattr(self.attrs, 'FILTERS_INDEX', None)
        self.indexprops = IndexProps(auto=autoindex, reindex=reindex,
                                     filters=filters)
        if self.indexed:
            self._indexedrows = indexobj.nelements
            self._unsaved_indexedrows = self.nrows - self._indexedrows

    def _get_container(self, shape):
        "Get the appropriate buffer for data depending on table nestedness."

        # The handling of a plain RecArray is usually faster than
        # NestedRecArray. This can be *critical* to certain benchmarks,
        # specially those whose run times are less than 1 msec.
        # So, from now on, the reading methods in Table will return a
        # plain RecArray when the table doesn't have nested columns
        # and a NestedRecArray when the table is nested.
        # F. Altet 2006-06-10
        colnames = self.description._v_nestedNames
        formats = self.description._v_nestedFormats
        if self.description._v_is_nested:
            recarr = nestedrecords.array(None, formats=formats, shape=shape,
                                         names = colnames)
        else:
            recarr = records.array(None, formats=formats, shape=shape,
                                   names = colnames)
        return recarr

    def _newBuffer(self, init=1):
        """Create a new recarray buffer for I/O purposes"""

        # The next optimization violates the rule that an iterator has to have
        # its own buffer. For achieving maximum speed in selects, a call to
        # Table.getWhereList followed by a Table.readCoordinates is the best.
#         if init == 0:
#             mydict = self.__dict__
#             if '_v_rbuffer' in mydict:
#                 recarr = mydict['_v_rbuffer']
#                 #recarr._fields = recarr._get_fields()
#                 return recarr
#             else:
#                 recarr = self._get_container(self._v_maxTuples)
#                 recarr._fields = recarr._get_fields()
#                 mydict['_v_rbuffer'] = recarr
#                 return mydict['_v_rbuffer']
#         else:
#             recarr = self._get_container(self._v_maxTuples)
        recarr = self._get_container(self._v_maxTuples)
        # Initialize the recarray with the defaults in description
        recarr._fields = recarr._get_fields()
        if init:
            for objcol in self.description._v_walk("Col"):
                colname = objcol._v_pathname
                recarr._fields[colname][:] =  objcol.dflt
        return recarr

    def _descrFromNRA(self, nra):
        "Get a description dictionary from a NestedRecArray"

        fields = {}
        i = 0
        for (colname, format) in nra.descr:
            if isinstance(format, str):
                # Column case
                shape, type = prog.search(format).groups()
                if shape == "":
                    shape = 1   # No shape. Put it to 1
                elif len(shape) == 1:
                    shape = int(shape)
                else:  # '(n, m, ...)'
                    # The next is safer and faster than eval(shape)
                    shape = tuple([ int(c) for c in shape[1:-1].split(',')
                                    if c.strip().isdigit() ])
                if type[0] == "a":
                    itemsize = int(type[1:])
                    type = "a"
                else:
                    type = numarray.typeDict[type]
                    itemsize = type.bytes
                # Special case for strings
                if type == 'a':
                    fields[colname] =  StringCol(length=itemsize,
                                                 dflt=None,
                                                 shape=shape,
                                                 pos=i)
                else:
                    fields[colname] = Col(dtype=type,
                                          shape=shape,
                                          pos=i)  # Position matters
            else:
                # Nested column
                fields[colname] = self._descrFromNRA(nra.field(colname))
                fields[colname]["_v_pos"] = i
            i += 1
        return fields

    def _descrFromRA(self, recarr):
        "Get a description dictionary from a RecArray"

        fields = {}
        for i in xrange(len(recarr._names)):
            colname = recarr._names[i]
            # Special case for strings
            if isinstance(recarr._fmt[i], records.Char):
                fields[colname] =  StringCol(length=recarr._itemsizes[i],
                                             dflt=None,
                                             shape=recarr._repeats[i],
                                             pos=i)
            else:
                fields[colname] = Col(dtype=recarr._fmt[i],
                                      shape=recarr._repeats[i],
                                      pos=i)  # Position matters
        return fields

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
        if hasattr(recarr, "descr"):  # Quacks like a NestedRecArray
            fields = self._descrFromNRA(recarr)
        else:
            fields = self._descrFromRA(recarr)
        # Set the byteorder
        fields['_v_byteorder'] = recarr._byteorder
        # Create an instance description to host the record fields
        self.description = Description(fields)
        # The rest of the info is automatically added when self.create()
        # is called

    def _getTypeColNames(self, stype):
        """Returns a list containing 'stype' column names."""

        return [ colobj._v_pathname
                 for colobj in self.description._v_walk('Col')
                 if colobj.stype == stype ]

    def _getEnumMap(self):
        """Return mapping from enumerated column names to `Enum` instances."""

        enumMap = {}
        for colobj in self.description._v_walk('Col'):
            if colobj.stype == 'Enum':
                enumMap[colobj._v_pathname] = colobj.enum
        return enumMap

    def _createIndexesTable(self, igroup):
        itgroup = IndexesTableG(
            igroup, _getEncodedTableName(self._v_name),
            "Indexes container for table "+self._v_pathname, new=True)
        # Assign the pathname table to this Group
        itgroup._v_attrs._g__setattr('PATHNAME', self._v_pathname)
        # Delete the FILTERS_INDEX attribute from table, so that
        # we don't have to syncronize it
        self._v_attrs._g__delattr('FILTERS_INDEX')
        return itgroup


    def _createIndexesDescr(self, igroup, dname, iname, filters):
        idgroup = IndexesDescG(
            igroup, iname,
            "Indexes container for sub-description "+dname,
            filters=filters, new=True)
        # Assign the pathname table to this Group
        pathname = "%s.cols.%s" % (self._v_pathname, dname)
        idgroup._v_attrs._g__setattr('PATHNAME', pathname)
        return idgroup


    def _g_create(self):
        """Create a new table on disk."""

        # Protection against too large row sizes
        # Set to a 512 KB limit (just because banana 640 KB limitation)
        # Protection removed. CSTables should be improved to deal with
        # this kind of situations. F. Altet 2006-01-11
#         if self.rowsize > 512*1024:
#             raise ValueError, \
# """Row size too large. Maximum size is 512 Kbytes, and you are asking
# for a row size of %s bytes.""" % (self.rowsize)

        # Find Time64 column names. (This should be generalised.)
        self._time64colnames = self._getTypeColNames('Time64')
        # Find CharType column names.
        self._strcolnames = self._getTypeColNames('CharType')
        # Get a mapping of enumerated columns to their `Enum` instances.
        self._colenums = self._getEnumMap()

        # Compute some values for buffering and I/O parameters
        (self._v_maxTuples, self._v_chunksize) = \
                            calcBufferSize(self.rowsize, self._v_expectedrows)


        # Create the table on disk
        # self._v_objectID needs to be assigned here because is needed for
        # setting attributes afterwards
        self._v_objectID = self._createTable(
            self._v_new_title, self.filters.complib, obversion)
        # self._v_recarray is not useful anymore. Get rid of it.
        self._v_recarray = None

        # Warning against assigning too much columns...
        # F. Altet 2005-06-05
        self.colnames = tuple(self.description._v_nestedNames)
        if (len(self.colnames) > MAX_COLUMNS):
            warnings.warn("""\
table ``%s`` is exceeding the recommended maximum number of columns (%d); \
be ready to see PyTables asking for *lots* of memory and possibly slow I/O"""
                          % (self._v_pathname, MAX_COLUMNS),
                          PerformanceWarning)

        # Compute some important parameters for createTable
        for colobj in self.description._v_walk(type="Col"):
            colname = colobj._v_pathname
            # Get the column types and string types
            self.coltypes[colname] = colobj.type
            self.colstypes[colname] = colobj.stype
            # Extract the shapes and itemsizes for columns
            self.colshapes[colname] = colobj.shape
            self.colitemsizes[colname] = colobj.itemsize
            self.coldflts[colname] = colobj.dflt
            # Indexed?
            colindexed = colobj.indexed
            self.colindexed[colname] = colindexed
            if colindexed:
                self.indexed = True

        setAttr = self._v_attrs._g__setattr
        # Assign the value of FLAVOR
        if self._flavor is None:
            self._flavor = getattr(self.description, '_v_flavor', "numarray")
        setAttr('FLAVOR', self._flavor)

        # We have to define indexprops here in order to propagate
        # index properties for eventual future index creation
        self.indexprops = getattr(self.description, '_v_indexprops',
                                  IndexProps())
        # Save AUTOMATIC_INDEX and REINDEX flags as attributes
        setAttr('AUTOMATIC_INDEX', self.indexprops.auto)
        setAttr('REINDEX', self.indexprops.reindex)
        # Filters is saved here until one index is created.
        setAttr('FILTERS_INDEX', self.indexprops.filters)
        if self.indexed:
            self._indexedrows = 0
            self._unsaved_indexedrows = 0

        # Attach the FIELD_N_FILL attributes. We write all the level defaults
        i = 0
        for colobj in self.description._v_walk(type="Col"):
            fieldname = "FIELD_%s_FILL" % i
            setAttr(fieldname, colobj.dflt)
            i += 1

        # Finally, return the object identifier.
        return self._v_objectID


    def _g_open(self):
        """Opens a table from disk and read the metadata on it.

        Creates an user description on the flight to easy the access to
        the actual data.

        """
        # Get table info
        self._v_objectID, description = self._getInfo()
        if self._v_file._isPTFile:
            # Checking validity names for fields is not necessary
            # when opening a PyTables file
            # Do this nested!
            description['__check_validity__'] = 0

        # Create an instance description to host the record fields
        self.description = Description(description)
        getAttr = self._v_attrs.__getattr__
        # Check if there is some "FIELD_0_FILL" attribute
        has_fill_attrs = "FIELD_0_FILL" in self._v_attrs._f_list("sys")
        i = 0
        for objcol in self.description._v_walk(type="Col"):
            colname = objcol._v_pathname
            if has_fill_attrs:
                # Get the default values for each column
                fieldname = "FIELD_%s_FILL" % i
                defval = getAttr(fieldname)
                objcol.dflt = defval
                # Set also the correct value in the desc._v_dflts dictionary
                self.description._v_dflts[colname] = defval
                i += 1
            # Add info for indexed columns
            indexname = _getIndexColName(self._v_parent, self._v_name, colname)
            colindexed = indexname in self._v_file
            objcol.indexed = colindexed

        # The expectedrows would be the actual number
        self._v_expectedrows = self.nrows

        # Extract the coltypes, colstypes, shapes and itemsizes
        # self.colnames, coltypes, col*... should be removed?
        self.colnames = tuple(self.description._v_nestedNames)

        # Find Time64 column names.
        self._time64colnames = self._getTypeColNames('Time64')
        # Find CharType column names.
        self._strcolnames = self._getTypeColNames('CharType')
        # Get a mapping of enumerated columns to their `Enum` instances.
        self._colenums = self._getEnumMap()

        # Compute buffer size
        (self._v_maxTuples, self._v_chunksize) = \
              calcBufferSize(self.rowsize, self.nrows)

        # Get info about columns
        for colobj in self.description._v_walk(type="Col"):
            colname = colobj._v_pathname
            # Get the column types and string types
            self.coltypes[colname] = colobj.type
            self.colstypes[colname] = colobj.stype
            # Extract the shapes and itemsizes for columns
            self.colshapes[colname] = colobj.shape
            self.colitemsizes[colname] = colobj.itemsize
            self.coldflts[colname] = colobj.dflt
        return self._v_objectID

    def _checkColumn(self, colname):
        """
        Check that the column named `colname` exists in the table.

        If it does not exist, a ``KeyError`` is raised.
        """

        for colobj in self.description._v_walk(type="All"):
            cname = colobj._v_pathname
            if colname == cname:
                break
        if colname <> cname:
            raise KeyError(
                "table ``%s`` does not have a column named ``%s``"
                % (self._v_pathname, colname))
        return colobj

    def where2XXX( self, condition, condvars,
                   start=None, stop=None, step=None ):
        return self._whereInRange2XXX(condition, condvars, start, stop, step)

    def _whereInRange2XXX( self, condition, condvars,
                           start=None, stop=None, step=None ):
        self.whereCondition = (condition, condvars)
        (start, stop, step) = processRangeRead(self.nrows, start, stop, step)
        if start < stop:
            row = TableExtension.Row(self)
            return row(start, stop, step, coords=None, ncoords=-1)
        return iter([])

    def _getWhereListXXX(self, condition, condvars):

        coords = [p.nrow for p in self._whereInRange2XXX(condition, condvars)]
        coords = numarray.array(coords, type=numarray.Int64)
        return coords

    def where(self, condition, start=None, stop=None, step=None):
        """
        Iterate over values fulfilling a `condition`.

        This method returns an iterator yielding `Row` instances built
        from rows in the table that satisfy the given `condition` over a
        column.  If that column is indexed, its index will be used in
        order to accelerate the search.  Else, the *in-kernel* iterator
        (with has still better performance than standard Python
        selections) will be chosen instead.

        Moreover, if a range is supplied (i.e. some of the `start`,
        `stop` or `step` parameters are passed), only the rows in that
        range *and* fullfilling the `condition` are returned.  The
        meaning of the `start`, `stop` and `step` parameters is the same
        as in the ``range()`` Python function, except that negative
        values of `step` are *not* allowed.  Moreover, if only `start`
        is specified, then `stop` will be set to ``start+1``.

        You can mix this method with standard Python selections in order
        to have complex queries.  It is strongly recommended that you
        pass the most restrictive condition as the parameter to this
        method if you want to achieve maximum performance.

        Example of use::

            passvalues=[]
            for row in table.where(0 < table.cols.col1 < 0.3, step=5):
                if row['col2'] <= 20:
                    passvalues.append(row['col3'])
            print "Values that pass the cuts:", passvalues

        """

        if not isinstance(condition, Column):
            raise TypeError("""\
Wrong 'condition' parameter type. Only Column instances are suported.""")

        if not condition.shape in [1, (1,)]:
            raise NotImplementedError, "You cannot use in-kernel or indexed searches along multimensional columns. Use the regular table iterator for that."

        colindex = condition.index
        if (colindex and not condition.dirty and colindex.nelements > 0):
            # Call the indexed version method
            return self._whereIndexed(condition, start, stop, step)
        # Fall back to in-kernel selection method
        return self._whereInRange(condition, start, stop, step)

    def _whereInRange(self, condition, start=None, stop=None, step=None):
        """
        Iterate over values fulfilling a `condition` avoiding indexes.

        This method is completely equivalent to `where()`, but it avoids
        the usage of the indexing capabilites on the column affected by
        the `condition` (whether indexed or not).

        This method is mainly for avoiding some pathological corner
        cases where automatically using indexation yields a poorer
        performance.  In that case, please contact the developers and
        explain your case.
        """

        if not isinstance(condition, Column):
            raise TypeError("""\
Wrong 'condition' parameter type. Only Column instances are suported.""")

        self.whereColname = condition.pathname   # Flag for Row.__iter__
        (start, stop, step) = processRangeRead(self.nrows, start, stop, step)
        if start < stop:
            # call row with coords=None and ncoords=-1 (in-kernel selection)
            row = TableExtension.Row(self)
            return row(start, stop, step, coords=None, ncoords=-1)

        # Fall-back action is to return an empty RecArray
        return iter([])

    def _whereIndexed2XXX( self, condition, condvars,
                           start=None, stop=None, step=None ):
        """
        Iterate over values fulfilling a `condition` using indexes.

        This method is completely equivalent to `where()`, but it forces
        the usage of the indexing capabilities on the column affected by
        the `condition`.
        """

        idxvar, ops, lims, rescond = \
                split_index_condXXX(condition, condvars, self)
        if not idxvar:
            raise ValueError( "could not find any usable indexes "
                              "for condition: %r" % condition )

        column = condvars[idxvar]
        index = column.index
        assert index is not None, "the chosen column is not indexed"
        assert not column.dirty, "the chosen column has a dirty index"
        assert index.is_pro or index.nelements > 0, \
               "the chosen column has too few elements to be indexed"

        # Set the index column and residual condition (if any)
        self.whereIndex = column.pathname
        if rescond:
            self.whereCondition = (rescond, condvars)
        # Get the coordinates to lookup
        ncoords = index.getLookupRange2XXX(ops, lims, self)
        if index.is_pro and ncoords == 0:
            # For the pro case, there are no interesting values
            # Reset the table variable conditions
            self.whereIndex = None
            self.whereCondition = None
            # Return the empty iterator
            return iter([])
        # Call the iterator even in case that there are no values satisfying
        # the conditions in the indexed region (ncoords = 0), because
        # we should look in the non-indexed region as well (for PyTables std).
        (start, stop, step) = processRangeRead(self.nrows, start, stop, step)
        row = TableExtension.Row(self)
        # Call the indexed version of Row iterator (coords=None,ncoords>=0)
        return row(start, stop, step, coords=None, ncoords=ncoords)

    def _whereIndexed(self, condition, start=None, stop=None, step=None):
        """
        Iterate over values fulfilling a `condition` using indexes.

        This method is completely equivalent to `where()`, but it forces
        the usage of the indexing capabilities on the column affected by
        the `condition`.  If the column is not indexed, a ``ValueError``
        is raised.
        """

        if not isinstance(condition, Column):
            raise TypeError("""\
Wrong 'condition' parameter type. Only Column instances are suported.""")
        index = condition.index
        if index is None:
            raise ValueError("""\
This method is intended only for indexed columns.""")
        if condition.dirty:
            raise ValueError("""\
This method is intended only for indexed columns, but this column has a dirty index. Try re-indexing it in order to put the index in a sane state.""")
        if index.nelements == 0:
            raise ValueError("""\
This method is intended only for indexed columns, but this column has not a minimum entries (%s) to be indexed.""" % condition.index.slicesize)

        self.whereColname = condition.pathname   # Flag for Row.__iter__
        # Get the coordinates to lookup
        ncoords = index.getLookupRange(condition)
        if index.is_pro and ncoords == 0:
            # For the pro case, there are no interesting values
            # Reset the table variable conditions
            self.ops = []
            self.opsValues = []
            self.whereColname = None
            # Return the empty iterator
            return iter([])
        # Call the iterator even in case that there are no values satisfying
        # the conditions in the indexed region (ncoords = 0), because
        # we should look in the non-indexed region as well (for PyTables std).
        (start, stop, step) = processRangeRead(self.nrows, start, stop, step)
        row = TableExtension.Row(self)
        # Call the indexed version of Row iterator (coords=None,ncoords>=0)
        return row(start, stop, step, coords=None, ncoords=ncoords)

    def readIndexed2XXX(self, condition, condvars):
        idxvar, ops, lims, rescond = split_index_condXXX(condition, condvars, self)
        if not idxvar:
            raise ValueError( "could not find any usable indexes "
                              "for condition: %r" % condition )

        column = condvars[idxvar]
        index = column.index
        assert index is not None, "the chosen column is not indexed"
        assert not column.dirty, "the chosen column has a dirty index"
        assert index.is_pro or index.nelements > 0, \
               "the chosen column has too few elements to be indexed"

        # Set the index column and residual condition (if any)
        self.whereIndex = column.pathname
        if rescond:
            self.whereCondition = (rescond, condvars)
        # Get the coordinates to lookup
        nrecords = index.getLookupRange2XXX(ops, lims, self)
        # Create a read buffer
        recarr = self._get_container(nrecords)
        if nrecords > 0:
            #coords = index.getCoords(0, nrecords)
            # The line below is the optimized call in pyrex
            coords = index.indices._getCoords(index, 0, nrecords)
            recout = self._read_elements(recarr, coords)
        # Delete indexation caches
        self.whereCondition = None
        self.whereIndex = None
        if numpy_imported and self.flavor == "numpy":
            # do an additional conversion conversion (without a copy)
            recarr = tonumpy(recarr, copy=False)
        return recarr

    def readIndexed(self, condition):
        """Return a record array fulfilling the given `condition`.

        The `condition` can be used to specify selections along a column in
        the form::

          condition = (0 < table.cols.col1 < 0.3)

        This method is only intended to be used for indexed columns.
        """

        if not isinstance(condition, Column):
            raise TypeError(
                "``condition`` argument is not an instance of ``Column``")
        index = condition.index
        if index is None:
            raise ValueError(
                "the column referenced by ``condition`` is not indexed")
        if condition.dirty:
            raise ValueError("""\
the column referenced by ``condition`` has a dirty index; \
please reindex the table to put the index in a sane state""")

        self.whereColname = condition.pathname   # Flag for Row.__iter__
        # Get the coordinates to lookup
        nrecords = index.getLookupRange(condition)
        # Create a read buffer
        recarr = self._get_container(nrecords)
        if nrecords > 0:
            #coords = index.getCoords(0, nrecords)
            # The line below is the optimized call in pyrex
            coords = index.indices._getCoords(index, 0, nrecords)
            recout = self._read_elements(recarr, coords)
        # Delete indexation caches
        self.ops = []
        self.opsValues = []
        self.opsColnames = []
        self.whereColname = None
        if numpy_imported and self.flavor == "numpy":
            # do an additional conversion conversion (without a copy)
            recarr = tonumpy(recarr, copy=False)
        return recarr

    def whereAppend(self, dstTable, condition, start=None, stop=None, step=None):
        """
        Append rows fulfulling the `condition` to the `dstTable` table.

        `dstTable` must be capable of taking the rows resulting from the
        query, i.e. it must have columns with the expected names and
        compatible types.  The meaning of the other arguments is the
        same as in the `where()` method.

        The number of rows appended to `dstTable` is returned as a
        result.
        """

        # Check that the destination file is not in read-only mode.
        dstTable._v_file._checkWritable()

        # Row objects do not support nested columns, so we must iterate
        # over the flat column paths.  When rows support nesting,
        # ``self.colnames`` can be directly iterated upon.
        colNames = [colName for colName in flattenNames(self.colnames)]
        dstRow = dstTable.row
        nrows = 0
        for srcRow in self.where(condition, start, stop, step):
            for colName in colNames:
                dstRow[colName] = srcRow[colName]
            dstRow.append()
            nrows += 1
        dstTable.flush()
        return nrows


    def getWhereList2XXX(self, condition, condvars, flavor=None, sort=False):
        if not flavor:
            flavor = self.flavor
        if flavor not in supportedFlavors:
            raise ValueError("""\
"%s" flavor is not allowed; please use some of %s.""" % \
                             (flavor, supportedFlavors))

        idxvar, ops, lims, rescond = split_index_condXXX(condition, condvars, self)

        # Take advantage of indexation, if present
        if idxvar is not None:

            column = condvars[idxvar]
            index = column.index
            assert index is not None, "the chosen column is not indexed"
            assert not column.dirty, "the chosen column has a dirty index"
            assert index.is_pro or index.nelements > 0, \
                   "the chosen column has too few elements to be indexed"

            # get the number of coords and set-up internal variables
            ncoords = index.getLookupRange2XXX(ops, lims, self)
            if ncoords > 0:
                coords = index.indices._getCoords_sparse(index, ncoords)
            else:
                #coords = numarray.array(None, type=numarray.Int64, shape=0)
                coords = self._g_getemptyarray("Int64")
            if not index.is_pro:
                # get the remaining rows from the table
                start = index.nelements
                if start < self.nrows:
                    remainCoords = [p.nrow for p in self._whereInRange2XXX(
                        condition, condvars, start, self.nrows)]
                    nremain = len(remainCoords)
                    # append the new values to the existing ones
                    coords.resize(ncoords+nremain)
                    coords[ncoords:] = remainCoords
        else:
            coords = [p.nrow for p in self._whereInRange2XXX(condition, condvars)]
            coords = numarray.array(coords, type=numarray.Int64)
        # re-initialize internal selection values
        self.whereCondition = None
        self.whereIndex = None
        if sort:
            coords = numarray.sort(coords)
        if flavor == "numarray":
            return coords
        if numpy_imported and flavor == "numpy":
            coords = numpy.asarray(coords)
        elif Numeric_imported and flavor == "numeric":
            coords = numeric.asarray(coords)
        elif flavor == "python":
            coords = coords.tolist()
        return coords

    def getWhereList(self, condition, flavor=None, sort=False):
        """Get the row coordinates that fulfill the `condition` param

        `condition` can be used to specify selections along a column
        in the form:

        condition=(0<table.cols.col1<0.3)

        `flavor` is the desired type of the returned list. It can take
        the 'numarray', 'numpy', 'numeric' or 'python' values.  If
        `flavor` is not provided, then it will take the value of
        self.flavor.

        `sort` means that you want to retrieve the coordinates ordered.  The
        default is to not sort them.

        `sort` means that you want to retrieve the coordinates ordered.  The
        default is to not sort them.

        """

        assert isinstance(condition, Column), """\
Wrong 'condition' parameter type. Only Column instances are suported."""

        if not flavor:
            flavor = self.flavor
        if flavor not in supportedFlavors:
            raise ValueError("""\
"%s" flavor is not allowed; please use some of %s.""" % \
                             (flavor, supportedFlavors))

        index = condition.index
        # Take advantage of indexation, if present
        if index is not None:
            # get the number of coords and set-up internal variables
            ncoords = index.getLookupRange(condition)
            if ncoords > 0:
                #coords = index.getCoords_sparse(ncoords)
                # The next call is the optimized one
                coords = index.indices._getCoords_sparse(index, ncoords)
            else:
                coords = numarray.array(None, type=numarray.Int64, shape=0)
            if not index.is_pro:
                # get the remaining rows from the table
                start = index.nelements
                if start < self.nrows:
                    remainCoords = [p.nrow for p in self._whereInRange(
                        condition, start, self.nrows)]
                    nremain = len(remainCoords)
                    # append the new values to the existing ones
                    coords.resize(ncoords+nremain)
                    coords[ncoords:] = remainCoords
        else:
            coords = [p.nrow for p in self.where(condition)]
            coords = numarray.array(coords, type=numarray.Int64)
        # re-initialize internal selection values
        self.ops = []
        self.opsValues = []
        self.opsColnames = []
        self.whereColname = None
        if sort:
            coords = numarray.sort(coords)
        if flavor == "numarray":
            return coords
        if numpy_imported and flavor == "numpy":
            coords = numpy.asarray(coords)
        elif Numeric_imported and flavor == "numeric":
            coords = numeric.asarray(coords)
        elif flavor == "python":
            coords = coords.tolist()
        return coords

    def itersequence(self, sequence, sort=False):
        """Iterate over a list of row coordinates.

        `sort` means that sequence will be sorted so that I/O *might* perform
        better. If your sequence is already sorted or you don't want to sort
        it, put this parameter to 0. The default is to do not sort the
        sequence.

        """

        if not hasattr(sequence, '__getitem__'):
            raise TypeError("""\
Wrong 'sequence' parameter type. Only sequences are suported.""")

        coords = numarray.array(sequence, type=numarray.Int64)
        # That would allow the retrieving on a sequential order
        # although this is not totally clear.
        if sort:
            coords.sort()
        row = TableExtension.Row(self)
        return row(coords=coords, ncoords=-1)

    def iterrows(self, start=None, stop=None, step=None):
        """Iterate over all the rows or a range.

        Specifying a negative value of step is not supported yet.

        """
        (start, stop, step) = processRangeRead(self.nrows, start, stop, step)
        if start < stop:
            row = TableExtension.Row(self)
            return row(start, stop, step, coords=None, ncoords=-1)
        # Fall-back action is to return an empty iterator
        return iter([])

    def __iter__(self):
        """Iterate over all the rows."""

        return self.iterrows()

    def _read(self, start, stop, step, field=None, coords=None):
        """Read a range of rows and return an in-memory object.
        """

        select_field = None
        if field:
            if field not in self.coltypes:
                if field in self.description._v_names:
                    # Remember to select this field
                    select_field = field
                    field = None
                else:
                    raise KeyError, "Field %s not found in table %s" % \
                          (field, self)
            else:
                # The column hangs directly from the top
                typeField = self.coltypes[field]

        # Return a rank-0 array if start > stop
        if start >= stop:
            if field == None:
                nra = self._get_container(0)
                return nra
            elif isinstance(typeField, records.Char):
                return numarray.strings.array(shape=(0,), itemsize = 0,
                                              padc='\x00')
            else:
                return numarray.array(shape=(0,), type=typeField)

        if coords is None:
            # (stop-start)//step  is not enough
            nrows = ((stop - start - 1) // step) + 1
        else:
            assert isinstance(coords, numarray.NumArray)
            assert coords.type() == numarray.Int64
            # I should test for stop and start values as well
            nrows = len(coords)

        # Compute the shape of the resulting column object
        if field and coords is None:  # coords handling expects a RecArray
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
                result = numarray.strings.array(shape=shape, itemsize=itemsize,
                                                padc='\x00')
            else:
                # Non-string column case
                result = numarray.array(shape=shape, type=typeField)
        else:
            # Recarray case
            result = self._get_container(nrows)

        # Handle coordinates separately.
        if coords is not None:
            if len(coords) > 0:
                self._read_elements(result, coords)
            if field:
                # result is always a RecArray
                return result.field(field)
            return result

        # Call the routine to fill-up the resulting array
        if step == 1 and not field:
            # This optimization works three times faster than
            # the row._fillCol method (up to 170 MB/s on a pentium IV @ 2GHz)
            self._read_records(start, stop-start, result)
        # Warning!: _read_field_name should not be used until
        # H5TBread_fields_name in TableExtension will be finished
        # F. Altet 2005/05/26
        elif field and step > 15 and 0:
            # For step>15, this seems to work always faster than row._fillCol.
            self._read_field_name(result, start, stop, step, field)
        else:
            self.row._fillCol(result, start, stop, step, field)

        if select_field:
            return result.field(select_field)
        else:
            return result

    def read(self, start=None, stop=None, step=None,
             field=None, flavor=None, coords = None):
        """Read a range of rows and return an in-memory object.

        If `start`, `stop`, or `step` parameters are supplied, a row
        range is selected. If `field` is specified, only this `field` is
        returned as a NumArray object. If `field` is not supplied all
        the fields are selected and a NestedRecArray (or equivalent
        numpy object, see description for `flavor` later on) is
        returned.  If `flavor` is provided, an additional conversion to
        an object of this flavor is made. `flavor` must have any of the
        next values: "numarray", "numpy", "python" or "numeric" (only if
        `field` has been specified). If `flavor` is not provided, then
        it will take the value of self.flavor.

        Nested fields can be specified in the `field` parameter by
        using a '/' as a separator between fields (e.g. 'Info/value').

        If `coords` is specified, only the indices in `coords` that
        are in the range of (`start`, `stop`) are returned. Also, if
        `coords` is specified, step only can be assigned to be 1,
        otherwise an error is issued.

        """

        if field:
            self._checkColumn(field)

        if not flavor:
            flavor = self.flavor
        if flavor not in supportedFlavors:
            raise ValueError("""\
"%s" flavor is not allowed; please use some of %s.""" % \
                             (flavor, supportedFlavors))
        if Numeric_imported and flavor == "numeric" and field is None:
            raise ValueError, \
"""Numeric does not support heterogeneous datasets. You cannot specify a 'numeric' flavor without specifying a field."""

        (start, stop, step) = processRangeRead(self.nrows, start, stop, step)

        if coords is not None:
            # Check step value.
            if len(coords) and step != 1:
                raise NotImplementedError("""\
``step`` must be 1 when the ``coords`` parameter is specified""")
            # Turn coords into an array of 64-bit indexes,
            # as expected by _read().
            if not (isinstance(coords, numarray.NumArray)
                    and coords.type() != numarray.Int64):
                coords = numarray.array(coords, type=numarray.Int64)

        arr = self._read(start, stop, step, field, coords)
        # Convert to Numeric, tuple or list if needed
        if field:
            # homogeneous conversion
            if numpy_imported and flavor == "numpy":
                arr = convertNAToNumPy(arr)
            elif Numeric_imported and flavor == "numeric":
                arr = convertNAToNumeric(arr)
        elif numpy_imported and flavor == "numpy":
            # heterogeneous conversion (without a copy)
            arr = tonumpy(arr, copy=False)
        elif flavor == "python":
            arr = self.tolist(arr)

        return arr

    def readCoordinates(self, coords, field=None, flavor=None):
        """
        Read a set of rows given their indexes into an in-memory object.

        This method works much like the `read()` method, but it uses a
        sequence (`coords`) of row indexes to select the wanted columns,
        instead of a column range.

        It returns the selected rows in a ``NestedRecArray`` object.  If
        `flavor` is provided, an additional conversion to an object of
        this flavor is made, just as in `read()`. If not specified, the
        default flavor for this table will be chosen as the output
        format. """

        if not flavor:
            flavor = self.flavor
        if flavor not in supportedFlavors:
            raise ValueError("""\
"%s" flavor is not allowed; please use some of %s.""" % \
                             (flavor, supportedFlavors))
        if Numeric_imported and flavor == "numeric" and field is None:
            raise ValueError, \
"""Numeric does not support heterogeneous datasets yet. You cannot specify a 'numeric' flavor without specifying a field."""

        # Turn coords into an array of 64-bit indexes
        if not (isinstance(coords, numarray.NumArray)
                and coords.type() != numarray.Int64):
            coords = numarray.array(coords, type=numarray.Int64)
        ncoords = len(coords)
        # Create a read buffer only if needed
        if field is None or ncoords > 0:
            if ncoords < self._v_maxTuples:
                result = self._v_rbuffer[:ncoords]
            else:
                result = self._get_container(ncoords)
        # Do the real read
        if ncoords > 0:
            self._read_elements(result, coords)

        if field is None:
            if numpy_imported and flavor == "numpy":
                na = tonumpy(result)
            else:
                na = result
        else:
            if ncoords > 0:
                na = result.field(field)
            else:
                # Get an empty array from the cache
                na = self._g_getemptyarray(self.colstypes[field],
                                           self.colitemsizes[field])
            # It's necessary to do a copy before conversion
            # because we need a well-behaved NumArray object
            # before feeding it into convertNATo*
            if Numeric_imported and flavor == "numeric":
                return convertNAToNumeric(na.copy())
            elif numpy_imported and flavor == "numpy":
                return convertNAToNumPy(na.copy())
        # Do an additional conversion, if needed
        if flavor == "python":
            return self.tolist(na)

        return na

    def tolist(self, arr):
        """Converts a NestedRecArray or NestedRecord to a list of rows"""
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

    def getEnum(self, colname):
        """
        Get the enumerated type associated with the named column.

        If the column named `colname` (a string) exists and is of an
        enumerated type, the corresponding `Enum` instance is returned.
        If it is not of an enumerated type, a ``TypeError`` is raised.
        If the column does not exist, a ``KeyError`` is raised.
        """

        self._checkColumn(colname)

        try:
            return self._colenums[colname]
        except KeyError:
            raise TypeError(
                "column ``%s`` of table ``%s`` is not of an enumerated type"
                % (colname, self._v_pathname))


    def col(self, name):
        """
        Get a column from the table.

        If a column called `name` exists in the table, it is read and
        returned as a ``numarray`` object or as a ``numpy`` object
        (depending on the flavor of the Table).  If it does not exist, a
        ``KeyError`` is raised.

        Example of use::

            narray = table.col('var2')

        That statement is equivalent to::

            narray = table.read(field='var2')

        Here you can see how this method can be used as a shorthand for
        the `read()` method.
        """
        return self.read(field=name)


    def __getitem__(self, key):
        """
        Get a row or a range of rows from the table.

        If the `key` argument is an integer, the corresponding table row is
        returned as a ``numarray.records.Record`` or as a
        ``tables.nestedrecords.NestedRecord`` object, what is more
        appropriate.  If `key` is a slice, the range of rows determined by it
        is returned as a ``numarray.records.RecArray`` or as a
        ``tables.nestedrecords.NestedRecArray`` object, what is more
        appropriate.

        Using a string as `key` to get a column is supported but
        deprecated.  Please use the `col()` method.

        Example of use::

            record = table[4]
            recarray = table[4:1000:2]

        Those statements are equivalent to::

            record = table.read(start=4)[0]
            recarray = table.read(start=4, stop=1000, step=2)

        Here you can see how indexing and slicing can be used as
        shorthands for the `read()` method.
        """

        if is_idx(key):
            # Index out of range protection
            if key >= self.nrows:
                raise IndexError, "Index out of range"
            if key < 0:
                # To support negative values
                key += self.nrows
            (start, stop, step) = processRange(self.nrows, key, key+1, 1)
            # For the scalar case, convert the Record and return it as a tuple
            # Fixes bug #972534
            # Reverted to return a numarray.records.Record in order
            # to support better the nested datatypes
            # return self.tolist(self.read(start, stop, step)[0])
            return self.read(start, stop, step)[0]
        elif isinstance(key, slice):
            (start, stop, step) = processRange(self.nrows,
                                               key.start, key.stop, key.step)
            return self.read(start, stop, step)
        elif isinstance(key, str):
            warnings.warn(DeprecationWarning("""\
``table['colname']`` is deprecated; please use ``table.col('colname')``"""),
                          stacklevel=2)
            return self.col(key)
        else:
            raise TypeError("invalid index or slice: %r" % (key,))


    def __setitem__(self, key, value):
        """Sets a table row or table slice.

        It takes different actions depending on the type of the 'key'
        parameter:

        If 'key' is an integer, the corresponding table row is set to
        'value' (List or Tuple). If 'key' is a slice, the row slice
        determined by key is set to value (a NestedRecArray or list of
        rows).

        """

        if self._v_file.mode == 'r':
            raise IOError("""\
Attempt to write over a file opened in read-only mode.""")

        if is_idx(key):
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

    def append(self, rows):
        """Append a series of rows to the end of the table

        rows can be either a recarray (both numarray and numpy flavors
        are supported) or a structure that is able to be converted to a
        recarray compliant with the table format.

        It raises an 'ValueError' in case the rows parameter could not
        be converted to an object compliant with table description.

        """

        if self._v_file.mode == 'r':
            raise IOError("""\
Attempt to write over a file opened in read-only mode.""")

        if not self._chunked:
            raise HDF5ExtError("""\
You cannot append rows to a non-chunked table.""")

        # Try to convert the object into a recarray compliant with table
        try:
            # This always makes a copy of the original,
            # so the resulting object is safe for in-place conversion.
            recarray = nestedrecords.array(
                rows,
                formats=self.description._v_nestedFormats,
                names=self.colnames)
        except Exception, exc:  #XXX
            raise ValueError, \
"rows parameter cannot be converted into a recarray object compliant with table '%s'. The error was: <%s>" % (str(self), exc)
        lenrows = recarray.shape[0]
        self._open_append(recarray)
        self._append_records(lenrows)
        self._close_append()
        # Update the number of saved rows
        self.nrows += lenrows
        # Save indexedrows
        if self.indexed:
            # Update the number of unsaved indexed rows
            self._unsaved_indexedrows += lenrows
            if self.indexprops.auto:
                self.flushRowsToIndex(lastrow=False)

    def _saveBufferedRows(self, flush=0):
        """Save buffered table rows"""
        # Save the records on disk
        # Data is copied to the buffer,
        # so it's safe to do an in-place conversion.
        # Open and close the table before and after appending to let
        # the dataset_id and other variables in a safe state for other
        # reading calls. Fixes #1186892
        # F. Altet 2005-04-05
        self._open_append(self._v_wbuffer)
        self._append_records(self._unsaved_nrows)
        self._close_append()
        # Update the number of saved rows in this buffer
        self.nrows += self._unsaved_nrows
        if self.indexed:
            self._unsaved_indexedrows += self._unsaved_nrows
            if self.indexprops.auto:
                # Flush the unindexed rows (this needs to read the table)
                self.flushRowsToIndex(lastrow=False)
        # Reset the number of unsaved rows
        self._unsaved_nrows = 0
        # Get a fresh copy of the default values
        # Note: It is important to do a copy only in the case that we are
        # not doing a flush. Doing the copy in the flush state, causes a fatal
        # error of the form:
        # *** glibc detected *** corrupted double-linked list: 0x08662d18 ***
        # I don't know the cause, but some tests seems to point out that this
        # *could* be related with the python garbage collector.
        # F. Altet 2006-04-28
        if not flush:
            self._v_wbuffer[:] = self._v_wbuffercpy[:]

    def modifyRows(self, start=None, stop=None, step=1, rows=None):
        """Modify a series of rows in the slice [start:stop:step]

        `rows` can be either a recarray or a structure that is able to
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
            # so the resulting object is safe for in-place conversion.
            recarray = nestedrecords.array(
                rows,
                formats=self.description._v_nestedFormats,
                names=self.colnames)
            # records.array does not seem to change the names
            # attibute in case rows is a recarray.
            # Change it manually and report this
            # 2004-08-08
            recarray._names = self.colnames
        except Exception, exc:  #XXX
            raise ValueError, \
"rows parameter cannot be converted into a recarray object compliant with table format '%s'. The error was: <%s>" % (str(self.description._v_nestedFormats), exc)
        lenrows = len(recarray)
        if start + lenrows > self.nrows:
            raise IndexError, \
"This modification will exceed the length of the table. Giving up."
        self._update_records(start, stop, step, recarray)
        # Redo the index if needed
        self._reIndex(self.colnames)

        return lenrows

    def modifyColumn(self, start=None, stop=None, step=1,
                     column=None, colname=None):
        """Modify one single column in the row slice [start:stop:step]

        `column` can be either a ``NestedRecArray``, ``RecArray``,
        ``NumArray``, ``numpy``, list or tuple that is able to be
        converted into a ``NestedRecArray`` compliant with the specified
        `colname` column of the table.

        `colname` specifies the column name of the table to be modified.

        Returns the number of modified rows.

        It raises a ``ValueError`` in case the columns parameter could
        not be converted into an object compliant with the column
        description.

        It raises an ``IndexError`` in case the modification will exceed
        the length of the table.

        """

        if not isinstance(colname, str):
            raise TypeError("""\
The 'colname' parameter must be a string.""")

        if column is None:      # Nothing to be done
            return 0
        if start is None:
            start = 0

        if start < 0:
            raise ValueError("'start' must have a positive value.")
        if step < 1:
            raise ValueError("'step' must have a value greater or equal than 1.")
        # Get the column format to be modified:
        objcol = self._checkColumn(colname)
        if isinstance(objcol, Description):
            selcolname = objcol._v_nestedNames
        else:
            selcolname = objcol._v_parent._v_nestedNames[objcol._v_pos]
        format = objcol._v_parent._v_nestedFormats[objcol._v_pos]
        # Try to convert the column object into a recarray
        try:
            if (isinstance(column, records.RecArray) or
                (numpy_imported and isinstance(column, numpy.ndarray))):
                recarray = nestedrecords.array(column, formats=format,
                                               names=selcolname)
            else:
                recarray = nestedrecords.fromarrays([column], formats=format,
                                                    names=selcolname)
            recarray._fields = recarray._get_fields()  # Refresh the cache
        except Exception, exc:  #XXX
            raise ValueError, \
"column parameter cannot be converted into a recarray object compliant with specified column '%s'. The error was: <%s>" % (str(column), exc)

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
        mod_recarr = self.read(start, stop, step, flavor="numarray")
        mod_recarr._fields = mod_recarr._get_fields()  # Refresh the cache
        # Modify the appropriate column in the original recarray
        if isinstance(objcol, Description):
            mod_recarr.field(colname)[:] = recarray
        else:
            # recarray should have one one field
            mod_recarr.field(colname)[:] = recarray.field(0)
        # save this modified rows in table
        self._update_records(start, stop, step, mod_recarr)
        # Redo the index if needed
        self._reIndex(colname)

        return nrows

    def modifyColumns(self, start=None, stop=None, step=1,
                      columns=None, names=None):
        """Modify a series of columns in the row slice [start:stop:step]

        `columns` can be either a ``NestedRecArray``, ``RecArray``,
        numpy object, a list of arrays or list of tuple (the columns)
        that are able to be converted into a ``NestedRecArray``
        compliant with the specified column `names` subset of the table
        format.

        `names` specifies the column names of the table to be modified.

        Returns the number of modified rows.

        It raises an ``ValueError`` in case the columns parameter could
        not be converted into an object compliant with table
        description.

        It raises an ``IndexError`` in case the modification will exceed
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
            raise ValueError("'step' must have a value greater or equal than 1.")        # Get the column formats to be modified:
        formats = []
        selcolnames = []
        for colname in names:
            objcol = self._checkColumn(colname)
            selcolnames.append(objcol._v_parent._v_nestedNames[objcol._v_pos])
            formats.append(objcol._v_parent._v_nestedFormats[objcol._v_pos])
        # Try to convert the columns object into a recarray
        try:
            if (isinstance(columns, records.RecArray) or
                (numpy_imported and isinstance(columns, numpy.ndarray))):
                recarray = nestedrecords.array(columns, formats=formats,
                                               names=selcolnames)
            else:
                recarray = nestedrecords.fromarrays(columns, formats=formats,
                                                    names=selcolnames)
            recarray._fields = recarray._get_fields()  # Refresh the cache
        except Exception, exc:  #XXX
            raise ValueError, \
"columns parameter cannot be converted into a recarray object compliant with table '%s'. The error was: <%s>" % (str(self), exc)

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
        mod_recarr = self.read(start, stop, step, flavor="numarray")
        mod_recarr._fields = mod_recarr._get_fields()  # Refresh the cache
        # Modify the appropriate columns in the original recarray
        for name in recarray._names:
            mod_recarr._fields[name][:] = recarray._fields[name]
        # save this modified rows in table
        self._update_records(start, stop, step, mod_recarr)
        # Redo the index if needed
        self._reIndex(names)

        return nrows

    def flushRowsToIndex(self, lastrow=True):
        "Add remaining rows in buffers to non-dirty indexes"
        rowsadded = 0
        if self.indexed:
            # Update the number of unsaved indexed rows
            start = self._indexedrows
            nrows = self._unsaved_indexedrows
            for (colname, colindexed) in self.colindexed.iteritems():
                if colindexed:
                    col = self.cols._f_col(colname)
                    if nrows > 0 and not col.dirty:
                        rowsadded = self._addRowsToIndex(
                            colname, start, nrows, lastrow )
            self._unsaved_indexedrows -= rowsadded
            self._indexedrows += rowsadded
        return rowsadded

    def _addRowsToIndex(self, colname, start, nrows, lastrow):
        """Add more elements to the existing index """

        # This method really belongs in Column, but since it makes extensive
        # use of the table, it gets dangerous when closing the file, since the
        # column may be accessing a table which is being destroyed.
        index = self.cols._f_col(colname).index
        slicesize = index.slicesize
        # The next loop does not rely on xrange so that it can
        # deal with long ints (i.e. more than 32-bit integers)
        # This allows to index columns with more than 2**31 rows
        # F. Altet 2005-05-09
        indexedrows = 0
        i = start
        stop = start+nrows-slicesize+1
        while i < stop:
            index.append(self._read(i, i+slicesize, 1, colname))
            indexedrows += slicesize
            i += slicesize
        # index the remaining rows
        nremain = nrows - indexedrows
        if lastrow and nremain > 0 and index.is_pro:
            index.appendLastRow(self._read(indexedrows, nrows, 1, colname),
                                self.nrows)
            indexedrows += nremain
        return indexedrows

    def removeRows(self, start, stop=None):
        """Remove a range of rows.

        If only "start" is supplied, this row is to be deleted.
        If "start" and "stop" parameters are supplied, a row
        range is selected to be removed.

        """

        (start, stop, step) = processRangeRead(self.nrows, start, stop, 1)
        nrows = stop - start
        if nrows >= self.nrows:
            raise NotImplementedError, \
"""You are trying to delete all the rows in table "%s". This is not supported right now due to limitations on the underlying HDF5 library. Sorry!""" % self._v_pathname
        nrows = self._remove_row(start, nrows)
        self.nrows -= nrows    # discount the removed rows from the total
        # removeRows is a invalidating index operation
        self._reIndex(self.colnames)

        return nrows


    def _g_updateDependent(self):
        super(Table, self)._g_updateDependent()

        self.cols._g_updateTableLocation(self)

    def _g_move(self, newParent, newName):
        """
        Move this node in the hierarchy.

        This overloads the Node._g_move() method.
        """

        oldparent = self._v_parent
        oldname = self._v_name

        # First, move the table to the new location.
        super(Table, self)._g_move(newParent, newName)

        # Then move the associated indexes (if any)
        if self.indexed:
            itgroup = self._v_file._getNode(_getIndexTableName(oldparent,
                                                               oldname))
            oldiname = itgroup._v_name
            newigroup = self._v_parent
            newiname = _getEncodedTableName(self._v_name)
            itgroup._g_move(newigroup, newiname)


    def _g_remove(self, recursive=False):
        # Remove the associated indexes (if they exist).
        if self.indexed:
            itgroup = self._v_file._getNode(
                _getIndexTableName(self._v_parent, self.name))
            itgroup._f_remove(recursive=True)
            self.indexed = False   # The indexes are no longer valid

        # Remove the leaf itself from the hierarchy.
        super(Table, self)._g_remove(recursive)


    def removeIndex(self, index):
        "Remove the index associated with the specified column"

        # Check that file is not in read-only mode
        if self._v_file.mode == 'r':
            raise IOError("""\
Attempt to write over a file opened in read-only mode.""")

        if not isinstance(index, Index):
            raise TypeError("""\
Wrong 'index' parameter type. Only Index instances are accepted.""")
        index.column.removeIndex()


    def _reIndex(self, colnames):
        """ re-index all the columns in colnames is self.reindex is true """

        if self.indexed:
            # Mark the proper indexes as dirty
            for (colname, colindexed) in self.colindexed.iteritems():
                if colindexed and colname in colnames:
                    col = self.cols._f_col(colname)
                    col.dirty = True
            # Now, re-index the dirty ones
            if self.indexprops.reindex:
                self.reIndex()

    def reIndex(self):
        """Recompute the existing indexes in table"""
        for (colname, colindexed) in self.colindexed.iteritems():
            if colindexed:
                indexcol = self.cols._f_col(colname)
                indexedrows = indexcol.reIndex()
        # Update counters
        self._indexedrows = indexedrows
        self._unsaved_indexedrows = self.nrows - indexedrows
        return indexedrows

    def reIndexDirty(self):
        """Recompute the existing indexes in table if they are dirty"""
        for (colname, colindexed) in self.colindexed.iteritems():
            if colindexed:
                indexcol = self.cols._f_col(colname)
                indexedrows = indexcol.reIndexDirty()
        # Update counters
        self._indexedrows = indexedrows
        self._unsaved_indexedrows = self.nrows - indexedrows
        return indexedrows

    def _g_copyRows(self, object, start, stop, step):
        "Copy rows from self to object"
        (start, stop, step) = processRangeRead(self.nrows, start, stop, step)
        nrowsinbuf = self._v_maxTuples
        object._open_append(self._v_wbuffer)
        nrowsdest = object.nrows
        for start2 in xrange(start, stop, step*nrowsinbuf):
            # Save the records on disk
            stop2 = start2+step*nrowsinbuf
            if stop2 > stop:
                stop2 = stop
            # Optimized version (it saves some conversions)
            nrows = ((stop2 - start2 - 1) // step) + 1
            self.row._fillCol(self._v_wbuffer, start2, stop2, step, None)
            # The output buffer is created anew,
            # so the operation is safe to in-place conversion.
            object._append_records(nrows)
            nrowsdest += nrows
        object._close_append()
        # Update the number of saved rows in this buffer
        object.nrows = nrowsdest
        return

    # This is an optimized version of copy
    def _g_copyWithStats(self, group, name, start, stop, step,
                         title, filters, _log):
        "Private part of Leaf.copy() for each kind of leaf"
        # Build the new Table object
        description = self.description
        # Checking validity names for fields in destination is not necessary
        description.__dict__['__check_validity__'] = 0
        # Ensure backward compatibility with old index format
        if hasattr(self.attrs, "VERSION") and self.attrs.VERSION < "2.3":
            # Set the appropriate indexes properties for these old indexes
            autoindex = getattr(self.attrs, 'AUTOMATIC_INDEX', None)
            reindex = getattr(self.attrs, 'REINDEX', None)
            self.indexprops = IndexProps(auto=autoindex, reindex=reindex)
            for colname in self.colnames:
                indexname = "_i_%s_%s" % (self.name, colname)
                indexpathname = joinPath(self._v_parent._v_pathname, indexname)
                try:
                    index = self._v_file._getNode(indexpathname)
                    # Get the filters values
                    self.indexprops.filters = self._g_getFilters()
                    getattr(description, colname).indexed = 1
                except NodeError:
                    getattr(description, colname).indexed = 0

        # Add a possible IndexProps property to that
        if self.indexprops is not None:
            description.__dict__["_v_indexprops"] = self.indexprops

        object = Table(
            group, name, description, title=title, filters=filters,
            expectedrows=self.nrows, _log=_log)
        # Now, fill the new table with values from the old one
        self._g_copyRows(object, start, stop, step)
        nbytes=self.nrows*self.rowsize
        if object.indexed:
            warnings.warn( \
"Regenerating indexes for destination table %s:%s. This may take a while, be patient please." % (object._v_file.filename, object._v_pathname))
            object._indexedrows = 0
            object._unsaved_indexedrows = object.nrows
            if object.indexprops.auto:
                object.flushRowsToIndex(lastrow=True)
        return (object, nbytes)

    def _g_cleanIOBuf(self):
        """Clean the I/O buffers."""

        mydict = self.__dict__
        if "_v_wbuffer" in mydict:
            del mydict['_v_wbuffer']     # Decrement the pointer to write buffer
            del mydict['_v_wbuffercpy']  # Decrement the pointer to write buffer copy

    def flush(self):
        """Flush the table buffers."""

        # Flush rows that remains to be appended
        if self._unsaved_nrows > 0:
            self._saveBufferedRows(flush=1)
        if self.indexed and self.indexprops.auto:
            # Flush any unindexed row
            rowsadded = self.flushRowsToIndex(lastrow=True)
            if rowsadded > 0 and self._indexedrows <> self.nrows:  ## XXX only for pro!
                raise RuntimeError , "Internal error: the number of indexed rows (%s) and rows in table (%s) must be equal!. Please, report this to the authors." % (self._indexedrows, self.nrows)

# #****************************** a test *************************************
#         # XXX For pro
#         if self.indexed:
#             # Optimize the indexed rows
#             for (colname, colindexed) in self.colindexed.iteritems():
#                 if colindexed:
#                     col = self.cols._f_col(colname)
#                     if nrows > 0 and not col.dirty:
#                         print "*optimizing col-->", colname
#                         col.index.optimize()
# #***************************** end test ************************************

        self._g_cleanIOBuf()
        super(Table, self).flush()

    def _g_preKillHook(self):
        """Code to be called before killing the node."""

        # Flush the buffers before to clean-up them
        # It seems that flushing during the __del__ phase is a sure receipt for
        # bringing all kind of problems:
        # 1. Illegal Instruction
        # 2. Malloc(): trying to call free() twice
        # 3. Bus Error
        # 4. Segmentation fault
        # So, the best would be doing *nothing* at all in this __del__ phase.
        # As a consequence, the I/O will not be cleaned until a call to
        # Table.flush() would be done. This could lead to a potentially large
        # memory consumption.
        # NOTE: The user should make a call to Table.flush() whenever he has
        #       finished working with his table.
        # F. Altet 2006-08-01
        self.flush()
        return

    def _f_close(self, flush=True):
        if not self._f_isOpen():
            return  # the node is already closed

        # .. note::
        #
        #   As long as ``Table`` objects access their indices on closing,
        #   ``File.close()`` will need to make *two separate passes*
        #   to first close ``Table`` objects and then ``Index`` hierarchies.
        #

        # Flush right now so the row object does not get in the middle.
        if flush:
            self.flush()

        # Some warnings can be issued after calling `self._g_setLocation()`
        # in `self.__init__()`.  If warnings are turned into exceptions,
        # `self._g_postInitHook` may not be called and `self.cols` not set.
        # One example of this is
        # ``test_create.createTestCase.test05_maxFieldsExceeded()``.
        cols = self.cols
        if cols is not None:
            cols._f_close()

        # Close myself as a leaf.
        super(Table, self)._f_close(False)


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

        _v_colnames -- List with all column names hanging from cols
        _v_colpathnames -- List with all column names hanging from cols
        _v_table -- The parent table instance
        _v_desc -- The associated Description instance

    Methods:

        __getitem__(slice)
        __f_col(colname)
        __len__()

    """

    def _g_gettable(self):
        return self._v__tableFile._getNode(self._v__tablePath)

    _v_table = property(_g_gettable)


    def __init__(self, table, desc):
        """Create the container to keep the column information.
        """

        myDict = self.__dict__
        myDict['_v__tableFile'] = table._v_file
        myDict['_v__tablePath'] = table._v_pathname
        myDict['_v_desc'] = desc
        myDict['_v_colnames'] = desc._v_names
        myDict['_v_colpathnames'] = table.description._v_pathnames
        # Put the column in the local dictionary
        for name in desc._v_names:
            if name in desc._v_types:
                myDict[name] = Column(table, name, desc)
            else:
                myDict[name] = Cols(table, desc._v_colObjects[name])


    def _g_updateTableLocation(self, table):
        """Updates the location information about the associated `table`."""

        myDict = self.__dict__
        myDict['_v__tableFile'] = table._v_file
        myDict['_v__tablePath'] = table._v_pathname

        # Update the locations in individual columns.
        for colname in self._v_colnames:
            myDict[colname]._g_updateTableLocation(table)


    def __len__(self):
        return len(self._v_colnames)

    def _f_col(self, colname):
        """Return the column named "colname"."""

        if not isinstance(colname, str):
            raise TypeError, \
"Parameter can only be an string. You passed object: %s" % colname
        if ((colname.find('/') > -1 and
             not colname in self._v_colpathnames) and
            not colname in self._v_colnames):
            raise KeyError(
"Cols accessor ``%s.cols%s`` does not have a column named ``%s``"
    % (self._v__tablePath, self._v_desc._v_pathname, colname))

        # Get the Column or Description object
        inames = colname.split('/')
        cols = self
        for iname in inames:
            cols = cols.__dict__[iname]
        return cols

    def __getitem__(self, key):
        """
        Get a row or a range of rows from a (nested) column.

        If the `key` argument is an integer, the corresponding nested
        type row is returned as a ``tables.nestedrecords.NestedRecord``
        object.  If `key` is a slice, the range of rows determined by it
        is returned as a ``tables.nestedrecords.NestedRecArray`` object.

        Using a string as `key` to get a column is supported but
        deprecated.  Please use the `_f_col()` method.

        Example of use::

            record = table.cols[4]  # equivalent to table[4]
            recarray = table.cols.Info[4:1000:2]

        Those statements are equivalent to::

            nrecord = table.read(start=4)[0]
            nrecarray = table.read(start=4, stop=1000, step=2).field('Info')

        Here you can see how a mix of natural naming, indexing and
        slicing can be used as shorthands for the `Table.read()` method.

        """

        table = self._v_table
        nrows = table.nrows
        if is_idx(key):
            # Index out of range protection
            if key >= nrows:
                raise IndexError, "Index out of range"
            if key < 0:
                # To support negative values
                key += nrows
            (start, stop, step) = processRange(nrows, key, key+1, 1)
            colgroup = self._v_desc._v_pathname
            if colgroup == "":  # The root group
                return table.read(start, stop, step)[0]
            else:
                crecord = table.read(start, stop, step)[0]
                return crecord.field(colgroup)
        elif isinstance(key, slice):
            (start, stop, step) = processRange(nrows,
                                               key.start, key.stop, key.step)
            colgroup = self._v_desc._v_pathname
            if colgroup == "":  # The root group
                return table.read(start, stop, step)
            else:
                crecarray = table.read(start, stop, step)
                if hasattr(crecarray, "field"):
                    return crecarray.field(colgroup)  # RecArray case
                else:
                    return crecarray[colgroup]  # numpy case
        elif isinstance(key, str):
            warnings.warn(DeprecationWarning("""\
``table.cols['colname']`` is deprecated; please use ``table.cols._f_col('colname')``"""),
                          stacklevel=2)
            return self._f_col(key)
        else:
            raise TypeError("invalid index or slice: %r" % (key,))

    def __setitem__(self, key, value):
        """
        Set a row or a range of rows to a (nested) column.

        If the `key` argument is an integer, the corresponding row is
        set to `value`.  If `key` is a slice, the range of rows
        determined by it is set to `value`.

        Example of use::

            table.cols[4] = record
            table.cols.Info[4:1000:2] = recarray

        Those statements are equivalent to::

            table.modifyRows(4, rows=record)
            table.modifyColumn(4, 1000, 2, colname='Info', column=recarray)

        Here you can see how a mix of natural naming, indexing and
        slicing can be used as shorthands for the `Table.modifyRows` and
        `Table.modifyColumn` methods.

        """

        table = self._v_table
        nrows = table.nrows
        if is_idx(key):
            # Index out of range protection
            if key >= nrows:
                raise IndexError, "Index out of range"
            if key < 0:
                # To support negative values
                key += nrows
            (start, stop, step) = processRange(nrows, key, key+1, 1)
        elif isinstance(key, slice):
            (start, stop, step) = processRange(nrows,
                                               key.start, key.stop, key.step)
        else:
            raise TypeError("invalid index or slice: %r" % (key,))

        # Actually modify the correct columns
        colgroup = self._v_desc._v_pathname
        if colgroup == "":  # The root group
            table.modifyRows(start, stop, step, rows=value)
        else:
            table.modifyColumn(start, stop, step, colname=colgroup, column=value)

    def _f_close(self):
        # First, close the columns (ie possible indices open)
        for col in self._v_colnames:
            colobj = self._f_col(col)
            if isinstance(colobj, Column):
                colobj.close()
                # Delete the reference to column
                del self.__dict__[col]
            else:
                colobj._f_close()

        self.__dict__.clear()

    # Cols does not accept comparisons
    def __lt__(self, other):
        raise TypeError, "Cols object can't be used in comparisons"

    def __le__(self, other):
        raise TypeError, "Cols object can't be used in comparisons"

    def __gt__(self, other):
        raise TypeError, "Cols object can't be used in comparisons"

    def __ge__(self, other):
        raise TypeError, "Cols object can't be used in comparisons"

    def __eq__(self, other):
        raise TypeError, "Cols object can't be used in comparisons"

    def __ne__(self, other):
        raise TypeError, "Cols object can't be used in comparisons"

    def __str__(self):
        """The string representation for this object."""
        # The pathname
        tablepathname = self._v__tablePath
        descpathname = self._v_desc._v_pathname
        # Get this class name
        classname = self.__class__.__name__
        # The number of columns
        ncols = len(self._v_colnames)
        return "%s.cols.%s (%s), %s columns" % \
               (tablepathname, descpathname, classname, ncols)

    def __repr__(self):
        """A detailed string representation for this object."""

        out = str(self) + "\n"
        for name in self._v_colnames:
            # Get this class name
            classname = getattr(self, name).__class__.__name__
            # The shape for this column
            shape = self._v_desc._v_shapes[name]
            # The type
            if name in self._v_desc._v_types:
                tcol = self._v_desc._v_types[name]
            else:
                tcol = "Description"
            if shape == 1:
                shape = (1,)
            out += "  %s (%s%s, %s)" % (name, classname, shape, tcol) + "\n"
        return out


class Column(object):
    """This is an accessor for the actual data in a table column

    Instance variables:

        table -- the parent table instance
        name -- the name of the associated column
        pathname -- the complete pathname of the column (the same as `name`
                    if column is non-nested)
        descr -- the parent description object
        type -- the type of the column
        shape -- the shape of the column
        index -- the Index object (None if doesn't exists)
        dirty -- whether the index is dirty or not (property)

    Methods:
        __getitem__(key)
        __setitem__(key, value)
        createIndex()
        reIndex()
        reIndexDirty()
        removeIndex()
    """

    def _gettable(self):
        return self._tableFile._getNode(self._tablePath)

    table = property(_gettable)


    def _getindex(self):
        if self._indexPath is None:
            return None  # the column is not indexed
        return self._indexFile._getNode(self._indexPath)

    index = property(_getindex)

    def _isindexed(self):
        if self._indexPath is None:
            return False
        else:
            return True

    is_indexed = property(_isindexed)


    def __init__(self, table, name, descr):
        """Create the container to keep the column information.

        Parameters:

        table -- The parent table instance
        name -- The name of the column that is associated with this object
        descr -- The parent description object

        """
        self._tableFile = tableFile = table._v_file
        self._tablePath = table._v_pathname
        self.name = name
        self.pathname = descr._v_colObjects[name]._v_pathname
        self.descr = descr
        self.type = descr._v_types[name]
        self.shape = descr._v_shapes[name]
        # Check whether an index exists or not
        indexname = _getIndexColName(table._v_parent, table._v_name,
                                     self.pathname)
        try:
            index = tableFile._getNode(indexname)
            index.column = self # points to this column
            self._indexFile = index._v_file
            self._indexPath = index._v_pathname
        except NodeError:
            self._indexFile = None
            self._indexPath = None
            # Only create indexes for newly-created tables
            # (where `table.colindexed` has already been initialized).
            if table._v_new and table.colindexed[self.pathname]:
                # The user wants to indexate this column,
                # but it doesn't exists yet. Create it without a warning.
                self.createIndex(warn=0)         # self.index is assigned here


    def _g_updateTableLocation(self, table):
        """Updates the location information about the associated `table`."""

        self._tableFile = table._v_file
        self._tablePath = table._v_pathname


    def _updateIndexLocation(self, index):
        """
        Updates the location information about the associated `index`.

        If the `index` is ``None``, no index will be set.
        """

        if index is None:
            self._indexFile = None
            self._indexPath = None
        else:
            self._indexFile = index._v_file
            self._indexPath = index._v_pathname


    # Define dirty as a property
    def _get_dirty(self):
        if hasattr(self, "_dirty"):
            return self._dirty
        index = self.index
        if index:
            if hasattr(index._v_attrs, "DIRTY"):
                self._dirty = dirty = getattr(index._v_attrs, "DIRTY")
                return dirty
            else:
                # If don't have a DIRTY attribute, index should be clean
                self._dirty = False
                return False
        else:
            self._dirty = True  # If don't have index, this is like dirty
            return True

    def _set_dirty(self, dirty):
        self._dirty = dirty
        index = self.index
        # Only set the index column as dirty if it exists
        if index:
            setattr(index._v_attrs, "DIRTY", dirty)
            if dirty:
                index.indicesLR[-1] = 0
                index.nelementsLR = 0
                index.nelements = 0

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

        table = self.table
        if is_idx(key):
            # Index out of range protection
            if key >= table.nrows:
                raise IndexError, "Index out of range"
            if key < 0:
                # To support negative values
                key += table.nrows
            (start, stop, step) = processRange(table.nrows, key, key+1, 1)
            return table.read(start, stop, step, self.pathname)[0]
        elif isinstance(key, slice):
            (start, stop, step) = processRange(table.nrows, key.start,
                                               key.stop, key.step)
            return table.read(start, stop, step, self.pathname)
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

        table = self.table

        if table._v_file.mode == 'r':
            raise IOError("""\
Attempt to write over a file opened in read-only mode.""")

        if is_idx(key):
            # Index out of range protection
            if key >= table.nrows:
                raise IndexError, "Index out of range"
            if key < 0:
                # To support negative values
                key += table.nrows
            return table.modifyColumns(key, key+1, 1,
                                       [[value]], names=[self.pathname])
        elif isinstance(key, slice):
            (start, stop, step) = processRange(table.nrows,
                                               key.start, key.stop, key.step)
            return table.modifyColumns(start, stop, step,
                                       [value], names=[self.pathname])
        else:
            raise ValueError, "Non-valid index or slice: %s" % key

    def _addComparison(self, noper, other):
        table = self.table
        table.ops.append(noper)
        table.opsValues.append(other)
        table.opsColnames.append(self.pathname)

    def __lt__(self, other):
        self._addComparison(1, other)
        return self

    def __le__(self, other):
        table = self.table
        table.ops.append(2)
        table.opsValues.append(other)
        return self

    def __gt__(self, other):
        table = self.table
        table.ops.append(3)
        table.opsValues.append(other)
        return self

    def __ge__(self, other):
        table = self.table
        table.ops.append(4)
        table.opsValues.append(other)
        return self

    def __eq__(self, other):
        table = self.table
        table.ops.append(5)
        table.opsValues.append(other)
        return self

    def __ne__(self, other):
        table = self.table
        table.ops.append(6)
        table.opsValues.append(other)
        return self

    def _addLogical(self, noper):
        table = self.table
        table.ops.append(noper)
        table.opsValues.append(None)
        table.opsColnames.append(None)

    def __and__(self, other):
        self._addLogical(10)
        return self

    def __or__(self, other):
        self._addLogical(11)
        return self

    def __xor__(self, other):
        self._addLogical(12)
        return self

    def createIndex(self, optlevel=0, warn=True, testmode=False, verbose=False):
        """Create an index for this column"""

        name = self.name
        table = self.table
        tableName = table._v_name
        tableParent = table._v_parent
        descr = self.descr
        index = self.index
        getNode = table._v_file._getNode

        assert descr._v_shapes[name] == 1, "only scalar columns can be indexed"

        # Warn if the index already exists
        if index:
            raise ValueError, \
"%s for column '%s' already exists. If you want to re-create it, please, try with reIndex() method better" % (str(index), str(self.pathname))

        if descr._v_shapes[name] != 1:
            raise ValueError("Only scalar columns can be indexed.")

        # Get the indexes group for table, and if not exists, create it
        try:
            itgroup = getNode(_getIndexTableName(tableParent, tableName))
        except NodeError:
            itgroup = table._createIndexesTable(tableParent)

        # Get the indexes group for table, and if not exists, create it
        filters = table.indexprops.filters

        # Create the necessary intermediate groups for descriptors
        idgroup = itgroup
        dname = ""
        pathname = descr._v_pathname
        if pathname != '':
            inames = pathname.split('/')
            for iname in inames:
                if dname == '':
                    dname = iname
                else:
                    dname += '/'+iname
                try:
                    idgroup = getNode('%s/%s' % (tableParent._v_name, dname))
                except NodeError:
                    idgroup = table._createIndexesDescr(
                        idgroup, dname, iname, filters)

        # Create the atom
        atomtype = descr._v_types[name]
        if str(atomtype) == "CharType":
            atom = StringAtom(shape=(0,), length=descr._v_itemsizes[name])
        else:
            atom = Atom(dtype=atomtype, shape=(0,))

        # Create the index itself
        index = Index(
            idgroup, name, atom=atom, column=self,
            title="Index for %s column" % name,
            filters=filters,
            optlevel=optlevel,
            testmode=testmode,
            expectedrows=table._v_expectedrows)
        self._updateIndexLocation(index)

        # Feed the index with values
        slicesize = index.slicesize
        if (not index.is_pro and
            table.nrows < slicesize):
            if warn:
                warnings.warn(
                    "not enough rows for indexing: "
                    "you need at least %d rows and the table only has %d"
                    % (slicesize, table.nrows))
            return 0
        # Add rows to the index if necessary
        if table.nrows > 0:
            indexedrows = table._addRowsToIndex(
                self.pathname, 0, table.nrows, lastrow=True )
        else:
            indexedrows = 0
        if index.is_pro:
            index.optimize(verbose=verbose)   # optimize indexes
        self.dirty = False
        # Set some flags in table parent
        table.indexed = True
        table.colindexed[self.pathname] = True
        # If the user has not defined properties, assign the default
        table.indexprops = getattr(
            table.description, '_v_indexprops', IndexProps())
        table._indexedrows = indexedrows
        table._unsaved_indexedrows = table.nrows - indexedrows
        return indexedrows

    def optimizeIndex(self, level=3, verbose=0):
        """Optimize the index for this column.

        `level` is the level optimization (from 1 to 9) and defaults to 1.
        `verbose` specifies the level of verbosity of the optimization process.

        """

        assert 0 < level < 10
        assert self.index, "This column is not indexed, so it can't be optimized."
        if level > 0:
            self.index.optimize(level, verbose)
        return

    def reIndex(self):
        """Recompute the existing index"""

        # Check that file is not in read-only mode
        if self._tableFile.mode == 'r':
            raise IOError("""\
Attempt to write over a file opened in read-only mode.""")

        index = self.index
        if index is not None:
            # Delete the existing Index
            index._f_remove()
            self._updateIndexLocation(None)
            # Create a new Index without warnings
            return self.createIndex(warn=0)
        else:
            return 0  # The column is not intended for indexing

    def reIndexDirty(self):
        """Recompute the existing index only if it is dirty"""

        # Check that file is not in read-only mode
        if self._tableFile.mode == 'r':
            raise IOError("""\
Attempt to write over a file opened in read-only mode.""")

        index = self.index
        if index is not None and self.dirty:
            # Delete the existing Index
            index._f_remove()
            # Create a new Index without warnings
            return self.createIndex(warn=0)
        else:
            # The column is not intended for indexing or is not dirty
            return 0

    def removeIndex(self):
        """Delete the associated column's index"""

        table = self.table

        # Check that file is not in read-only mode
        if table._v_file.mode == 'r':
            raise IOError("""\
Attempt to write over a file opened in read-only mode.""")

        # delete some references
        index = self.index
        if index:
            index._f_remove()
            self._updateIndexLocation(None)
            table.colindexed[self.name] = 0
        else:
            return  # Do nothing


    def close(self):
        """Close this column"""
        self.__dict__.clear()


    def __str__(self):
        """The string representation for this object."""
        # The pathname
        tablepathname = self._tablePath
        pathname = self.pathname.replace('/', '.')
        # Get this class name
        classname = self.__class__.__name__
        # The shape for this column
        shape = self.descr._v_shapes[self.name]
        if shape == 1:
            shape = (1,)
        # The type
        tcol = self.descr._v_types[self.name]
        return "%s.cols.%s (%s%s, %s, idx=%s)" % \
               (tablepathname, pathname, classname, shape, tcol, self.index)

    def __repr__(self):
        """A detailed string representation for this object."""
        return str(self)



## Local Variables:
## mode: python
## py-indent-offset: 4
## tab-width: 4
## fill-column: 78
## End:
