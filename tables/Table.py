#######################################################################
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

import sys
import warnings
import re
import copy
from time import time

import numpy

import tables.TableExtension as TableExtension
from tables.conditions import split_condition, call_on_recarr
from tables.numexpr.compiler import getType as numexpr_getType
from tables.numexpr.expressions import functions as numexpr_functions
from tables.flavor import flavor_of, array_as_internal, internal_to_flavor
from tables.utils import processRange, processRangeRead, \
     joinPath, is_idx, flattenNames, byteorders
from tables.Leaf import Leaf
from tables.Index import Index, IndexProps
from tables.IsDescription import IsDescription, Description, Col
from tables.atom import Atom
from tables.Group import IndexesTableG, IndexesDescG
from tables.exceptions import NodeError, HDF5ExtError, PerformanceWarning
from tables.constants import MAX_COLUMNS, EXPECTED_ROWS_TABLE, CHUNKTIMES, \
     LIMDATA_MAX_SLOTS, LIMDATA_MAX_SIZE, TABLE_MAX_SLOTS, MB
from tables.utilsExtension import getNestedField

from tables.lrucacheExtension import ObjectCache, NumCache


__version__ = "$Revision$"


# 2.2: Added support for complex types. Introduced in version 0.9.
# 2.2.1: Added suport for time types.
# 2.3: Changed the indexes naming schema.
# 2.4: Changed indexes naming schema (again).
# 2.5: Added the FIELD_%d_FILL attributes.
# 2.6: Added the FLAVOR attribute (optional).
obversion = "2.6"  # The Table VERSION number

# Paths and names for hidden nodes related with indexes.
_indexName   = '_i_%s'  # %s -> encoded table path

# Compile a regular expression for expressions like '(2,2)Int8'
prog = re.compile(r'([\(\),\d\s]*)([A-Za-z]+[0-9]*)')


# Maps NumPy types to the types used by Numexpr.
_nxTypeFromNPType = {
    numpy.bool_: bool,
    numpy.int8: int,
    numpy.int16: int,
    numpy.int32: int,
    numpy.int64: long,
    numpy.uint8: int,
    numpy.uint16: int,
    numpy.uint32: long,
    numpy.uint64: long,
    numpy.float32: float,
    numpy.float64: float,
    numpy.complex64: complex,
    numpy.complex128: complex,
    numpy.str_: str, }


def _getEncodedTableName(tablename):
    return _indexName % tablename

def _getIndexTableName(parent, tablename):
    return joinPath(parent._v_pathname, _getEncodedTableName(tablename))

def _getIndexColName(parent, tablename, colname):
    return joinPath(_getIndexTableName(parent, tablename), colname)


class NailedDict(object):

    """A dictionary which ignores its items when it has nails on it."""

    def __init__(self):
        self._cache = {}
        self._nailcount = 0

    # Only a restricted set of dictionary methods are supported.  That
    # is why we buy instead of inherit.

    # The following are intended to be used by ``Table`` code changing
    # the set of usable indexes.

    def clear(self):
        self._cache.clear()
    def nail(self):
        self._nailcount -= 1
    def unnail(self):
        self._nailcount += 1

    # The following are intended to be used by ``Table`` code handling
    # conditions.

    def __contains__(self, key):
        if self._nailcount > 0:
            return False
        return key in self._cache

    def __getitem__(self, key):
        if self._nailcount > 0:
            raise KeyError(key)
        return self._cache[key]

    def get(self, key, default=None):
        if self._nailcount > 0:
            return default
        return self._cache.get(key, default)

    def __setitem__(self, key, value):
        if self._nailcount > 0:
            return
        self._cache[key] = value


class Table(TableExtension.Table, Leaf):
    """Represent a table in the object tree.

    It provides methods to create new tables or open existing ones, as
    well as to write/read data to/from table objects over the
    file. A method is also provided to iterate over the rows without
    loading the entire table or column in memory.

    Data can be written or read both as Row instances and as homogeneous
    or heterogeneous (record) arrays of the supported flavors (such as
    NumPy and numarray, including NestedRecArray objects).

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
        read([start] [, stop] [, step] [, field])
        readCoordinates(coords [, field])
        readWhere(condition [, condvars] [, field])
        reIndex()
        reIndexDirty()
        removeRows(start [, stop])
        where(condition [, condvars] [, start] [, stop] [, step])
        whereAppend(dstTable, condition [, condvars] [, start] [, stop] [, step])
        getWhereList(condition [, condvars] [, sort])

    Instance variables:

        description -- the metaobject describing this table
        row -- a reference to the Row object associated with this table
        nrows -- the number of rows in this table
        rowsize -- the size, in bytes, of each row
        cols -- accessor to the columns using a natural name schema
        colnames -- the field names for the table (list)
        colinstances -- the column instances for the table fields (dictionary)
        coldtypes -- the dtype class for the table fields (dictionary)
        coltypes -- the PyTables type for the table fields (dictionary)
        coldflts -- the defaults for each column (dictionary)
        colindexed -- whether the table fields are indexed (dictionary)
        indexed -- whether or not some field in Table is indexed
        indexprops -- properties of an indexed Table

    """

    # Class identifier.
    _c_classId = 'TABLE'


    # Properties
    # ~~~~~~~~~~
    row = property(
        lambda self: TableExtension.Row(self), None, None,
        "The associated `Row` instance.")

    # Read-only shorthands
    # ````````````````````

    shape = property(
        lambda self: (self.nrows,), None, None,
        "The shape of this table.")

    rowsize = property(
        lambda self: self.description._v_dtype.itemsize, None, None,
        "The size in bytes of each row in the table.")

    # itemsize & rowsize are the same for a unidimensional table.
    # This property is needed in order to allow a generalization
    # for buffersize/chunkshape calculation.
    itemsize = property(
        lambda self: self.description._v_dtype.itemsize, None, None,
        "The size in bytes of each element in the table.")

    byteorder = property(
        lambda self: self.description._v_byteorder, None, None,
        "The endianness of data in memory "
        "('big', 'little' or 'irrelevant').")

    # Lazy attributes
    # ```````````````
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

    # Other
    # `````
    def _setindexprops(self, value):
        if not isinstance(value, IndexProps):
            raise TypeError("not an instance of ``IndexProps``: %r" % value)
        oldprops, newprops = self._indexprops, value

        setAttr = self._v_attrs._g__setattr
        for (prop, attr) in [ ('auto', 'AUTOMATIC_INDEX'),
                              ('reindex', 'REINDEX'),
                              ('filters', 'FILTERS_INDEX') ]:
            # Only store values that have changed or were undefined.
            newvalue = getattr(newprops, prop)
            if not oldprops or getattr(oldprops, prop) != newvalue:
                setAttr(attr, newvalue)

        self._indexprops = value

    indexprops = property(
        lambda self: self._indexprops, _setindexprops, None,
        """
        Properties of the indexes of this table.

        This is an `IndexProps` instance.  You may replace it at any
        time, but only some changes will affect existing indexes.
        Particularly, changing automatic and reindexing parameters
        affects existing indexes immediatly, while changing the filters
        parameter only affects newly created indexes.
        """ )


    # Other methods
    # ~~~~~~~~~~~~~
    def __init__(self, parentNode, name,
                 description=None, title="", filters=None,
                 expectedrows=EXPECTED_ROWS_TABLE,
                 chunkshape=None, _log=True):
        """Create an instance of Table.

        Keyword arguments:

        description -- A IsDescription subclass or a dictionary where
            the keys are the field names, and the values the type
            definitions. And it can be also a recarray NumPy object,
            RecArray numarray object or NestedRecArray. If None, the
            table metadata is read from disk, else, it's taken from
            previous parameters.

        title -- Sets a TITLE attribute on the HDF5 table entity.

        filters -- An instance of the Filters class that provides
            information about the desired I/O filters to be applied
            during the life of this object.

        expectedrows -- An user estimate about the number of rows that
            will be on table. If not provided, the default value is
            appropiate for tables until 1 MB in size (more or less,
            depending on the record size). If you plan to save bigger
            tables, try providing a guess; this will optimize the HDF5
            B-Tree creation and management process time and memory used.

        chunkshape -- The shape of the data chunk to be read or written
            as a single HDF5 I/O operation. The filters are applied to
            those chunks of data. Its rank for tables has to be 1.  If
            None, a sensible value is calculated (which is recommended).
        """

        self._v_new = new = description is not None
        """Is this the first time the node has been created?"""
        self._v_new_title = title
        """New title for this node."""
        self._v_new_filters = filters
        """New filter properties for this node."""
        self.extdim = 0   # Tables only have one dimension currently
        """The index of the enlargeable dimension."""

        self._v_recarray = None
        """A record array to be stored in the table."""
        self._v_expectedrows = expectedrows
        """The expected number of rows to be stored in the table."""
        self.nrows = 0L
        """Current number of rows in the table."""
        self._unsaved_nrows = 0
        """Number of rows in buffers but still not in disk."""
        self.description = None
        """A `Description` instance describing the structure of the table."""
        self._time64colnames = []
        """The names of ``Time64`` columns."""
        self._strcolnames = []
        """The names of ``String`` columns."""
        self._colenums = {}
        """Maps the name of an enumerated column to its ``Enum`` instance."""
        self._v_nrowsinbuf = None
        """The number of rows that fit in the table buffer."""
        self._v_chunkshape = None
        """The HDF5 chunk shape."""

        self.indexed = False
        """Does this table have any indexed columns?"""
        self._indexprops = None
        """Properties of the indexes of this table."""
        self._indexedrows = 0
        """Number of rows indexed in disk."""
        self._unsaved_indexedrows = 0
        """Number of rows indexed in memory but still not in disk."""

        self.colnames = ()
        """
        A tuple containing the (possibly nested) names of the columns in
        the table.
        """
        self.colinstances = {}
        """Maps the name of a column to its `Column` or `Cols` instance."""
        self.coldtypes = {}
        """Maps the name of a column to its NumPy data type."""
        self.coltypes = {}
        """Maps the name of a column to its PyTables data type."""
        self.coldflts = {}
        """Maps the name of a column to its default value."""
        self.colindexed = {}
        """Is the column which name is used as a key indexed? (dictionary)"""

        self._whereCondition = None
        """Condition function and argument list for selection of values."""
        self._whereIndex = None
        """Path of the indexed column to be used in an indexed search."""
        self._conditionCache = NailedDict()
        """Cache of already splitted conditions."""
        self._exprvarsCache = {}
        """Cache of variables participating in numexpr expressions."""
        self._enabledIndexingInQueries = True
        """Is indexing enabled in queries?  *Use only for testing.*"""
        self._emptyArrayCache = {}
        """Cache of empty arrays."""

        self._v_dtype = None
        """The NumPy datatype fopr this table."""
        self.cols = None
        """
        A `Cols` instance that serves as an accessor to `Column` objects.
        """
        self._dirtycache = True
        """Whether the data caches are dirty or not."""

        self._descflavor = None
        """Temporarily keeps the flavor of a description with data."""

        # Initialize this object in case is a new Table

        # Try purely descriptive description objects.
        if new and isinstance(description, dict):
            # Dictionary case
            self.description = Description(description)
        elif new and ( type(description) == type(IsDescription)
                       and issubclass(description, IsDescription) ):
            # IsDescription subclass case
            descr = description()
            self.description = Description(descr.columns)
        elif new and isinstance(description, Description):
            # It is a Description instance already
            self.description = description

        # No description yet?
        if new and self.description is None:
            # Try record array description objects.
            try:
                self._descflavor = flavor = flavor_of(description)
            except TypeError:  # probably not an array
                pass
            else:
                if flavor == 'python':
                    nparray = numpy.rec.array(description)
                else:
                    nparray = array_as_internal(description, flavor)
                self.nrows = nrows = long(nparray.size)
                # If `self._v_recarray` is set, it will be used as the
                # initial buffer.
                if nrows > 0:
                    self._v_recarray = nparray
                fields = self._descrFromRA(nparray)
                self.description = Description(fields)

        # No description yet?
        if new and self.description is None:
            raise TypeError(
                "the ``description`` argument is not of a supported type: "
                "``IsDescription`` subclass, ``Description`` instance, "
                "dictionary, or record array" )

        # Check the chunkshape parameter
        if new and chunkshape is not None:
            if type(chunkshape) in (int, long):
                chunkshape = (long(chunkshape),)
            if type(chunkshape) not in (tuple, list):
                raise ValueError, """\
chunkshape parameter should be an int, tuple or list and you passed a %s.
""" % type(chunkshape)
            elif len(chunkshape) != 1:
                    raise ValueError, """\
the chunkshape (%s) rank must be equal to 1.""" % (chunkshape)
            else:
                self._v_chunkshape = chunkshape
                
        super(Table, self).__init__(parentNode, name, new, filters, _log)


    def _g_postInitHook(self):
        # We are putting here the index-related issues
        # as well as filling general info for table
        # This is needed because we need first the index objects created

        # First, get back the flavor of input data (if any) for
        # `Leaf._g_postInitHook()`.
        self._flavor, self._descflavor = self._descflavor, None
        super(Table, self)._g_postInitHook()

        # Create a cols accessor.
        self.cols = Cols(self, self.description)

        # Place the `Cols` and `Column` objects into `self.colinstances`.
        colinstances, cols = self.colinstances, self.cols
        for colpathname in self.description._v_pathnames:
            colinstances[colpathname] = cols._g_col(colpathname)

        if self._v_new:
            return
        # The following code is only for opened tables.

        # Do the indexes group exist?
        indexesGroupPath = _getIndexTableName(self._v_parent, self._v_name)
        igroup = indexesGroupPath in self._v_file
        for colobj in self.description._f_walk(type="Col"):
            colname = colobj._v_pathname
            # Is this column indexed?
            if igroup:
                indexname = _getIndexColName(
                    self._v_parent, self._v_name, colname)
                indexed = indexname in self._v_file
                self.colindexed[colname] = indexed
                if indexed:
                    column = self.cols._g_col(colname)
                    indexobj = column.index  # to query properties later
                    # Tell the condition cache about dirty indexed columns.
                    if column.dirty:
                        self._conditionCache.nail()
            else:
                indexed = False
                self.colindexed[colname] = False
            if indexed:
                self.indexed = True

        # Create an index properties object.
        # It does not matter to which column 'indexobj' belongs,
        # since their respective index objects share
        # the same filters and number of elements.
        autoindex = getattr( self.attrs, 'AUTOMATIC_INDEX',
                             IndexProps.auto_default )
        reindex = getattr(self.attrs, 'REINDEX', IndexProps.reindex_default)
        if self.indexed:
            filters = indexobj.filters
        else:
            filters = getattr(self.attrs, 'FILTERS_INDEX', None)
        self._indexprops = IndexProps( auto=autoindex, reindex=reindex,
                                       filters=filters )
        if self.indexed:
            self._indexedrows = indexobj.nelements
            self._unsaved_indexedrows = self.nrows - self._indexedrows


    def _restorecache(self):
        # Define a cache for sparse table reads
        self._sparsecache = NumCache(
            shape=(TABLE_MAX_SLOTS, 1),
            itemsize=self.rowsize, name="sparse rows")
        """A cache for row data based on row number."""
        self._limdatacache = ObjectCache(LIMDATA_MAX_SLOTS,
                                         LIMDATA_MAX_SIZE,
                                         'data limits')
        """A cache for data based on search limits and table colum."""
        self._dirtycache = False


    def _getemptyarray(self, dtype):
        # Acts as a cache for empty arrays
        key = dtype
        if key in self._emptyArrayCache:
            return self._emptyArrayCache[key]
        else:
            self._emptyArrayCache[key] = arr = numpy.empty(shape=0, dtype=key)
            return arr


    def _get_container(self, shape):
        "Get the appropriate buffer for data depending on table nestedness."

        # This is *much* faster than the numpy.rec.array counterpart
        return numpy.empty(shape=shape, dtype=self._v_dtype)


    def _newBuffer(self, init=1):
        """Create a new recarray buffer for I/O purposes"""

        recarr = self._get_container(self._v_nrowsinbuf)
        # Initialize the recarray with the defaults in description
        if init:
            for objcol in self.description._f_walk("Col"):
                colname = objcol._v_pathname
                ra = getNestedField(recarr, colname)
                ra[:] = objcol.dflt
        return recarr


    def _descrFromRA(self, recarr):
        """
        Get a description dictionary from a (nested) record array.

        This method is aware of byteswapped record arrays.
        """

        fields = {}
        fbyteorder = '|'
        for (name, (dtype, pos)) in recarr.dtype.fields.items():
            kind = dtype.base.kind
            byteorder = dtype.base.byteorder
            if byteorder in '<>=':
                if fbyteorder not in ['|', byteorder]:
                    raise NotImplementedError(
                        "record arrays with mixed byteorders "
                        "are not supported yet, sorry" )
                fbyteorder = byteorder
            # Non-nested column
            if kind in 'biufSc':
                col = Col.from_dtype(dtype, pos=pos)
            # Nested column
            elif kind == 'V' and dtype.shape in [(), (1,)]:
                col = self._descrFromRA(recarr[name])
                col['_v_pos'] = pos
            else:
                raise NotImplementedError(
                    "record arrays with columns with type description ``%s`` "
                    "are not supported yet, sorry" % dtype )
            fields[name] = col

        # Set the byteorder
        # The columns in records should always have the same byteorder
        # (until different byterorder in columns would be implemented
        # at the HDF5 extension level)
        fields['_v_byteorder'] = byteorders[fbyteorder]
        return fields


    def _getTypeColNames(self, type_):
        """Returns a list containing 'type_' column names."""

        return [ colobj._v_pathname
                 for colobj in self.description._f_walk('Col')
                 if colobj.type == type_ ]


    def _getEnumMap(self):
        """Return mapping from enumerated column names to `Enum` instances."""

        enumMap = {}
        for colobj in self.description._f_walk('Col'):
            if colobj.kind == 'enum':
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
        self._time64colnames = self._getTypeColNames('time64')
        # Find String column names.
        self._strcolnames = self._getTypeColNames('string')
        # Get a mapping of enumerated columns to their `Enum` instances.
        self._colenums = self._getEnumMap()

        # Compute the optimal chunk size
        if self._v_chunkshape is None:
            self._v_chunkshape = self._calc_chunkshape(self._v_expectedrows,
                                                       self.rowsize)
        # Compute the optimal nrowsinbuf
        self._v_nrowsinbuf = self._calc_nrowsinbuf(self._v_chunkshape,
                                                   self.rowsize)

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
        for colobj in self.description._f_walk(type="Col"):
            colname = colobj._v_pathname
            # Get the column dtypes, types and defaults
            self.coldtypes[colname] = colobj.dtype
            self.coltypes[colname] = colobj.type
            self.coldflts[colname] = colobj.dflt
            self.colindexed[colname] = False  # never indexed on creation

        # Attach the FIELD_N_FILL attributes. We write all the level defaults
        i = 0
        setAttr = self._v_attrs._g__setattr
        for colobj in self.description._f_walk(type="Col"):
            fieldname = "FIELD_%s_FILL" % i
            setAttr(fieldname, colobj.dflt)
            i += 1

        # Create index properties attributes by triggering changes in
        # the ``indexprops`` property.
        self.indexprops = IndexProps()

        # Cache _v_dtype for this table
        self._v_dtype = self.description._v_dtype

        return self._v_objectID


    def _g_open(self):
        """Opens a table from disk and read the metadata on it.

        Creates an user description on the flight to easy the access to
        the actual data.

        """
        # Get table info
        self._v_objectID, description, chunksize = self._getInfo()

        # Checking validity names for fields is not necessary
        # when opening a PyTables file
        # Do this nested!
        validate = not self._v_file._isPTFile

        # Create an instance description to host the record fields
        self.description = Description(description, validate=validate)
        getAttr = self._v_attrs.__getattr__
        # Check if there is some "FIELD_0_FILL" attribute
        has_fill_attrs = "FIELD_0_FILL" in self._v_attrs._f_list("sys")
        i = 0
        for objcol in self.description._f_walk(type="Col"):
            colname = objcol._v_pathname
            if has_fill_attrs:
                # Get the default values for each column
                fieldname = "FIELD_%s_FILL" % i
                defval = getAttr(fieldname)
                objcol.dflt = defval
                # Set also the correct value in the desc._v_dflts dictionary
                self.description._v_dflts[colname] = defval
                i += 1

        # The expectedrows would be the actual number
        self._v_expectedrows = self.nrows

        # Extract the coldtypes, coltypes, coldflts
        # self.colnames, coldtypes, col*... should be removed?
        self.colnames = tuple(self.description._v_nestedNames)

        # Find Time64 column names.
        self._time64colnames = self._getTypeColNames('time64')
        # Find String column names.
        self._strcolnames = self._getTypeColNames('string')
        # Get a mapping of enumerated columns to their `Enum` instances.
        self._colenums = self._getEnumMap()

        if chunksize == 0:
            # Not a chunked dataset. Compute the optimal chunkshape
            self._v_chunkshape = self._calc_chunkshape(self.nrows,
                                                       self.rowsize)
        else:
            self._v_chunkshape = (chunksize,)
        # Compute the optimal nrowsinbuf
        self._v_nrowsinbuf = self._calc_nrowsinbuf(self._v_chunkshape,
                                                   self.rowsize)

        # Get info about columns
        for colobj in self.description._f_walk(type="Col"):
            colname = colobj._v_pathname
            # Get the column types, types and defaults
            self.coldtypes[colname] = colobj.dtype
            self.coltypes[colname] = colobj.type
            self.coldflts[colname] = colobj.dflt

        # Assign _v_dtype for this table
        self._v_dtype = self.description._v_dtype

        return self._v_objectID


    def _checkColumn(self, colname):
        """
        Check that the column named `colname` exists in the table.

        If it does not exist, a ``KeyError`` is raised.
        """

        for colobj in self.description._f_walk(type="All"):
            cname = colobj._v_pathname
            if colname == cname:
                break
        if colname <> cname:
            raise KeyError(
                "table ``%s`` does not have a column named ``%s``"
                % (self._v_pathname, colname))
        return colobj


    def _disableIndexingInQueries(self):
        """Force queries not to use indexing.  *Use only for testing.*"""
        if not self._enabledIndexingInQueries:
            return  # already disabled
        # The nail avoids setting/getting splitted conditions in/from
        # the cache where indexing is used.
        self._conditionCache.nail()
        self._enabledIndexingInQueries = False

    def _enableIndexingInQueries(self):
        """Allow queries to use indexing.  *Use only for testing.*"""
        if self._enabledIndexingInQueries:
            return  # already enabled
        self._conditionCache.unnail()
        self._enabledIndexingInQueries = True

    def _requiredExprVars(self, expression, uservars):
        """
        Get the variables required by the `expression`.

        A new dictionary defining the variables used in the `expression`
        is returned.  Required variables are first looked up in the
        `uservars` mapping, then in the set of top-level columns of the
        table.  Unknown variables cause a `NameError` to be raised.

        When `uservars` is `None`, the local and global namespace where
        the API callable which uses this method is called is sought
        instead.  This mechanism will not work as expected if this
        method is not used *directly* from an API callable.  To disable
        this mechanism, just specify a mapping as `uservars`.

        Nested columns and columns from other tables are not allowed
        (`TypeError` and `ValueError` are raised, respectively).  Also,
        non-column variable values are converted to NumPy arrays.
        """
        # Get the names of variables used in the expression.
        if not expression in self._exprvarsCache:
            cexpr = compile(expression, '<string>', 'eval')
            exprvars = [ var for var in cexpr.co_names
                         if var not in ['None', 'False', 'True']
                         and var not in numexpr_functions ]
            self._exprvarsCache[expression] = exprvars
        else:
            exprvars = self._exprvarsCache[expression]

        # Get the local and global variable mappings of the user frame
        # if no mapping has been explicitly given for user variables.
        user_locals, user_globals = {}, {}
        if uservars is None:
            # We use depth 2 to get the frame where the API callable
            # using this method is called.  For instance:
            #
            # * ``table._requiredExprVars()`` (depth 0) is called by
            # * ``table.where()`` (depth 1) is called by
            # * the user (depth 2)
            user_frame = sys._getframe(2)
            user_locals = user_frame.f_locals
            user_globals = user_frame.f_globals

        colinstances = self.colinstances
        tblfile, tblpath = self._v_file, self._v_pathname
        # Look for the required variables first among the ones
        # explicitly provided by the user, then among implicit columns,
        # then among external variables (only if no explicit variables).
        reqvars = {}
        for var in exprvars:
            # Get the value.
            if uservars is not None and var in uservars:
                val = uservars[var]
            elif var in colinstances:
                val = colinstances[var]
            elif uservars is None and var in user_locals:
                val = user_locals[var]
            elif uservars is None and var in user_globals:
                val = user_globals[var]
            else:
                raise NameError("name ``%s`` is not defined" % var)

            # Check the value.
            if hasattr(val, 'pathname'):  # non-nested column
                if val.dtype.shape != ():
                    raise NotImplementedError(
                        "variable ``%s`` refers to "
                        "a multidimensional column, "
                        "not yet supported in conditions, sorry" % var )
                if val._tableFile is not tblfile or val._tablePath != tblpath:
                    raise ValueError( "variable ``%s`` refers to a column "
                                      "which is not part of table ``%s``"
                                      % (var, tblpath) )
                if val.dtype.str[1:] == 'u8':
                    raise NotImplementedError(
                        "variable ``%s`` refers to "
                        "a 64-bit unsigned integer column, "
                        "not yet supported in conditions, sorry; "
                        "please use regular Python selections" % var )
            elif hasattr(val, '_v_colpathnames'):  # nested column
                raise TypeError(
                    "variable ``%s`` refers to a nested column, "
                    "not allowed in conditions" % var )
            else:  # only non-column values are converted to arrays
                val = numpy.asarray(val)
            reqvars[var] = val
        return reqvars

    def _getConditionKey(self, condition, condvars):
        """
        Get the condition cache key for `condition` with `condvars`.

        Currently, the key is a tuple of `condition`, column variables
        names, normal variables names, column paths and variable paths
        (all are tuples).
        """

        # Variable names for column and normal variables.
        colnames, varnames = [], []
        # Column paths and types for each of the previous variable.
        colpaths, vartypes = [], []
        for (var, val) in condvars.items():
            if hasattr(val, 'pathname'):  # column
                colnames.append(var)
                colpaths.append(val.pathname)
            else:  # array
                assert hasattr(val, '__array_struct__')
                try:
                    varnames.append(var)
                    vartypes.append(numexpr_getType(val))  # expensive
                except ValueError:
                    # This is more clear than the error given by Numexpr.
                    raise TypeError( "variable ``%s`` has data type ``%s``, "
                                     "not allowed in conditions"
                                     % (var, val.dtype.name) )
        colnames, varnames = tuple(colnames), tuple(varnames)
        colpaths, vartypes = tuple(colpaths), tuple(vartypes)
        condkey = (condition, colnames, varnames, colpaths, vartypes)
        return condkey

    def _splitCondition(self, condition, condvars):
        """
        Split the `condition` into indexable and non-indexable parts.

        This method returns an instance of ``SplittedCondition``.  See
        the ``split_condition()`` function in the ``conditions`` module
        for more information about the splitting process.

        This method makes use of the condition cache when possible.
        """

        # Look up the condition in the condition cache.
        condcache = self._conditionCache
        condkey = self._getConditionKey(condition, condvars)
        splitted = condcache.get(condkey)
        if splitted:
            return splitted.with_replaced_vars(condvars)  # bingo!

        # Bad luck, the condition must be parsed and splitted.
        # Fortunately, the key provides some valuable information. ;)
        (condition, colnames, varnames, colpaths, vartypes) = condkey

        # Extract types from *all* the given variables.
        typemap = dict(zip(varnames, vartypes))  # start with normal variables
        for colname in colnames:  # then add types of columns
            coltype = condvars[colname].dtype.type
            typemap[colname] = _nxTypeFromNPType[coltype]

        # Get the set of columns with usable indexes.
        def can_use_index(column):
            if not self._enabledIndexingInQueries:
                return False  # looks like testing in-kernel searches
            return self.colindexed[column.pathname] and not column.dirty
        indexedcols = frozenset(
            colname for colname in colnames
            if can_use_index(condvars[colname]) )

        # Now let ``split_condition()`` do the Numexpr-related job.
        splitted = split_condition(condition, typemap, indexedcols)

        # Check that there actually are columns in the condition.
        resparams = splitted.residual_parameters
        if ( not splitted.index_variable
             and not set(resparams).intersection(set(colnames)) ):
            raise ValueError( "there are no columns taking part "
                              "in condition ``%s``" % (condition,) )

        # Store the splitted condition in the cache and return it.
        condcache[condkey] = splitted
        return splitted.with_replaced_vars(condvars)


    def willQueryUseIndexing(self, condition, condvars=None):
        """
        Will a query for the `condition` use indexing?

        The meaning of the `condition` and `condvars` arguments is the
        same as in the `self.where()` method.  If the `condition` can
        use indexing, this method returns the path name of the column
        whose index is usable.  Otherwise, it returns `None`.

        This method is mainly intended for testing.  Keep in mind that
        changing the set of indexed columns or their dirtyness may make
        this method return different values for the same arguments at
        different times.
        """
        # Split the condition into indexable and residual parts.
        condvars = self._requiredExprVars(condition, condvars)
        splitted = self._splitCondition(condition, condvars)
        if not splitted.index_variable:
            return None
        return condvars[splitted.index_variable].pathname


    def where( self, condition, condvars=None,
               start=None, stop=None, step=None ):
        """
        Iterate over values fulfilling a `condition`.

        This method returns an iterator yielding `Row` instances built
        from rows in the table that satisfy the given `condition` (an
        expression-like string).

        The `condvars` mapping may be used to define the variable names
        appearing in the `condition`.  `condvars` should consist of
        identifier-like strings pointing to `Column` instances *of this
        table*, or to other values (which will be converted to arrays).

        When `condvars` is not provided or `None`, the current local and
        global namespace is sought instead of `condvars`.  The previous
        mechanism is mostly intended for interactive usage.  To disable
        it, just specify a (maybe empty) mapping as `condvars`.

        A default set of condition variables is always provided where
        each top-level column with an identifier-like name appears.
        Only variables in `condvars` can override the default variables.

        If a range is supplied (by setting some of the `start`, `stop`
        or `step` parameters), only the rows in that range *and*
        fullfilling the `condition` are used.  The meaning of the
        `start`, `stop` and `step` parameters is the same as in the
        ``range()`` Python function, except that negative values of
        `step` are *not* allowed.  Moreover, if only `start` is
        specified, then `stop` will be set to ``start+1``.

        When possible, indexed columns participating in the condition
        will be used to speed up the search.  It is recommended that you
        place the indexed columns as left and out in the condition as
        possible.  Anyway, this method has always better performance
        than standard Python selections on the table.

        You can mix this method with standard Python selections in order
        to have complex queries.  It is strongly recommended that you
        pass the most restrictive condition as the parameter to this
        method if you want to achieve maximum performance.

        Example of use:

        >>> passvalues = [ row['col3'] for row in
        ...                table.where('(col1 > 0) & (col2 <= 20)', step=5)
        ...                if your_function(row['col2']) ]
        >>> print \"Values that pass the cuts:\", passvalues

        Note that, from PyTables 1.1 on, you can nest several iterators
        over the same table.  For example:

        >>> for p in rout.where('pressure < 16'):
        ...   for q in rout.where('pressure < 9'):
        ...     for n in rout.where('energy < 10'):
        ...       print \"pressure, energy:\", p['pressure'], n['energy']

        In this example, iterators returned by ``where()`` have been
        used, but you may as well use any of the other reading iterators
        that ``Table`` objects offer.  See ``examples/nested-iter.py``
        for the full code.
        """
        # Split the condition into indexable and residual parts.
        condvars = self._requiredExprVars(condition, condvars)
        splitted = self._splitCondition(condition, condvars)
        return self._where(splitted, condvars, start, stop, step)

    def _where( self, splitted, condvars,
                start=None, stop=None, step=None ):
        """
        Low-level counterpart of `self.where()`.

        This version needs the condition to already be `splitted`.  It
        also uses `condvars` as is.  This is on purpose; if you want
        default variables and the like, use `self._requiredExprVars()`.
        """

        # Set the index column and residual condition (if any)
        # for the ``Row`` iterator.
        idxvar = splitted.index_variable
        if idxvar:
            idxcol = condvars[idxvar]
            index = idxcol.index
            assert index is not None, "the chosen column is not indexed"
            assert not idxcol.dirty, "the chosen column has a dirty index"
            self._whereIndex = idxcol.pathname
        rescond = splitted.residual_function
        if rescond:
            resparams = splitted.residual_parameters
            resargs = [condvars[param] for param in resparams]
            self._whereCondition = (rescond, resargs)

        # Get the number of rows that the indexed condition yields.
        # This also signals ``Row`` whether to use indexing or not.
        ncoords = -1  # do not use indexing by default
        if idxvar:
            range_ = index.getLookupRange(
                splitted.index_operators, splitted.index_limits, self )
            ncoords = index.search(range_)  # do use indexing (always >= 0)
            if ncoords == 0 and not rescond:
                # No values neither from index nor from residual condition.
                self._whereIndex = self._whereCondition = None
                return iter([])

        # Adjust the slice to be used.
        (start, stop, step) = processRangeRead(self.nrows, start, stop, step)
        if start >= stop:  # empty range, reset conditions
            self._whereIndex = self._whereCondition = None
            return iter([])

        # Iterate according to the index and residual conditions.
        row = TableExtension.Row(self)
        return row(start, stop, step, coords=None, ncoords=ncoords)


    def _checkFieldIfNumeric(self, field):
        """Check that `field` has been selected with ``numeric`` flavor."""
        if self.flavor == 'numeric' and field is None:
            raise ValueError(
                "Numeric does not support heterogeneous datasets; "
                "you must specify a field when using the ``numeric`` flavor" )


    def readWhere(self, condition, condvars=None, field=None):
        """
        Read table data fulfilling the given `condition`.

        This method is similar to `self.read()`, having their common
        arguments and return values the same meanings.  However, only
        the rows fulfilling the `condition` are included in the result.

        The meaning of the `condition` and `condvars` arguments is the
        same as in the `self.where()` method.
        """
        self._checkFieldIfNumeric(field)

        # Split the condition into indexable and residual parts.
        condvars = self._requiredExprVars(condition, condvars)
        splitted = self._splitCondition(condition, condvars)

        idxvar = splitted.index_variable
        if not idxvar:
            coords = [p.nrow for p in self._where(splitted, condvars)]
            self._whereCondition = None  # reset the conditions
            return self.readCoordinates(coords, field)

        # Retrieve the array of rows fulfilling the index condition.

        column = condvars[idxvar]
        index = column.index
        assert index is not None, "the chosen column is not indexed"
        assert not column.dirty, "the chosen column has a dirty index"

        # Clean the cache if needed
        if self._dirtycache:
            self._restorecache()

        # Get the coordinates to lookup
        range_ = index.getLookupRange(
            splitted.index_operators, splitted.index_limits, self )

        # Check whether the array is in the limdata cache or not.
        key = (column.name, range_)
        nslot = self._limdatacache.getslot(key)
        if nslot >= 0:
            # Cache hit. Use the array kept there.
            recarr = self._limdatacache.getitem(nslot)
            nrecords = len(recarr)
        else:
            # No luck with cached data. Proceed with the regular search.
            nrecords = index.search(range_)
            # Create a buffer and read the values in.
            recarr = self._get_container(nrecords)
            if nrecords > 0:
                coords = index.indices._getCoords(index, 0, nrecords)
                recout = self._read_elements(recarr, coords)
            # Put this recarray in limdata cache.
            size = len(recarr) * self.rowsize + 1  # approx. size of array
            self._limdatacache.setitem(key, recarr, size)

        # Filter out rows not fulfilling the residual condition.
        rescond = splitted.residual_function
        if rescond and nrecords > 0:
            indexValid = call_on_recarr(
                rescond, splitted.residual_parameters,
                recarr, param2arg=condvars.__getitem__ )
            recarr = recarr[indexValid]

        if field:
            recarr = getNestedField(recarr, field)
        return internal_to_flavor(recarr, self.flavor)


    def whereAppend( self, dstTable, condition, condvars=None,
                     start=None, stop=None, step=None ):
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

        # Split the condition into indexable and residual parts.
        condvars = self._requiredExprVars(condition, condvars)
        splitted = self._splitCondition(condition, condvars)

        # Row objects do not support nested columns, so we must iterate
        # over the flat column paths.  When rows support nesting,
        # ``self.colnames`` can be directly iterated upon.
        colNames = [colName for colName in flattenNames(self.colnames)]
        dstRow = dstTable.row
        nrows = 0
        for srcRow in self._where(splitted, condvars, start, stop, step):
            for colName in colNames:
                dstRow[colName] = srcRow[colName]
            dstRow.append()
            nrows += 1
        dstTable.flush()
        return nrows


    def getWhereList(self, condition, condvars=None, sort=False):
        """
        Get the row coordinates fulfilling the given `condition`.

        The coordinates are returned as a list of the current flavor.

        The meaning of the `condition` and `condvars` arguments is the
        same as in the `self.where()` method.

        `sort` means that you want to retrieve the coordinates ordered.
        The default is to not sort them.
        """

        # Split the condition into indexable and residual parts.
        condvars = self._requiredExprVars(condition, condvars)
        splitted = self._splitCondition(condition, condvars)

        # Take advantage of indexation, if present
        idxvar = splitted.index_variable
        if idxvar is not None:
            column = condvars[idxvar]
            index = column.index
            assert index is not None, "the chosen column is not indexed"
            assert not column.dirty, "the chosen column has a dirty index"

            # get the number of coords and set-up internal variables
            range_ = index.getLookupRange(
                splitted.index_operators, splitted.index_limits, self )
            ncoords = index.search(range_)
            if ncoords > 0:
                coords = index.indices._getCoords_sparse(index, ncoords)
                # Get a copy of the internal buffer to handle it to the user
                coords = coords.copy()
            else:
                #coords = numpy.empty(type=numpy.int64, shape=0)
                coords = self._getemptyarray("int64")

            # Filter out rows not fulfilling the residual condition.
            rescond = splitted.residual_function
            if rescond and ncoords > 0:
                indexValid = call_on_recarr(
                    rescond, splitted.residual_parameters,
                    recarr=self._readCoordinates(coords),
                    param2arg=condvars.__getitem__ )
                coords = coords[indexValid]
                ncoords = len(coords)
        else:
            coords = [p.nrow for p in self._where(splitted, condvars)]
            coords = numpy.array(coords, dtype=numpy.int64)
            # Reset the conditions
            self._whereCondition = None
        if sort:
            coords = numpy.sort(coords)
        return internal_to_flavor(coords, self.flavor)


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

        coords = numpy.asarray(sequence, dtype=numpy.int64)
        # That might allow the retrieving on a sequential order
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
            if field not in self.coldtypes:
                if field in self.description._v_names:
                    # Remember to select this field
                    select_field = field
                    field = None
                else:
                    raise KeyError, "Field %s not found in table %s" % \
                          (field, self)
            else:
                # The column hangs directly from the top
                dtypeField = self.coldtypes[field]
                typeField = self.coltypes[field]

        # Return a rank-0 array if start > stop
        if start >= stop:
            if field == None:
                nra = self._get_container(0)
                return nra
            return numpy.empty(shape=0, dtype=dtypeField)

        if coords is None:
            nrows = len(xrange(start, stop, step))
        else:
            # We should test for stop and start values as well
            nrows = coords.size

        # Compute the shape of the resulting column object
        if field and coords is None:  # coords handling expects a RecArray
            # Create a container for the results
            result = numpy.empty(shape=nrows, dtype=dtypeField)
        else:
            # Recarray case
            result = self._get_container(nrows)

        # Handle coordinates separately.
        if coords is not None:
            if coords.size > 0:
                self._read_elements(result, coords)
            if field:
                # result is always a RecArray
                return result[field]
            return result

        # Call the routine to fill-up the resulting array
        if step == 1 and not field:
            # This optimization works three times faster than
            # the row._fillCol method (up to 170 MB/s on a pentium IV @ 2GHz)
            self._read_records(start, stop-start, result)
        # Warning!: _read_field_name should not be used until
        # H5TBread_fields_name in TableExtension will be finished
        # F. Altet 2005/05/26
        # XYX Ho implementem per a PyTables 2.0??
        elif field and step > 15 and 0:
            # For step>15, this seems to work always faster than row._fillCol.
            self._read_field_name(result, start, stop, step, field)
        else:
            self.row._fillCol(result, start, stop, step, field)

        if select_field:
            return result[select_field]
        else:
            return result


    def read(self, start=None, stop=None, step=None,
             field=None, coords = None):
        """Read a range of rows and return an in-memory object.

        If `start`, `stop`, or `step` parameters are supplied, a row
        range is selected. If `field` is specified, only this `field` is
        returned as an array of the current flavor. If `field` is not
        supplied all the fields are selected and a record array of the
        current flavor is returned.

        Nested fields can be specified in the `field` parameter by
        using a '/' as a separator between fields (e.g. 'Info/value').

        If `coords` is specified, only the indices in `coords` that
        are in the range of (`start`, `stop`) are returned. Also, if
        `coords` is specified, step only can be assigned to be 1,
        otherwise an error is issued.

        """

        if field:
            self._checkColumn(field)
        else:
            self._checkFieldIfNumeric(field)

        (start, stop, step) = processRangeRead(self.nrows, start, stop, step)

        if coords is not None:
            # Check step value.
            if coords.size and step != 1:
                raise NotImplementedError("""\
``step`` must be 1 when the ``coords`` parameter is specified""")
            # Turn coords into an array of 64-bit indexes,
            # as expected by _read().
            if not (type(coords) == numpy.ndarray and
                    coords.dtype.type == numpy.int64):
                coords = numpy.asarray(coords, dtype=numpy.int64)

        arr = self._read(start, stop, step, field, coords)
        return internal_to_flavor(arr, self.flavor)


    def _readCoordinates(self, coords, field=None):
        """Private part of `readCoordinates()` with no flavor conversion."""

        ncoords = len(coords)
        # Create a read buffer only if needed
        if field is None or ncoords > 0:
            # Doing a copy is faster when ncoords is small (<1000)
            if ncoords < min(1000, self._v_nrowsinbuf):
                result = self._v_rbuffer[:ncoords].copy()
            else:
                result = self._get_container(ncoords)

        # Do the real read
        if ncoords > 0:
            # Turn coords into an array of 64-bit indexes, if necessary
            if not (type(coords) is numpy.ndarray and
                    coords.dtype.type is numpy.int64):
                coords = numpy.asarray(coords, dtype=numpy.int64)
            self._read_elements(result, coords)

        # Do the final conversions, if needed
        if field:
            if ncoords > 0:
                result = getNestedField(result, field)
            else:
                # Get an empty array from the cache
                result = self._getemptyarray(self.coldtypes[field])
        return result

    def readCoordinates(self, coords, field=None):
        """
        Read a set of rows given their indexes into an in-memory object.

        This method works much like the `read()` method, but it uses a
        sequence (`coords`) of row indexes to select the wanted columns,
        instead of a column range.

        It returns the selected rows in a record array of the current
        flavor.
        """
        self._checkFieldIfNumeric(field)
        result = self._readCoordinates(coords, field)
        return internal_to_flavor(result, self.flavor)


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
        returned as a ``numpy`` object or as a ``numarray`` object
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

        If the `key` argument is an integer, the corresponding table row
        is returned as a record of the current flavor.  If `key` is a
        slice, the range of rows determined by it is returned as a
        record array of the current flavor.

        Using a string as `key` to get a column is supported but deprecated.
        Please use the `col()` method.

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
            # Reverted to return a numpy.void in order
            # to support better the nested datatypes
            # return self.tolist(self.read(start, stop, step)[0])
            return self.read(start, stop, step)[0]
        elif isinstance(key, slice):
            (start, stop, step) = processRange(self.nrows,
                                               key.start, key.stop, key.step)
            return self.read(start, stop, step)
        elif isinstance(key, str):
            warnings.warn( "``table['colname']`` is deprecated; "
                           "please use ``table.col('colname')``",
                           DeprecationWarning, stacklevel=2 )
            return self.col(key)
        else:
            raise TypeError("invalid index or slice: %r" % (key,))


    def __setitem__(self, key, value):
        """Sets a table row or table slice.

        It takes different actions depending on the type of the `key`
        parameter: if it is an integer, the corresponding table row is
        set to `value` (a record, list or tuple capable of being
        converted to the table field format).  If the `key` is a slice,
        the row slice determined by it is set to `value` (a NumPy record
        array, ``NestedRecArray`` or list of rows).

        Example of use::

            # Modify just one existing row
            table[2] = [456,'db2',1.2]
            # Modify two existing rows
            rows = numpy.rec.array([[457,'db1',1.2],[6,'de2',1.3]],
                                   formats="i4,a3,f8")
            table[1:3:2] = rows

        Which is equivalent to::

            table.modifyRows(start=2, rows=[456,'db2',1.2])
            rows = numpy.rec.array([[457,'db1',1.2],[6,'de2',1.3]],
                                   formats="i4,a3,f8")
            table.modifyRows(start=1, stop=3, step=2, rows=rows)

        """

        self._v_file._checkWritable()

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

        rows can be either a recarray (both numpy and numarray flavors
        are supported) or a structure that is able to be converted to a
        recarray compliant with the table format.

        It raises an 'ValueError' in case the rows parameter could not
        be converted to an object compliant with table description.

        """

        self._v_file._checkWritable()

        if not self._chunked:
            raise HDF5ExtError("""\
You cannot append rows to a non-chunked table.""")

        # Try to convert the object into a recarray compliant with table
        try:
            iflavor = flavor_of(rows)
            if iflavor != 'python':
                rows = array_as_internal(rows, iflavor)
            # Works for Python structures and always copies the original,
            # so the resulting object is safe for in-place conversion.
            recarray = numpy.rec.array(rows, dtype=self._v_dtype)
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
        # Compute the number of rows to read.
        nrows = len(xrange(start, stop, step))
        if len(rows) < nrows:
            raise ValueError, \
           "The value has not enough elements to fill-in the specified range"
        # Try to convert the object into a recarray
        try:
            iflavor = flavor_of(rows)
            if iflavor != 'python':
                rows = array_as_internal(rows, iflavor)
            if hasattr(rows, "shape") and rows.shape == ():
                # To allow conversion of scalars (void type) into arrays.
                # See http://projects.scipy.org/scipy/numpy/ticket/315
                # for discussion on how to pass buffers to constructors
                # See also http://projects.scipy.org/scipy/numpy/ticket/348
                recarray = numpy.array([rows], dtype=self._v_dtype)
            else:
                # Works for Python structures and always copies the original,
                # so the resulting object is safe for in-place conversion.
                recarray = numpy.rec.array(rows, dtype=self._v_dtype)
        except Exception, exc:  #XXX
            raise ValueError, \
"""rows parameter cannot be converted into a recarray object compliant with
table format '%s'. The error was: <%s>
""" % (self.description._v_nestedDescr, exc)
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
        """Modify one single column in the row slice [start:stop:step].

        `column` can be either a NumPy record array, ``NestedRecArray``,
        ``RecArray``, NumPy array, ``NumArray``, list or tuple that is
        able to be converted into a record array compliant with the
        specified `colname` column of the table.

        `colname` specifies the column name of the table to be modified.

        Returns the number of modified rows.

        It raises a ``ValueError`` in case the columns parameter could
        not be converted into an object compliant with the column
        description.

        It raises an ``IndexError`` in case the modification will exceed
        the length of the table.

        """

        if not isinstance(colname, str):
            raise TypeError("The 'colname' parameter must be a string.")
        self._v_file._checkWritable()

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
        descr = [objcol._v_parent._v_nestedDescr[objcol._v_pos]]
        # Try to convert the column object into a recarray
        try:
            # Make sure the result is always a *copy* of the original,
            # so the resulting object is safe for in-place conversion.
            iflavor = flavor_of(column)
            if iflavor != 'python':
                column = array_as_internal(column, iflavor)
                recarray = numpy.rec.array(column, dtype=descr)
            else:
                recarray = numpy.rec.fromarrays([column], dtype=descr)
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
        # Compute the number of rows to read.
        nrows = len(xrange(start, stop, step))
        if len(recarray) < nrows:
            raise ValueError, \
                  "The value has not enough elements to fill-in the specified range"
        # Now, read the original values:
        mod_recarr = self._read(start, stop, step)
        # Modify the appropriate column in the original recarray
        mod_recarr[colname] = recarray[colname]
        # save this modified rows in table
        self._update_records(start, stop, step, mod_recarr)
        # Redo the index if needed
        self._reIndex(colname)

        return nrows


    def modifyColumns(self, start=None, stop=None, step=1,
                      columns=None, names=None):
        """Modify a series of columns in the row slice [start:stop:step].

        `column` can be either a NumPy record array, ``NestedRecArray``,
        ``RecArray``, NumPy array, ``NumArray``, list or tuple that is
        able to be converted into a record array compliant with the
        specified `colname` column of the table.

        `columns` can be either a NumPy record array,
        ``NestedRecArray``, ``RecArray``, list of arrays or list of
        tuples (the columns) that is able to be converted into a record
        array compliant with the format of the specified column `names`
        in the table.

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
        descr = []
        for colname in names:
            objcol = self._checkColumn(colname)
            descr.append(objcol._v_parent._v_nestedDescr[objcol._v_pos])
            #descr.append(objcol._v_parent._v_dtype[objcol._v_pos])
        # Try to convert the columns object into a recarray
        try:
            # Make sure the result is always a *copy* of the original,
            # so the resulting object is safe for in-place conversion.
            iflavor = flavor_of(columns)
            if iflavor != 'python':
                columns = array_as_internal(columns, iflavor)
                recarray = numpy.rec.array(columns, dtype=descr)
            else:
                recarray = numpy.rec.fromarrays(columns, dtype=descr)
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
        # Compute the number of rows to read.
        nrows = len(xrange(start, stop, step))
        if len(recarray) < nrows:
            raise ValueError, \
           "The value has not enough elements to fill-in the specified range"
        # Now, read the original values:
        mod_recarr = self._read(start, stop, step)
        # Modify the appropriate columns in the original recarray
        for name in recarray.dtype.names:
            mod_recarr[name] = recarray[name]
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
                    col = self.cols._g_col(colname)
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
        index = self.cols._g_col(colname).index
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
        # index the remaining rows in last row
        nremain = nrows - indexedrows
        if lastrow and nremain > 0:
            remainvalues = self._read(indexedrows, self.nrows, 1, colname)
            index.appendLastRow(remainvalues, self.nrows)
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


    def _setColumnIndexing(self, colpathname, indexed):
        """Mark the referred column as indexed or non-indexed."""

        colindexed = self.colindexed
        isindexed, wasindexed = bool(indexed), colindexed[colpathname]
        if isindexed == wasindexed:
            return  # indexing state is unchanged

        # Changing the set of indexed columns
        # invalidates the condition cache.
        self._conditionCache.clear()
        colindexed[colpathname] = isindexed
        self.indexed = max(colindexed.values())  # this is an OR :)


    def _reIndex(self, colnames):
        """ re-index all the columns in colnames is self.reindex is true """

        if self.indexed:
            # Mark the proper indexes as dirty
            for (colname, colindexed) in self.colindexed.iteritems():
                if colindexed and colname in colnames:
                    col = self.cols._g_col(colname)
                    col.dirty = True
            # Now, re-index the dirty ones
            if self.indexprops.reindex:
                self.reIndex()


    def reIndex(self):
        """Recompute the existing indexes in table"""
        for (colname, colindexed) in self.colindexed.iteritems():
            if colindexed:
                indexcol = self.cols._g_col(colname)
                indexedrows = indexcol.reIndex()
        # Update counters
        self._indexedrows = indexedrows
        self._unsaved_indexedrows = self.nrows - indexedrows
        return indexedrows


    def reIndexDirty(self):
        """Recompute the existing indexes in table if they are dirty"""
        for (colname, colindexed) in self.colindexed.iteritems():
            if colindexed:
                indexcol = self.cols._g_col(colname)
                indexedrows = indexcol.reIndexDirty()
        # Update counters
        self._indexedrows = indexedrows
        self._unsaved_indexedrows = self.nrows - indexedrows
        return indexedrows


    def _g_copyRows(self, object, start, stop, step):
        "Copy rows from self to object"
        (start, stop, step) = processRangeRead(self.nrows, start, stop, step)
        nrowsinbuf = self._v_nrowsinbuf
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

    def _g_copyIndexes(self, other):
        """Generate index in `other` table for every indexed column here."""
        oldcols, newcols = self.colinstances, other.colinstances
        for colname in newcols:
            oldcolindex = oldcols[colname].index
            if oldcolindex:
                optlevel = oldcolindex.optlevel
                testmode = oldcolindex.testmode
                newcol = newcols[colname]
                newcol.createIndex(optlevel=optlevel, testmode=testmode)


    def _g_copyWithStats(self, group, name, start, stop, step,
                         title, filters, _log):
        "Private part of Leaf.copy() for each kind of leaf"
        # Create the new table and copy the selected data.
        newtable = Table( group, name, self.description, title=title,
                          filters=filters, expectedrows=self.nrows,
                          _log=_log )
        self._g_copyRows(newtable, start, stop, step)
        nbytes = newtable.nrows * newtable.rowsize
        # Generate equivalent indexes in the new table, if any.
        newtable.indexprops = copy.copy(self.indexprops)
        if self.indexed:
            warnings.warn(
                "generating indexes for destination table ``%s:%s``; "
                "please be patient"
                % (newtable._v_file.filename, newtable._v_pathname) )
            self._g_copyIndexes(newtable)

        return (newtable, nbytes)


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
#                     col = self.cols._g_col(colname)
#                     if nrows > 0 and not col.dirty:
#                         print "*optimizing col-->", colname
#                         col.index.optimize()
# #***************************** end test ************************************

        self._g_cleanIOBuf()
        super(Table, self).flush()


    def _g_preKillHook(self):
        """Code to be called before killing the node."""

        # Flush the buffers before to clean-up them
        #self.flush()
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
        # I've added a Performance warning in order to compel the user to
        # call self.flush() in case the tables is being preempted before doing it.
        # F. Altet 2006-08-03
        if (self._unsaved_nrows > 0 or (self.indexed and
                                        self.indexprops.auto and
                                        self._unsaved_indexedrows > 0)):
            warnings.warn("""\
table ``%s`` is being preempted from alive nodes without its buffers being flushed. This may lead to very ineficient use of resources and even to fatal errors in certain situations. Please, do a call to the .flush() method on this table before start using other nodes."""
                          % (self._v_pathname),
                          PerformanceWarning)
        return


    def _f_close(self, flush=True):
        if not self._v_isopen:
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
  byteorder := %r
  indexprops := %r""" % \
        (str(self), self.description, self.byteorder, self.indexprops)
        else:
            return "%s\n  description := %r\n  byteorder := %r\n" % \
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
        # Bound the index table group because it will be referenced
        # quite a lot when populating the attrs with column objects.
        try:
            itgroup = table._v_file._getNode(
                _getIndexTableName(table._v_parent, table.name))
        except NodeError:
            if table._v_new and table.indexed:
                # The indexes group for table does not exist, create it
                itgroup = table._createIndexesTable(table._v_parent)
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

        return self._g_col(colname)

    def _g_col(self, colname):
        """Like `self._f_col()` but it does not check arguments."""
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
        type row is returned as a record of the current flavor.  If
        `key` is a slice, the range of rows determined by it is returned
        as a record array of the current flavor.

        Using a string as `key` to get a column is supported but deprecated.
        Please use the `_f_col()` method.

        Example of use::

            record = table.cols[4]  # equivalent to table[4]
            recarray = table.cols.Info[4:1000:2]

        Those statements are equivalent to::

            nrecord = table.read(start=4)[0]
            nrecarray = table.read(start=4, stop=1000, step=2)['Info']

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
                return crecord[colgroup]
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
                    return getNestedField(crecarray, colgroup)  # numpy case
        elif isinstance(key, str):
            warnings.warn( "``table.cols['colname']`` is deprecated; "
                           "please use ``table.cols._f_col('colname')``",
                           DeprecationWarning, stacklevel=2 )
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
            colobj = self._g_col(col)
            if isinstance(colobj, Column):
                colobj.close()
                # Delete the reference to column
                del self.__dict__[col]
            else:
                colobj._f_close()

        self.__dict__.clear()


    def __str__(self):
        """The string representation for this object."""
        # The pathname
        tablepathname = self._v__tablePath
        descpathname = self._v_desc._v_pathname
        if descpathname:
            descpathname = "."+descpathname
        # Get this class name
        classname = self.__class__.__name__
        # The number of columns
        ncols = len(self._v_colnames)
        return "%s.cols%s (%s), %s columns" % \
               (tablepathname, descpathname, classname, ncols)


    def __repr__(self):
        """A detailed string representation for this object."""

        out = str(self) + "\n"
        for name in self._v_colnames:
            # Get this class name
            classname = getattr(self, name).__class__.__name__
            # The type
            if name in self._v_desc._v_dtypes:
                tcol = self._v_desc._v_dtypes[name]
                # The shape for this column
                shape = self._v_desc._v_dtypes[name].shape
            else:
                tcol = "Description"
                # Description doesn't have a shape currently
                shape = ()
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
        type -- the PyTables type of the column
        dtype -- the NumPy data type of the column
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
        self.dtype = descr._v_dtypes[name]
        self.type = descr._v_types[name]
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
                dirty = getattr(index._v_attrs, "DIRTY")
                # Turn numbers into logical values
                if dirty:
                    dirty = True
                else:
                    dirty = False
                self._dirty = dirty
                return dirty
            else:
                # If don't have a DIRTY attribute, index should be clean
                self._dirty = False
                return False
        else:
            self._dirty = True  # If don't have index, this is like dirty
            return True


    def _set_dirty(self, dirty):
        wasdirty = getattr(self, "_dirty", False)
        self._dirty = isdirty = dirty
        index = self.index
        # Only set the index column as dirty if it exists
        if index:
            setattr(index._v_attrs, "DIRTY", dirty)
            if dirty:
                index.nelementsLR = 0
                index.nelements = 0
        # If an *actual* change in dirtiness happens,
        # notify the condition cache by setting or removing a nail.
        if index and not wasdirty and isdirty:
            self.table._conditionCache.nail()
        if index and wasdirty and not isdirty:
            self.table._conditionCache.unnail()

    # Define a property.  The 'delete this attribute'
    # method is defined as None, so the attribute can't be deleted.
    dirty = property(_get_dirty, _set_dirty, None, "Column dirtiness")


    def __len__(self):
        return self.table.nrows


    def __getitem__(self, key):
        """Returns a column element or slice

        It takes different actions depending on the type of the 'key'
        parameter:

        If 'key' is an integer, the corresponding element in the column is
        returned as a NumPy or scalar object, depending on its shape. If 'key'
        is a slice, the row slice determined by this slice is returned as a
        NumPy object.

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

        If 'key' is an integer, the corresponding element in the column is set
        to 'value' (scalar or NumPy, depending on column's shape). If 'key' is
        a slice, the row slice determined by 'key' is set to 'value' (a NumPy
        or list of elements).

        """

        table = self.table
        table._v_file._checkWritable()

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


    def createIndex(self, optlevel=5, warn=True, testmode=False,
                    verbose=False):
        """Create an index for this column.

        optlevel -- The default level of optimization for the index.
        """

        name = self.name
        table = self.table
        tableName = table._v_name
        tableParent = table._v_parent
        dtype = self.dtype
        descr = self.descr
        index = self.index
        getNode = table._v_file._getNode

        # Warn if the index already exists
        if index:
            raise ValueError, \
"%s for column '%s' already exists. If you want to re-create it, please, try with reIndex() method better" % (str(index), str(self.pathname))

        # Check that the datatype is indexable.
        if dtype.str[1:] == 'u8':
            raise NotImplementedError(
                "indexing 64-bit unsigned integer columns "
                "is not supported yet, sorry" )
        if dtype.kind == 'c':
            raise TypeError("complex columns can not be indexed")
        if dtype.shape != ():
            raise TypeError("multidimensional columns can not be indexed")

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
                    idgroup = getNode('%s/%s' % (itgroup._v_pathname, dname))
                except NodeError:
                    idgroup = table._createIndexesDescr(
                        idgroup, dname, iname, filters)

        # Create the atom
        assert dtype.shape == ()
        atom = Atom.from_dtype(numpy.dtype((dtype, (0,))))

        # Create the index itself
        index = Index(
            idgroup, name, atom=atom, column=self,
            title="Index for %s column" % name,
            filters=filters,
            optlevel=optlevel,
            testmode=testmode,
            expectedrows=table._v_expectedrows)
        self._updateIndexLocation(index)

        table._setColumnIndexing(self.pathname, True)

        # Feed the index with values
        slicesize = index.slicesize
        # Add rows to the index if necessary
        if table.nrows > 0:
            indexedrows = table._addRowsToIndex(
                self.pathname, 0, table.nrows, lastrow=True )
        else:
            indexedrows = 0
        # Optimize indexes with computed parameters for optimising
        # (i.e. we should not pass the optlevel parameter here!)
        index.optimize(verbose=verbose)
        self.dirty = False
        table._indexedrows = indexedrows
        table._unsaved_indexedrows = table.nrows - indexedrows
        return indexedrows


    def optimizeIndex(self, level=9, verbose=0):
        """Optimize the index for this column.

        `level` is the level optimization (from 0 to 9).
        """

        if type(level) not in (int, long) or level < 0 or level > 9:
            raise ValueError, "Optimization level should be in the range 0-9."
        if not self.index:
            warnings.warn("""\
column '%s' is not indexed, so it can't be optimized."""
                          % (self.pathname), UserWarning)
            return
        if level > 0:
            self.index.optimize(level, verbose)


    def reIndex(self):
        """Recompute the existing index"""

        self._tableFile._checkWritable()

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

        self._tableFile._checkWritable()

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
        """
        Remove the index associated with this column.

        If the column is not indexed, nothing happens.  The index can be
        created again by calling the `self.createIndex()` method.
        """

        self._tableFile._checkWritable()

        # Remove the index if existing.
        index = self.index
        if index:
            index._f_remove()
            self._updateIndexLocation(None)
            self.table._setColumnIndexing(self.pathname, False)


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
        shape = self.descr._v_dtypes[self.name].shape
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
## fill-column: 72
## End:
