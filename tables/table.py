######################################################################
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
import os.path
from time import time

import numpy

from tables import tableExtension
from tables.utilsExtension import lrange
from tables.conditions import compile_condition
from tables.numexpr.compiler import getType as numexpr_getType
from tables.numexpr.expressions import functions as numexpr_functions
from tables.flavor import flavor_of, array_as_internal, internal_to_flavor
from tables.utils import is_idx, lazyattr, SizeType
from tables.leaf import Leaf
from tables.description import IsDescription, Description, Col
from tables.exceptions import NodeError, HDF5ExtError, PerformanceWarning, \
     OldIndexWarning, NoSuchNodeError
from tables.parameters import MAX_COLUMNS, EXPECTED_ROWS_TABLE, CHUNKTIMES
from tables.utilsExtension import getNestedField
from tables.numexpr.compiler import stringToExpression, numexpr

from tables._table_common import (
    _indexNameOf, _indexPathnameOf, _indexPathnameOfColumn )

try:
    from tables.index import IndexesDescG
    from tables.index import IndexesTableG, OldIndex
    from tables.index import defaultIndexFilters
    from tables._table_pro import (
        NailedDict,
        _table__autoIndex, _table__indexFilters,
        _table__restorecache,
        _column__createIndex )
except ImportError:
    from tables.exceptions import NoIndexingError, NoIndexingWarning
    from tables.node import NotLoggedMixin
    from tables.group import Group

    # The following classes are registered to avoid extra warnings when
    # checking for the existence of indexes and to avoid logging node
    # renames and the like on them.
    class _DummyIndexesTableG(NotLoggedMixin, Group):
        _c_classId = 'TINDEX'
    class _DummyIndex(NotLoggedMixin, Group):
        _c_classId = 'INDEX'
    class _DummyOldIndex(NotLoggedMixin, Group):
        _c_classId = 'CINDEX'
    class _DummyIndexesDescG(NotLoggedMixin, Group):
        _c_classId = 'DINDEX'

    NailedDict = dict

    # Forbid accesses to these attributes.
    _table__autoIndex = _table__indexFilters = property()

    def _table__restorecache(self):
        pass

    def _checkIndexingAvailable():
        raise NoIndexingError
    _is_pro = False
else:
    def _checkIndexingAvailable():
        pass
    _is_pro = True


__version__ = "$Revision$"


# 2.2: Added support for complex types. Introduced in version 0.9.
# 2.2.1: Added suport for time types.
# 2.3: Changed the indexes naming schema.
# 2.4: Changed indexes naming schema (again).
# 2.5: Added the FIELD_%d_FILL attributes.
# 2.6: Added the FLAVOR attribute (optional).
obversion = "2.6"  # The Table VERSION number


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

# The NumPy scalar type corresponding to `SizeType`.
_npSizeType = numpy.array(SizeType(0)).dtype.type


class Table(tableExtension.Table, Leaf):
    """
    This class represents heterogeneous datasets in an HDF5 file.

    Tables are leaves (see the `Leaf` class) whose data consists of a
    unidimensional sequence of *rows*, where each row contains one or
    more *fields*.  Fields have an associated unique *name* and
    *position*, with the first field having position 0.  All rows have
    the same fields, which are arranged in *columns*.

    Fields can have any type supported by the `Col` class and its
    descendants, which support multidimensional data.  Moreover, a field
    can be *nested* (to an arbitrary depth), meaning that it includes
    further fields inside.  A field named ``x`` inside a nested field
    ``a`` in a table can be accessed as the field ``a/x`` (its *path
    name*) from the table.

    The structure of a table is declared by its description, which is
    made available in the `Table.description` attribute.

    This class provides new methods to read, write and search table data
    efficiently.  It also provides special Python methods to allow
    accessing the table as a normal sequence or array (with extended
    slicing supported).

    PyTables supports *in-kernel* searches working simultaneously on
    several columns using complex conditions.  These are faster than
    selections using Python expressions.  See the `Tables.where()`
    method for more information on in-kernel searches.

    Non-nested columns can be *indexed*.  Searching an indexed column
    can be several times faster than searching a non-nested one.  Search
    methods automatically take advantage of indexing where available.

    .. Note:: Column indexing is only available in PyTables Pro.

    When iterating a table, an object from the `Row` class is used.
    This object allows to read and write data one row at a time, as well
    as to perform queries which are not supported by in-kernel syntax
    (at a much lower speed, of course).

    Objects of this class support access to individual columns via
    *natural naming* through the `Table.cols` accessor.  Nested columns
    are mapped to `Cols` instances, and non-nested ones to `Column`
    instances.  See the `Column` class for examples of this feature.

    Instance variables
    ------------------

    The following instance variables are provided in addition to those
    in `Leaf`.  Please note that there are several ``col*`` dictionaries
    to ease retrieving information about a column directly by its path
    name, avoiding the need to walk through `Table.description` or
    `Table.cols`.

    autoIndex
        Automatically keep column indexes up to date?

        Setting this value states whether existing indexes should be
        automatically updated after an append operation or recomputed
        after an index-invalidating operation (i.e. removal and
        modification of rows).  The default is true.

        This value gets into effect whenever a column is altered.  If
        you don't have automatic indexing activated and you want to do
        an an immediate update use `Table.flushRowsToIndex()`; for an
        immediate reindexing of invalidated indexes, use
        `Table.reIndexDirty()`.

        This value is persistent.

        .. Note:: Column indexing is only available in PyTables Pro.

    coldescrs
        Maps the name of a column to its `Col` description.
    coldflts
        Maps the name of a column to its default value.
    coldtypes
        Maps the name of a column to its NumPy data type.
    colindexed
        Is the column which name is used as a key indexed?

        .. Note:: Column indexing is only available in PyTables Pro.

    colinstances
        Maps the name of a column to its `Column` or `Cols` instance.
    colnames
        A list containing the names of *top-level* columns in the table.
    colpathnames
        A list containing the pathnames of *bottom-level* columns in the
        table.

        These are the leaf columns obtained when walking the table
        description left-to-right, bottom-first.  Columns inside a
        nested column have slashes (``/``) separating name components in
        their pathname.

    cols
        A `Cols` instance that provides *natural naming* access to
        non-nested (`Column`) and nested (`Cols`) columns.
    coltypes
        Maps the name of a column to its PyTables data type.
    description
        A `Description` instance reflecting the structure of the table.
    extdim
        The index of the enlargeable dimension (always 0 for tables).
    indexed
        Does this table have any indexed columns?

        .. Note:: Column indexing is only available in PyTables Pro.

    indexedcolpathnames
        List of the pathnames of indexed columns in the table.

        .. Note:: Column indexing is only available in PyTables Pro.

    indexFilters
        Filters used to compress indexes.

        Setting this value to a `Filters` instance determines the
        compression to be used for indexes.  Setting it to ``None``
        means that no filters will be used for indexes.  The default is
        zlib compression level 1 with shuffling.

        This value is used when creating new indexes or recomputing old
        ones.  To apply it to existing indexes, use `Table.reIndex()`.

        This value is persistent.

        .. Note:: Column indexing is only available in PyTables Pro.

    nrows
        Current number of rows in the table.
    row
        The associated `Row` instance.
    rowsize
        The size in bytes of each row in the table.

    Public methods -- reading
    -------------------------

    * col(name)
    * iterrows([start][, stop][, step])
    * itersequence(sequence[, sort])
    * read([start][, stop][, step][, field][, coords])
    * readCoordinates(coords[, field])
    * __getitem__(key)
    * __iter__()

    Public methods -- writing
    -------------------------

    * append(rows)
    * modifyColumn([start][, stop][, step][, column][, colname])
    * modifyColumns([start][, stop][, step][, columns][, names])
    * modifyRows([start][, stop][, step][, rows])
    * removeRows(start[, stop])
    * __setitem__(key, value)

    Public methods -- querying
    --------------------------

    * getWhereList(condition[, condvars][, sort][, start][, stop][, step])
    * readWhere(condition[, condvars][, field][, start][, stop][, step])
    * where(condition[, condvars][, start][, stop][, step])
    * whereAppend(dstTable, condition[, condvars][, start][, stop][, step])
    * willQueryUseIndexing(condition[, condvars])

    Public methods -- other
    -----------------------

    * flushRowsToIndex()
    * getEnum(colname)
    * reIndex()
    * reIndexDirty()
    """

    # Class identifier.
    _c_classId = 'TABLE'


    # Properties
    # ~~~~~~~~~~
    @lazyattr
    def row(self):
        """The associated `Row` instance."""
        return tableExtension.Row(self)

    # Read-only shorthands
    # ````````````````````

    shape = property(
        lambda self: (self.nrows,), None, None,
        "The shape of this table.")

    rowsize = property(
        lambda self: self.description._v_dtype.itemsize, None, None,
        "The size in bytes of each row in the table.")

    # Lazy attributes
    # ```````````````
    @lazyattr
    def _v_iobuf(self):
        """A buffer for doing I/O."""
        return self._get_container(self.nrowsinbuf)

    @lazyattr
    def _v_wdflts(self):
        """The defaults for writing in recarray format."""
        wdflts = self._get_container(1)
        for colname, coldflt in self.coldflts.iteritems():
           ra = getNestedField(wdflts, colname)
           ra[:] = coldflt
        return wdflts

    @lazyattr
    def _colunaligned(self):
        """The pathnames of unaligned, *unidimensional* columns."""
        colunaligned, rarr = [], self._get_container(0)
        for colpathname in self.colpathnames:
            carr = getNestedField(rarr, colpathname)
            if not carr.flags.aligned and carr.ndim == 1:
                colunaligned.append(colpathname)
        return frozenset(colunaligned)

    # Index-related properties
    # ````````````````````````
    autoIndex = _table__autoIndex
    indexFilters = _table__indexFilters

    indexedcolpathnames = property(
        lambda self: [ _colpname for _colpname in self.colpathnames
                       if self.colindexed[_colpname] ],
        None, None,
        """
        The pathnames of the indexed columns of this table.

        .. Note:: Column indexing is only available in PyTables Pro.
        """ )

    # Other methods
    # ~~~~~~~~~~~~~
    def __init__(self, parentNode, name,
                 description=None, title="", filters=None,
                 expectedrows=EXPECTED_ROWS_TABLE,
                 chunkshape=None, byteorder=None, _log=True):
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

        byteorder -- The byteorder of the data *on-disk*, specified as
            'little' or 'big'. If this is not specified, the byteorder
            is that of the platform, unless you passed a recarray as the
            `description`, in which case the recarray byteorder will be
            chosen.

        """

        self._v_new = new = description is not None
        """Is this the first time the node has been created?"""
        self._v_new_title = title
        """New title for this node."""
        self._v_new_filters = filters
        """New filter properties for this node."""
        self.extdim = 0   # Tables only have one dimension currently
        """The index of the enlargeable dimension (always 0 for tables)."""
        self._v_recarray = None
        """A record array to be stored in the table."""
        self._rabyteorder = None
        """The computed byteorder of the self._v_recarray."""
        self._v_expectedrows = expectedrows
        """The expected number of rows to be stored in the table."""
        self.nrows = SizeType(0)
        """The current number of rows in the table."""
        self.description = None
        """A `Description` instance reflecting the structure of the table."""
        self._time64colnames = []
        """The names of ``Time64`` columns."""
        self._strcolnames = []
        """The names of ``String`` columns."""
        self._colenums = {}
        """Maps the name of an enumerated column to its ``Enum`` instance."""
        self._v_chunkshape = None
        """Private storage for the `chunkshape` property of the leaf."""

        self.indexed = False
        """
        Does this table have any indexed columns?

        .. Note:: Column indexing is only available in PyTables Pro.
        """
        self._indexedrows = 0
        """Number of rows indexed in disk."""
        self._unsaved_indexedrows = 0
        """Number of rows indexed in memory but still not in disk."""
        self._listoldindexes = []
        """The list of columns with old indexes."""
        self._autoIndex = None
        """Private variable that caches the value for autoIndex."""

        self.colnames = []
        """
        A list containing the names of *top-level* columns in the table.
        """
        self.colpathnames = []
        """
        A list containing the pathnames of *bottom-level* columns in the
        table.  These are the leaf columns obtained when walking the
        table description left-to-right, bottom-first.  Columns inside a
        nested column have slashes (``/``) separating name components in
        their pathname.
        """
        self.colinstances = {}
        """Maps the name of a column to its `Column` or `Cols` instance."""
        self.coldescrs = {}
        """Maps the name of a column to its `Col` description."""
        self.coltypes = {}
        """Maps the name of a column to its PyTables data type."""
        self.coldtypes = {}
        """Maps the name of a column to its NumPy data type."""
        self.coldflts = {}
        """Maps the name of a column to its default value."""
        self.colindexed = {}
        """
        Is the column which name is used as a key indexed?

        .. Note:: Column indexing is only available in PyTables Pro.
        """

        self._whereCondition = None
        """Condition function and argument list for selection of values."""
        self._whereIndex = None
        """Path of the indexed column to be used in an indexed search."""
        self._conditionCache = NailedDict()
        """Cache of already compiled conditions."""
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
        A `Cols` instance that provides *natural naming* access to
        non-nested (`Column`) and nested (`Cols`) columns.
        """
        self._dirtycache = True
        """Whether the data caches are dirty or not. Initially set to yes."""

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
                self.nrows = nrows = SizeType(nparray.size)
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
            if isinstance(chunkshape, (int, numpy.integer, long)):
                chunkshape = (chunkshape,)
            try:
                chunkshape = tuple(chunkshape)
            except TypeError:
                raise TypeError(
                    "`chunkshape` parameter must be an integer or sequence "
                    "and you passed a %s" % type(chunkshape) )
            if len(chunkshape) != 1:
                raise ValueError( "`chunkshape` rank (length) must be 1: %r"
                                  % (chunkshape,) )
            self._v_chunkshape = tuple(SizeType(s) for s in chunkshape)

        super(Table, self).__init__(parentNode, name, new, filters,
                                    byteorder, _log)



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
            # Columns are never indexed on creation.
            self.colindexed = dict((cpn, False) for cpn in self.colpathnames)
            return

        # The following code is only for opened tables.

        # Do the indexes group exist?
        indexesGroupPath = _indexPathnameOf(self)
        igroup = indexesGroupPath in self._v_file
        oldindexes = False
        for colobj in self.description._f_walk(type="Col"):
            colname = colobj._v_pathname
            # Is this column indexed?
            if igroup:
                indexname = _indexPathnameOfColumn(self, colname)
                indexed = indexname in self._v_file
                if indexed and not _is_pro:
                    warnings.warn( "table ``%s`` has column indexes"
                                   % self._v_pathname, NoIndexingWarning )
                    indexed = False
                self.colindexed[colname] = indexed
                if indexed:
                    column = self.cols._g_col(colname)
                    indexobj = column.index
                    if isinstance(indexobj, OldIndex):
                        indexed = False  # Not a vaild index
                        oldindexes = True
                        self._listoldindexes.append(colname)
                    else:
                        # Tell the condition cache about dirty indexed columns.
                        if indexobj.dirty:
                            self._conditionCache.nail()
            else:
                indexed = False
                self.colindexed[colname] = False
            if indexed:
                self.indexed = True

        if oldindexes:  # this should only appear under 2.x Pro
            warnings.warn(
                "table ``%s`` has column indexes with PyTables 1.x format. "
                "Unfortunately, this format is not supported in "
                "PyTables 2.x series. Note that you can use the "
                "``ptrepack`` utility in order to recreate the indexes. "
                "The 1.x indexed columns found are: %s" %
                (self._v_pathname, self._listoldindexes),
                OldIndexWarning )

        # It does not matter to which column 'indexobj' belongs,
        # since their respective index objects share
        # the same number of elements.
        if self.indexed:
            self._indexedrows = indexobj.nelements
            self._unsaved_indexedrows = self.nrows - self._indexedrows
            # Put the autoIndex value in a cache variable
            self._autoIndex = self.autoIndex


    def _restorecache(self):
        _table__restorecache(self)  # restore caches used by indexes
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

        self._rabyteorder = fbyteorder

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


    def _createIndexesTable(self):
        itgroup = IndexesTableG(
            self._v_parent, _indexNameOf(self),
            "Indexes container for table "+self._v_pathname, new=True)
        return itgroup


    def _createIndexesDescr(self, igroup, dname, iname, filters):
        idgroup = IndexesDescG(
            igroup, iname,
            "Indexes container for sub-description "+dname,
            filters=filters, new=True)
        return idgroup


    def _g_create(self):
        """Create a new table on disk."""

        # Warning against assigning too much columns...
        # F. Altet 2005-06-05
        if (len(self.description._v_names) > MAX_COLUMNS):
            warnings.warn(
                "table ``%s`` is exceeding the recommended "
                "maximum number of columns (%d); "
                "be ready to see PyTables asking for *lots* of memory "
                "and possibly slow I/O" % (self._v_pathname, MAX_COLUMNS),
                PerformanceWarning )

        # 1. Create the HDF5 table (some parameters need to be computed).

        # Fix the byteorder of the recarray and update the number of
        # expected rows if necessary
        if self._v_recarray is not None:
            self._v_recarray = self._g_fix_byteorder_data(self._v_recarray,
                                                          self._rabyteorder)
            if len(self._v_recarray) > self._v_expectedrows:
                self._v_expectedrows = len(self._v_recarray)
        # Compute a sensible chunkshape
        if self._v_chunkshape is None:
            self._v_chunkshape = self._calc_chunkshape(
                self._v_expectedrows, self.rowsize, self.rowsize)
        # Correct the byteorder, if still needed
        if self.byteorder is None:
            self.byteorder = sys.byteorder
        # After creating the table, ``self._v_objectID`` needs to be
        # set because it is needed for setting attributes afterwards.
        self._v_objectID = self._createTable(
            self._v_new_title, self.filters.complib or '', obversion )
        self._v_recarray = None  # not useful anymore
        self._rabyteorder = None # not useful anymore

        # 2. Compute or get chunk shape and buffer size parameters.
        self.nrowsinbuf = self._calc_nrowsinbuf(
            self._v_chunkshape, self.rowsize, self.rowsize)

        # 3. Get field fill attributes from the table description and
        #    set them on disk.
        i = 0
        setAttr = self._v_attrs._g__setattr
        for colobj in self.description._f_walk(type="Col"):
            fieldname = "FIELD_%d_FILL" % i
            setAttr(fieldname, colobj.dflt)
            i += 1

        # 4. Cache some data which is already in the description.
        self._cacheDescriptionData()

        return self._v_objectID


    def _g_open(self):
        """Opens a table from disk and read the metadata on it.

        Creates an user description on the flight to easy the access to
        the actual data.

        """

        # 1. Open the HDF5 table and get some data from it.
        self._v_objectID, description, chunksize = self._getInfo()
        self._v_expectedrows = self.nrows  # the actual number of rows

        # 2. Create an instance description to host the record fields.
        validate = not self._v_file._isPTFile  # only for non-PyTables files
        self.description = Description(description, validate=validate)

        # 3. Compute or get chunk shape and buffer size parameters.
        if chunksize == 0:
            self._v_chunkshape = self._calc_chunkshape(
                self._v_expectedrows, self.rowsize, self.rowsize)
        else:
            self._v_chunkshape = (chunksize,)
        self.nrowsinbuf = self._calc_nrowsinbuf(
            self._v_chunkshape, self.rowsize, self.rowsize)

        # 4. If there are field fill attributes, get them from disk and
        #    set them in the table description.
        if "FIELD_0_FILL" in self._v_attrs._f_list("sys"):
            i = 0
            getAttr = self._v_attrs.__getattr__
            for objcol in self.description._f_walk(type="Col"):
                colname = objcol._v_pathname
                # Get the default values for each column
                fieldname = "FIELD_%s_FILL" % i
                defval = getAttr(fieldname)
                if defval is not None:
                    objcol.dflt = defval
                else:
                    warnings.warn( "could not load default value "
                                   "for the ``%s`` column of table ``%s``; "
                                   "using ``%r`` instead"
                                   % (colname, self._v_pathname, objcol.dflt) )
                    defval = objcol.dflt
                i += 1

            # Set also the correct value in the desc._v_dflts dictionary
            for descr in self.description._f_walk(type="Description"):
                names = descr._v_names
                for i in range(len(names)):
                    objcol = descr._v_colObjects[names[i]]
                    if isinstance(objcol, Col):
                        descr._v_dflts[objcol._v_name] = objcol.dflt

        # 5. Cache some data which is already in the description.
        self._cacheDescriptionData()

        return self._v_objectID


    def _cacheDescriptionData(self):
        """
        Cache some data which is already in the description.

        Some information is extracted from `self.description` to build
        some useful (but redundant) structures:

        * `self.colnames`
        * `self.colpathnames`
        * `self.coldescrs`
        * `self.coltypes`
        * `self.coldtypes`
        * `self.coldflts`
        * `self._v_dtype`
        * `self._time64colnames`
        * `self._strcolnames`
        * `self._colenums`
        """

        self.colnames = list(self.description._v_names)
        self.colpathnames = [
            col._v_pathname for col in self.description._f_walk()
            if not hasattr(col, '_v_names') ]  # bottom-level

        # Find ``time64`` column names.
        self._time64colnames = self._getTypeColNames('time64')
        # Find ``string`` column names.
        self._strcolnames = self._getTypeColNames('string')
        # Get a mapping of enumerated columns to their `Enum` instances.
        self._colenums = self._getEnumMap()

        # Get info about columns
        for colobj in self.description._f_walk(type="Col"):
            colname = colobj._v_pathname
            # Get the column types, types and defaults
            self.coldescrs[colname] = colobj
            self.coltypes[colname] = colobj.type
            self.coldtypes[colname] = colobj.dtype
            self.coldflts[colname] = colobj.dflt

        # Assign _v_dtype for this table
        self._v_dtype = self.description._v_dtype


    def _getColumnInstance(self, colpathname):
        """
        Get the instance of the column with the given `colpathname`.

        If the column does not exist in the table, a ``KeyError`` is
        raised.
        """
        try:
            return reduce(getattr, colpathname.split('/'), self.description)
        except AttributeError:
            raise KeyError( "table ``%s`` does not have a column named ``%s``"
                            % (self._v_pathname, colpathname) )

    _checkColumn = _getColumnInstance


    def _disableIndexingInQueries(self):
        """Force queries not to use indexing.  *Use only for testing.*"""
        if not self._enabledIndexingInQueries:
            return  # already disabled
        # The nail avoids setting/getting compiled conditions in/from
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

    def _compileCondition(self, condition, condvars):
        """
        Compile the `condition` and extract usable index conditions.

        This method returns an instance of ``CompiledCondition``.  See
        the ``compile_condition()`` function in the ``conditions``
        module for more information about the compilation process.

        This method makes use of the condition cache when possible.
        """

        # Look up the condition in the condition cache.
        condcache = self._conditionCache
        condkey = self._getConditionKey(condition, condvars)
        compiled = condcache.get(condkey)
        if compiled:
            return compiled.with_replaced_vars(condvars)  # bingo!

        # Bad luck, the condition must be parsed and compiled.
        # Fortunately, the key provides some valuable information. ;)
        (condition, colnames, varnames, colpaths, vartypes) = condkey

        # Extract more information from referenced columns.
        typemap = dict(zip(varnames, vartypes))  # start with normal variables
        indexedcols, copycols = [], []
        for colname in colnames:
            col = condvars[colname]

            # Extract types from *all* the given variables.
            coltype = col.dtype.type
            typemap[colname] = _nxTypeFromNPType[coltype]

            # Get the set of columns with usable indexes.
            if ( self._enabledIndexingInQueries  # not test in-kernel searches
                 and self.colindexed[col.pathname] and not col.index.dirty ):
                indexedcols.append(colname)

            # Get the list of unaligned, unidimensional columns.  See
            # the comments in `tables.numexpr.evaluate()` for the
            # reasons of inserting copy operators for these columns.
            if col.pathname in self._colunaligned:
                copycols.append(colname)
        indexedcols = frozenset(indexedcols)

        # Now let ``compile_condition()`` do the Numexpr-related job.
        compiled = compile_condition(condition, typemap, indexedcols, copycols)

        # Check that there actually are columns in the condition.
        if not set(compiled.parameters).intersection(set(colnames)):
            raise ValueError( "there are no columns taking part "
                              "in condition ``%s``" % (condition,) )

        # Store the compiled condition in the cache and return it.
        condcache[condkey] = compiled
        return compiled.with_replaced_vars(condvars)


    def willQueryUseIndexing(self, condition, condvars=None):
        """
        Will a query for the `condition` use indexing?

        The meaning of the `condition` and `condvars` arguments is the
        same as in the `Table.where()` method.  If the `condition` can
        use indexing, this method returns the path name of the column
        whose index is usable.  Otherwise, it returns `None`.

        This method is mainly intended for testing.  Keep in mind that
        changing the set of indexed columns or their dirtyness may make
        this method return different values for the same arguments at
        different times.

        .. Note:: Column indexing is only available in PyTables Pro.
        """
        # Compile the condition and extract usable index conditions.
        condvars = self._requiredExprVars(condition, condvars)
        compiled = self._compileCondition(condition, condvars)
        if not compiled.index_variable:
            return None
        return condvars[compiled.index_variable].pathname


    def where( self, condition, condvars=None,
               start=None, stop=None, step=None ):
        """
        Iterate over values fulfilling a `condition`.

        This method returns a `Row` iterator which only selects rows in
        the table that satisfy the given `condition` (an expression-like
        string).

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

        .. Note:: Column indexing is only available in PyTables Pro.

        You can mix this method with standard Python selections in order
        to support even more complex queries.  It is strongly
        recommended that you pass the most restrictive condition as the
        parameter to this method if you want to achieve maximum
        performance.

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

        In this example, iterators returned by ``Table.where()`` have
        been used, but you may as well use any of the other reading
        iterators that ``Table`` objects offer.  See the file
        ``examples/nested-iter.py`` for the full code.

        .. Warning:: When in the middle of a table row iterator, you
           should not use methods that can change the number of rows in
           the table (like ``Table.append()`` or ``Table.removeRows()``)
           or unexpected errors will happen.
        """
        # Compile the condition and extract usable index conditions.
        condvars = self._requiredExprVars(condition, condvars)
        compiled = self._compileCondition(condition, condvars)
        return self._where(compiled, condvars, start, stop, step)

    def _where( self, compiled, condvars,
                start=None, stop=None, step=None ):
        """
        Low-level counterpart of `self.where()`.

        This version needs the condition to already be `compiled`.  It
        also uses `condvars` as is.  This is on purpose; if you want
        default variables and the like, use `self._requiredExprVars()`.
        """

        # Set the index column (if any) and condition
        # for the ``Row`` iterator.
        idxvar = compiled.index_variable
        if idxvar:
            idxcol = condvars[idxvar]
            index = idxcol.index
            assert index is not None, "the chosen column is not indexed"
            assert not index.dirty, "the chosen column has a dirty index"
            self._whereIndex = idxcol.pathname
            # Clean the table caches for indexed queries if needed
            if self._dirtycache:
                self._restorecache()

        args = [condvars[param] for param in compiled.parameters]
        self._whereCondition = (compiled.function, args)

        # Get the number of rows that the indexed condition yields.
        if idxvar:
            range_ = index.getLookupRange(
                compiled.index_operators, compiled.index_limits, self )
            ncoords = index.search(range_)  # do use indexing (always >= 0)
            if index.reduction == 1 and ncoords == 0:
                # No values from index condition, thus no resulting rows.
                self._whereIndex = self._whereCondition = None
                return iter([])

        # Adjust the slice to be used.
        (start, stop, step) = self._processRangeRead(start, stop, step)
        if start >= stop:  # empty range, reset conditions
            self._whereIndex = self._whereCondition = None
            return iter([])

        chunkmap = None  # default to an in-kernel query
        if idxvar:
            # Iterate according to the chunkmap and condition
            chunkmap = index.get_chunkmap()
            #print "chunks to read-->", chunkmap.sum(), chunkmap.nonzero()
            if chunkmap.sum() == 0:
                # The chunkmap is empty
                self._whereIndex = self._whereCondition = None
                return iter([])

        row = tableExtension.Row(self)
        return row._iter(start, stop, step, chunkmap=chunkmap)


    def _checkFieldIfNumeric(self, field):
        """Check that `field` has been selected with ``numeric`` flavor."""
        if self.flavor == 'numeric' and field is None:
            raise ValueError(
                "Numeric does not support heterogeneous datasets; "
                "you must specify a field when using the ``numeric`` flavor" )


    def readWhere( self, condition, condvars=None, field=None,
                   start=None, stop=None, step=None ):
        """
        Read table data fulfilling the given `condition`.

        This method is similar to `Table.read()`, having their common
        arguments and return values the same meanings.  However, only
        the rows fulfilling the `condition` are included in the result.

        The meaning of the other arguments is the same as in the
        `Table.where()` method.
        """
        self._checkFieldIfNumeric(field)

        # Compile the condition and extract usable index conditions.
        condvars = self._requiredExprVars(condition, condvars)
        compiled = self._compileCondition(condition, condvars)

        coords = [ p.nrow for p in
                   self._where(compiled, condvars, start, stop, step) ]
        self._whereCondition = None  # reset the conditions
        return self.readCoordinates(coords, field)


    def whereAppend( self, dstTable, condition, condvars=None,
                     start=None, stop=None, step=None ):
        """
        Append rows fulfulling the `condition` to the `dstTable` table.

        `dstTable` must be capable of taking the rows resulting from the
        query, i.e. it must have columns with the expected names and
        compatible types.  The meaning of the other arguments is the
        same as in the `Table.where()` method.

        The number of rows appended to `dstTable` is returned as a
        result.
        """
        # Check that the destination file is not in read-only mode.
        dstTable._v_file._checkWritable()

        # Compile the condition and extract usable index conditions.
        condvars = self._requiredExprVars(condition, condvars)
        compiled = self._compileCondition(condition, condvars)

        # Row objects do not support nested columns, so we must iterate
        # over the flat column paths.  When rows support nesting,
        # ``self.colnames`` can be directly iterated upon.
        colNames = [colName for colName in self.colpathnames]
        dstRow = dstTable.row
        nrows = 0
        for srcRow in self._where(compiled, condvars, start, stop, step):
            for colName in colNames:
                dstRow[colName] = srcRow[colName]
            dstRow.append()
            nrows += 1
        dstTable.flush()
        return nrows


    def getWhereList( self, condition, condvars=None, sort=False,
                      start=None, stop=None, step=None ):
        """
        Get the row coordinates fulfilling the given `condition`.

        The coordinates are returned as a list of the current flavor.
        `sort` means that you want to retrieve the coordinates ordered.
        The default is to not sort them.

        The meaning of the other arguments is the same as in the
        `Table.where()` method.
        """

        # Compile the condition and extract usable index conditions.
        condvars = self._requiredExprVars(condition, condvars)
        compiled = self._compileCondition(condition, condvars)
        coords = [ p.nrow for p in
                   self._where(compiled, condvars, start, stop, step) ]
        coords = numpy.array(coords, dtype=SizeType)
        # Reset the conditions
        self._whereCondition = None
        if sort:
            coords = numpy.sort(coords)
        return internal_to_flavor(coords, self.flavor)


    def itersequence(self, sequence, sort=False):
        """
        Iterate over a `sequence` of row coordinates.

        A true value for `sort` means that the `sequence` will be sorted
        so that I/O *might* perform better.  If your sequence is already
        sorted or you don't want to sort it, leave this parameter as
        false.  The default is not to sort the `sequence`.

        .. Note:: This iterator can be nested (see `Table.where()` for
           an example).
        """

        if not hasattr(sequence, '__getitem__'):
            raise TypeError("""\
Wrong 'sequence' parameter type. Only sequences are suported.""")

        coords = numpy.asarray(sequence, dtype=SizeType)
        # That might allow the retrieving on a sequential order
        # although this is not totally clear.
        if sort:
            sequence.sort()
        row = tableExtension.Row(self)
        return row._iter(coords=sequence)


    def iterrows(self, start=None, stop=None, step=None):
        """
        Iterate over the table using a `Row` instance.

        If a range is not supplied, *all the rows* in the table are
        iterated upon --you can also use the `Table.__iter__()` special
        method for that purpose.  If you only want to iterate over a
        given *range of rows* in the table, you may use the `start`,
        `stop` and `step` parameters, which have the same meaning as in
        `Table.read()`.

        Example of use::

            result = [ row['var2'] for row in table.iterrows(step=5)
                       if row['var1'] <= 20 ]

        .. Note:: This iterator can be nested (see `Table.where()` for
           an example).

        .. Warning:: When in the middle of a table row iterator, you
           should not use methods that can change the number of rows in
           the table (like ``Table.append()`` or ``Table.removeRows()``)
           or unexpected errors will happen.
        """
        (start, stop, step) = self._processRangeRead(start, stop, step)
        if start < stop:
            row = tableExtension.Row(self)
            return row._iter(start, stop, step)
        # Fall-back action is to return an empty iterator
        return iter([])


    def __iter__(self):
        """
        Iterate over the table using a `Row` instance.

        This is equivalent to calling `Table.iterrows()` with default
        arguments, i.e. it iterates over *all the rows* in the table.

        Example of use::

            result = [ row['var2'] for row in table
                       if row['var1'] <= 20 ]

        Which is equivalent to::

            result = [ row['var2'] for row in table.iterrows()
                       if row['var1'] <= 20 ]

        .. Note:: This iterator can be nested (see `Table.where()` for
           an example).
        """
        return self.iterrows()


    def _read(self, start, stop, step, field=None):
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

        nrows = lrange(start, stop, step).length

        # Compute the shape of the resulting column object
        if field:
            # Create a container for the results
            result = numpy.empty(shape=nrows, dtype=dtypeField)
        else:
            # Recarray case
            result = self._get_container(nrows)

        # Call the routine to fill-up the resulting array
        if step == 1 and not field:
            # This optimization works three times faster than
            # the row._fillCol method (up to 170 MB/s on a pentium IV @ 2GHz)
            self._read_records(start, stop-start, result)
        # Warning!: _read_field_name should not be used until
        # H5TBread_fields_name in tableExtension will be finished
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


    def read(self, start=None, stop=None, step=None, field=None):
        """
        Get data in the table as a (record) array.

        The `start`, `stop` and `step` parameters can be used to select
        only a *range of rows* in the table.  Their meanings are the
        same as in the built-in `range()` Python function, except that
        negative values of `step` are not allowed yet.  Moreover, if
        only `start` is specified, then `stop` will be set to
        ``start+1``.  If you do not specify neither `start` nor `stop`,
        then *all the rows* in the table are selected.

        If `field` is supplied only the named column will be selected.
        If the column is not nested, an *array* of the current flavor
        will be returned; if it is, a *record array* will be used
        instead.  I no `field` is specified, all the columns will be
        returned in a record array of the current flavor.

        Columns under a nested column can be specified in the `field`
        parameter by using a slash character (``/``) as a separator
        (e.g. ``'position/x'``).
        """

        if field:
            self._checkColumn(field)
        else:
            self._checkFieldIfNumeric(field)

        (start, stop, step) = self._processRangeRead(start, stop, step)

        arr = self._read(start, stop, step, field)
        return internal_to_flavor(arr, self.flavor)


    def _readCoordinates(self, coords, field=None):
        """Private part of `readCoordinates()` with no flavor conversion."""

        ncoords = len(coords)
        # Create a read buffer only if needed
        if field is None or ncoords > 0:
            # Doing a copy is faster when ncoords is small (<1000)
            if ncoords < min(1000, self.nrowsinbuf):
                result = self._v_iobuf[:ncoords].copy()
            else:
                result = self._get_container(ncoords)

        # Do the real read
        if ncoords > 0:
            # Turn coords into an array of coordinate indexes, if necessary
            if not (type(coords) is numpy.ndarray and
                    coords.dtype.type is _npSizeType and
                    coords.flags.contiguous and
                    coords.flags.aligned):
                # Get a contiguous and aligned coordinate array
                coords = numpy.array(coords, dtype=SizeType)
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
        Get a set of rows given their indexes as a (record) array.

        This method works much like the `read()` method, but it uses a
        sequence (`coords`) of row indexes to select the wanted columns,
        instead of a column range.

        The selected rows are returned in an array or record array of
        the current flavor.
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
        returned as a NumPy object or as a ``numarray`` object
        (depending on the flavor of the table).  If it does not exist, a
        ``KeyError`` is raised.

        Example of use::

            narray = table.col('var2')

        That statement is equivalent to::

            narray = table.read(field='var2')

        Here you can see how this method can be used as a shorthand for
        the `Table.read()` method.
        """
        return self.read(field=name)

    def __getitem__(self, key):
        """
        Get a row or a range of rows from the table.

        If the `key` argument is an integer, the corresponding table row
        is returned as a record of the current flavor.  If `key` is a
        slice, the range of rows determined by it is returned as a
        record array of the current flavor.

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
            (start, stop, step) = self._processRange(key, key+1, 1)
            # For the scalar case, convert the Record and return it as a tuple
            # Fixes bug #972534
            # Reverted to return a numpy.void in order
            # to support better the nested datatypes
            # return self.tolist(self.read(start, stop, step)[0])
            return self.read(start, stop, step)[0]
        elif isinstance(key, slice):
            (start, stop, step) = self._processRange(
                key.start, key.stop, key.step )
            return self.read(start, stop, step)
        else:
            raise TypeError("invalid index or slice: %r" % (key,))


    def __setitem__(self, key, value):
        """
        Set a row or a range of rows in the table.

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
                                   formats='i4,a3,f8')
            table[1:3:2] = rows

        Which is equivalent to::

            table.modifyRows(start=2, rows=[456,'db2',1.2])
            rows = numpy.rec.array([[457,'db1',1.2],[6,'de2',1.3]],
                                   formats='i4,a3,f8')
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
            (start, stop, step) = self._processRange(
                key.start, key.stop, key.step )
            return self.modifyRows(start, stop, step, value)
        else:
            raise ValueError, "Non-valid index or slice: %s" % key


    def append(self, rows):
        """
        Append a sequence of `rows` to the end of the table.

        The `rows` argument may be any object which can be converted to
        a record array compliant with the table structure (otherwise, a
        ``ValueError`` is raised).  This includes NumPy record arrays,
        ``RecArray`` or ``NestedRecArray`` objects if ``numarray`` is
        available, lists of tuples or array records, and a string or
        Python buffer.

        Example of use::

            from tables import *
            class Particle(IsDescription):
                name        = StringCol(16, pos=1) # 16-character String
                lati        = IntCol(pos=2)        # integer
                longi       = IntCol(pos=3)        # integer
                pressure    = Float32Col(pos=4)    # float  (single-precision)
                temperature = FloatCol(pos=5)      # double (double-precision)

            fileh = openFile('test4.h5', mode='w')
            table = fileh.createTable(fileh.root, 'table', Particle, \"A table\")
            # Append several rows in only one call
            table.append([(\"Particle:     10\", 10, 0, 10*10, 10**2),
                          (\"Particle:     11\", 11, -1, 11*11, 11**2),
                          (\"Particle:     12\", 12, -2, 12*12, 12**2)])
            fileh.close()
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
            wbufRA = numpy.rec.array(rows, dtype=self._v_dtype)
        except Exception, exc:  #XXX
            raise ValueError, \
"rows parameter cannot be converted into a recarray object compliant with table '%s'. The error was: <%s>" % (str(self), exc)
        lenrows = wbufRA.shape[0]
        # If the number of rows to append is zero, don't do anything else
        if lenrows > 0:
            # Save write buffer to disk
            self._saveBufferedRows(wbufRA, lenrows)


    def _saveBufferedRows(self, wbufRA, lenrows):
        """Update the indexes after a flushing of rows"""
        self._open_append(wbufRA)
        self._append_records(lenrows)
        self._close_append()
        self.nrows += lenrows
        if self.indexed:
            self._unsaved_indexedrows += lenrows
            # The table caches for indexed queries are dirty now
            self._dirtycache = True
            if self.autoIndex:
                # Flush the unindexed rows (this needs to read the table)
                self.flushRowsToIndex(_lastrow=False)


    def modifyRows(self, start=None, stop=None, step=1, rows=None):
        """
        Modify a series of rows in the slice ``[start:stop:step]``.

        The values in the selected rows will be modified with the data
        given in `rows`.  This method returns the number of rows
        modified.  Should the modification exceed the length of the
        table, an ``IndexError`` is raised before changing data.

        The possible values for the `rows` argument are the same as in
        `Table.append()`.
        """

        if rows is None:      # Nothing to be done
            return SizeType(0)
        if start is None:
            start = 0

        if start < 0:
            raise ValueError("'start' must have a positive value.")
        if step < 1:
            raise ValueError("'step' must have a value greater or equal than 1.")
        if stop is None:
            # compute the stop value. start + len(rows)*step does not work
            stop = start + (len(rows)-1)*step + 1

        (start, stop, step) = self._processRange(start, stop, step)
        if stop > self.nrows:
            raise IndexError, \
"This modification will exceed the length of the table. Giving up."
        # Compute the number of rows to read.
        nrows = lrange(start, stop, step).length
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
        self._reIndex(self.colpathnames)

        return SizeType(lenrows)


    def modifyColumn(self, start=None, stop=None, step=1,
                     column=None, colname=None):
        """
        Modify one single column in the row slice ``[start:stop:step]``.

        The `colname` argument specifies the name of the column in the
        table to be modified with the data given in `column`.  This
        method returns the number of rows modified.  Should the
        modification exceed the length of the table, an ``IndexError``
        is raised before changing data.

        The `column` argument may be any object which can be converted
        to a (record) array compliant with the structure of the column
        to be modified (otherwise, a ``ValueError`` is raised).  This
        includes NumPy (record) arrays, ``NumArray``, ``RecArray`` or
        ``NestedRecArray`` objects if ``numarray`` is available, Numeric
        arrays if available, lists of scalars, tuples or array records,
        and a string or Python buffer.
        """

        if not isinstance(colname, str):
            raise TypeError("The 'colname' parameter must be a string.")
        self._v_file._checkWritable()

        if column is None:      # Nothing to be done
            return SizeType(0)
        if start is None:
            start = 0

        if start < 0:
            raise ValueError("'start' must have a positive value.")
        if step < 1:
            raise ValueError("'step' must have a value greater or equal than 1.")
        # Get the column format to be modified:
        objcol = self._getColumnInstance(colname)
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
        (start, stop, step) = self._processRange(start, stop, step)
        if stop > self.nrows:
            raise IndexError, \
"This modification will exceed the length of the table. Giving up."
        # Compute the number of rows to read.
        nrows = lrange(start, stop, step).length
        if len(recarray) < nrows:
            raise ValueError, \
                  "The value has not enough elements to fill-in the specified range"
        # Now, read the original values:
        mod_recarr = self._read(start, stop, step)
        # Modify the appropriate column in the original recarray
        mod_col = getNestedField(mod_recarr, colname)
        mod_col[:] = recarray.field(0)
        # save this modified rows in table
        self._update_records(start, stop, step, mod_recarr)
        # Redo the index if needed
        self._reIndex([colname])

        return SizeType(nrows)


    def modifyColumns(self, start=None, stop=None, step=1,
                      columns=None, names=None):
        """
        Modify a series of columns in the row slice ``[start:stop:step]``.

        The `names` argument specifies the names of the columns in the
        table to be modified with the data given in `columns`.  This
        method returns the number of rows modified.  Should the
        modification exceed the length of the table, an ``IndexError``
        is raised before changing data.

        The `columns` argument may be any object which can be converted
        to a record array compliant with the structure of the columns to
        be modified (otherwise, a ``ValueError`` is raised).  This
        includes NumPy record arrays, ``RecArray`` or ``NestedRecArray``
        objects if ``numarray`` is available, lists of tuples or array
        records, and a string or Python buffer.
        """

        if type(names) not in (list, tuple):
            raise TypeError("""\
The 'names' parameter must be a list of strings.""")

        if columns is None:      # Nothing to be done
            return SizeType(0)
        if start is None:
            start = 0
        if start < 0:
            raise ValueError("'start' must have a positive value.")
        if step < 1:
            raise ValueError("'step' must have a value greater or equal than 1.")        # Get the column formats to be modified:
        descr = []
        for colname in names:
            objcol = self._getColumnInstance(colname)
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
        (start, stop, step) = self._processRange(start, stop, step)
        if stop > self.nrows:
            raise IndexError, \
"This modification will exceed the length of the table. Giving up."
        # Compute the number of rows to read.
        nrows = lrange(start, stop, step).length
        if len(recarray) < nrows:
            raise ValueError, \
           "The value has not enough elements to fill-in the specified range"
        # Now, read the original values:
        mod_recarr = self._read(start, stop, step)
        # Modify the appropriate columns in the original recarray
        for i, name in enumerate(recarray.dtype.names):
            mod_col = getNestedField(mod_recarr, names[i])
            mod_col[:] = recarray[name]
        # save this modified rows in table
        self._update_records(start, stop, step, mod_recarr)
        # Redo the index if needed
        self._reIndex(names)

        return SizeType(nrows)


    def flushRowsToIndex(self, _lastrow=True):
        """
        Add remaining rows in buffers to non-dirty indexes.

        This can be useful when you have chosen non-automatic indexing
        for the table (see the `Table.autoIndex` property) and you want
        to update the indexes on it.

        .. Note:: Column indexing is only available in PyTables Pro.
        """

        rowsadded = 0
        if self.indexed:
            # Update the number of unsaved indexed rows
            start = self._indexedrows
            nrows = self._unsaved_indexedrows
            for (colname, colindexed) in self.colindexed.iteritems():
                if colindexed:
                    col = self.cols._g_col(colname)
                    if nrows > 0 and not col.index.dirty:
                        rowsadded = self._addRowsToIndex(
                            colname, start, nrows, _lastrow, update=True )
            self._unsaved_indexedrows -= rowsadded
            self._indexedrows += rowsadded
        return rowsadded


    def _addRowsToIndex(self, colname, start, nrows, lastrow, update):
        """Add more elements to the existing index """

        # This method really belongs to Column, but since it makes extensive
        # use of the table, it gets dangerous when closing the file, since the
        # column may be accessing a table which is being destroyed.
        index = self.cols._g_col(colname).index
        slicesize = index.slicesize
        # The next loop does not rely on xrange so that it can
        # deal with long ints (i.e. more than 32-bit integers)
        # This allows to index columns with more than 2**31 rows
        # F. Altet 2005-05-09
        startLR = index.sorted.nrows*slicesize
        indexedrows = startLR - start
        stop = start+nrows-slicesize+1
        while startLR < stop:
            index.append(
                [self._read(startLR, startLR+slicesize, 1, colname)],
                update=update)
            indexedrows += slicesize
            startLR += slicesize
        # index the remaining rows in last row
        if lastrow and startLR < self.nrows:
            index.appendLastRow([self._read(startLR, self.nrows, 1, colname)])
            indexedrows += self.nrows - startLR
        return indexedrows


    def removeRows(self, start, stop=None):
        """
        Remove a range of rows in the table.

        If only `start` is supplied, only this row is to be deleted.  If
        a range is supplied, i.e. both the `start` and `stop` parameters
        are passed, all the rows in the range are removed.  A ``step``
        parameter is not supported, and it is not foreseen to be
        implemented anytime soon.

        `start`
            Sets the starting row to be removed.  It accepts negative
            values meaning that the count starts from the end.  A value
            of 0 means the first row.

        `stop`
            Sets the last row to be removed to ``stop-1``, i.e. the end
            point is omitted (in the Python ``range()`` tradition).
            Negative values are also accepted.  A special value of
            ``None`` (the default) means removing just the row supplied
            in `start`.
        """

        (start, stop, step) = self._processRangeRead(start, stop, 1)
        nrows = stop - start
        if nrows >= self.nrows:
            raise NotImplementedError, \
"""You are trying to delete all the rows in table "%s". This is not supported right now due to limitations on the underlying HDF5 library. Sorry!""" % self._v_pathname
        nrows = self._remove_row(start, nrows)
        self.nrows -= nrows    # discount the removed rows from the total
        # removeRows is a invalidating index operation
        self._reIndex(self.colpathnames)

        return SizeType(nrows)


    def _g_updateDependent(self):
        super(Table, self)._g_updateDependent()

        self.cols._g_updateTableLocation(self)


    def _g_move(self, newParent, newName):
        """
        Move this node in the hierarchy.

        This overloads the Node._g_move() method.
        """

        itgpathname = _indexPathnameOf(self)

        # First, move the table to the new location.
        super(Table, self)._g_move(newParent, newName)

        # Then move the associated index group (if any).
        try:
            itgroup = self._v_file._getNode(itgpathname)
        except NoSuchNodeError:
            pass
        else:
            oldiname = itgroup._v_name
            newigroup = self._v_parent
            newiname = _indexNameOf(self)
            itgroup._g_move(newigroup, newiname)


    def _g_remove(self, recursive=False):
        # Remove the associated index group (if any).
        itgpathname = _indexPathnameOf(self)
        try:
            itgroup = self._v_file._getNode(itgpathname)
        except NoSuchNodeError:
            pass
        else:
            itgroup._f_remove(recursive=True)
            self.indexed = False   # there are indexes no more

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
        """Re-index columns in `colnames` if automatic indexing is true."""

        if self.indexed:
            colstoindex = []
            # Mark the proper indexes as dirty
            for (colname, colindexed) in self.colindexed.iteritems():
                if colindexed and colname in colnames:
                    col = self.cols._g_col(colname)
                    col.index.dirty = True
                    colstoindex.append(colname)
            # Now, re-index the dirty ones
            if self.autoIndex and colstoindex:
                self._doReIndex(dirty=True)
            # The table caches for indexed queries are dirty now
            self._dirtycache = True


    def _doReIndex(self, dirty):
        """Common code for `reIndex()` and `reIndexDirty()`."""

        indexedrows = 0
        for (colname, colindexed) in self.colindexed.iteritems():
            if colindexed:
                indexcol = self.cols._g_col(colname)
                indexedrows = indexcol._doReIndex(dirty)
        # Update counters in case some column has been updated
        if indexedrows > 0:
            self._indexedrows = indexedrows
            self._unsaved_indexedrows = self.nrows - indexedrows
        return SizeType(indexedrows)


    def reIndex(self):
        """
        Recompute all the existing indexes in the table.

        This can be useful when you suspect that, for any reason, the
        index information for columns is no longer valid and want to
        rebuild the indexes on it.

        .. Note:: Column indexing is only available in PyTables Pro.
        """
        self._doReIndex(dirty=False)


    def reIndexDirty(self):
        """
        Recompute the existing indexes in table, *if* they are dirty.

        This can be useful when you have set `Table.autoIndex` to false
        for the table and you want to update the indexes after a
        invalidating index operation (`Table.removeRows()`, for
        example).

        .. Note:: Column indexing is only available in PyTables Pro.
        """
        self._doReIndex(dirty=True)


    def _g_copyRows(self, object, start, stop, step):
        "Copy rows from self to object"
        (start, stop, step) = self._processRangeRead(start, stop, step)
        nrowsinbuf = self.nrowsinbuf
        object._open_append(self._v_iobuf)
        nrowsdest = object.nrows
        for start2 in lrange(start, stop, step*nrowsinbuf):
            # Save the records on disk
            stop2 = start2+step*nrowsinbuf
            if stop2 > stop:
                stop2 = stop
            # Optimized version (it saves some conversions)
            nrows = ((stop2 - start2 - 1) // step) + 1
            self.row._fillCol(self._v_iobuf, start2, stop2, step, None)
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
                newcol = newcols[colname]
                newcol._createIndex(
                    oldcolindex.optlevel, oldcolindex.filters,
                    tmp_dir=None, _indsize=oldcolindex.indsize)


    def _g_copyWithStats(self, group, name, start, stop, step,
                         title, filters, _log):
        "Private part of Leaf.copy() for each kind of leaf"
        # Create the new table and copy the selected data.
        newtable = Table( group, name, self.description, title=title,
                          filters=filters, expectedrows=self.nrows,
                          _log=_log )
        self._g_copyRows(newtable, start, stop, step)
        nbytes = newtable.nrows * newtable.rowsize
        # We need to look at the HDF5 attribute to tell whether an index
        # property was explicitly set by the user.
        try:
            indexgroup = self._v_file._getNode(_indexPathnameOf(self))
        except NoSuchNodeError:
            pass
        else:
            if _is_pro and 'AUTO_INDEX' in indexgroup._v_attrs:
                newtable.autoIndex = self.autoIndex
            # There may be no filters; this is also a explicit change if
            # the default is having filters.  This is the reason for the
            # second part of the condition.
            if ( _is_pro and ('FILTERS' in indexgroup._v_attrs
                 or self.indexFilters != defaultIndexFilters) ):
                # Ignoring the DeprecationWarning temporarily here
                warnings.filterwarnings('ignore', category=DeprecationWarning)
                newtable.indexFilters = self.indexFilters
                warnings.filterwarnings('default', category=DeprecationWarning)
        # Generate equivalent indexes in the new table, if any.
        if self.indexed:
            warnings.warn(
                "generating indexes for destination table ``%s:%s``; "
                "please be patient"
                % (newtable._v_file.filename, newtable._v_pathname) )
            self._g_copyIndexes(newtable)

        return (newtable, nbytes)


    def flush(self):
        """Flush the table buffers."""

        # Flush rows that remains to be appended
        if 'row' in self.__dict__:
            self.row._flushBufferedRows()
        if self.indexed and self.autoIndex:
            # Flush any unindexed row
            rowsadded = self.flushRowsToIndex(_lastrow=True)
            assert rowsadded <= 0 or self._indexedrows == self.nrows, \
                   ( "internal error: the number of indexed rows (%d) "
                     "and rows in the table (%d) is not equal; "
                     "please report this to the authors."
                     % (self._indexedrows, self.nrows) )

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
        # call self.flush() before the table is being preempted.
        # F. Altet 2006-08-03
        if ('row' in self.__dict__ and
            self.row._getUnsavedNrows() > 0 or
            (self.indexed and self.autoIndex and
             self._unsaved_indexedrows > 0)):
            warnings.warn("""\
table ``%s`` is being preempted from alive nodes without its buffers being flushed. This may lead to very ineficient use of resources and even to fatal errors in certain situations. Please do a call to the .flush() method on this table before start using other nodes."""
                          % (self._v_pathname),
                          PerformanceWarning)
        # Get rid of the IO buffers (if they have been created at all)
        mydict = self.__dict__
        if '_v_iobuf' in mydict:
            del mydict['_v_iobuf']
        if '_v_wdflts' in mydict:
            del mydict['_v_wdflts']


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
            cols._g_close()

        # Close myself as a leaf.
        super(Table, self)._f_close(False)


    def __repr__(self):
        """This provides column metainfo in addition to standard __str__"""

        if self.indexed:
            format = """\
%s
  description := %r
  byteorder := %r
  chunkshape := %r
  autoIndex := %r
  indexFilters := %r
  indexedcolpathnames := %r"""
            return format % ( str(self), self.description, self.byteorder,
                              self.chunkshape, self.autoIndex,
                              self.indexFilters, self.indexedcolpathnames )
        else:
            return """\
%s
  description := %r
  byteorder := %r
  chunkshape := %r""" % \
        (str(self), self.description, self.byteorder, self.chunkshape)



class Cols(object):
    """
    Container for columns in a table or nested column.

    This class is used as an *accessor* to the columns in a table or
    nested column.  It supports the *natural naming* convention, so that
    you can access the different columns as attributes which lead to
    `Column` instances (for non-nested columns) or other `Cols`
    instances (for nested columns).

    For instance, if ``table.cols`` is a `Cols` instance with a column
    named ``col1`` under it, the later can be accessed as
    ``table.cols.col1``.  If ``col1`` is nested and contains a ``col2``
    column, this can be accessed as ``table.cols.col1.col2`` and so on.
    Because of natural naming, the names of members start with special
    prefixes, like in the `Group` class.

    Like the `Column` class, `Cols` supports item access to read and
    write ranges of values in the table or nested column.

    Public instance variables
    -------------------------

    _v_colnames
        A list of the names of the columns hanging directly from the
        associated table or nested column.  The order of the names
        matches the order of their respective columns in the containing
        table.

    _v_colpathnames
        A list of the pathnames of all the columns under the associated
        table or nested column (in preorder).  If it does not contain
        nested columns, this is exactly the same as the
        `Cols._v_colnames` attribute.

    _v_desc
        The associated `Description` instance.

    _v_table
        The parent `Table` instance.

    Public Methods
    --------------

    _f_col(colname)
        Get an accessor to the column ``colname``.
    __getitem__(key)
        Get a row or a range of rows from a table or nested column.
    __len__()
        Get the number of elements in the column.
    __setitem__(key, value)
        Set a row or a range of rows in a table or nested column.
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
            itgroup = table._v_file._getNode(_indexPathnameOf(table))
        except NodeError:
            pass
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
        """
        Get the number of elements in the column.

        This matches the length in rows of the parent table.
        """
        return self._v_table.nrows


    def _f_col(self, colname):
        """
        Get an accessor to the column `colname`.

        This method returns a `Column` instance if the requested column
        is not nested, and a `Cols` instance if it is.  You may use full
        column pathnames in `colname`.

        Calling ``cols._f_col('col1/col2')`` is equivalent to using
        ``cols.col1.col2``.  However, the first syntax is more intended
        for programmatic use.  It is also better if you want to access
        columns with names that are not valid Python identifiers.
        """

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
        Get a row or a range of rows from a table or nested column.

        If the `key` argument is an integer, the corresponding nested
        type row is returned as a record of the current flavor.  If
        `key` is a slice, the range of rows determined by it is returned
        as a record array of the current flavor.

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
            (start, stop, step) = table._processRange(key, key+1, 1)
            colgroup = self._v_desc._v_pathname
            if colgroup == "":  # The root group
                return table.read(start, stop, step)[0]
            else:
                crecord = table.read(start, stop, step)[0]
                return crecord[colgroup]
        elif isinstance(key, slice):
            (start, stop, step) = table._processRange(
                key.start, key.stop, key.step )
            colgroup = self._v_desc._v_pathname
            if colgroup == "":  # The root group
                return table.read(start, stop, step)
            else:
                crecarray = table.read(start, stop, step)
                if hasattr(crecarray, "field"):
                    return crecarray.field(colgroup)  # RecArray case
                else:
                    return getNestedField(crecarray, colgroup)  # numpy case
        else:
            raise TypeError("invalid index or slice: %r" % (key,))


    def __setitem__(self, key, value):
        """
        Set a row or a range of rows in a table or nested column.

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
        slicing can be used as shorthands for the `Table.modifyRows()`
        and `Table.modifyColumn()` methods.
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
            (start, stop, step) = table._processRange(key, key+1, 1)
        elif isinstance(key, slice):
            (start, stop, step) = table._processRange(
                key.start, key.stop, key.step )
        else:
            raise TypeError("invalid index or slice: %r" % (key,))

        # Actually modify the correct columns
        colgroup = self._v_desc._v_pathname
        if colgroup == "":  # The root group
            table.modifyRows(start, stop, step, rows=value)
        else:
            table.modifyColumn(
                start, stop, step, colname=colgroup, column=value)


    def _g_close(self):
        # First, close the columns (ie possible indices open)
        for col in self._v_colnames:
            colobj = self._g_col(col)
            if isinstance(colobj, Column):
                colobj.close()
                # Delete the reference to column
                del self.__dict__[col]
            else:
                colobj._g_close()

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
    """
    Accessor for a non-nested column in a table.

    Each instance of this class is associated with one *non-nested*
    column of a table.  These instances are mainly used to read and
    write data from the table columns using item access (like the `Cols`
    class), but there are a few other associated methods to deal with
    indexes.

    .. Note:: Column indexing is only available in PyTables Pro.

    Public instance variables
    -------------------------

    descr
        The `Description` instance of the parent table or nested column.
    dtype
        The NumPy ``dtype`` that most closely matches this column.
    index
        The `Index` instance associated with this column (``None`` if
        the column is not indexed).

        .. Note:: Column indexing is only available in PyTables Pro.

    is_indexed
        True if the column is indexed, false otherwise.

        .. Note:: Column indexing is only available in PyTables Pro.

    name
        The name of the associated column.
    pathname
        The complete pathname of the associated column (the same as
        `Column.name` if the column is not inside a nested column).
    table
        The parent `Table` instance.
    type
        The PyTables type of the column (a string).

    Public methods
    --------------

    createIndex([kind][, filters][, tmp_dir])
        Create an index for this column.
    createUltraLightIndex([optlevel][, filters][, tmp_dir])
        Create an ``ultralight`` index for this column.
    createLightIndex([optlevel][, filters][, tmp_dir])
        Create an ``light`` index for this column.
    createMediumIndex([optlevel][, filters][, tmp_dir])
        Create an ``medium`` index for this column.
    createFullIndex([optlevel][, filters][, tmp_dir])
        Create an ``full`` index for this column.
    reIndex()
        Recompute the index associated with this column.
    reIndexDirty()
        Recompute the associated index only if it is dirty.
    removeIndex()
        Remove the index associated with this column.

    Special methods
    ---------------

    __getitem__(key)
        Get an element or a range of elements from a column.
    __len__()
        Get the number of elements in the column.
    __setitem__(key, value)
        Set an element or a range of elements in a column.
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
        indexname = _indexPathnameOfColumn(table, self.pathname)
        try:
            index = tableFile._getNode(indexname)
            index.column = self # points to this column
            self._indexFile = index._v_file
            self._indexPath = index._v_pathname
        except NodeError:
            self._indexFile = None
            self._indexPath = None


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


    def __len__(self):
        """
        Get the number of elements in the column.

        This matches the length in rows of the parent table.
        """
        return self.table.nrows


    def __getitem__(self, key):
        """
        Get a row or a range of rows from a column.

        If the `key` argument is an integer, the corresponding element
        in the column is returned as an object of the current flavor.
        If `key` is a slice, the range of elements determined by it is
        returned as an array of the current flavor.

        Example of use::

            print \"Column handlers:\"
            for name in table.colnames:
                print table.cols._f_col(name)

            print \"Select table.cols.name[1]-->\", table.cols.name[1]
            print \"Select table.cols.name[1:2]-->\", table.cols.name[1:2]
            print \"Select table.cols.name[:]-->\", table.cols.name[:]
            print \"Select table.cols._f_col('name')[:]-->\", table.cols._f_col('name')[:]

        The output of this for a certain arbitrary table is::

            Column handlers:
            /table.cols.name (Column(), string, idx=None)
            /table.cols.lati (Column(), int32, idx=None)
            /table.cols.longi (Column(), int32, idx=None)
            /table.cols.vector (Column(2,), int32, idx=None)
            /table.cols.matrix2D (Column(2, 2), float64, idx=None)
            Select table.cols.name[1]--> Particle:     11
            Select table.cols.name[1:2]--> ['Particle:     11']
            Select table.cols.name[:]--> ['Particle:     10'
             'Particle:     11' 'Particle:     12'
             'Particle:     13' 'Particle:     14']
            Select table.cols._f_col('name')[:]--> ['Particle:     10'
             'Particle:     11' 'Particle:     12'
             'Particle:     13' 'Particle:     14']</screen>

        See the ``examples/table2.py`` file for a more complete example.
        """

        table = self.table
        if is_idx(key):
            # Index out of range protection
            if key >= table.nrows:
                raise IndexError, "Index out of range"
            if key < 0:
                # To support negative values
                key += table.nrows
            (start, stop, step) = table._processRange(key, key+1, 1)
            return table.read(start, stop, step, self.pathname)[0]
        elif isinstance(key, slice):
            (start, stop, step) = table._processRange(
                key.start, key.stop, key.step )
            return table.read(start, stop, step, self.pathname)
        else:
            raise TypeError, "'%s' key type is not valid in this context" % \
                  (key)


    def __setitem__(self, key, value):
        """
        Set a row or a range of rows in a column.

        If the `key` argument is an integer, the corresponding element
        is set to `value`.  If `key` is a slice, the range of elements
        determined by it is set to `value`.

        Example of use::

            # Modify row 1
            table.cols.col1[1] = -1
            # Modify rows 1 and 3
            table.cols.col1[1::2] = [2,3]

        Which is equivalent to::

            # Modify row 1
            table.modifyColumns(start=1, columns=[[-1]], names=['col1'])
            # Modify rows 1 and 3
            columns = numpy.rec.fromarrays([[2,3]], formats='i4')
            table.modifyColumns(start=1, step=2, columns=columns, names=['col1'])
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
            return table.modifyColumn(key, key+1, 1,
                                      [[value]], self.pathname)
        elif isinstance(key, slice):
            (start, stop, step) = table._processRange(
                key.start, key.stop, key.step )
            return table.modifyColumn(start, stop, step,
                                      value, self.pathname)
        else:
            raise ValueError, "Non-valid index or slice: %s" % key


    def createIndex( self, kind="light", filters=None, tmp_dir=None,
                     _blocksizes=None, _indsize=4,
                     _testmode=False, _verbose=False ):
        """
        Create an index for this column.

        You can select the kind of the index by setting `kind` from
        'ultralight', 'light', 'medium' or 'full' values.  Lighter kinds
        ('ultralight' and 'light') mean that the index takes less space
        on disk.  Heavier kinds ('medium' and 'full') mean better
        chances for reducing the entropy of the index at the price of
        using more disk space as well as more CPU, memory and I/O
        resources for creating the index.

        The `filters` argument can be used to set the `Filters` used to
        compress the index.  If ``None``, default index filters will be
        used (currently, zlib level 1 with shuffling).

        When `kind` is heavier than 'ultralight', a temporary file is
        created during the index build process.  You can use the
        `tmp_dir` argument to specify the directory for this temporary
        file.  The default is to create it in the same directory as the
        file containing the original table.

        .. Note:: Column indexing is only available in PyTables Pro.
        """
        if kind == "ultralight":
            return self.createUltraLightIndex(
                filters=filters, tmp_dir=tmp_dir,
                _blocksizes=_blocksizes,
                _testmode=_testmode, _verbose=_verbose)
        elif kind == "light":
            return self.createLightIndex(
                filters=filters, tmp_dir=tmp_dir,
                _blocksizes=_blocksizes,
                _testmode=_testmode, _verbose=_verbose)
        elif kind == "medium":
            return self.createMediumIndex(
                filters=filters, tmp_dir=tmp_dir,
                _blocksizes=_blocksizes,
                _testmode=_testmode, _verbose=_verbose)
        elif kind == "full":
            return self.createFullIndex(
                filters=filters, tmp_dir=tmp_dir,
                _blocksizes=_blocksizes,
                _testmode=_testmode, _verbose=_verbose)

        else:
            raise (ValueError,
                   "`kind` argument can only be 'ultralight', 'light', 'medium' or 'full'.")


    def createUltraLightIndex( self, optlevel=6, filters=None,
                               tmp_dir=None, _blocksizes=None,
                               _testmode=False, _verbose=False ):
        """Create an index of kind 'ultralight' for this column.

        You can select the optimization level of the index by setting
        `optlevel` from 0 (no optimization) to 9 (maximum optimization).
        Higher levels of optimization mean better chances for reducing
        the entropy of the index at the price of using more CPU, memory
        and I/O resources for creating the index.

        For the meaning of `filters` and `tmp_dir` arguments see
        ``Column.createIndex()``.

        .. Note:: Column indexing is only available in PyTables Pro.
        """
        return self._createIndex(optlevel, filters, tmp_dir,
                                 _blocksizes, _indsize=1,
                                 _testmode=_testmode, _verbose=_verbose)


    def createLightIndex( self, optlevel=6, filters=None,
                           tmp_dir=None, _blocksizes=None,
                          _testmode=False, _verbose=False ):
        """Create an index of kind 'light' for this column.

        You can select the optimization level of the index by setting
        `optlevel` from 0 (no optimization) to 9 (maximum optimization).
        Higher levels of optimization mean better chances for reducing
        the entropy of the index at the price of using more CPU, memory
        and I/O resources for creating the index.

        For the meaning of `filters` and `tmp_dir` arguments see
        ``Column.createIndex()``.

        .. Note:: Column indexing is only available in PyTables Pro.
        """
        return self._createIndex(optlevel, filters, tmp_dir,
                                 _blocksizes, _indsize=2,
                                 _testmode=_testmode, _verbose=_verbose)


    def createMediumIndex( self, optlevel=6, filters=None,
                           tmp_dir=None,_blocksizes=None,
                           _testmode=False, _verbose=False ):
        """Create an index of kind 'medium' for this column.

        You can select the optimization level of the index by setting
        `optlevel` from 0 (no optimization) to 9 (maximum optimization).
        Higher levels of optimization mean better chances for reducing
        the entropy of the index at the price of using more CPU, memory
        and I/O resources for creating the index.

        For the meaning of `filters` and `tmp_dir` arguments see
        ``Column.createIndex()``.

        .. Note:: Column indexing is only available in PyTables Pro.
        """
        return self._createIndex(optlevel, filters, tmp_dir,
                                 _blocksizes, _indsize=4,
                                 _testmode=_testmode, _verbose=_verbose)


    def createFullIndex( self, optlevel=6, filters=None,
                          tmp_dir=None, _blocksizes=None,
                         _testmode=False, _verbose=False ):
        """Create an index of kind 'full' for this column.

        You can select the optimization level of the index by setting
        `optlevel` from 0 (no optimization) to 9 (maximum optimization).
        Higher levels of optimization mean better chances for reducing
        the entropy of the index at the price of using more CPU, memory
        and I/O resources for creating the index.

        For the meaning of `filters` and `tmp_dir` arguments see
        ``Column.createIndex()``.

        .. Note:: Column indexing is only available in PyTables Pro.
        """
        return self._createIndex(optlevel, filters, tmp_dir,
                                 _blocksizes, _indsize=8,
                                 _testmode=_testmode, _verbose=_verbose)


    def _createIndex( self, optlevel, filters, tmp_dir,
                      _blocksizes=None, _indsize=2,
                      _testmode=False, _verbose=False ):
        """Private method to call the real index code."""
        _checkIndexingAvailable()
        if (not isinstance(optlevel, (int, long)) or
            (optlevel < 0 or optlevel > 9)):
            raise (ValueError,
                   "Optimization level should be an integer in the range 0-9.")
        if tmp_dir is None:
            tmp_dir = os.path.dirname(self._tableFile.filename)
        else:
            if not os.path.isdir(tmp_dir):
                raise (ValueError,
                       "Temporary directory '%s' does not exist." % tmp_dir)
        if (_blocksizes is not None and
            (type(_blocksizes) is not tuple or len(_blocksizes) != 4)):
            raise (ValueError,
                   "_blocksizes must be a tuple with exactly 4 elements.")
        idxrows = _column__createIndex(self, optlevel, filters, tmp_dir,
                                       _blocksizes, _indsize,
                                       _verbose)
        return SizeType(idxrows)


    def _doReIndex(self, dirty):
        "Common code for reIndex() and reIndexDirty() codes."

        index = self.index
        dodirty = True
        if dirty and not index.dirty: dodirty = False
        if index is not None and dodirty:
            self._tableFile._checkWritable()
            # Delete the existing Index
            index._f_remove()
            self._updateIndexLocation(None)
            # Create a new Index without warnings
            return SizeType(self.createIndex())
        else:
            return SizeType(0)  # The column is not intended for indexing


    def reIndex(self):
        """
        Recompute the index associated with this column.

        This can be useful when you suspect that, for any reason, the
        index information is no longer valid and you want to rebuild it.

        This method does nothing if the column is not indexed.

        .. Note:: Column indexing is only available in PyTables Pro.
        """

        self._doReIndex(dirty=False)


    def reIndexDirty(self):
        """
        Recompute the associated index only if it is dirty.

        This can be useful when you have set `Table.autoIndex` to false
        for the table and you want to update the column's index after an
        invalidating index operation (like `Table.removeRows()`).

        This method does nothing if the column is not indexed.

        .. Note:: Column indexing is only available in PyTables Pro.
        """

        self._doReIndex(dirty=True)


    def removeIndex(self):
        """
        Remove the index associated with this column.

        This method does nothing if the column is not indexed.  The
        removed index can be created again by calling the
        `Column.createIndex()` method.

        .. Note:: Column indexing is only available in PyTables Pro.
        """

        _checkIndexingAvailable()

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
