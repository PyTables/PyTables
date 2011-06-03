########################################################################
#
#       License: BSD
#       Created: October 14, 2002
#       Author:  Francesc Alted - faltet@pytables.com
#
#       $Id$
#
########################################################################

"""Here is defined the Leaf class.

See Leaf class docstring for more info.

Classes:

    Leaf

Functions:

    calc_chunksize

Misc variables:

    __version__


"""

import sys
import warnings
import math

import numpy

import tables
from tables.flavor import ( check_flavor, internal_flavor,
                            alias_map as flavor_alias_map )
from tables import hdf5Extension
from tables.node import Node
from tables.filters import Filters
from tables.utils import byteorders, idx2long, lazyattr, SizeType
from tables.utilsExtension import whichLibVersion
from tables.exceptions import PerformanceWarning
from tables import utilsExtension


__version__ = "$Revision$"


def csformula(expectedsizeinMB):
    """Return the fitted chunksize for expectedsizeinMB."""
    # For a basesize of 8 KB, this will return:
    # 8 KB for datasets <= 1 MB
    # 1 MB for datasets >= 10 TB
    basesize = 8*1024   # 8 KB is a good minimum
    return basesize * int(2**math.log10(expectedsizeinMB))


def limit_es(expectedsizeinMB):
    """Protection against creating too small or too large chunks."""
    if expectedsizeinMB < 1:        # < 1 MB
        expectedsizeinMB = 1
    elif expectedsizeinMB > 10**7:  # > 10 TB
        expectedsizeinMB = 10**7
    return expectedsizeinMB


def calc_chunksize(expectedsizeinMB):
    """Compute the optimum HDF5 chunksize for I/O purposes.

    Rational: HDF5 takes the data in bunches of chunksize length to
    write the on disk. A BTree in memory is used to map structures on
    disk. The more chunks that are allocated for a dataset the larger
    the B-tree. Large B-trees take memory and causes file storage
    overhead as well as more disk I/O and higher contention for the meta
    data cache.  You have to balance between memory and I/O overhead
    (small B-trees) and time to access to data (big B-trees).

    The tuning of the chunksize parameter affects the performance and
    the memory consumed. This is based on my own experiments and, as
    always, your mileage may vary.
    """

    expectedsizeinMB = limit_es(expectedsizeinMB)
    zone = int(math.log10(expectedsizeinMB))
    expectedsizeinMB = 10**zone
    chunksize = csformula(expectedsizeinMB)
    return chunksize*8     # XXX: Multiply by 8 seems optimal for
                           # sequential access



class Leaf(Node):
    """
    Abstract base class for all PyTables leaves.

    A leaf is a node (see the `Node` class) which hangs from a group
    (see the `Group` class) but, unlike a group, it can not have any
    further children below it (i.e. it is an end node).

    This definition includes all nodes which contain actual data
    (datasets handled by the `Table`, `Array`, `CArray`, `EArray` and
    `VLArray` classes) and unsupported nodes (the `UnImplemented` class)
    --these classes do in fact inherit from `Leaf`.

    Public instance variables
    -------------------------

    The following instance variables are provided in addition to those
    in `Node`:

    byteorder
        The byte ordering of the leaf data *on disk*.
    chunkshape
        The HDF5 chunk size for chunked leaves (a tuple).

        This is read-only because you cannot change the chunk size of a
        leaf once it has been created.
    filters
        Filter properties for this leaf --see `Filters`.
    flavor
        The type of the data object read from this leaf.

        It can be any of 'numpy', 'numarray', 'numeric' or 'python' (the
        set of supported flavors depends on which packages you have
        installed on your system).

        You can (and are encouraged to) use this property to get, set
        and delete the ``FLAVOR`` HDF5 attribute of the leaf.  When the
        leaf has no such attribute, the default flavor is used.
    maindim
        The dimension along which iterators work.

        Its value is 0 (i.e. the first dimension) when the dataset is
        not extendable, and `Leaf.extdim` (where available) for
        extendable ones.
    nrows
        The length of the main dimension of the leaf data.
    nrowsinbuf
        The number of rows that fit in internal input buffers.

        You can change this to fine-tune the speed or memory
        requirements of your application.
    shape
        The shape of data in the leaf.

    Public instance variables -- aliases
    ------------------------------------

    The following instance variables are just easier-to-write aliases to
    their `Node` counterparts (indicated between parentheses):

    attrs
        The associated `AttributeSet` instance (`Node._v_attrs`).
    name
        The name of this node in its parent group (`Node._v_name`).
    objectID
        A node identifier (may change from run to run).
        (`Node._v_objectID`).
    title
        A description for this node (`Node._v_title`).

    Public methods
    --------------

    * close([flush])
    * copy([newparent][, newname][, overwrite][, createparents][, **kwargs])
    * delAttr(name)
    * flush()
    * getAttr(name)
    * isVisible()
    * move([newparent][, newname][, overwrite])
    * remove()
    * rename(newname)
    * truncate(size)
    * setAttr(name, value)
    * _f_close([flush])
    * __len__()
    """

    # Properties
    # ~~~~~~~~~~

    # Node property aliases
    # `````````````````````
    # These are a little hard to override, but so are properties.
    attrs = Node._v_attrs
    title = Node._v_title

    # Read-only node property aliases
    # ```````````````````````````````
    name = property(
        lambda self: self._v_name, None, None,
        "The name of this node in its parent group (a string)." )

    chunkshape = property(
        lambda self: self._v_chunkshape, None, None,
        """
        The HDF5 chunk size for chunked leaves (a tuple).

        This is read-only because you cannot change the chunk size of a
        leaf once it has been created.
        """ )

    objectID = property(
        lambda self: self._v_objectID, None, None,
        "A node identifier (may change from run to run)." )

    # Lazy read-only attributes
    # `````````````````````````
    @lazyattr
    def filters(self):
        """Filter properties for this leaf."""
        return Filters._from_leaf(self)

    # Other properties
    # ````````````````
    def _getmaindim(self):
        if self.extdim < 0:
            return 0  # choose the first dimension
        return self.extdim

    maindim = property(
        _getmaindim, None, None,
        """
        The dimension along which iterators work.

        Its value is 0 (i.e. the first dimension) when the dataset is
        not extendable, and `Leaf.extdim` (where available) for
        extendable ones.
        """ )

    def _setflavor(self, flavor):
        self._v_file._checkWritable()
        check_flavor(flavor)
        self._v_attrs.FLAVOR = self._flavor = flavor  # logs the change

    def _delflavor(self):
        del self._v_attrs.FLAVOR
        self._flavor = internal_flavor

    flavor = property(
        lambda self: self._flavor, _setflavor, _delflavor,
        """
        The representation of data read from this leaf.

        It can be any of 'numpy', 'numarray', 'numeric' or 'python' (the
        set of supported flavors depends on which packages you have
        installed on your system).

        You can (and are encouraged to) use this property to get, set
        and delete the ``FLAVOR`` HDF5 attribute of the leaf.  When the
        leaf has no such attribute, the default flavor is used.
        """ )


    # Special methods
    # ~~~~~~~~~~~~~~~
    def __init__(self, parentNode, name,
                 new=False, filters=None,
                 byteorder=None, _log=True):
        self._v_new = new
        """Is this the first time the node has been created?"""
        self.nrowsinbuf = None
        """
        The number of rows that fits in internal input buffers.

        You can change this to fine-tune the speed or memory
        requirements of your application.
        """
        self._flavor = None
        """Private storage for the `flavor` property."""

        if new:
            # Get filter properties from parent group if not given.
            if filters is None:
                filters = parentNode._v_filters
            self.__dict__['filters'] = filters  # bypass the property

            if byteorder not in (None, 'little', 'big'):
                raise ValueError(
                    "the byteorder can only take 'little' or 'big' values "
                    "and you passed: %s" % byteorder)
            self.byteorder = byteorder
            """The byte ordering of the leaf data *on disk*."""

        # Existing filters need not be read since `filters`
        # is a lazy property that automatically handles their loading.

        super(Leaf, self).__init__(parentNode, name, _log)


    def __len__(self):
        """
        Return the length of the main dimension of the leaf data.

        Please note that this may raise an ``OverflowError`` on 32-bit
        platforms for datasets having more than 2**31-1 rows.  This is a
        limitation of Python that you can work around by using the
        ``nrows`` or ``shape`` attributes.
        """
        return self.nrows


    def __str__(self):

        """The string representation for this object is its pathname in
        the HDF5 object tree plus some additional metainfo.
        """

        # Get this class name
        classname = self.__class__.__name__
        # The title
        title = self._v_title
        # The filters
        filters = ""
        if self.filters.fletcher32:
            filters += ", fletcher32"
        if self.filters.complevel:
            if self.filters.shuffle:
                filters += ", shuffle"
            filters += ", %s(%s)" % (self.filters.complib,
                                     self.filters.complevel)
        return "%s (%s%s%s) %r" % \
               (self._v_pathname, classname, self.shape, filters, title)


    # Private methods
    # ~~~~~~~~~~~~~~~
    def _g_postInitHook(self):
        """
        Code to be run after node creation and before creation logging.

        This method gets or sets the flavor of the leaf.
        """

        super(Leaf, self)._g_postInitHook()
        if self._v_new:  # set flavor of new node
            if self._flavor is None:
                self._flavor = internal_flavor
            else:  # flavor set at creation time, do not log
                if self._v_file.params['PYTABLES_SYS_ATTRS']:
                    self._v_attrs._g__setattr('FLAVOR', self._flavor)
        else:  # get flavor of existing node (if any)
            if self._v_file.params['PYTABLES_SYS_ATTRS']:
                flavor = getattr(self._v_attrs, 'FLAVOR', internal_flavor)
                self._flavor = flavor_alias_map.get(flavor, flavor)
            else:
                self._flavor = internal_flavor


    def _calc_chunkshape(self, expectedrows, rowsize, itemsize):
        """Calculate the shape for the HDF5 chunk."""

        # In case of a scalar shape, return the unit chunksize
        if self.shape == ():
            return (SizeType(1),)

        # Compute the chunksize
        MB = 1024 * 1024
        expectedsizeinMB = (expectedrows * rowsize) / MB
        chunksize = calc_chunksize(expectedsizeinMB)

        maindim = self.maindim
        # Compute the chunknitems
        chunknitems = chunksize // itemsize
        # Safeguard against itemsizes being extremely large
        if chunknitems == 0:
            chunknitems = 1
        chunkshape = list(self.shape)
        # Check whether trimming the main dimension is enough
        chunkshape[maindim] = 1
        newchunknitems = numpy.prod(chunkshape, dtype=SizeType)
        if newchunknitems <= chunknitems:
            chunkshape[maindim] = chunknitems // newchunknitems
        else:
            # No, so start trimming other dimensions as well
            for j in xrange(len(chunkshape)):
                # Check whether trimming this dimension is enough
                chunkshape[j] = 1
                newchunknitems = numpy.prod(chunkshape, dtype=SizeType)
                if newchunknitems <= chunknitems:
                    chunkshape[j] = chunknitems // newchunknitems
                    break
            else:
                # Ops, we ran out of the loop without a break
                # Set the last dimension to chunknitems
                chunkshape[-1] = chunknitems

        return tuple(SizeType(s) for s in chunkshape)


    def _calc_nrowsinbuf(self):
        """Calculate the number of rows that fits on a PyTables buffer."""

        params = self._v_file.params
        # Compute the nrowsinbuf
        rowsize = self.rowsize
        buffersize = params['IO_BUFFER_SIZE']
        nrowsinbuf = buffersize // rowsize
        # Safeguard against row sizes being extremely large
        if nrowsinbuf == 0:
            nrowsinbuf = 1
            # If rowsize is too large, issue a Performance warning
            maxrowsize = params['BUFFER_TIMES'] * buffersize
            if rowsize > maxrowsize:
                warnings.warn("""\
The Leaf ``%s`` is exceeding the maximum recommended rowsize (%d bytes);
be ready to see PyTables asking for *lots* of memory and possibly slow
I/O.  You may want to reduce the rowsize by trimming the value of
dimensions that are orthogonal (and preferably close) to the *main*
dimension of this leave.  Alternatively, in case you have specified a
very small/large chunksize, you may want to increase/decrease it."""
                              % (self._v_pathname, maxrowsize),
                                 PerformanceWarning)
        return nrowsinbuf


    # This method is appropriate for calls to __getitem__ methods
    def _processRange(self, start, stop, step, dim=None, warn_negstep=True):
        if dim is None:
            nrows = self.nrows  # self.shape[self.maindim]
        else:
            nrows = self.shape[dim]

        if warn_negstep and step and step < 0 :
            raise ValueError("slice step cannot be negative")
        # (start, stop, step) = slice(start, stop, step).indices(nrows)
        # The next function is a substitute for slice().indices in order to
        # support full 64-bit integer for slices even in 32-bit machines.
        # F. Alted 2005-05-08
        (start, stop, step) = utilsExtension.getIndices(
            start, stop, step, long(nrows) )

        return (start, stop, step)


    # This method is appropiate for calls to read() methods
    def _processRangeRead(self, start, stop, step, warn_negstep=True):
        nrows = self.nrows
        if start is None and stop is None:
            start = 0
            stop = nrows
        if start is not None and stop is None:
            # Protection against start greater than available records
            # nrows == 0 is a special case for empty objects
            if nrows > 0 and start >= nrows:
                raise IndexError( "start of range (%s) is greater than "
                                  "number of rows (%s)" % (start, nrows) )
            step = 1
            if start == -1:  # corner case
                stop = nrows
            else:
                stop = start + 1
        # Finally, get the correct values (over the main dimension)
        start, stop, step = self._processRange(
            start, stop, step, warn_negstep=warn_negstep)

        return (start, stop, step)


    def _g_copy(self, newParent, newName, recursive, _log=True, **kwargs):
        # Compute default arguments.
        start = kwargs.pop('start', None)
        stop = kwargs.pop('stop', None)
        step = kwargs.pop('step', None)
        title = kwargs.pop('title', self._v_title)
        filters = kwargs.pop('filters', self.filters)
        chunkshape = kwargs.pop('chunkshape', self.chunkshape)
        copyuserattrs = kwargs.pop('copyuserattrs', True)
        stats = kwargs.pop('stats', None)
        if chunkshape == 'keep':
            chunkshape = self.chunkshape  # Keep the original chunkshape
        elif chunkshape == 'auto':
            chunkshape = None             # Will recompute chunkshape

        # Fix arguments with explicit None values for backwards compatibility.
        if title is None:  title = self._v_title
        if filters is None:  filters = self.filters

        # Create a copy of the object.
        (newNode, bytes) = self._g_copyWithStats(
            newParent, newName, start, stop, step,
            title, filters, chunkshape, _log, **kwargs)

        # Copy user attributes if requested (or the flavor at least).
        if copyuserattrs == True:
            self._v_attrs._g_copy(newNode._v_attrs, copyClass=True)
        elif 'FLAVOR' in self._v_attrs:
            if self._v_file.params['PYTABLES_SYS_ATTRS']:
                newNode._v_attrs._g__setattr('FLAVOR', self._flavor)
        newNode._flavor = self._flavor  # update cached value

        # Update statistics if needed.
        if stats is not None:
            stats['leaves'] += 1
            stats['bytes'] += bytes

        return newNode


    def _g_fix_byteorder_data(self, data, dbyteorder):
        "Fix the byteorder of data passed in constructors."
        dbyteorder = byteorders[dbyteorder]
        # If self.byteorder has not been passed as an argument of
        # the constructor, then set it to the same value of data.
        if self.byteorder is None:
            self.byteorder = dbyteorder
        # Do an additional in-place byteswap of data if the in-memory
        # byteorder doesn't match that of the on-disk.  This is the only
        # place that we have to do the conversion manually. In all the
        # other cases, it will be HDF5 the responsible of doing the
        # byteswap properly.
        if dbyteorder in ['little', 'big']:
            if dbyteorder != self.byteorder:
                # if data is not writeable, do a copy first
                if not data.flags.writeable:
                    data = data.copy()
                data.byteswap(True)
        else:
            # Fix the byteorder again, no matter which byteorder have
            # specified the user in the constructor.
            self.byteorder = "irrelevant"
        return data


    def _pointSelection(self, key):
        """Perform a point-wise selection.

        `key` can be any of the following items:

        * A boolean array with the same shape than self. Those positions
          with True values will signal the coordinates to be returned.

        * A numpy array (or list or tuple) with the point coordinates.
          This has to be a two-dimensional array of size len(self.shape)
          by num_elements containing a list of of zero-based values
          specifying the coordinates in the dataset of the selected
          elements. The order of the element coordinates in the array
          specifies the order in which the array elements are iterated
          through when I/O is performed. Duplicate coordinate locations
          are not checked for.

        Return the coordinates array.  If this is not possible, raise a
        `TypeError` so that the next selection method can be tried out.

        This is useful for whatever `Leaf` instance implementing a
        point-wise selection.
        """

        if type(key) in (list, tuple):
            if type(key) is tuple and len(key) > len(self.shape):
                raise IndexError("Invalid index or slice: %r" % (key,))
            # Try to convert key to a numpy array.  If not possible,
            # a TypeError will be issued (to be catched later on).
            try:
                key = numpy.array(key)
            except ValueError:
                raise TypeError("Invalid index or slice: %r" % (key,))
        elif not isinstance(key, numpy.ndarray):
            raise TypeError("Invalid index or slice: %r" % (key,))

        # Protection against empty keys
        if len(key) == 0:
            return numpy.array([], dtype="i8")

        if key.dtype.kind == 'b':
            if not key.shape == self.shape:
                raise IndexError(
                    "Boolean indexing array has incompatible shape")
            # Get the True coordinates (64-bit indices!)
            coords = numpy.asarray(key.nonzero(), dtype='i8')
            coords = numpy.transpose(coords)
        elif key.dtype.kind == 'i':
            if len(key.shape) > 2:
                raise IndexError(
                    "Coordinate indexing array has incompatible shape")
            elif len(key.shape) == 2:
                if key.shape[0] <> len(self.shape):
                    raise IndexError(
                        "Coordinate indexing array has incompatible shape")
                coords = numpy.asarray(key, dtype="i8")
                coords = numpy.transpose(coords)
            else:
                # For 1-dimensional datasets
                coords = numpy.asarray(key, dtype="i8")
        else:
            raise TypeError("Only integer coordinates allowed.")
        # We absolutely need a contiguous array
        if not coords.flags.contiguous:
            coords = coords.copy()
        return coords


    # Public methods
    # ~~~~~~~~~~~~~~
    # Tree manipulation
    # `````````````````
    def remove(self):
        """
        Remove this node from the hierarchy.

        This method has the behavior described in `Node._f_remove()`.
        Please note that there is no ``recursive`` flag since leaves
        do not have child nodes.
        """
        self._f_remove(False)


    def rename(self, newname):
        """
        Rename this node in place.

        This method has the behavior described in `Node._f_rename()`.
        """
        self._f_rename(newname)


    def move( self, newparent=None, newname=None,
              overwrite=False, createparents=False ):
        """
        Move or rename this node.

        This method has the behavior described in `Node._f_move()`.
        """
        self._f_move(newparent, newname, overwrite, createparents)


    def copy( self, newparent=None, newname=None,
              overwrite=False, createparents=False, **kwargs ):
        """
        Copy this node and return the new one.

        This method has the behavior described in `Node._f_copy()`.
        Please note that there is no ``recursive`` flag since leaves
        do not have child nodes.  In addition, this method recognises
        the following keyword arguments:

        `title`
            The new title for the destination.  If omitted or
            ``None``, the original title is used.
        `filters`
            Specifying this parameter overrides the original filter
            properties in the source node.  If specified, it must be
            an instance of the `Filters` class.  The default is to
            copy the filter properties from the source node.
        `copyuserattrs`
            You can prevent the user attributes from being copied by
            setting this parameter to ``False``.  The default is to
            copy them.
        `start`, `stop`, `step`
            Specify the range of rows to be copied; the default is to
            copy all the rows.
        `chunkshape`
            The chunkshape of the new leaf.  It supports a couple of
            special values.  A value of 'keep' means that the chunkshape
            will be the same than original leaf (this is the default).
            A value of 'auto' means that a new shape will be computed
            automatically in order to ensure best performance when
            accessing the dataset through the main dimension.  Any other
            value should be an integer or a tuple matching the
            dimensions of the leaf.
        `stats`
            This argument may be used to collect statistics on the
            copy process.  When used, it should be a dictionary whith
            keys ``'groups'``, ``'leaves'`` and ``'bytes'`` having a
            numeric value.  Their values will be incremented to
            reflect the number of groups, leaves and bytes,
            respectively, that have been copied during the operation.

        .. Warning:: Note that unknown parameters passed to this method
           will be ignored, so may want to double check the spell of
           these (i.e. if you write them incorrectly, they will most
           probably be ignored).
        """
        return self._f_copy(
            newparent, newname, overwrite, createparents, **kwargs )


    def truncate(self, size):
        """Truncate the main dimension to be `size` rows.

        If the main dimension previously was larger than this `size`,
        the extra data is lost.  If the main dimension previously was
        shorter, it is extended, and the extended part is filled with
        the default values.

        The truncation operation can only be applied to *enlargeable*
        datasets, else a `TypeError` will be raised.

        .. Warning:: If you are using the HDF5 1.6.x series, and due to
           limitations of them, `size` must be greater than zero
           (i.e. the dataset can not be completely emptied).  A
           `ValueError` will be issued if you are using HDF5 1.6.x and
           try to pass a zero size to this method.  HDF5 1.8.x doesn't
           undergo this problem.
        """
        # A non-enlargeable arrays (Array, CArray) cannot be truncated
        if self.extdim < 0:
            raise TypeError("non-enlargeable datasets cannot be truncated")
        if (size > 0 or
            (size == 0 and whichLibVersion("hdf5")[1] >= "1.8.0")):
                self._g_truncate(size)
        else:
            raise ValueError("""
`size` must be greater than 0 if you are using HDF5 < 1.8.0.
With HDF5 1.8.0 and higher, `size` can also be 0 or greater.""")


    def isVisible(self):
        """
        Is this node visible?

        This method has the behavior described in `Node._f_isVisible()`.
        """
        return self._f_isVisible()


    # Attribute handling
    # ``````````````````
    def getAttr(self, name):
        """
        Get a PyTables attribute from this node.

        This method has the behavior described in `Node._f_getAttr()`.
        """
        return self._f_getAttr(name)


    def setAttr(self, name, value):
        """
        Set a PyTables attribute for this node.

        This method has the behavior described in `Node._f_setAttr()`.
        """
        self._f_setAttr(name, value)


    def delAttr(self, name):
        """
        Delete a PyTables attribute from this node.

        This method has the behavior described in `Node._f_delAttr()`.
        """
        self._f_delAttr(name)


    # Data handling
    # `````````````
    def flush(self):
        """
        Flush pending data to disk.

        Saves whatever remaining buffered data to disk.  It also
        releases I/O buffers, so if you are filling many datasets in the
        same PyTables session, please call ``flush()`` extensively so as
        to help PyTables to keep memory requirements low.
        """
        self._g_flush()


    def _f_close(self, flush=True):
        """
        Close this node in the tree.

        This method has the behavior described in `Node._f_close()`.
        Besides that, the optional argument `flush` tells whether to
        flush pending data to disk or not before closing.
        """

        if not self._v_isopen:
            return  # the node is already closed or not initialized

        # Only do a flush in case the leaf has an IO buffer.  The
        # internal buffers of HDF5 will be flushed afterwards during the
        # self._g_close() call.  Avoiding an unnecessary flush()
        # operation accelerates the closing for the unbuffered leaves.
        if flush and hasattr(self, "_v_iobuf"):
            self.flush()

        # Close the dataset and release resources
        self._g_close()

        # Close myself as a node.
        super(Leaf, self)._f_close()


    def close(self, flush=True):
        """
        Close this node in the tree.

        This method is completely equivalent to `Leaf._f_close()`.
        """
        self._f_close(flush)



## Local Variables:
## mode: python
## py-indent-offset: 4
## tab-width: 4
## fill-column: 72
## End:
