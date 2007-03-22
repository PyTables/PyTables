########################################################################
#
#       License: BSD
#       Created: October 14, 2002
#       Author:  Francesc Altet - faltet@carabos.com
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

import numpy

import tables
from tables.flavor import ( check_flavor, internal_flavor,
                            alias_map as flavor_alias_map )
from tables import hdf5Extension
from tables import utilsExtension
from tables.node import Node
from tables.filters import Filters
from tables.utils import idx2long, byteorders
from tables.parameters import CHUNKTIMES, BUFFERTIMES
from tables.exceptions import PerformanceWarning



__version__ = "$Revision$"


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

    basesize = 1024*4   # 4KB is one page of memory
    if expectedsizeinMB < 1:
        # Values for files less than 1 MB of size
        chunksize = basesize
    elif (expectedsizeinMB >= 1 and
        expectedsizeinMB < 10):
        # Values for files between 1 MB and 10 MB
        chunksize = 2 * basesize
    elif (expectedsizeinMB >= 10 and
          expectedsizeinMB < 100):
        # Values for sizes between 10 MB and 100 MB
        chunksize = 4 * basesize
    elif (expectedsizeinMB >= 100 and
          expectedsizeinMB < 1000):
        # Values for sizes between 100 MB and 1 GB
        chunksize = 8 * basesize
    elif (expectedsizeinMB >= 1000 and
          expectedsizeinMB < 10000):
        # Values for sizes between 1 GB and 10 GB
        chunksize = 16 * basesize
    else:  # Greater than 10 GB
        chunksize = 32 * basesize

    return chunksize



class Leaf(Node):
    """A class to place common functionality of all Leaf objects.

    A Leaf object is all the nodes that can hang directly from a
    Group, but that are not groups nor attributes. Right now this set
    is composed by Table and Array objects.

    Leaf objects (like Table or Array) will inherit the next methods
    and variables using the mix-in technique.

    Instance variables (in addition to those in `Node`):

    shape -- The shape of data in the leaf.
    maindim -- The dimension along which iterators work.
    nrows -- The length of the main dimension of the leaf data.
    nrowsinbuf -- The number of rows that fits in internal input
        buffers.
        You can change this to fine-tune the speed or memory
        requirements of your application.
    byteorder -- The byte ordering of the leaf data *on disk*.
    filters -- Filter properties for this leaf --see `Filters`.
    name -- The name of this node in its parent group (a string).  An
        alias for `Node._v_name`.
    hdf5name -- The name of this node in the hosting HDF5 file (a string).
        An alias for `Node._v_hdf5name`.
    objectID -- The identifier of this node in the hosting HDF5 file.  An
        alias for `Node._v_objectID`.
    attrs -- The associated `AttributeSet` instance.  An alias for
        `Node._v_attrs`.
    title -- A description for this node.  An alias for `Node._v_title`.
    flavor -- The type of the data object read from this array.  It can
        be any of 'numpy', 'numarray', 'numeric' or 'python' (the set of
        supported flavors depends on which packages you have installed
        on your system).
        Note that, during the reads of ``VLArray`` objects, `flavor`
        only applies to the *components* of the returned python list,
        not to the list itself.
        You can (and are encouraged to) use this property to get, set
        and delete the ``FLAVOR`` HDF5 attribute of the leaf.  When the
        leaf has no such attribute, the default flavor is used.
    chunkshape -- The HDF5 chunk size for chunked leaves (a tuple).
        This is read-only because you cannot change the chunk size of a
        leaf once it has been created.


    Public methods (in addition to those in `Node`):

    flush()
    _f_close([flush])
    close([flush])
    remove()
    rename(newname)
    move([newparent][, newname][, overwrite])
    copy([newparent][, newname][, overwrite][, **kwargs])
    isVisible()
    getAttr(name)
    setAttr(name, value)
    delAttr(name)
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

    hdf5name = property(
        lambda self: self._v_hdf5name, None, None,
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
        "The identifier of this node in the hosting HDF5 file." )

    # Lazy read-only attributes
    # `````````````````````````
    def _getfilters(self):
        mydict = self.__dict__
        if 'filters' in mydict:
            return mydict['filters']
        mydict['filters'] = filters = Filters._from_leaf(self)
        return filters

    filters = property(_getfilters, None, None,
                       "Filter properties for this leaf.")

    # Other properties
    # ````````````````
    def _getmaindim(self):
        if self.extdim < 0:
            return 0  # choose the first dimension
        return self.extdim

    maindim = property(
        _getmaindim, None, None,
        "The main (enlargeable or first) dimension of the array." )

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
        The representation of data read from this array.

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
        "Useful for dealing with Leaf objects as sequences"
        return self.nrows


    def __str__(self):

        """The string reprsentation choosed for this object is its pathname
        in the HDF5 object tree.
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
                self._v_attrs._g__setattr('FLAVOR', self._flavor)
        else:  # get flavor of existing node (if any)
            flavor = getattr(self._v_attrs, 'FLAVOR', internal_flavor)
            self._flavor = flavor_alias_map.get(flavor, flavor)
        assert self._flavor is not None


    def _calc_chunkshape(self, expectedrows, rowsize, itemsize):
        """Calculate the shape for the HDF5 chunk."""

        # Compute the chunksize
        MB = 1024 * 1024
        expectedsizeinMB = (expectedrows * rowsize) / MB
        chunksize = calc_chunksize(expectedsizeinMB)

        # In case of a scalar shape, return the unit chunksize
        if self.shape == ():
            return (1,)

        maindim = self.maindim
        # Compute the chunknitems
        chunknitems = chunksize // itemsize
        # Safeguard against itemsizes being extremely large
        if chunknitems == 0:
            chunknitems = 1
        chunkshape = list(self.shape)
        # Check whether trimming the main dimension is enough
        chunkshape[maindim] = 1
        newchunknitems = numpy.prod(chunkshape)
        if newchunknitems <= chunknitems:
            chunkshape[maindim] = chunknitems // newchunknitems
        else:
            # No, so start trimming other dimensions as well
            for j in xrange(len(chunkshape)):
                # Check whether trimming this dimension is enough
                chunkshape[j] = 1
                newchunknitems = numpy.prod(chunkshape)
                if newchunknitems <= chunknitems:
                    chunkshape[j] = chunknitems // newchunknitems
                    break
            else:
                # Ops, we ran out of the loop without a break
                # Set the last dimension to chunknitems
                chunkshape[-1] = chunknitems

        return tuple(chunkshape)


    def _calc_nrowsinbuf(self, chunkshape, rowsize, itemsize):
        """Calculate the number of rows that fits on a PyTables buffer."""

        # Compute the nrowsinbuf
        chunksize = numpy.prod(chunkshape) * itemsize
        buffersize = chunksize * CHUNKTIMES
        nrowsinbuf = buffersize // rowsize
        # Safeguard against row sizes being extremely large
        if nrowsinbuf == 0:
            nrowsinbuf = 1
            # If rowsize is too large, issue a Performance warning
            maxrowsize = BUFFERTIMES * buffersize
            if rowsize > maxrowsize:
                warnings.warn("""\
array or table ``%s`` is exceeding the maximum recommended rowsize (%d bytes);
be ready to see PyTables asking for *lots* of memory and possibly slow I/O.
You may want to reduce the rowsize by trimming the value of dimensions
that are orthogonal to the main dimension of this array or table.
Alternatively, in case you have specified a very small chunksize,
you may want to increase it."""
                              % (self._v_pathname, maxrowsize),
                                 PerformanceWarning)
# It is difficult to forsee the level of code nesting to reach user code.
#f = sys._getframe(8)
###Caller --> %s (%s:%s)"""
#                    f.f_code.co_name,
#                    f.f_code.co_filename, f.f_lineno,),
        return nrowsinbuf


    # This method is appropriate for calls to __getitem__ methods
    def _processRange(self, start, stop, step, dim=None):
        if dim is None:
            nrows = self.nrows  # self.shape[self.maindim]
        else:
            nrows = self.shape[dim]

        if step and step < 0:
            raise ValueError("slice step cannot be negative")
        # (start, stop, step) = slice(start, stop, step).indices(nrows)  # Python > 2.3
        # The next function is a substitute for slice().indices in order to
        # support full 64-bit integer for slices (Python 2.4 does not
        # support that yet)
        # F. Altet 2005-05-08
        # In order to convert possible numpy.integer values to long ones
        # F. Altet 2006-05-02
        if start is not None: start = idx2long(start)
        if stop is not None: stop = idx2long(stop)
        if step is not None: step = idx2long(step)
        (start, stop, step) = utilsExtension.getIndices(
            slice(start, stop, step), long(nrows) )

        # Some protection against empty ranges
        if start > stop:
            start = stop
        return (start, stop, step)


    # This method is appropiate for calls to read() methods
    def _processRangeRead(self, start, stop, step):
        nrows = self.nrows
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
        start, stop, step = self._processRange(start, stop, step)

        return (start, stop, step)


    def _g_copy(self, newParent, newName, recursive, _log=True, **kwargs):
        # Compute default arguments.
        start = kwargs.get('start', 0)
        stop = kwargs.get('stop', self.nrows)
        step = kwargs.get('step', 1)
        title = kwargs.get('title', self._v_title)
        filters = kwargs.get('filters', self.filters)
        stats = kwargs.get('stats', None)

        # Fix arguments with explicit None values for backwards compatibility.
        if stop is None:  stop = self.nrows
        if title is None:  title = self._v_title
        if filters is None:  filters = self.filters

        # Compute the correct indices.
        (start, stop, step) = self._processRangeRead(start, stop, step)

        # Create a copy of the object.
        (newNode, bytes) = self._g_copyWithStats(
            newParent, newName, start, stop, step, title, filters, _log)

        # Copy user attributes if requested (or the flavor at least).
        if kwargs.get('copyuserattrs', True):
            self._v_attrs._g_copy(newNode._v_attrs)
        elif 'FLAVOR' in self._v_attrs:
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
            Specify the range of rows in child leaves to be copied;
            the default is to copy all the rows.
        `stats`
            This argument may be used to collect statistics on the
            copy process.  When used, it should be a dictionary whith
            keys ``'groups'``, ``'leaves'`` and ``'bytes'`` having a
            numeric value.  Their values will be incremented to
            reflect the number of groups, leaves and bytes,
            respectively, that have been copied during the operation.
        """
        return self._f_copy(
            newparent, newname, overwrite, createparents, **kwargs )


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

        Saves whatever remaining buffered data to disk.
        """
        self._g_flush()


    def _f_close(self, flush=True):
        """
        Close this node in the tree.

        This method has the behavior described in `Node._f_close()`.
        Besides that, the optional argument `flush` tells whether to
        flush pending data to disk or not before closing.
        """

        if not hasattr(self, "_v_isopen"):
            return  # the node is probably being aborted during creation time
        if not self._v_isopen:
            return  # the node is already closed

        if flush:
            self.flush()

        # Close the dataset and release resources
        self._g_close()

        # Close myself as a node.
        super(Leaf, self)._f_close()


    def close(self, flush=True):
        """
        Close this node in the tree.

        This method is completely equivalent to ``_f_close()``.
        """
        self._f_close(flush)



## Local Variables:
## mode: python
## py-indent-offset: 4
## tab-width: 4
## fill-column: 72
## End:
