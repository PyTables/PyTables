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

    Filters
    Leaf

Functions:


Misc variables:

    __version__


"""

import warnings

import tables
import tables.hdf5Extension as hdf5Extension
import tables.utilsExtension as utilsExtension
from tables.utils import processRangeRead
from tables.Node import Node



__version__ = "$Revision$"



class Filters(object):
    """Container for filter properties

    Instance variables:

        complevel -- the compression level (0 means no compression)
        complib -- the compression filter used (in case of compressed dataset)
        shuffle -- whether the shuffle filter is active or not
        fletcher32 -- whether the fletcher32 filter is active or not

    """

    def __init__(self, complevel=0, complib="zlib", shuffle=True,
                 fletcher32=False):
        """Create a new Filters instance

        complevel -- Specifies a compress level for data. The allowed
            range is 0-9. A value of 0 disables compression and this
            is the default.

        complib -- Specifies the compression library to be used. Right
            now, 'zlib', 'lzo', 'ucl' and 'bzip2' values are supported.
            If None, then 'zlib' is chosen.

        shuffle -- Whether or not to use the shuffle filter in the HDF5
            library. This is normally used to improve the compression
            ratio. A value of False disables shuffling and True makes it
            active. The default value depends on whether compression is
            enabled or not; if compression is enabled, shuffling
            defaults to be active, else shuffling is disabled.

        fletcher32 -- Whether or not to use the fletcher32 filter in the
            HDF5 library. This is used to add a checksum on each data
            chunk. A value of False disables the checksum and it is the
            default.
        """

        libnames = ('zlib', 'lzo', 'ucl', 'bzip2', 'szip')

        if complib is None:
            complib = "zlib"
        if complib not in libnames:
            raise ValueError("unsupported library %r; it must be one of %s"
                             % (complib, str(libnames)[1:-1]))
        if complib == "ucl":
            warnings.warn(DeprecationWarning("""\
UCL is being deprecated. Please, try to avoid using it if you can.
You can use the ptrepack utility to migrate datafiles compressed with UCL."""))

        if shuffle and not complevel:
            # Shuffling and not compressing makes non sense
            shuffle = False
        self.complevel = complevel
        self.shuffle = shuffle
        self.fletcher32 = fletcher32
        # Select the library to do compression
        if utilsExtension.whichLibVersion(complib) is not None:
            self.complib = complib
        else:
            warnings.warn( \
"%s compression library is not available. Using zlib instead!." %(complib))
            self.complib = "zlib"   # Should always exists

    def __repr__(self):
        """The string reprsentation choosed for this object.
        """
        filters = "Filters("
#         if self.complevel:
#             filters += "complevel=%s" % (self.complevel)
#             filters += ", complib='%s'" % (self.complib)
#             if self.shuffle:
#                 filters += ", shuffle=%s" % (self.shuffle)
#             if self.fletcher32:
#                 filters += ", "
#         if self.fletcher32:
#             filters += "fletcher32=%s" % (self.fletcher32)
        filters += "complevel=%s" % (self.complevel)
        filters += ", complib='%s'" % (self.complib)
        filters += ", shuffle=%s" % (self.shuffle)
        filters += ", fletcher32=%s" % (self.fletcher32)
        filters += ")"
        return filters

    def __str__(self):
        """The string reprsentation choosed for this object.
        """

        return repr(self)



class Leaf(Node):
    """A class to place common functionality of all Leaf objects.

    A Leaf object is all the nodes that can hang directly from a
    Group, but that are not groups nor attributes. Right now this set
    is composed by Table and Array objects.

    Leaf objects (like Table or Array) will inherit the next methods
    and variables using the mix-in technique.

    Instance variables (in addition to those in `Node`):

    shape
        The shape of data in the leaf.
    byteorder
        The byte ordering of data in the leaf.
    filters
        Filter properties for this leaf --see `Filters`.

    name
        The name of this node in its parent group (a string).  An
        alias for `Node._v_name`.
    hdf5name
        The name of this node in the hosting HDF5 file (a string).  An
        alias for `Node._v_hdf5name`.
    objectID
        The identifier of this node in the hosting HDF5 file.  An
        alias for `Node._v_objectID`.
    attrs
        The associated `AttributeSet` instance.  An alias for
        `Node._v_attrs`.
    title
        A description for this node.  An alias for `Node._v_title`.

    Public methods (in addition to those in `Node`):

    flush()
        Flush pending data to disk.
    _f_close([flush])
        Close this node in the tree.
    close([flush])
        Close this node in the tree.
    remove()
        Remove this node from the hierarchy.
    rename(newname)
        Rename this node in place.
    move([newparent][, newname][, overwrite])
        Move or rename this node.
    copy([newparent][, newname][, overwrite][, **kwags])
        Copy this node and return the new one.
    isVisible()
        Is this node visible?

    getAttr(name)
        Get a PyTables attribute from this node.
    setAttr(name, value)
        Set a PyTables attribute for this node.
    delAttr(name)
        Delete a PyTables attribute from this node.
    """

    # <properties>

    # These are a little hard to override, but so are properties.

    # `attrs` is an alias of `_v_attrs`.
    attrs = Node._v_attrs
    # `title` is an alias of `_v_title`.
    title = Node._v_title


    # The following are read-only aliases of their `Node` counterparts.

    def _g_getname(self):
        return self._v_name
    name = property(
        _g_getname, None, None,
        "The name of this node in its parent group (a string).")

    def _g_gethdf5name(self):
        return self._v_hdf5name
    hdf5name = property(
        _g_gethdf5name, None, None,
        "The name of this node in its parent group (a string).")

    def _g_getobjectid(self):
        return self._v_objectID
    objectID = property(
        _g_getobjectid, None, None,
        "The identifier of this node in the hosting HDF5 file.")


    # `filters` is defined as a lazy read-only attribute.

    def _getfilters(self):
        mydict = self.__dict__
        if 'filters' in mydict:
            return mydict['filters']
        else:
            mydict['filters'] = filters = self._g_getFilters()
            return filters

    filters = property(_getfilters, None, None,
                       "Filter properties for this leaf.")

    # </properties>


    def __init__(self, parentNode, name,
                 new=False, filters=None,
                 _log=True):
        self._v_new = new
        """Is this the first time the node has been created?"""

        if new:
            if filters is None:
                # If no filter properties have been given,
                # get the default onesfilter from the parent.
                filters = parentNode._v_filters
            self.__dict__['filters'] = filters  # bypass the property
            # Writing the `Filters` object to an attribute on disk is
            # not necessary for now, as retrieving the filters using
            # `utilsExtension.getFilters()` is safer and faster.
            # Also, cPickling the `filters` attribute is very slow (it
            # is as much as twice slower than the normal overhead for
            # creating a Table, for example).

        # Existing filters need not be read since `filters`
        # is a lazy property that automatically handles their loading.

        super(Leaf, self).__init__(parentNode, name, _log)


    def _g_getFilters(self):
        # Create a filters instance with default values
        filters = Filters()
        # Get a dictionary with all the filters
        filtersDict = utilsExtension.getFilters(self._v_parent._v_objectID,
                                                self._v_hdf5name)
        if filtersDict:
            for name in filtersDict:
                if name.startswith("lzo"):
                    filters.complib = "lzo"
                    filters.complevel = filtersDict[name][0]
                elif name.startswith("ucl"):
                    filters.complib = "ucl"
                    filters.complevel = filtersDict[name][0]
                elif name.startswith("bzip2"):
                    filters.complib = "bzip2"
                    filters.complevel = filtersDict[name][0]
                elif name.startswith("deflate"):
                    filters.complib = "zlib"
                    filters.complevel = filtersDict[name][0]
                elif name.startswith("szip"):
                    filters.complib = "szip"
                    #filters.complevel = filtersDict[name][0]
                    filters.complevel = 1  # Because there is not a compression
                                           # level equivalent for szip
                elif name.startswith("shuffle"):
                    filters.shuffle = True
                elif name.startswith("fletcher32"):
                    filters.fletcher32 = True

        return filters


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
        (start, stop, step) = processRangeRead(self.nrows, start, stop, step)

        # Create a copy of the object.
        (newNode, bytes) = self._g_copyWithStats(
            newParent, newName, start, stop, step, title, filters, _log)

        # Copy user attributes if needed.
        if kwargs.get('copyuserattrs', True):
            self._v_attrs._g_copy(newNode._v_attrs)

        # Update statistics if needed.
        if stats is not None:
            stats['leaves'] += 1
            stats['bytes'] += bytes

        return newNode


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


    def move(self, newparent=None, newname=None, overwrite=False):
        """
        Move or rename this node.

        This method has the behavior described in `Node._f_move()`.
        """
        self._f_move(newparent, newname, overwrite)


    def copy(self, newparent=None, newname=None, overwrite=False, **kwargs):
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
        return self._f_copy(newparent, newname, overwrite, **kwargs)


    def isVisible(self):
        """
        Is this node visible?

        This method has the behavior described in `Node._f_isVisible()`.
        """
        return self._f_isVisible()


    # <attribute handling>

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

    # </attribute handling>


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



## Local Variables:
## mode: python
## py-indent-offset: 4
## tab-width: 4
## fill-column: 72
## End:
