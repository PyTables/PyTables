########################################################################
#
#       License: BSD
#       Created: October 14, 2002
#       Author:  Francesc Alted - falted@openlc.org
#
#       $Source: /home/ivan/_/programari/pytables/svn/cvs/pytables/pytables/tables/Leaf.py,v $
#       $Id: Leaf.py,v 1.35 2004/02/04 10:28:27 falted Exp $
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

__version__ = "$Revision: 1.35 $"

import types, warnings
from utils import checkNameValidity, calcBufferSize, processRangeRead
from AttributeSet import AttributeSet
import Group
import hdf5Extension

class Filters:
    """Container for filter properties

    Instance variables:

        complevel -- the compression level (0 means no compression)
        complib -- the compression filter used (in case of compressed dataset)
        shuffle -- whether the shuffle filter is active or not
        fletcher32 -- whether the fletcher32 filter is active or not

    """

    def __init__(self, complevel=0, complib="zlib", shuffle=1, fletcher32=0):
        """Create a new Filters instance
        
        compress -- Specifies a compress level for data. The allowed
            range is 0-9. A value of 0 disables compression and this
            is the default.

        complib -- Specifies the compression library to be used. Right
            now, "zlib", "lzo" and "ucl" values are supported.

        shuffle -- Whether or not to use the shuffle filter in the
            HDF5 library. This is normally used to improve the
            compression ratio. A value of 0 disables shuffling and 1
            makes it active. The default value depends on whether
            compression is enabled or not; if compression is enabled,
            shuffling defaults to be active, else shuffling is
            disabled.

        fletcher32 -- Whether or not to use the fletcher32 filter in
            the HDF5 library. This is used to add a checksum on each
            data chunk. A value of 0 disables the checksum and it is
            the default.

            """
        if complib not in ["zlib","lzo","ucl"]:
            raise ValueError, "Wrong \'complib\' parameter value: '%s'. It only can take the values: 'zlib', 'lzo' and 'ucl'." %  (str(complib))
        if shuffle and not complevel:
            # Shuffling and not compressing makes non sense
            shuffle = 0
        self.complevel = complevel
        self.shuffle = shuffle
        self.fletcher32 = fletcher32
        # Select the library to do compression
        if hdf5Extension.whichLibVersion(complib)[0]:
            self.complib = complib
        else:
            warnings.warn( \
"%s compression library is not available. Using zlib instead!." %(complib))
            self.complib = "zlib"   # Should always exists

    def __str__(self):
        """The string reprsentation choosed for this object is its pathname
        in the HDF5 object tree.
        """
        
        filters = ""
        if self.fletcher32:
            filters += ", fletcher32"
        if self.complevel:
            if self.shuffle:
                filters += ", shuffle"
            filters += ", %s(%s)" % (self.complib, self.complevel)
        return filters

class Leaf:
    """A class to place common functionality of all Leaf objects.

    A Leaf object is all the nodes that can hang directly from a
    Group, but that are not groups nor attributes. Right now this set
    is composed by Table and Array objects.

    Leaf objects (like Table or Array) will inherit the next methods
    and variables using the mix-in technique.

    Methods:

        close()
        flush()
        getAttr(attrname)
        rename(newname)
        remove()
        setAttr(attrname, attrvalue)

    Instance variables:

        name -- the leaf node name
        hdf5name -- the HDF5 leaf node name
        objectID -- the HDF5 object ID of the Leaf node
        title -- the leaf title (actually a property)
        shape -- the leaf shape
        byteorder -- the byteorder of the leaf
        filters -- information for active filters
        attrs -- The associated AttributeSet instance

    """

    
    def _g_putObjectInTree(self, name, parent):
        """Given a new Leaf object (fresh or in a HDF5 file), set
        links and attributes to include it in the object tree."""
        
        # New attributes for the this Leaf instance
        parent._g_setproperties(name, self)
        self.name = self._v_name     # This is a standard attribute for Leaves
        # Call the new method in Leaf superclass 
        self._g_new(parent, self._v_hdf5name)
        # Update this instance attributes
        parent._v_leaves[self._v_name] = self
        # Update class variables
        parent._v_file.leaves[self._v_pathname] = self
        if self._v_new:
            self._create()
        else:
            self._open()

    # Define attrs as a property. This saves us 0.7s/3.8s
    def _get_attrs (self):
        return AttributeSet(self)
    # attrs can't be set or deleted by the user
    attrs = property(_get_attrs, None, None, "Attrs of this object")

    # Define title as a property
    def _get_title (self):
        return self.attrs.TITLE
    
    def _set_title (self, title):
        self.attrs.TITLE = title
    # Define a property.  The 'delete this attribute'
    # method is defined as None, so the attribute can't be deleted.
    title = property(_get_title, _set_title, None, "Title of this object")

    def _g_setFilters(self, filters):
        if filters is None:
            # If no filters, return the defaults
            return Filters()
        else:
            return filters

    def _g_getFilters(self):
        # Create a filters instance with default values
        filters = Filters()
        # Get a dictionary with all the filters
        filtersDict = hdf5Extension._getFilters(self._v_parent._v_objectID,
                                                self._v_hdf5name)
        if filtersDict:
            for name in filtersDict:
                if name.startswith("lzo"):
                    filters.complib = "lzo"
                    filters.complevel = filtersDict[name][0]
                elif name.startswith("ucl"):
                    filters.complib = "ucl"
                    filters.complevel = filtersDict[name][0]
                elif name.startswith("deflate"):
                    filters.complib = "zlib"
                    filters.complevel = filtersDict[name][0]
                elif name.startswith("shuffle"):
                    filters.shuffle = 1
                elif name.startswith("fletcher32"):
                    filters.fletcher32 = 1
        return filters
        
    def _g_renameObject(self, newname):
        """Rename this leaf in the object tree as well as in the HDF5 file."""

        parent = self._v_parent
        newattr = self.__dict__

        # Delete references to the oldname
        del parent._v_file.leaves[self._v_pathname]
        del parent._v_file.objects[self._v_pathname]
        del parent._v_leaves[self._v_name]
        del parent._v_childs[self._v_name]
        del parent.__dict__[self._v_name]

        # Get the alternate name (if any)
        trMap = self._v_rootgroup._v_parent.trMap
        
        # New attributes for the this Leaf instance
        newattr["_v_name"] = newname
        newattr["_v_hdf5name"] = trMap.get(newname, newname)
        newattr["_v_pathname"] = parent._g_join(newname)
        
        # Update class variables
        parent._v_file.objects[self._v_pathname] = self
        parent._v_file.leaves[self._v_pathname] = self

        # Standard attribute for Leaves
        self.name = newname
        self.hdf5name = trMap.get(newname, newname)
        
        # Call the _g_new method in Leaf superclass 
        self._g_new(parent, self._v_hdf5name)
        
        # Update this instance attributes
        parent._v_childs[newname] = self
        parent._v_leaves[newname] = self
        parent.__dict__[newname] = self
        
    def copy(self, where, name, start=0, stop=None, step=1,
             title=None, filters=None, copyuserattrs=1):
        """Copy this leaf to other location

        where -- the group where the leaf will be copied.
        name -- the name of the new leaf.
        start -- the row to start copying.
        stop -- the row to cease copying. None means last row.
        step -- the increment of the row number during the copy
        title -- the new title for destination. If None, the original
            title is kept.
        filters -- An instance of the Filters class. A None value means
            that the source properties are copied as is.
        copyuserattrs -- Whether copy the user attributes of the source leaf
            to the destination or not. The default is copy them.

        """

        # First, check if the copy() method has been defined for this object
        if not hasattr(self, "_g_copy"):
            raise NotImplementedError, \
                  "<%s> has not a copy() method" % str(self)
             
        if isinstance(where, str):
            if where not in self._v_file.objects:
                raise LookupError, "'%s' path cannot be found in file '%s'" % \
                      (where, self._v_filename)
            if where in self._v_file.groups:
                group = self._v_file.groups[where]
            else:
                raise LookupError, "Path '%s' is not a group '%s'"
        elif isinstance(where, Group.Group):
            group = where
        elif where == None:
            group = self._v_parent
        else:
            raise TypeError, \
"'where' has to be a Group or string instance, not type '%s'" % (type(where))
        # Get the correct indices (all the Leafs have nrows attribute)
        if stop == None:
            stop = self.nrows
        (start, stop, step) = processRangeRead(self.nrows, start, stop, step)
        if title == None: title = self.title
        if filters == None: filters = self.filters

        # Call the part of copy() that depends on the kind of the leaf
        object = self._g_copy(group, name, start, stop, step, title, filters)

        # Finally, copy the user attributes, if needed
        if copyuserattrs:
            for attrname in self.attrs._v_attrnamesuser:
                setattr(object.attrs, attrname, getattr(self.attrs, attrname))
        
        return object

    def remove(self):
        "Remove a leaf"
        parent = self._v_parent
        parent._g_deleteLeaf(self._v_name)
        self.close()

    def rename(self, newname):
        """Rename a leaf"""

        # Check for name validity
        checkNameValidity(newname)
        # Check if self has a child with the same name
        if newname in self._v_parent._v_childs:
            raise RuntimeError, \
        """Another sibling (%s) already has the name '%s' """ % \
                   (self._v_parent._v_childs[newname], newname)
        # Rename all the appearances of oldname in the object tree
        oldname = self._v_name
        self._g_renameObject(newname)
        self._v_parent._g_renameNode(oldname, newname)
        
    def getAttr(self, attrname):
        """Get a leaf attribute as a string"""

        return getattr(self.attrs, attrname, None)
        
    def setAttr(self, attrname, attrvalue):
        """Set a leaf attribute as a string"""

        setattr(self.attrs, attrname, attrvalue)

    def flush(self):
        """Save whatever remaining data in buffer"""
        # This is a do-nothing fall-back method

    def close(self):
        """Flush the buffers and close this object on tree"""
        self.flush()
        parent = self._v_parent
        del parent._v_leaves[self._v_name]
        del parent.__dict__[self._v_name]
        del parent._v_childs[self._v_name]
        parent.__dict__["_v_nchilds"] -= 1
        del parent._v_file.leaves[self._v_pathname]
        del parent._v_file.objects[self._v_pathname]
        del self._v_parent
        del self._v_rootgroup
        del self._v_file
        # Detach the AttributeSet instance
        # This has to called in this manner
        #del self.__dict__["attrs"]
        # The next also work!
        # In some situations, this maybe undefined
        if hasattr(self, "attrs"): 
            self.attrs._f_close()
            del self.attrs

        # After the objects are disconnected, destroy the
        # object dictionary using the brute force ;-)
        # This should help to the garbage collector
        #self.__dict__.clear()

    def __str__(self):
        """The string reprsentation choosed for this object is its pathname
        in the HDF5 object tree.
        """
        
        # Get this class name
        classname = self.__class__.__name__
        # The title
        title = self.attrs.TITLE
        return "%s (%s%s%s) %r" % \
               (self._v_pathname, classname, self.shape, self.filters, title)

