########################################################################
#
#       License: BSD
#       Created: October 10, 2002
#       Author:  Francesc Alted - falted@openlc.org
#
#       $Source: /home/ivan/_/programari/pytables/svn/cvs/pytables/pytables/tables/Array.py,v $
#       $Id: Array.py,v 1.33 2003/09/15 19:22:48 falted Exp $
#
########################################################################

"""Here is defined the Array class.

See Array class docstring for more info.

Classes:

    Array

Functions:


Misc variables:

    __version__


"""

__version__ = "$Revision: 1.33 $"
import types, warnings, sys
from Leaf import Leaf
import hdf5Extension
import numarray
import numarray.strings as strings

try:
    import Numeric
    Numeric_imported = 1
except:
    Numeric_imported = 0

class Array(Leaf, hdf5Extension.Array, object):
    """Represent an homogeneous dataset in HDF5 file.

    It enables to create new datasets on-disk from Numeric, numarray,
    lists, tuples, strings or scalars, or open existing ones.

    All Numeric and numarray typecodes are supported except for complex
    datatypes.

    Methods:

      Common to all leaves:
        close()
        flush()
        getAttr(attrname)
        rename(newname)
        remove()
        setAttr(attrname, attrvalue)
        
      Specific of Array:
        read()

    Instance variables:

      Common to all leaves:
        name -- the leaf node name
        hdf5name -- the HDF5 leaf node name
        title -- the leaf title
        shape -- the leaf shape
        byteorder -- the byteorder of the leaf
        
      Specific of Array:
        type -- the type class for the array
        flavor -- the object type of this object (Numarray, Numeric, List,
                  Tuple, String, Int of Float)

    """
    
    def __init__(self, object = None, title = "", atomictype = 1):
        """Create the instance Array.

        Keyword arguments:

        object -- Regular object to be saved. It can be any of
                  Numarray, Numeric, List, Tuple, String, Int of Float
                  types, provided that they are regular (i.e. they are
                  not like [[1,2],2]). If None, the metadata for the
                  array will be taken from disk.

        "title" -- Sets a TITLE attribute on the HDF5 array entity.
        "atomictype" -- Whether an HDF5 atomic datatype or H5T_ARRAY
                        is to be used.

        """
        # Check if we have to create a new object or read their contents
        # from disk
        if object is not None:
            self._v_new = 1
            self.object = object
            self.new_title = title
            self.atomictype = atomictype
        else:
            self._v_new = 0
            
    def _create(self):
        """Save a fresh array (i.e., not present on HDF5 file)."""
        # Call the _createArray superclass method to create the table
        # on disk

        obversion = "1.0"    # default version for ARRAY objects
        arr = self.object
        self.byteorder = sys.byteorder  # Default byteorder
        # Check for Numeric objects
        if isinstance(arr, numarray.NumArray):
            flavor = "NumArray"
            naarr = arr
            self.byteorder = arr._byteorder
        elif (Numeric_imported and type(arr) == type(Numeric.array(1))):
            flavor = "Numeric"
            if arr.typecode() == "c":
                # To emulate as close as possible Numeric character arrays,
                # itemsize for chararrays will be always 1
                if arr.iscontiguous():
                    # This the fastest way to convert from Numeric to numarray
                    # because no data copy is involved
                    naarr = strings.array(buffer(arr),
                                            itemsize=1,
                                            shape=arr.shape)
                else:
                    # Here we absolutely need a copy so as to obtain a buffer.
                    # Perhaps this can be avoided or optimized by using
                    # the tolist() method, but this should be tested.
                    naarr = strings.array(buffer(arr.copy()),
                                            itemsize=1,
                                            shape=arr.shape)
            else:
                if arr.iscontiguous():
                    # This the fastest way to convert from Numeric to numarray
                    # because no data copy is involved
                    naarr = numarray.array(buffer(arr),
                                           type=arr.typecode(),
                                           shape=arr.shape)
                else:
                    # Here we absolutely need a copy in order
                    # to obtain a buffer.
                    # Perhaps this can be avoided or optimized by using
                    # the tolist() method, but this should be tested.
                    naarr = numarray.array(buffer(arr.copy()),
                                           type=arr.typecode(),
                                           shape=arr.shape)                    

        elif (isinstance(arr, strings.CharArray)):
            flavor = "CharArray"
            naarr = arr
            self.byteorder = "non-relevant" 
        elif (isinstance(arr, types.TupleType) or
              isinstance(arr, types.ListType)):
            # Test if can convert to numarray object
            try:
                naarr = numarray.array(arr)
            # If not, test with a chararray
            except TypeError:
                try:
                    naarr = strings.array(arr)
                # If still doesn't, issues an error
                except:
                    raise ValueError, \
"""The object '%s' can't be converted to a numerical or character array.
  Sorry, but this object is not supported.""" % (arr)
            if isinstance(arr, types.TupleType):
                flavor = "Tuple"
            else:
                flavor = "List"
        elif isinstance(arr, types.IntType):
            naarr = numarray.array(arr)
            flavor = "Int"
        elif isinstance(arr, types.FloatType):
            naarr = numarray.array(arr)

            flavor = "Float"
        elif isinstance(arr, types.StringType):
            naarr = strings.array(arr)
            flavor = "String"
        else:
            raise ValueError, \
"""The object '%s' is not in the list of supported objects (numarray,
  chararray,homogeneous list or homogeneous tuple, int, float or str).
  Sorry, but this object is not supported.""" % (arr)

        if naarr.shape == (0,):
            raise ValueError, \
"""The object '%s' has a zero sized dimension.
  Sorry, but this object is not supported.""" % (arr)
            
            
        self.type = self._createArray(naarr, self.new_title,
                                           flavor, obversion, self.atomictype)
        # Get some important attributes
        self.shape = naarr.shape
        self.itemsize = naarr._itemsize
        self.flavor = flavor

    def _open(self):
        """Get the metadata info for an array in file."""
        (self.type, self.shape, self.itemsize, self.byteorder) = \
                        self._openArray()

    # Accessor for the _readArray method in superclass
    def read(self):
        """Read the array from disk and return it as numarray."""

        if repr(self.type) == "CharType":
            #print "self.shape ==>", self.shape
            #print "self.shape 2 ==>", self.itemsize
            arr = strings.array(None, itemsize=self.itemsize,
                                  shape=self.shape)
        else:
            arr = numarray.array(buffer=None,
                                 type=self.type,
                                 shape=self.shape)
            # Set the same byteorder than on-disk
            arr._byteorder = self.byteorder
        # Do the actual data read
        self._readArray(arr._data)

        # Numeric, NumArray, CharArray, Tuple, List, String, Int or Float
        self.flavor = self.getAttr("FLAVOR")
        
        # Convert to Numeric, tuple or list if needed
        if self.flavor == "Numeric":
            if Numeric_imported:
                # This works for both numeric and chararrays
                # arr=Numeric.array(arr, typecode=arr.typecode())
                # The next is 10 times faster (for tolist(),
                # we should check for tostring()!)
                if repr(self.type) == "CharType":
                    arrstr = arr.tostring()
                    arr=Numeric.reshape(Numeric.array(arrstr), arr.shape)
                else:
                    # tolist() method creates a list with a sane byteorder
                    if arr.shape <> ():
                        arr=Numeric.array(arr.tolist(), typecode=arr.typecode())
                    else:
                        # This works for rank-0 arrays
                        # (but is slower for big arrays)
                        arr=Numeric.array(arr, typecode=arr.typecode())
                        
            else:
                # Warn the user
                warnings.warn( \
"""The object on-disk is type Numeric, but Numeric is not installed locally.
  Returning a numarray object instead!.""")
        elif self.flavor == "Tuple":
            arr = tuple(arr.tolist())
        elif self.flavor == "List":
            arr = arr.tolist()
        elif self.flavor == "Int":
            arr = int(arr)
        elif self.flavor == "Float":
            arr = float(arr)
        elif self.flavor == "String":
            arr = arr.tostring()
        
        return arr
        
    # Moved out of scope
    def _g_del__(self):
        """Delete some objects"""
        print "Deleting Array object"
        pass

    def __repr__(self):
        """This provides more metainfo in addition to standard __str__"""

        return "%s\n  type = %r\n  itemsize = %r\n  flavor = %r\n  byteorder = %r" % \
               (self, self.type, self.itemsize, self.attrs.FLAVOR, self.byteorder)
