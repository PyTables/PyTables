########################################################################
#
#       License: BSD
#       Created: October 10, 2002
#       Author:  Francesc Alted - falted@openlc.org
#
#       $Source: /home/ivan/_/programari/pytables/svn/cvs/pytables/pytables/tables/Array.py,v $
#       $Id: Array.py,v 1.20 2003/02/24 12:06:00 falted Exp $
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

__version__ = "$Revision: 1.20 $"
import types, warnings, sys
from Leaf import Leaf
import hdf5Extension
import chararray
import numarray

try:
    import Numeric
    Numeric_imported = 1
except:
    Numeric_imported = 0

class Array(Leaf, hdf5Extension.Array):
    """Represent a Numeric Array in HDF5 file.

    It provides methods to create new arrays or open existing ones, as
    well as methods to write/read data and metadata to/from array
    objects over the HDF5 file.

    All Numeric typecodes are supported except "F" and "D" which
    corresponds to complex datatypes.

    Methods:

        read()
        flush()  # This can be moved to Leaf
        close()  # This can be moved to Leaf

    Instance variables:

        name -- the node name
        title -- the node title  # This can be moved to Leaf
        shape -- tuple with the array shape (in Numeric sense)
        type -- the type class for the array

    """
    
    def __init__(self, NumericObject = None, title = "", atomictype = 1):
        """Create the instance Array.

        Keyword arguments:

        NumericObject -- Numeric array to be saved. If None, the
            metadata for the array will be taken from disk.

        "title" -- Sets a TITLE attribute on the HDF5 array entity.
        "atomictype" -- Whether an HDF5 atomic datatype or H5T_ARRAY
                        is to be used.

        """
        # Check if we have to create a new object or read their contents
        # from disk
        if NumericObject is not None:
            self._v_new = 1
            self.object = NumericObject
            self.title = title
            self.atomictype = atomictype
        else:
            self._v_new = 0
            
    def create(self):
        """Save a fresh array (i.e., not present on HDF5 file)."""
        # Call the createArray superclass method to create the table
        # on disk

        obversion = "1.0"    # default version for ARRAY objects
        arr = self.object
        self.byteorder = sys.byteorder  # Default byteorder
        # Check for Numeric objects
        if isinstance(arr, numarray.NumArray):
            flavor = "NUMARRAY"
            naarr = arr
            self.byteorder = arr._byteorder
        elif (Numeric_imported and type(arr) == type(Numeric.array(1))):
            flavor = "NUMERIC"
            if arr.typecode() == "c":
                # To emulate as close as possible Numeric character arrays,
                # itemsize for chararrays will be always 1
                if arr.iscontiguous():
                    # This the fastest way to convert from Numeric to numarray
                    # because no data copy is involved
                    naarr = chararray.array(buffer(arr),
                                            itemsize=1,
                                            shape=arr.shape)
                else:
                    # Here we absolutely need a copy so as to obtain a buffer.
                    # Perhaps this can be avoided or optimized by using
                    # the tolist() method, but this should be tested.
                    naarr = chararray.array(buffer(arr.copy()),
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

        elif (isinstance(arr, chararray.CharArray)):
            flavor = "CHARARRAY"
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
                    naarr = chararray.array(arr)
                # If still doesn't, issues an error
                except:
                    raise ValueError, \
"""The object '%s' can't be converted to a numerical or character array.
  Sorry, but this object is not supported.""" % (arr)
            if isinstance(arr, types.TupleType):
                flavor = "TUPLE"
            else:
                flavor = "LIST"
        elif isinstance(arr, types.IntType):
            naarr = numarray.array(arr)
            flavor = "INT"
        elif isinstance(arr, types.FloatType):
            naarr = numarray.array(arr)
            flavor = "FLOAT"
        elif isinstance(arr, types.StringType):
            naarr = chararray.array(arr)
            flavor = "STRING"
        else:
            raise ValueError, \
"""The object '%s' is not in the list of supported objects (numarray,
  chararray,homogeneous list or homogeneous tuple, int, float or str).
  Sorry, but this object is not supported.""" % (arr)

        if naarr.shape == (0,):
            raise ValueError, \
"""The object '%s' has a zero sized dimension.
  Sorry, but this object is not supported.""" % (arr)
            
            
        self.typeclass = self.createArray(naarr, self.title,
                                     flavor, obversion, self.atomictype)
        # Get some important attributes
        self.shape = naarr.shape
        self.itemsize = naarr._itemsize

    def open(self):
        """Get the metadata info for an array in file."""
        (self.typeclass, self.shape, self.itemsize, self.byteorder) = \
                        self.openArray()

        self.title = self.getAttrStr("TITLE")
        # NUMERIC, NUMARRAY, TUPLE, LIST or other flavor 
        self.flavor = self.getAttrStr("FLAVOR")
        
    # Accessor for the readArray method in superclass
    def read(self):
        """Read the array from disk and return it as numarray."""

        if repr(self.typeclass) == "CharType":
            #print "self.shape ==>", self.shape
            #print "self.shape 2 ==>", self.itemsize
            arr = chararray.array(None, itemsize=self.itemsize,
                                  shape=self.shape)
        else:
            arr = numarray.array(buffer=None,
                                 type=self.typeclass,
                                 shape=self.shape)
            # Set the same byteorder than on-disk
            arr._byteorder = self.byteorder
        # Do the actual data read
        self.readArray(arr._data)

        # Convert to Numeric, tuple or list if needed
        if self.flavor == "NUMERIC":
            if Numeric_imported:
                # This works for both numeric and chararrays
                # arr=Numeric.array(arr, typecode=arr.typecode())
                # The next is 10 times faster (for tolist(),
                # we should check for tostring()!)
                if repr(self.typeclass) == "CharType":
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
        elif self.flavor == "TUPLE":
            arr = tuple(arr.tolist())
        elif self.flavor == "LIST":
            arr = arr.tolist()
        elif self.flavor == "INT":
            arr = int(arr)
        elif self.flavor == "FLOAT":
            arr = float(arr)
        elif self.flavor == "STRING":
            arr = arr.tostring()
        
        return arr
        
    def flush(self):
        """Save whatever remaining data in buffer."""
        # This is a do nothing method because, at the moment the Array
        # class don't support buffers
    
    def close(self):
        """Flush the array buffers and close this object on file."""
        self.flush()
        # Delete the reference to the array object
        #if hasattr(self, "object"):
        #    del self.object

    # Moved out of scope
    def _f_del__(self):
        """Delete some objects"""
        print "Deleting Array object"
        pass

    def __repr__(self):
        """This provides more metainfo in addition to standard __str__"""

        return "%s\n  Typeclass: %s\n  Itemsize: %s\n  Byteorder: %s\n" % \
               (self, repr(self.typeclass), self.itemsize, self.byteorder)
