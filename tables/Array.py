########################################################################
#
#       License: BSD
#       Created: October 10, 2002
#       Author:  Francesc Alted - falted@openlc.org
#
#       $Source: /home/ivan/_/programari/pytables/svn/cvs/pytables/pytables/tables/Array.py,v $
#       $Id: Array.py,v 1.3 2002/11/10 13:31:50 falted Exp $
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

__version__ = "$Revision: 1.3 $"

from Leaf import Leaf
import hdf5Extension

class Array(Leaf, hdf5Extension.Array):
    """Represent a Numeric Array in HDF5 file.

    It provides methods to create new arrays or open existing ones, as
    well as methods to write/read data and metadata to/from array
    objects over the HDF5 file.

    All Numeric typecodes supported except "F" and "D" which
    corresponds to complex datatypes.

    Methods:

        read()
        flush()
        close()

    Instance variables:

        name -- the Leaf node name
        title -- the title for this node
        shape -- tuple with the array shape (in Numeric sense)
        typecode -- the typecode for the array

    """
    
    def __init__(self, NumericObject = None, title = ""):
        """Create the instance Array.

        Keyword arguments:

        NumericObject -- Numeric array to be saved. If None, the
            metadata for the array will be taken from disk.

        "title" -- Sets a TITLE attribute on the HDF5 array entity.

        """
        # Check if we have to create a new object or read their contents
        # from disk
        if NumericObject is not None:
            self._v_new = 1
            self.object = NumericObject
            self.title = title
        else:
            self._v_new = 0

    def create(self):
        """Save a fresh array (i.e., not present on HDF5 file)."""
        # Call the createArray superclass method to create the table on disk
        self.createArray(self.object, self.title)
        # Get some important attributes
        self.typecode = self.object.typecode()
        self.shape = self.object.shape

    def open(self):
        """Get the metadata info for an array in file."""
        #(self.typecode, self.shape, self.title) = self.openArray()
        #print "passing open..."
        (self.typecode, self.shape) = self.openArray()
        #print "Shape ==>", self.shape
        #self.typecode = self.openArray()
        self.title = self.getArrayTitle()
        # This still does not work!
        #self.shape = (1,)
        
    # Accessor for the readArray method in superclass
    def read(self):
        """Read the array from disk and return it as Numeric."""
        return self.readArray()
    
    # Accessor for the getArrayTitle method in superclass
    def getTitle(self):
        """Read the atribute TITLE from disk and return it."""
        return self.getArrayTitle()

    def flush(self):
        """Save whatever remaining data in buffer."""
        # This is a do nothing method because, at the moment the Array
        # class don't support buffers
    
    def close(self):
        """Flush the array buffers and close this object on file."""
        self.flush()
