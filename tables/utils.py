########################################################################
#
#       License: BSD
#       Created: March 4, 2003
#       Author:  Francesc Alted - falted@pytables.org
#
#       $Source: /home/ivan/_/programari/pytables/svn/cvs/pytables/pytables/tables/utils.py,v $
#       $Id: utils.py,v 1.28 2004/12/09 11:34:56 falted Exp $
#
########################################################################

"""Utility functions

"""

import types, re
# The second line is better for some installations
#from tables.hdf5Extension import getIndices
from hdf5Extension import getIndices
import numarray
from numarray import strings
try:
    import Numeric
    Numeric_imported = 1
except:
    Numeric_imported = 0

# Reserved prefixes for special attributes in Group and other classes
reservedprefixes = [
  '_c_',   # For class variables
  '_f_',   # For class public functions
  '_g_',   # For class private functions
  '_v_',   # For instance variables
]

pat = re.compile('^[a-zA-Z_][a-zA-Z0-9_]*$')

def checkNameValidity(name):
    "Check the validity of a name to be put in the object tree"
    global pat

    # First, some checks for avoid execution of arbitrary (malign) code
    # Suggested by I. Vilata
    if not pat.match(name):
        raise NameError, \
"""Sorry, you must use a name compliant with '[a-zA-Z_][a-zA-Z0-9_]*' regexp"""

    # Check if name starts with a reserved prefix
    for prefix in reservedprefixes:
        if (name.startswith(prefix)):
            raise NameError, \
"""Sorry, you cannot use a name like "%s" with the following reserved prefixes:\
  %s in this context""" % (name, reservedprefixes)
                
    # Check if new  node name have the appropriate set of characters
    # and is not one of the Python reserved word!
    # We use the next trick: exec the assignment 'name = 1' and
    # if a SyntaxError raises, catch it and re-raise a personalized error.
    testname = '_' + name + '_'
    try:
        exec(testname + ' = 1')  # Test for trailing and ending spaces
        exec(name + '= 1')  # Test for name validity
    except SyntaxError:
        raise NameError, \
"""\'%s\' is not a valid python identifier and cannot be used in this context.
  Check for special symbols ($, %%, @, ...), spaces or reserved words.""" % \
  (name)


def _calcBufferSize(rowsize, expectedrows):
    # A bigger buffer makes the writing faster and reading slower (!)
    #bufmultfactor = 1000 * 10
    # A smaller buffer also makes the tests to not take too much memory
    # We choose the smaller one
    # In addition, with the new iterator in the Row class, this seems to
    # be the best choice in terms of performance!
    #bufmultfactor = int(1000 * 1.0) # Original value
    # Best value with latest in-core selections optimisations
    # 5% to 10% of improvement in Pentium4 and non-noticeable in AMD64
    # 2004-05-16
    #bufmultfactor = int(1000 * 20.0) # A little better (5%) but
                                      # consumes more memory
    bufmultfactor = int(1000 * 10.0) # Optimum for Table objects
    rowsizeinfile = rowsize
    expectedfsizeinKb = (expectedrows * rowsizeinfile) / 1024

    # Some code to compute appropiate values for chunksize & buffersize
    # chunksize:  The chunksize for the HDF5 library
    # buffersize: The Table internal buffer size
    #
    # Rational: HDF5 takes the data in bunches of chunksize length
    # to write the on disk. A BTree in memory is used to map structures
    # on disk. The more chunks that are allocated for a dataset the
    # larger the B-tree. Large B-trees take memory and causes file
    # storage overhead as well as more disk I/O and higher contention
    # for the meta data cache.
    # You have to balance between memory and I/O overhead (small B-trees)
    # and time to access to data (big B-trees).
    #
    # The tuning of the chunksize & buffersize parameters affects the
    # performance and the memory size consumed. This is based on
    # experiments on a Intel arquitecture and, as always, your mileage
    # may vary.
    if expectedfsizeinKb <= 100:
        # Values for files less than 100 KB of size
        buffersize = 5 * bufmultfactor
    elif (expectedfsizeinKb > 100 and
        expectedfsizeinKb <= 1000):
        # Values for files less than 1 MB of size
        buffersize = 20 * bufmultfactor
    elif (expectedfsizeinKb > 1000 and
          expectedfsizeinKb <= 20 * 1000):
        # Values for sizes between 1 MB and 20 MB
        buffersize = 40  * bufmultfactor
        #buffersize = 80  * bufmultfactor  # New value (experimental)
    elif (expectedfsizeinKb > 20 * 1000 and
          expectedfsizeinKb <= 200 * 1000):
        # Values for sizes between 20 MB and 200 MB
        buffersize = 50 * bufmultfactor
        #buffersize = 320 * bufmultfactor  # New value (experimental)
    else:  # Greater than 200 MB
        # These values gives an increment of memory of 50 MB for a table
        # size of 2.2 GB. I think this increment should be attributed to
        # the BTree which is created to save the table data.
        # If we increment these values more than that, the HDF5 takes
        # considerably more CPU. If you don't want to spend 50 MB
        # (or more, depending on the final table size) to
        # the BTree, and want to save files bigger than 2 GB,
        # try to increment these values, but be ready for a quite big
        # overhead needed to traverse the BTree.
        buffersize = 60 * bufmultfactor
        #buffersize = 1280 * bufmultfactor  # New value (experimental)

    return buffersize

def calcBufferSize(rowsize, expectedrows, compress):
    """Calculate the buffer size and the HDF5 chunk size.

    The logic followed here is based purely in experiments playing
    with different buffer sizes, chunksize and compression flag. It is
    obvious that using big buffers optimize the I/O speed when dealing
    with tables. This might (should) be further optimized doing more
    experiments.

    """

    buffersize = _calcBufferSize(rowsize, expectedrows)

    # Max Tuples to fill the buffer
    maxTuples = buffersize // rowsize
    # Set the chunksize as the 10% of maxTuples
    #chunksize = maxTuples // 10
    chunksize = maxTuples // 2  # Makes the BTree hash to consume less memory
                                # This is experimental
    # Safeguard against row sizes being extremely large
    if maxTuples == 0:
        maxTuples = 1
    if chunksize == 0:
        chunksize = 1
    # A new correction for avoiding too many calls to HDF5 I/O calls
    # But this does not bring advantages rather the contrary,
    # the memory comsumption grows, and performance becomes worse.
    #if expectedrows//maxTuples > 50:
    #    buffersize *= 4
    #    maxTuples = buffersize // rowsize
    #chunksize *= 10  # just to test
    #print "maxTuples, chunksize -->", (maxTuples, chunksize)
    return (maxTuples, chunksize)

# This function is appropriate for calls to __getitem__ methods
def processRange(nrows, start=None, stop=None, step=1):
    if step and step < 0:
        raise ValueError, "slice step canot be negative"
    # slice object does not have a indices method in python 2.2
    # the next is a workaround for that (basically the code for indices
    # has been copied from python2.3 to hdf5Extension.pyx)
    #(start1, stop1, step1) = slice(start, stop, step).indices(nrows)
    (start, stop, step) = getIndices(slice(start, stop, step), nrows)
    # Some protection against empty ranges
    if start > stop:
        start = stop
    #print "start, stop, step(2)-->", (start, stop, step)
    return (start, stop, step)

# This function is appropiate for calls to read() methods
def processRangeRead(nrows, start=None, stop=None, step=1):
#     assert isinstance(start, types.IntType), "start must be an integer"
#     assert isinstance(stop, types.IntType), "stop must be an integer"
#     assert isinstance(step, types.IntType), "step must be an integer"
    if start is not None and stop is None:
        # Protection against start greater than available records
        # nrows == 0 is a special case for empty objects
        if (isinstance(start, types.IntType) or
            (isinstance(start, types.LongType))):
            if nrows > 0 and start >= nrows:
                raise IndexError, \
"Start (%s) value is greater than number of rows (%s)." % (start, nrows)
            step = 1
            if start == -1:  # corner case
                stop = nrows
            else:
                stop = start + 1
        else:
            raise IndexError, "start must be an integer and you passed: %s which os of type %s" % (repr(start), type(start))
    #print "start, stop, step -->", start, stop, step
    # Finally, get the correct values
    start, stop, step = processRange(nrows, start, stop, step)

    return (start, stop, step)

# This is used in VLArray and EArray to produce a numarray object
# of type atom from a generic python type 
def convertIntoNA(arr, atom):
    "Convert a generic object into a numarray object"
    # Check for Numeric objects
    if (isinstance(arr, numarray.NumArray) or
        isinstance(arr, strings.CharArray)):
        naarr = arr
    elif (Numeric_imported and type(arr) == type(Numeric.array(1))
          and not arr.typecode() == 'c'):
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
    elif (Numeric_imported and type(arr) == type(Numeric.array(1))
          and arr.typecode() == 'c'):
        # Special case for Numeric objects of type Char
        try:
            naarr = strings.array(arr.tolist(), itemsize=atom.itemsize)
            # If still doesn't, issues an error
        except:
            raise TypeError, \
"""The object '%s' can't be converted into a CharArray object of type '%s'. Sorry, but this object is not supported in this context.""" % (arr, atom)
    else:
        # Test if arr can be converted to a numarray object of the
        # correct type
        try:
            naarr = numarray.array(arr, type=atom.type)
        # If not, test with a chararray
        except TypeError:
            try:
                naarr = strings.array(arr, itemsize=atom.itemsize)
            # If still doesn't, issues an error
            except:
                raise TypeError, \
"""The object '%s' can't be converted into a numarray object of type '%s'. Sorry, but this object is not supported in this context.""" % (arr, atom)

    # Convert to the atom type, if necessary
    if (isinstance(naarr, numarray.NumArray) and naarr.type() <> atom.type):
        naarr = naarr.astype(atom.type)         # Force a cast
        
    # We always want a contiguous buffer
    # (no matter if has an offset or not; that will be corrected later)
    if not naarr.iscontiguous():
        # Do a copy of the array in case is not contiguous
        naarr = numarray.NDArray.copy(naarr)

    return naarr



if __name__=="__main__":
    import sys
    import getopt

    usage = \
"""usage: %s [-v] name
  -v means ...\n""" \
    % sys.argv[0]
    try:
        opts, pargs = getopt.getopt(sys.argv[1:], 'v')
    except:
        sys.stderr.write(usage)
        sys.exit(0)
    # if we pass too much parameters, abort
    if len(pargs) <> 1:
        sys.stderr.write(usage)
        sys.exit(0)
    name = sys.argv[1]
    # default options
    verbose = 0
    # Get the options
    for option in opts:
        if option[0] == '-v':
            verbose = 1
    # Catch the name to be validated
    name = pargs[0]
    checkNameValidity(name)
    print "Correct name: '%s'" % name
