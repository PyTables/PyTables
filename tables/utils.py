########################################################################
#
#       License: BSD
#       Created: March 4, 2003
#       Author:  Francesc Altet - faltet@carabos.com
#
#       $Source: /home/ivan/_/programari/pytables/svn/cvs/pytables/pytables/tables/utils.py,v $
#       $Id$
#
########################################################################

"""Utility functions

"""

import re
import warnings
import keyword

import numarray
from numarray import strings

try:
    import Numeric
    Numeric_imported = True
except ImportError:
    Numeric_imported = False

from tables.hdf5Extension import getIndices
from tables.exceptions import NaturalNameWarning



# Python identifier regular expression.
pythonIdRE = re.compile('^[a-zA-Z_][a-zA-Z0-9_]*$')
# PyTables reserved identifier regular expression.
#   c: class variables
#   f: class public methods
#   g: class private methods
#   v: instance variables
#   i: indexed columns
reservedIdRE = re.compile('^_[cfgvi]_')


def checkNameValidity(name):
    """
    Check the validity of the `name` of an object.

    If the name is not valid, a ``ValueError`` is raised.  If it is
    valid but it can not be used with natural naming, a
    `NaturalNameWarning` is issued.
    """

    warnInfo = """\
you will not be able to use natural naming to acces this object \
(using ``getattr()`` will still work)"""

    if not isinstance(name, basestring):  # Python >= 2.3
        raise TypeError("object name is not a string: %r" % (name,))

    # Check whether `name` is a valid HDF5 name.
    # http://hdf.ncsa.uiuc.edu/HDF5/doc/UG/03_Model.html#Structure
    if name == '':
        raise ValueError("the empty string is not allowed as an object name")
    if name == '.':
        raise ValueError("``.`` is not allowed as an object name")
    if '/' in name:
        raise ValueError(
            "the ``/`` character is not allowed in object names: %r" % (name,))

    # Check whether `name` is a valid Python identifier.
    if not pythonIdRE.match(name):
        warnings.warn("""\
object name is not a valid Python identifier: %r; \
it does not match the pattern ``%s``; %s"""
                      % (name, pythonIdRE.pattern, warnInfo),
                      NaturalNameWarning)
        return

    # However, Python identifiers and keywords have the same form.
    if keyword.iskeyword(name):
        warnings.warn("object name is a Python keyword: %r; %s"
                      % (name, warnInfo), NaturalNameWarning)
        return

    # Still, names starting with reserved prefixes are not allowed.
    if reservedIdRE.match(name):
        raise ValueError("""\
object name starts with a reserved prefix: %r; \
it matches the pattern ``%s``""" % (name, reservedIdRE.pattern))


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

def calcBufferSize(rowsize, expectedrows):
    """Calculate the buffer size and the HDF5 chunk size.

    The logic followed here is based purely in experiments playing with
    different buffer sizes and chunksize. It is obvious that using big
    buffers optimize the I/O speed when dealing with tables. This might
    (should) be further optimized doing more experiments.

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
        raise ValueError, "slice step cannot be negative"
    # slice object does not have a indices method in python 2.2
    # the next is a workaround for that (basically the code for indices
    # has been copied from python2.3 to hdf5Extension.pyx)
    #(start1, stop1, step1) = slice(start, stop, step).indices(nrows)
    (start, stop, step) = getIndices(slice(start, stop, step), nrows)
    # Some protection against empty ranges
    if start > stop:
        start = stop
    return (start, stop, step)

# This function is appropiate for calls to read() methods
def processRangeRead(nrows, start=None, stop=None, step=1):
    if start is not None and stop is None:
        # Protection against start greater than available records
        # nrows == 0 is a special case for empty objects
        if type(start) not in (int,long):
            raise TypeError, "Start must be an integer and you passed: %s which is of type %s" % (repr(start), type(start))
        if nrows > 0 and start >= nrows:
            raise IndexError, "Start of range (%s) is greater than number of rows (%s)." % (start, nrows)
        step = 1
        if start == -1:  # corner case
            stop = nrows
        else:
            stop = start + 1
    # Finally, get the correct values
    start, stop, step = processRange(nrows, start, stop, step)

    return (start, stop, step)

# This is used in VLArray and EArray to produce a numarray object
# of type atom from a generic python type.  If stated as true,
# it is assured that it will return a copy of the object and never
# the same object or a new one sharing the same memory.
def convertIntoNA(arr, atom, copy = False):
    "Convert a generic object into a numarray object"
    # Check for Numeric objects
    if (isinstance(arr, numarray.NumArray) or
        isinstance(arr, strings.CharArray)):
        if not copy:
            naarr = arr
        else:
            naarr = arr.copy()
    elif (Numeric_imported and type(arr) == type(Numeric.array(1))
          and not arr.typecode() == 'c'):
        if copy or not arr.iscontiguous():
            # Here we absolutely need a copy in order
            # to obtain a buffer.
            # Perhaps this can be avoided or optimized by using
            # the tolist() method, but this should be tested.
            carr = arr.copy()
        else:
            # This the fastest way to convert from Numeric to numarray
            # because no data copy is involved
            carr = arr
        naarr = numarray.array(buffer(carr),
                               type=arr.typecode(),
                               shape=arr.shape)
    elif (Numeric_imported and type(arr) == type(Numeric.array(1))
          and arr.typecode() == 'c'):
        # Special case for Numeric objects of type Char
        try:
            naarr = strings.array(arr.tolist(), itemsize=atom.itemsize)
            # If still doesn't, issues an error
        except:  #XXX
            raise TypeError, """The object '%s' can't be converted into a CharArray object of type '%s'. Sorry, but this object is not supported in this context.""" % (arr, atom)
    else:
        # Test if arr can be converted to a numarray object of the
        # correct type
        try:
            # 2005-02-04: The 'copy' argument appears in __doc__
            # but not in documentation.
            naarr = numarray.array(arr, type=atom.type, copy=copy)
        # If not, test with a chararray
        except TypeError:
            try:
                naarr = strings.array(arr, itemsize=atom.itemsize)
            # If still doesn't, issues an error
            except:  #XXX
                raise TypeError, """The object '%s' can't be converted into a numarray object of type '%s'. Sorry, but this object is not supported in this context.""" % (arr, atom)

    # Convert to the atom type, if necessary
    if (isinstance(naarr, numarray.NumArray) and naarr.type() <> atom.type):
        naarr = naarr.astype(atom.type)         # Force a cast
        
    # We always want a contiguous buffer
    # (no matter if has an offset or not; that will be corrected later)
    if not naarr.iscontiguous():
        # Do a copy of the array in case is not contiguous
        naarr = numarray.NDArray.copy(naarr)

    return naarr


def joinPath(parentPath, name):
    """joinPath(parentPath, name) -> path.  Joins a canonical path with a name.

    Joins the given canonical path with the given child node name.
    """

    if parentPath == '/':
        pstr = '%s%s'
    else:
        pstr = '%s/%s'
    return pstr % (parentPath, name)


def splitPath(path):
    """splitPath(path) -> (parentPath, name).  Splits a canonical path.

    Splits the given canonical path into a parent path (without the trailing
    slash) and a node name.
    """

    lastSlash = path.rfind('/')
    ppath = path[:lastSlash]
    name = path[lastSlash+1:]

    if ppath == '':
        ppath = '/'

    return (ppath, name)



if __name__=="__main__":
    import sys
    import getopt

    usage = \
"""usage: %s [-v] name
  -v means ...\n""" \
    % sys.argv[0]
    try:
        opts, pargs = getopt.getopt(sys.argv[1:], 'v')
    except getopt.GetoptError:
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



## Local Variables:
## mode: python
## py-indent-offset: 4
## tab-width: 4
## fill-column: 72
## End:
