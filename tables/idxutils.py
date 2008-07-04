########################################################################
#
#       License: BSD
#       Created: April 02, 2007
#       Author:  Francesc Alted - faltet@pytables.com
#
#       $Id$
#
########################################################################

"""Utilities to be used mainly by the Index class."""

import math, os, subprocess
from time import time, clock
import numpy


# Hints for chunk/slice/block/superblock computations:
# - The slicesize should not exceed 2**32 elements (because of
# implementation reasons).  Such an extreme case would make the
# sorting algorithms to consume up to 64 GB of memory.
# - In general, one should favor a small chunksize ( < 128 KB) if one
# wants to reduce the latency for indexed queries. However, keep in
# mind that a very low value of chunksize for big datasets may hurt
# the performance by requering the HDF5 to use a lot of memory and CPU
# for its internal B-Tree.

def csformula(nrows):
    """Return the fitted chunksize (a float value) for nrows."""
    # This formula has been computed using two points:
    # 2**12 = m * 2**(n + log10(10**6))
    # 2**15 = m * 2**(n + log10(10**9))
    # where 2**12 and 2**15 are reasonable values for chunksizes for indexes
    # with 10**6 and 10**9 elements respectively.
    # Yes, return a floating point number!
    return 64 * 2**math.log10(nrows)


def limit_er(expectedrows):
    """Protection against creating too small or too large chunks or slices."""
    if expectedrows < 10**5:
        expectedrows = 10**5
    elif expectedrows > 10**12:
        expectedrows = 10**12
    return expectedrows


def computechunksize(expectedrows):
    """Get the optimum chunksize based on expectedrows."""

    expectedrows = limit_er(expectedrows)
    zone = int(math.log10(expectedrows))
    nrows = 10**zone
    return int(csformula(nrows))


def computeslicesize(expectedrows, memlevel):
    """Get the optimum slicesize based on expectedrows and memorylevel."""

    expectedrows = limit_er(expectedrows)
    # First, the optimum chunksize
    cs = csformula(expectedrows)
    # Now, the actual chunksize
    chunksize = computechunksize(expectedrows)
    # The optimal slicesize
    ss = int(cs * memlevel**2)
    # We *need* slicesize to be an exact multiple of the actual chunksize
    ss = (ss // chunksize) * chunksize
    ss *= 4    # slicesize should be at least divisible by 4
    # ss cannot be bigger than 2**32 - 1 elements because of
    # implementation reasons (this limitation can be overridden when
    # keysort would be implemented for the string type)
    if ss >= 2**32:
        ss = 2**32 - 2
    return ss


def computeblocksize(expectedrows, compoundsize, lowercompoundsize):
    """Calculate the optimum number of superblocks made from compounds blocks.

    This is useful for computing the sizes of both blocks and
    superblocks (using the PyTables terminology for blocks in indexes).
    """

    nlowerblocks = (expectedrows // lowercompoundsize) + 1
    if nlowerblocks > 2**20:
        # Protection against too large number of compound blocks
        nlowerblocks = 2**20
    size = lowercompoundsize * nlowerblocks
    # We *need* superblocksize to be an exact multiple of the actual
    # compoundblock size (a ceil must be performed here!)
    size = ((size // compoundsize) + 1) * compoundsize
    return size


def calcChunksize(expectedrows, optlevel=6, memlevel=4):
    """Calculate the HDF5 chunk size for index and sorted arrays.

    The logic to do that is based purely in experiments playing with
    different chunksizes and compression flag. It is obvious that
    using big chunks optimizes the I/O speed, but if they are too
    large, the uncompressor takes too much time. This might (should)
    be further optimized by doing more experiments.
    """

    chunksize = computechunksize(expectedrows)
    slicesize = computeslicesize(expectedrows, memlevel)
    # Correct the slicesize and the chunksize based on optlevel
    if optlevel in (0,1,2):
        slicesize /= 2
    elif optlevel in (3,4,5):
        pass
    elif optlevel in (6,7,8):
        chunksize /= 2
    elif optlevel == 9:
        # Reducing the chunksize and enlarging the slicesize is the
        # best way to reduce the entropy with the current algorithm.
        chunksize /= 2; slicesize *= 2
    blocksize = computeblocksize(expectedrows, slicesize, chunksize)
    superblocksize = computeblocksize(expectedrows, blocksize, slicesize)
    # The size for different blocks information
    sizes = (superblocksize, blocksize, slicesize, chunksize)
    return sizes


def calcoptlevels(nblocks, optlevel):
    """Compute the optimizations to be done.

    The calculation is based on the number of blocks and the optlevel.
    """

    optmedian, optstarts, optstops, optfull = (False,)*4

    # Regular case
    if nblocks <= 1:
        if 0 < optlevel <= 3:
            optmedian = True
        elif 3 < optlevel <= 6:
            optmedian, optstarts = (True, True)
        elif 6 < optlevel <= 9:
            optfull = 1
    else:  # More than a block
        if 0 < optlevel <= 3:
            optfull = 1
        elif 3 < optlevel <= 6:
            optfull = 2
        elif 6 < optlevel <= 9:
            optfull = 3

    return optmedian, optstarts, optstops, optfull


# Python implementations of NextAfter and NextAfterF
#
# These implementations exist because the standard function
# nextafterf is not available on Microsoft platforms.
#
# These implementations are based on the IEEE representation of
# floats and doubles.
# Author:  Shack Toms - shack@livedata.com
#
# Thanks to Shack Toms shack@livedata.com for NextAfter and NextAfterF
# implementations in Python. 2004-10-01

epsilon  = math.ldexp(1.0, -53) # smallest double such that 0.5+epsilon != 0.5
epsilonF = math.ldexp(1.0, -24) # smallest float such that 0.5+epsilonF != 0.5

maxFloat = float(2**1024 - 2**971)  # From the IEEE 754 standard
maxFloatF = float(2**128 - 2**104)  # From the IEEE 754 standard

minFloat  = math.ldexp(1.0, -1022) # min positive normalized double
minFloatF = math.ldexp(1.0, -126)  # min positive normalized float

smallEpsilon  = math.ldexp(1.0, -1074) # smallest increment for doubles < minFloat
smallEpsilonF = math.ldexp(1.0, -149)  # smallest increment for floats < minFloatF

infinity = math.ldexp(1.0, 1023) * 2
infinityF = math.ldexp(1.0, 128)
#Finf = float("inf")  # Infinite in the IEEE 754 standard (not avail in Win)

# A portable representation of NaN
# if sys.byteorder == "little":
#     testNaN = struct.unpack("d", '\x01\x00\x00\x00\x00\x00\xf0\x7f')[0]
# elif sys.byteorder == "big":
#     testNaN = struct.unpack("d", '\x7f\xf0\x00\x00\x00\x00\x00\x01')[0]
# else:
#     raise ValueError, "Byteorder '%s' not supported!" % sys.byteorder
# This one seems better
testNaN = infinity - infinity

# "infinity" for several types
infinityMap = {
    'bool':    [0,          1],
    'int8':    [-2**7,      2**7-1],
    'uint8':   [0,          2**8-1],
    'int16':   [-2**15,     2**15-1],
    'uint16':  [0,          2**16-1],
    'int32':   [-2**31,     2**31-1],
    'uint32':  [0,          2**32-1],
    'int64':   [-2**63,     2**63-1],
    'uint64':  [0,          2**64-1],
    'float32': [-infinityF, infinityF],
    'float64': [-infinity,  infinity], }


# Utility functions
def infType(dtype, itemsize, sign=+1):
    """Return a superior limit for maximum representable data type"""
    assert sign in [-1, +1]

    if dtype.kind == "S":
        if sign < 0:
            return "\x00"*itemsize
        else:
            return "\xff"*itemsize
    try:
        return infinityMap[dtype.name][sign >= 0]
    except KeyError:
        raise TypeError, "Type %s is not supported" % dtype.name


# This check does not work for Python 2.2.x or 2.3.x (!)
def IsNaN(x):
    """a simple check for x is NaN, assumes x is float"""
    return x != x


def PyNextAfter(x, y):
    """returns the next float after x in the direction of y if possible, else returns x"""
    # if x or y is Nan, we don't do much
    if IsNaN(x) or IsNaN(y):
        return x

    # we can't progress if x == y
    if x == y:
        return x

    # similarly if x is infinity
    if x >= infinity or x <= -infinity:
        return x

    # return small numbers for x very close to 0.0
    if -minFloat < x < minFloat:
        if y > x:
            return x + smallEpsilon
        else:
            return x - smallEpsilon  # we know x != y

    # it looks like we have a normalized number
    # break x down into a mantissa and exponent
    m, e = math.frexp(x)

    # all the special cases have been handled
    if y > x:
        m += epsilon
    else:
        m -= epsilon

    return math.ldexp(m, e)


def PyNextAfterF(x, y):
    """returns the next IEEE single after x in the direction of y if possible, else returns x"""

    # if x or y is Nan, we don't do much
    if IsNaN(x) or IsNaN(y):
        return x

    # we can't progress if x == y
    if x == y:
        return x

    # similarly if x is infinity
    if x >= infinityF:
        return infinityF
    elif x <= -infinityF:
        return -infinityF

    # return small numbers for x very close to 0.0
    if -minFloatF < x < minFloatF:
        # since Python uses double internally, we
        # may have some extra precision to toss
        if x > 0.0:
            extra = x % smallEpsilonF
        elif x < 0.0:
            extra = x % -smallEpsilonF
        else:
            extra = 0.0
        if y > x:
            return x - extra + smallEpsilonF
        else:
            return x - extra - smallEpsilonF  # we know x != y

    # it looks like we have a normalized number
    # break x down into a mantissa and exponent
    m, e = math.frexp(x)

    # since Python uses double internally, we
    # may have some extra precision to toss
    if m > 0.0:
        extra = m % epsilonF
    else:  # we have already handled m == 0.0 case
        extra = m % -epsilonF

    # all the special cases have been handled
    if y > x:
        m += epsilonF - extra
    else:
        m -= epsilonF - extra

    return math.ldexp(m, e)


def StringNextAfter(x, direction, itemsize):
    "Return the next representable neighbor of x in the appropriate direction."
    assert direction in [-1, +1]

    # Pad the string with \x00 chars until itemsize completion
    padsize = itemsize - len(x)
    if padsize > 0:
        x += "\x00"*padsize
    xlist = list(x); xlist.reverse()
    i = 0
    if direction > 0:
        if xlist == "\xff"*itemsize:
            # Maximum value, return this
            return "".join(xlist)
        for xchar in xlist:
            if ord(xchar) < 0xff:
                xlist[i] = chr(ord(xchar)+1)
                break
            else:
                xlist[i] = "\x00"
            i += 1
    else:
        if xlist == "\x00"*itemsize:
            # Minimum value, return this
            return "".join(xlist)
        for xchar in xlist:
            if ord(xchar) > 0x00:
                xlist[i] = chr(ord(xchar)-1)
                break
            else:
                xlist[i] = "\xff"
            i += 1
    xlist.reverse()
    return "".join(xlist)


def IntTypeNextAfter(x, direction, itemsize):
    "Return the next representable neighbor of x in the appropriate direction."
    assert direction in [-1, +1]

    # x is guaranteed to be either an int or a float
    if direction < 0:
        if type(x) is int:
            return x-1
        else:
            return int(PyNextAfter(x,x-1))
    else:
        if type(x) is int:
            return x+1
        else:
            return int(PyNextAfter(x,x+1))+1


def nextafter(x, direction, dtype, itemsize):
    "Return the next representable neighbor of x in the appropriate direction."
    assert direction in [-1, 0, +1]
    assert dtype.kind == "S" or type(x) in (int, long, float)

    if direction == 0:
        return x

    if dtype.kind == "S":
        return StringNextAfter(x, direction, itemsize)

    if dtype.kind in ['i', 'u']:
        return IntTypeNextAfter(x, direction, itemsize)
    elif dtype.name == "float32":
        if direction < 0:
            return PyNextAfterF(x,x-1)
        else:
            return PyNextAfterF(x,x+1)
    elif dtype.name == "float64":
        if direction < 0:
            return PyNextAfter(x,x-1)
        else:
            return PyNextAfter(x,x+1)

    raise TypeError("data type ``%s`` is not supported" % dtype)


def show_stats(explain, tref):
    "Show the used memory"
    # Build the command to obtain memory info (only for Linux 2.6.x)
    cmd = "cat /proc/%s/status" % os.getpid()
    sout = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE).stdout
    for line in sout:
        if line.startswith("VmSize:"):
            vmsize = int(line.split()[1])
        elif line.startswith("VmRSS:"):
            vmrss = int(line.split()[1])
        elif line.startswith("VmData:"):
            vmdata = int(line.split()[1])
        elif line.startswith("VmStk:"):
            vmstk = int(line.split()[1])
        elif line.startswith("VmExe:"):
            vmexe = int(line.split()[1])
        elif line.startswith("VmLib:"):
            vmlib = int(line.split()[1])
    sout.close()
    print "Memory usage: ******* %s *******" % explain
    print "VmSize: %7s kB\tVmRSS: %7s kB" % (vmsize, vmrss)
    print "VmData: %7s kB\tVmStk: %7s kB" % (vmdata, vmstk)
    print "VmExe:  %7s kB\tVmLib: %7s kB" % (vmexe, vmlib)
    tnow = time()
    print "WallClock time:", round(tnow - tref, 3)
    return tnow



## Local Variables:
## mode: python
## py-indent-offset: 4
## tab-width: 4
## fill-column: 72
## End:
