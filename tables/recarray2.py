import numarray as num
import ndarray as mda
import memory
import chararray
import sys, copy, os, re, types, string

__version__ = '1.0'

class Char:
    """ data type Char class"""
    bytes = 1
    def __repr__(self):
        return "CharType"

CharType = Char()

# translation table to the num data types
# This table modified by F. Alted 2003-28-01
# All Numeric charcodes are removed. "a" means CharArray type
numfmt = {'i1':num.Int8, 'u1':num.UInt8,
          'i2':num.Int16, 'u2':num.UInt16,
          'i4':num.Int32, 'u4':num.UInt32,
          'i8':num.Int64, 'u8':num.UInt64,
          'f4':num.Float32, 'f8':num.Float64,
          'Int8':num.Int8, 'UInt8':num.UInt8,
          'Int16':num.Int16, 'UInt16':num.UInt16,
          'Int32':num.Int32, 'UInt32':num.UInt32, 
          'Int64':num.Int64, 'UInt64':num.UInt64, 
          'Float32':num.Float32, 'Float64':num.Float64,
          'a':CharType,
          }

# the reverse translation table of the above (for recarray only)
# This table modified by F. Alted 2003-28-01
revfmt = {num.Int8:'i1',   num.UInt8:'u1',
          num.Int16:'i2',  num.UInt16:'u2',
          num.Int32:'i4',  num.UInt32:'u4',
          num.Int64:'i8',  num.UInt64:'u8',
          num.Float32:'f4', num.Float64:'f8',
          CharType:'a'
          }

# Tform regular expression
format_re = re.compile(r'(?P<repeat>^[0-9]*)(?P<dtype>[A-Za-z0-9.]+)')

def fromrecords (recList, formats=None, names=None):
    """ create a Record Array from a list of records in text form

        The data in the same field can be heterogeneous, they will be promoted
        to the highest data type.  This method is intended for creating
        smaller record arrays.  If used to create large array e.g.

        r=recarray.fromrecords([[2,3.,'abc']]*100000)

        it is slow.

    >>> r=fromrecords([[456,'dbe',1.2],[2,'de',1.3]],names='col1,col2,col3')
    >>> print r[0]
    (456, 'dbe', 1.2)
    >>> r.field('col1')
    array([456,   2])
    >>> r.field('col2')
    CharArray(['dbe', 'de'])
    >>> import cPickle
    >>> print cPickle.loads(cPickle.dumps(r))
    RecArray[ 
    (456, 'dbe', 1.2),
    (2, 'de', 1.3)
    ]
    """

    _shape = len(recList)
    _nfields = len(recList[0])
    for _rec in recList:
        if len(_rec) != _nfields:
            raise ValueError, "inconsistent number of objects in each record"
    arrlist = [0]*_nfields
    for col in range(_nfields):
        tmp = [0]*_shape
        for row in range(_shape):
            tmp[row] = recList[row][col]
        try:
            arrlist[col] = num.array(tmp)
        except:
            try:
                arrlist[col] = chararray.array(tmp)
            except:
                raise ValueError, "inconsistent data at row %d,field %d" % (row, col)
    _array = fromarrays(arrlist, formats=formats, names=names)
    del arrlist
    del tmp
    return _array

def fromarrays (arrayList, formats=None, names=None):
    """ create a Record Array from a list of num/char arrays

    >>> x1=num.array([1,2,3,4])
    >>> x2=chararray.array(['a','dd','xyz','12'])
    >>> x3=num.array([1.1,2,3,4])
    >>> r=fromarrays([x1,x2,x3],names='a,b,c')
    >>> print r[1]
    (2, 'dd', 2.0)
    >>> x1[1]=34
    >>> r.field('a')
    array([1, 2, 3, 4])
    """

    _shape = len(arrayList[0])

    if formats == None:

        # go through each object in the list to see if it is a numarray or
        # chararray and determine the formats
        formats = ''
        for obj in arrayList:
            if isinstance(obj, chararray.CharArray):
                formats += `obj._itemsize` + 'a,'
            elif isinstance(obj, num.NumArray):
                if len(obj._shape) == 1: _repeat = ''
                elif len(obj._shape) == 2: _repeat = `obj._shape[1]`
                else: raise ValueError, "doesn't support numarray more than 2-D"

                formats += _repeat + revfmt[obj._type] + ','
            else:
                raise ValueError, "item in the array list must be numarray or chararray"
        formats=formats[:-1]

    for obj in arrayList:
        if len(obj) != _shape:
            raise ValueError, "array has different lengths"

    _array = RecArray(None, formats=formats, shape=_shape, names=names)

    # populate the record array (make a copy)
    for i in range(len(arrayList)):
        try:
            _array.field(_array._names[i])[:] = arrayList[i]
        except:
            print "Incorrect CharArray format %s, copy unsuccessful." % _array._formats[i]
    return _array

def fromstring (datastring, formats, shape=0, names=None):
    """ create a Record Array from binary data contained in a string"""
    _array = RecArray(chararray._stringToBuffer(datastring), formats, shape, names)
    if mda.product(_array._shape)*_array._itemsize > len(datastring):
        raise ValueError("Insufficient input data.")
    else: return _array

def fromfile(file, formats, shape=-1, names=None):
    """Create an array from binary file data

    If file is a string then that file is opened, else it is assumed
    to be a file object. No options at the moment, all file positioning
    must be done prior to this function call with a file object

    >>> import testdata, sys
    >>> fd=open(testdata.filename)
    >>> fd.seek(2880*2)
    >>> r=fromfile(fd, formats='d,i,5a', shape=3)
    >>> r._byteorder = "big"
    >>> print r[0]
    (5.1000000000000005, 61, 'abcde')
    >>> r._shape
    (3,)
    """

    if isinstance(shape, types.IntType) or isinstance(shape, types.LongType):
        shape = (shape,)
    name = 0
    if isinstance(file, types.StringType):
        name = 1
        file = open(file, 'rb')
    size = os.path.getsize(file.name) - file.tell()

    dummy = array(None, formats=formats, shape=0)
    itemsize = dummy._itemsize

    if shape and itemsize:
        shapesize = mda.product(shape)*itemsize
        if shapesize < 0:
            shape = list(shape)
            shape[ shape.index(-1) ] = size / -shapesize
            shape = tuple(shape)

    nbytes = mda.product(shape)*itemsize

    if nbytes > size:
        raise ValueError(
                "Not enough bytes left in file for specified shape and type")

    # create the array
    _array = RecArray(None, formats=formats, shape=shape, names=names)
    nbytesread = memory.file_readinto(file, _array._data)
    if nbytesread != nbytes:
        raise IOError("Didn't read as many bytes as expected")
    if name:
        file.close()
    return _array

# The test below was factored out of "array" due to platform specific
# floating point formatted results:  e+020 vs. e+20
if sys.platform == "win32":
    _fnumber = "2.5984589414244182e+020"
else:
    _fnumber = "2.5984589414244182e+20"

__test__ = {}
__test__["array_platform_test_workaround"] = """
        >>> r=array('a'*200,'r,3s,5a,i',3)
        >>> print r[0]
        (%(_fnumber)s, array([24929, 24929, 24929], type=Int16), 'aaaaa', 1633771873)
        >>> print r[1]
        (%(_fnumber)s, array([24929, 24929, 24929], type=Int16), 'aaaaa', 1633771873)
        """ % globals()
del _fnumber

def array(buffer=None, formats=None, shape=0, names=None):
    """This function will creates a new instance of a RecArray.

    buffer      specifies the source of the array's initialization data.
                buffer can be: RecArray, list of records in text, list of
                numarray/chararray, None, string, buffer.

    formats     specifies the fromat definitions of the array's records.

    shape       specifies the array dimensions.

    names       specifies the field names.

    >>> r=array([[456,'dbe',1.2],[2,'de',1.3]],names='col1,col2,col3')
    >>> print r[0]
    (456, 'dbe', 1.2)
    >>> r=array('a'*200,'r,3i,5a,s',3)
    >>> r._bytestride
    23
    >>> r._names
    ['c1', 'c2', 'c3', 'c4']
    >>> r._repeats
    [1, 3, 5, 1]
    >>> r._shape
    (3,)
    """

    if (buffer is None) and (formats is None):
        raise ValueError("Must define formats if buffer=None")
    elif buffer is None or isinstance(buffer, types.BufferType):
        return RecArray(buffer, formats=formats, shape=shape, names=names)
    elif isinstance(buffer, types.StringType):
        return fromstring(buffer, formats=formats, shape=shape, names=names)
    elif isinstance(buffer, types.ListType) or isinstance(buffer, types.TupleType):
        if isinstance(buffer[0], num.NumArray) or isinstance(buffer[0], chararray.CharArray):
            return fromarrays(buffer, formats=formats, names=names)
        else:
            return fromrecords(buffer, formats=formats, names=names)
    elif isinstance(buffer, RecArray):
        return buffer.copy()
    elif isinstance(buffer, types.FileType):
        return fromfile(buffer, formats=formats, shape=shape, names=names)
    else:
        raise ValueError("Unknown input type")

def _RecGetType(name):
    """Converts a type repr string into a type."""
    if name == "CharType":
        return CharType
    else:
        return num._getType(name)

class RecArray(mda.NDArray):
    """Record Array Class"""

    def __init__(self, buffer, formats, shape=0, names=None, byteoffset=0,
                 bytestride=None, byteorder=sys.byteorder, aligned=1):

        # names and formats can be either a string with components separated
        # by commas or a list of string values, e.g. ['i4', 'f4'] and 'i4,f4'
        # are equivalent formats

        self._parseFormats(formats)
        self._fieldNames(names)

        itemsize = self._stops[-1] + 1

        if shape != None:
            if type(shape) in [types.IntType, types.LongType]: shape = (shape,)
            elif (type(shape) == types.TupleType and type(shape[0]) in [types.IntType, types.LongType]):
                pass
            else: raise NameError, "Illegal shape %s" % `shape`

        #XXX need to check shape*itemsize == len(buffer)?

        self._shape = shape
        mda.NDArray.__init__(self, self._shape, itemsize, buffer=buffer,
                             byteoffset=byteoffset,
                             bytestride=bytestride,
                             aligned=aligned)
        self._byteorder = byteorder

        # Build the column arrays
        self._fields = self._get_fields()

        # Associate a record object for accessing values in each row
        # in a efficient way (i.e. without creating a new object each time)
        self._row = Row(self)

    def _parseFormats(self, formats):
        """ Parse the field formats """

        if (type(formats) in [types.ListType, types.TupleType]):
            _fmt = formats[:]           ### make a copy
        elif (type(formats) == types.StringType):
            _fmt = string.split(formats, ',')
        else:
            raise NameError, "illegal input formats %s" % `formats`

        self._nfields = len(_fmt)
        self._repeats = [1] * self._nfields
        self._sizes = [0] * self._nfields
        self._stops = [0] * self._nfields

        # preserve the input for future reference
        self._formats = [''] * self._nfields

        sum = 0
        for i in range(self._nfields):

            # parse the formats into repeats and formats
            try:
                (_repeat, _dtype) = format_re.match(string.strip(_fmt[i])).groups()
            except: print 'format %s is not recognized' % _fmt[i]

            if _repeat == '': _repeat = 1
            else: _repeat = eval(_repeat)
            _fmt[i] = numfmt[_dtype]
            self._repeats[i] = _repeat

            self._sizes[i] = _fmt[i].bytes * _repeat
            sum += self._sizes[i]
            self._stops[i] = sum - 1

            # Unify the appearance of _format, independent of input formats
            self._formats[i] = `_repeat`+revfmt[_fmt[i]]

        self._fmt = _fmt

    def __getstate__(self):
        """returns pickled state dictionary for RecArray"""
        state = mda.NDArray.__getstate__(self)
        state["_fmt"] = map(repr, self._fmt)
        return state
    
    def __setstate__(self, state):
        mda.NDArray.__setstate__(self, state)
        self._fmt = map(_RecGetType, state["_fmt"])

    def _byteswap(self):
        """Byteswap the data in place.  Does *not* change _byteorder."""
        for fieldName in self._names:
            column = self._fields[fieldName]
            # Only deal with array objects which column size is
            # greather than 1
            if isinstance(column, num.NumArray) and column._itemsize != 1:
                column._byteswap()

    def togglebyteorder(self):
        "reverses the state of the _byteorder attribute:  little <-> big."""
        self._byteorder = {"little":"big","big":"little"}[self._byteorder]

    def byteswap(self):
        """Byteswap data in place and change the _byteorder attribute."""
        self._byteswap()
        self.togglebyteorder()

    def _fieldNames(self, names=None):
        """convert input field names into a list and assign to the _names
        attribute """

        if (names):
            if (type(names) in [types.ListType, types.TupleType]):
                pass
            elif (type(names) == types.StringType):
                names = string.split(names, ',')
            else:
                raise NameError, "illegal input names %s" % `names`

            self._names = map(lambda n:string.strip(n), names)
        else: self._names = []

        # if the names are not specified, they will be assigned as "c1, c2,..."
        # if not enough names are specified, they will be assigned as "c[n+1],
        # c[n+2],..." etc. where n is the number of specified names..."
        self._names += map(lambda i: 'c'+`i`, range(len(self._names)+1,self._nfields+1))

    def _get_fields(self):
        """ get a dictionary with fields as numeric arrays """

        # Iterate over all the fields
        fields = {}
        for fieldName in self._names:
            # determine the offset within the record
            indx = index_of(self._names, fieldName)
            _start = self._stops[indx] - self._sizes[indx] + 1

            _shape = self._shape
            _type = self._fmt[indx]
            _buffer = self._data
            _offset = self._byteoffset + _start

            # don't use self._itemsize due to possible slicing
            _stride = self._strides[0]

            _order = self._byteorder

            if isinstance(_type, Char):
                arr = chararray.CharArray(buffer=_buffer, shape=_shape,
                          itemsize=self._repeats[indx], byteoffset=_offset,
                          bytestride=_stride)
            else:
                arr = num.NumArray(shape=_shape, type=_type, buffer=_buffer,
                          byteoffset=_offset, bytestride=_stride,
                          byteorder = _order)

                # modify the _shape and _strides for array elements
                if (self._repeats[indx] > 1):
                    arr._shape = self._shape + (self._repeats[indx],)
                    arr._strides = (self._strides[0], _type.bytes)

            # Put this array as a value in dictionary
            fields[fieldName] = arr

        return fields

    def field(self, fieldName):
        """ get the field data as a numeric array """
        # Check if stride has changed from last call
        # I think this would be safe when multidimensional extensions comes
        #print "Strides2:", self._strides[0],"recarray"
        #print "Strides2:", self._fields[fieldName]._strides[0],"numarray"
        if self._fields[fieldName]._strides[0] <> self._strides[0]:
            self._fields = self._get_fields()  # Refresh the cache
            #self._row._array = self
            #self._row = Row(self)  # Refreseh the Row instance
        return self._fields[fieldName]

    def info(self):
        """display instance's attributes (except _data)"""
        _attrList = dir(self)
        _attrList.remove('_data')
        _attrList.remove('_fmt')
        for attr in _attrList:
            print '%s = %s' % (attr, getattr(self,attr))

    def __str__(self):
        outstr = 'RecArray[ \n'
        for i in self:
            outstr += Record.__str__(i) + ',\n'
        return outstr[:-2] + '\n]'

    # This doesn't work if printing strided recarrays
    # this should be further investigated
    def __str__0(self):
        """ return a string representation of this object """

        # This __str__ is around 30 times faster than the original one
        # Byteswap temporarily the byte order for presentation (if needed)
        if self._byteorder != sys.byteorder:
            self._byteswap()
        outlist = []
        for row in range(self.nelements()):
            outlist.append(str(self._row(row)))
        # When finished, restore the byte order (if needed)
        if self._byteorder != sys.byteorder:
            # Byteswap temporarily the byte order for presentation
            self._byteswap()
        return "RecArray[ \n" + ",\n".join(outlist) + "\n]"

    ### The followng  __getitem__ is not in the requirements
    ### and is here for experimental purposes
    def __getitem__(self, key):
        if type(key) == types.TupleType:
            if len(key) == 1:
                return mda.NDArray.__getitem__(self,key[0])
            elif len(key) == 2 and type(key[1]) == types.StringType:
                return mda.NDArray.__getitem__(self,key[0]).field(key[1])
            else:
                raise NameError, "Illegal key %s" % `key`
        return mda.NDArray.__getitem__(self,key)

    def _getitem(self, key):
        byteoffset = self._getByteOffset(key)
        row = (byteoffset - self._byteoffset) / self._strides[0]
        return Record(self, row)

    def _setitem(self, key, value):
        byteoffset = self._getByteOffset(key)
        row = (byteoffset - self._byteoffset) / self._strides[0]
        for i in range(self._nfields):
            self.field(self._names[i])[row] = value.field(self._names[i])

    def reshape(*value):
        print "Cannot reshape record array."

    # Moved out of scope
    def _f_del__(self):
        print "Deleting RecArray2 object"

import copy
class Row(object):
    """Row Class

    This class is similar to Record except for the fact that it is
    created and associated with a recarray in their creation
    time. When speed in traversing the recarray is required this
    approach is more convenient than create a new Record object for
    each row that is visited.

    """

    def __init__(self, input):

        self.__dict__["_array"] = input
        self.__dict__["_fields"] = input._fields
        self.__dict__["_row"] = 0

    def __call__(self, row):
        """ set the row for this record object """
        
        if row < self._array.shape[0]:
            self.__dict__["_row"] = row
            return self
        else:
            return None

    def __getattr__(self, fieldName):
        """ get the field data of the record"""

        # In case that the value is an array, the user should be responsible to
        # copy it if he wants to keep it.
        try:
            #value = self._fields[fieldName][self._row]
            # The next line gives place to a nasty bug: memory  consumption
            # grows without limit!
            # Why the heck??
            #return self._fields[fieldName][self._row]
            return self.__dict__["_fields"][fieldName][self.__dict__['_row']]
            #return -1
            #return self._array.field(fieldName)[self._row]
        except:
            (type, value, traceback) = sys.exc_info()
            raise AttributeError, "Error accessing \"%s\" attr.\n %s" % \
                  (fieldName, "Error was: \"%s: %s\"" % (type,value))

        if isinstance(value, num.NumArray):
            return copy.deepcopy(value)
        else:
             return value

    def __setattr__(self, fieldName, value):
        """ set the field data of the record"""

        # The next line commented out for the same reason than in __getattr_
        #self._fields[fieldName][self._row] = value
        self.__dict__["_fields"][fieldName][self.__dict__['_row']] = value
        #self._array.field(fieldName)[self._row] = value

    def __str__(self):
        """ represent the record as an string """
        
        outlist = []
        for name in self._array._names:
            outlist.append(`self._fields[name][self._row]`)
            #outlist.append(`self._array.field(name)[self._row]`)
        return "(" + ", ".join(outlist) + ")"

    def _all(self):
        """ represent the record as a list """
        
        outlist = []
        for name in self._fields:
            outlist.append(self._fields[name][self._row])
            #outlist.append(self._array.field(name)[self._row])
        return outlist

    # Moved out of scope
    def _f_del__(self):
        print "Deleting Row object"
        pass

class Record:
    """Record Class"""

    def __init__(self, input, row=0):
        if isinstance(input, types.ListType) or isinstance(input, types.TupleType):
            input = fromrecords([input])
        if isinstance(input, RecArray):
            self.array = input
            self.row = row

    def __getattr__(self, fieldName):
        """ get the field data of the record"""

        #return self.array.field(fieldName)[self.row]
        if fieldName in self.array._names:
            #return self.array.field(fieldName)[self.row]
            return self.array._fields[fieldName][self.row]

    def field(self, fieldName):
        """ get the field data of the record"""

        return self.array.field(fieldName)[self.row]

    def __str__(self):
        outstr = '('
        for name in self.array._names:
            ### this is not efficient, need to know how to convert N-bytes to each data type
            outstr += `self.array.field(name)[self.row]` + ', '
            # The next line doesn't work well with strided arrays
            #outstr += `self.array._fields[name][self.row]` + ', '
        return outstr[:-2] + ')'

def index_of(nameList, key):
    """ Get the index of the key in the name list.

        The key can be an integer or string.  If integer, it is the index
        in the list.  If string, the name matching will be case-insensitive and
        trailing blank-insensitive.
    """
    if (type(key) in [types.IntType, types.LongType]):
        indx = key
    elif (type(key) == types.StringType):
        _names = nameList[:]
        for i in range(len(_names)):
            _names[i] = string.lower(_names[i])
        try:
            indx = _names.index(string.strip(string.lower(key)))
        except:
            raise NameError, "Key %s does not exist" % key
    else:
        raise NameError, "Illegal key %s" % `key`

    return indx

def find_duplicate (list):
    """Find duplication in a list, return a list of dupicated elements"""
    dup = []
    for i in range(len(list)):
        if (list[i] in list[i+1:]):
            if (list[i] not in dup):
                dup.append(list[i])
    return dup

def test():
    import doctest, recarray
    return doctest.testmod(recarray)

if __name__ == "__main__":
    test()
