"""
PyTables NetCDF version 3 emulation API.

This package provides an API is nearly identical to Scientific.IO.NetCDF
(http://starship.python.net/~hinsen/ScientificPython/ScientificPythonManual/Scientific.html).
Some key differences between the Scientific.IO.NetCDF API and the pytables
NetCDF emulation API to keep in mind are:

1) data is stored in an HDF5 file instead of a netCDF file.
2) Although each variable can have only one unlimited
   dimension, it need not be the first as in a true NetCDF file.
   Complex data types 'F' (complex64) and 'D' (complex128) are supported
   in tables.netcdf3, but are not supported in netCDF
   (or Scientific.IO.NetCDF). Files with variables that have
   these datatypes, or an unlimited dimension other than the first,
   cannot be converted to netCDF using h5tonc.
3) variables are compressed on disk by default using
   HDF5 zlib compression with the 'shuffle' filter.
   If the 'least_significant_digit' keyword is used when a
   variable is created with the createVariable method, data will be
   truncated (quantized) before being written to the file.
   This can significantly improve compression.  For example, if
   least_significant_digit=1, data will be quantized using
   numpy.around(scale*data)/scale, where scale = 2**bits, and
   bits is determined so that a precision of 0.1 is retained (in
   this case bits=4).
   From http://www.cdc.noaa.gov/cdc/conventions/cdc_netcdf_standard.shtml:
   "least_significant_digit -- power of ten of the smallest decimal
   place in unpacked data that is a reliable value."
4) data must be appended to a variable with an unlimited
   dimension using the 'append' method of the netCDF
   variable object. In Scientific.IO.NetCDF, data can be added
   along an unlimited dimension by assigning it to a slice (there
   is no append method).
   The 'sync' method synchronizes the size
   of all variables with an unlimited dimension by filling in
   data using the default netCDF _FillValue, and
   is invoked automatically when the NetCDFFile object is closed.
   In the Scientific.IO.NetCDF, the 'sync' method flushes the data to disk.
5) the createVariable method has three extra optional keyword
   arguments not found in the Scientific.IO.NetCDF interface,
   'least_significant_digit' (see item (2) above), 'expectedsize'
   and 'filters'.
   The 'expectedsize' keyword applies only to variables with an
   unlimited dimension, and is an estimate of the number
   of entries that will be added along that dimension
   (default 1000). This estimate is used to optimize
   HDF5 file access and memory usage.
   The 'filters' keyword is a PyTables filters instance
   that describes how to store the data on disk.
   The default corresponds to complevel=6, complib='zlib',
   shuffle=1 and fletcher32=0.
6) data can be saved to a real netCDF file using the NetCDFFile class
   method 'h5tonc' (if Scientific.IO.NetCDF is installed). The
   unlimited dimension must be the first (for all variables in the file)
   in order to use the 'h5tonc' method.
   Data can also be imported from a true netCDF file and saved
   in an HDF5 file using the 'nctoh5' class method.
7) A list of attributes corresponding to global netCDF attributes
   defined in the file can be obtained with the NetCDFFile ncattrs method.
   Similarly, netCDF variable attributes can be obtained with
   the NetCDFVariable ncattrs method.
8) you should not define global or variable attributes that start
   with '_NetCDF_', those names are reserved for internal use.
9) output similar to 'ncdump -h' can be obtained by simply
   printing the NetCDFFile instance.

A tables.netcdf3 file consists of array objects (either EArrays or
CArrays) located in the root group of a pytables hdf5 file.  Each of
the array objects must have a dimensions attribute, consisting of a
tuple of dimension names (the length of this tuple should be the same
as the rank of the array object). Any such objects with one
of the supported data types in a pytables file that conforms to
this simple structure can be read with the tables.netcdf3 package.

Note: This package does not yet create HDF5 files that are compatible
with netCDF version 4.

Datasets created with the PyTables netCDF emulation API can be shared
over the internet with the OPeNDAP protocol (http://opendap.org), via
the python opendap module (http://opendap.oceanografia.org).  A plugin
for the python opendap server is included with the pytables
distribution (contrib/h5_dap_plugin.py).  Simply copy that file into
the 'plugins' directory of the opendap python module source
distribution, run 'setup.py install', point the opendap server to the
directory containing your hdf5 files, and away you go. Any OPeNDAP
aware client (such as Matlab or IDL) can now access your data over
http as if it were a local disk file.

Jeffrey Whitaker <jeffrey.s.whitaker@noaa.gov>

Version: 20051110
"""
__version__ = '20051110'

import warnings
warnings.warn('The tables.netcdf3 is not actively maintained anymore. '
              'This module is deprecated and will be removed in the future '
              'versions.', DeprecationWarning)

import numpy

# need Numeric for h5 <--> netCDF conversion.
try:
    import Numeric
    Numeric_imported = True
except:
    Numeric_imported = False

# need Scientific to convert to/from real netCDF files.
if Numeric_imported:
    try:
        import Scientific.IO.NetCDF as RealNetCDF
        ScientificIONetCDF_imported = True
    except:
        ScientificIONetCDF_imported = False
else:
    ScientificIONetCDF_imported = False


import tables

# dictionary that maps pytables types to single-character Numeric typecodes.
_typecode_dict = {'float64':'d',
                  'float32':'f',
                  'int32':'i',
                  'int16':'s',
                  'int8':'1',
                  'string':'c',
                  'complex64':'F',
                  'complex128':'D',
                  }

# The reverse typecode dict
_rev_typecode_dict = {}
for key, value in _typecode_dict.iteritems():
    _rev_typecode_dict[value] = key

# dictionary that maps single character Numeric typecodes to netCDF
# data types (False if no corresponding netCDF datatype exists).
_netcdftype_dict = {'s':'short','1':'byte','l':'int','i':'int',
          'f':'float','d':'double','c':'character','F':False,'D':False}
# values to print out in __repr__ method.
_reprtype_dict = {'s':'short','1':'byte','l':'int','i':'int',
          'f':'float','d':'double','c':'character','F':'complex','D':'double_complex'}

# _NetCDF_FillValue defaults taken netCDF 3.6.1 header file.
_fillvalue_dict = {'f': 9.9692099683868690e+36,
                   'd': 9.9692099683868690e+36, # near 15 * 2^119
                   'F': 9.9692099683868690e+36+0j, # next two I made up
                   'D': 9.9692099683868690e+36+0j, # (no complex in netCDF)
                   'i': -2147483647,
                   'l': -2147483647,
                   's': -32767,
                   '1': -127,   # (signed char)-127
                   'c': chr(0)} # (char)0

def quantize(data,least_significant_digit):
    """quantize data to improve compression.
    data is quantized using around(scale*data)/scale,
    where scale is 2**bits, and bits is determined from
    the least_significant_digit.
    For example, if least_significant_digit=1, bits will be 4."""
    precision = 10.**-least_significant_digit
    exp = math.log(precision,10)
    if exp < 0:
        exp = int(math.floor(exp))
    else:
        exp = int(math.ceil(exp))
    bits = math.ceil(math.log(10.**-exp,2))
    scale = 2.**bits
    return numpy.around(scale*data)/scale

class NetCDFFile:
    """
    netCDF file Constructor: NetCDFFile(filename, mode="r",history=None)

    Arguments:

    filename -- Name of hdf5 file to hold data.

    mode -- access mode. "r" means read-only; no data can be modified.
            "w" means write; a new file is created, an existing
            file with the same name is deleted. "a" means append
            (in analogy with serial files); an existing file is
            opened for reading and writing.

    history -- a string that is used to define the global NetCDF
    attribute 'history'.

    A NetCDFFile object has two standard attributes: 'dimensions' and
    'variables'. The values of both are dictionaries, mapping
    dimension names to their associated lengths and variable names to
    variables, respectively. Application programs should never modify
    these dictionaries.

    A list of attributes corresponding to global netCDF attributes
    defined in the file can be obtained with the ncattrs method.
    Global file attributes are created by assigning to an attribute of
    the NetCDFFile object.
    """

    def __init__(self,filename,mode='r',history=None):
        # open an hdf5 file.
        self._NetCDF_h5file = tables.openFile(filename, mode=mode)
        self._NetCDF_mode = mode
        # file already exists, set up variable and dimension dicts.
        if mode != 'w':
            self.dimensions = {}
            self.variables = {}
            for var in self._NetCDF_h5file.root:
                if not isinstance(var,tables.CArray) and not isinstance(var,tables.EArray):
                    print 'object',var,'is not a EArray or CArray, skipping ..'
                    continue
                if var.atom.type not in _typecode_dict.keys():
                    print 'object',var.name,'is not a supported datatype (',var.atom.type,'), skipping ..'
                    continue
                if var.attrs.__dict__.has_key('dimensions'):
                    n = 0
                    for dim in var.attrs.__dict__['dimensions']:
                        if var.extdim >= 0 and n == var.extdim:
                            val=None
                        else:
                            val=int(var.shape[n])
                        if not self.dimensions.has_key(dim):
                            self.dimensions[dim] = val
                        else:
                            # raise an exception of a dimension of that
                            # name has already been encountered with a
                            # different value.
                            if self.dimensions[dim] != val:
                                raise KeyError,'dimension lengths not consistent'
                        n = n + 1
                else:
                    print 'object',var.name,'does not have a dimensions attribute, skipping ..'
                    continue
                self.variables[var.name]=_NetCDFVariable(var,self)
            if len(self.variables.keys()) == 0:
                raise IOError, 'file does not contain any objects compatible with tables.netcdf3'
        else:
        # initialize dimension and variable dictionaries for a new file.
            self.dimensions = {}
            self.variables = {}
        # set history attribute.
        if mode != 'r':
            if history != None:
                self.history = history

    def createDimension(self,dimname,size):
        """Creates a new dimension with the given "dimname" and
        "size". "size" must be a positive integer or 'None',
        which stands for the unlimited dimension. There can
        be only one unlimited dimension per dataset."""
        self.dimensions[dimname] = size
        # make sure there is only one unlimited dimension.
        if self.dimensions.values().count(None) > 1:
            raise ValueError, 'only one unlimited dimension allowed!'

    def createVariable(self,varname,datatype,dimensions,least_significant_digit=None,expectedsize=1000,filters=None):
        """Creates a new variable with the given "varname", "datatype", and
        "dimensions". The "datatype" is a one-letter string with the same
        meaning as the typecodes for arrays in module Numeric; in
        practice the predefined type constants from Numeric should
        be used. "dimensions" must be a tuple containing dimension
        names (strings) that have been defined previously.
        The unlimited dimension must be the first (leftmost)
        dimension of the variable.

        If the optional keyword parameter 'least_significant_digit' is
        specified, multidimensional variables will be truncated
        (quantized).  This can significantly improve compression.  For
        example, if least_significant_digit=1, data will be quantized
        using Numeric.around(scale*data)/scale, where scale = 2**bits,
        and bits is determined so that a precision of 0.1 is retained
        (in this case bits=4).
        From http://www.cdc.noaa.gov/cdc/conventions/cdc_netcdf_standard.shtml:
        "least_significant_digit -- power of ten of the smallest decimal
        place in unpacked data that is a reliable value."

        The 'expectedsize' keyword applies only to variables with an
        unlimited dimension - it is the expected number of entries
        that will be added along the unlimited dimension (default
        1000).  If think the actual number of entries will be an order
        of magnitude different than the default, consider providing a
        guess; this will optimize the HDF5 B-Tree creation, management
        process time, and memory usage.

        The 'filters' keyword also applies only to variables with
        an unlimited dimension, and is a PyTables filters instance
        that describes how to store an enlargeable array on disk.
        The default is tables.Filters(complevel=6, complib='zlib',
        shuffle=1, fletcher32=0).

        The return value is the NetCDFVariable object describing the
        new variable."""
        # create NetCDFVariable instance.
        var = NetCDFVariable(varname,self,datatype,dimensions,least_significant_digit=least_significant_digit,expectedsize=expectedsize,filters=filters)
        # update shelf variable dictionary, global variable
        # info dict.
        self.variables[varname] = var
        return var

    def close(self):
        """Closes the file (after calling the sync method)"""
        self.sync()
        self._NetCDF_h5file.close()

    def sync(self):
        """
 synchronize variables along unlimited dimension, filling in data
 with default netCDF _FillValue. Returns the length of the
 unlimited dimension. Invoked automatically when the NetCDFFile
 object is closed.
        """
        # find max length of unlimited dimension.
        len_unlim_dims = []
        hasunlimdim = False
        for varname,var in self.variables.iteritems():
            if var.extdim >= 0:
                hasunlimdim = True
                len_unlim_dims.append(var.shape[var.extdim])
        if not hasunlimdim:
            return 0
        len_max = max(len_unlim_dims)
        if self._NetCDF_mode == 'r':
            return len_max # just returns max length of unlim dim if read-only
        # fill in variables that have an unlimited
        # dimension with _FillValue if they have fewer
        # entries along unlimited dimension than the max.
        for varname,var in self.variables.iteritems():
            len_var = var.shape[var.extdim]
            if var.extdim >= 0 and len_var < len_max:
                shp = list(var.shape)
                shp[var.extdim]=len_max-len_var
                dtype = _rev_typecode_dict[var.typecode()]
                var._NetCDF_varobj.append(
                    var._NetCDF_FillValue*numpy.ones(shp, dtype=dtype))
        return len_max

    def __repr__(self):
        """produces output similar to 'ncdump -h'."""
        info=[self._NetCDF_h5file.filename+' {\n']
        info.append('dimensions:\n')
        n = 0
        len_unlim = int(self.sync())
        for key,val in self.dimensions.iteritems():
            if val == None:
                size = len_unlim
                info.append('    '+key+' = UNLIMITED ; // ('+repr(size)+' currently)\n')
            else:
                info.append('    '+key+' = '+repr(val)+' ;\n')
            n = n + 1
        info.append('variables:\n')
        for varname in self.variables.keys():
            var = self.variables[varname]
            dim = var.dimensions
            type = _reprtype_dict[var.typecode()]
            info.append('    '+type+' '+varname+str(dim)+' ;\n')
            for key in var.ncattrs():
                val = getattr(var,key)
                info.append('        '+varname+':'+key+' = '+repr(val)+' ;\n')
        info.append('// global attributes:\n')
        for key in self.ncattrs():
            val = getattr(self,key)
            info.append('        :'+key+' = '+repr(val)+' ;\n')
        info.append('}')
        return ''.join(info)

    def __setattr__(self,name,value):
        # if name = 'dimensions', 'variables', or begins with
        # '_NetCDF_', it is a temporary at the python level
        # (not stored in the hdf5 file).
        if not name.startswith('_') and name not in ['dimensions','variables']:
            setattr(self._NetCDF_h5file.root._v_attrs,name,value)
        elif not name.endswith('__'):
            self.__dict__[name]=value

    def __getattr__(self,name):
        if name.startswith('__') and name.endswith('__'):
            raise AttributeError
        elif name.startswith('_NetCDF_') or name in ['dimensions','variables']:
            return self.__dict__[name]
        else:
            if self.__dict__.has_key(name):
                return self.__dict__[name]
            else:
                return self._NetCDF_h5file.root._v_attrs.__dict__[name]

    def ncattrs(self):
        """return attributes corresponding to netCDF file attributes"""
        return [attr for attr in self._NetCDF_h5file.root._v_attrs._v_attrnamesuser]

    def h5tonc(self,filename,packshort=False,scale_factor=None,add_offset=None):
        """convert to a true netcdf file (filename).  Requires
        Scientific.IO.NetCDF module. If packshort=True, variables are
        packed as short integers using the dictionaries scale_factor
        and add_offset. The dictionary keys are the the variable names
        in the hdf5 file to be packed as short integers. Each
        variable's unlimited dimension must be the slowest varying
        (the first dimension for C/Python, the last for Fortran)."""

        if not ScientificIONetCDF_imported or not Numeric_imported:
            print 'Scientific.IO.NetCDF and Numeric must be installed to convert to NetCDF'
            return
        ncfile = RealNetCDF.NetCDFFile(filename,'w')
        # create dimensions.
        for dimname,size in self.dimensions.iteritems():
            ncfile.createDimension(dimname,size)
        # create global attributes.
        for key in self.ncattrs():
            setattr(ncfile,key,getattr(self,key))
        # create variables.
        for varname,varin in self.variables.iteritems():
            packvar = False
            dims = varin.dimensions
            dimsizes = [self.dimensions[dim] for dim in dims]
            if None in dimsizes:
                if dimsizes.index(None) != 0:
                    raise ValueError,'unlimited or enlargeable dimension must be most significant (slowest changing, or first) one in order to convert to a true netCDF file'
            if packshort and scale_factor.has_key(varname) and add_offset.has_key(varname):
                print 'packing %s as short integers ...'%(varname)
                datatype = 's'
                packvar = True
            else:
                datatype = varin.typecode()
            if not _netcdftype_dict[datatype]:
                raise ValueError,'datatype not supported in netCDF, cannot convert to a true netCDF file'

            varout = ncfile.createVariable(varname,datatype,dims)
            for key in varin.ncattrs():
                setattr(varout,key,getattr(varin,key))
                if packvar:
                    setattr(varout,'scale_factor',scale_factor[varname])
                    setattr(varout,'add_offset',add_offset[varname])
            for n in range(varin.shape[0]):
                if packvar:
                    varout[n] = ((1./scale_factor[varname])*(varin[n] - add_offset[varname])).astype('s')
                else:
                    if datatype == 'c':
                        tmp = Numeric.array(varin[n].flatten(),'c')
                        varout[n] = Numeric.reshape(tmp, varin.shape[1:])
                    else:
                        varout[n] = varin[n]
        # close file.
        ncfile.close()

    def nctoh5(self,filename,unpackshort=True,filters=None):
        """convert a true netcdf file (filename) to a hdf5 file
        compatible with this package.  Requires Scientific.IO.NetCDF
        module. If unpackshort=True, variables stored as short
        integers with a scale and offset are unpacked to Float32
        variables in the hdf5 file.  If the least_significant_digit
        attribute is set, the data is quantized to improve
        compression.  Use the filters keyword to change the default
        tables.Filters instance used for compression (see the
        createVariable docstring for details)."""

        if not ScientificIONetCDF_imported or not Numeric_imported:
            print 'Scientific.IO.NetCDF and Numeric must be installed to convert from NetCDF'
            return
        ncfile = RealNetCDF.NetCDFFile(filename,'r')
        # create dimensions.
        hasunlimdim = False
        for dimname,size in ncfile.dimensions.iteritems():
            self.createDimension(dimname,size)
            if size == None:
                hasunlimdim = True
                unlimdim = dimname
        # create variables.
        for varname,ncvar in ncfile.variables.iteritems():
            if hasattr(ncvar,'least_significant_digit'):
                lsd = ncvar.least_significant_digit
            else:
                lsd = None
            if unpackshort and hasattr(ncvar,'scale_factor') and hasattr(ncvar,'add_offset'):
                dounpackshort = True
                datatype = 'f'
            else:
                dounpackshort = False
                datatype = ncvar.typecode()
            var = self.createVariable(varname,datatype,ncvar.dimensions,least_significant_digit=lsd,filters=filters)
            for key,val in ncvar.__dict__.iteritems():
                if dounpackshort and key in ['add_offset','scale_factor']: continue
                if dounpackshort and key == 'missing_value': val=1.e30
                # convert rank-0 Numeric array.to python float/int/string
                if isinstance(val,type(Numeric.array([1]))) and len(val)==1:
                    val = val[0]
                setattr(var,key,val)
        # fill variables with data.
        nobjects = 0; nbytes = 0  # Initialize counters
        for varname,ncvar in ncfile.variables.iteritems():
            var = self.variables[varname]
            extdim = var._NetCDF_varobj.extdim
            if extdim >= 0:
                hasunlimdim = True
            else:
                hasunlimdim = False
            if unpackshort and hasattr(ncvar,'scale_factor') and hasattr(ncvar,'add_offset'):
                dounpackshort = True
            else:
                dounpackshort = False
            if hasunlimdim:
                # write data to enlargeable array one chunk of records at a
                # time (so the whole array doesn't have to be kept in memory).
                nrowsinbuf = var._NetCDF_varobj.nrowsinbuf
                # The slices parameter for var.__getitem__()
                slices = [slice(0, dim, 1) for dim in ncvar.shape]
                # range to copy
                start = 0; stop = ncvar.shape[extdim]; step = nrowsinbuf
                if step < 1: step = 1
                # Start the copy itself
                for start2 in range(start, stop, step):
                    # Save the records on disk
                    stop2 = start2+step
                    if stop2 > stop:
                        stop2 = stop
                    # Set the proper slice in the extensible dimension
                    slices[extdim] = slice(start2, stop2, 1)
                    idata = ncvar[tuple(slices)]
                    if dounpackshort:
                        tmpdata = (ncvar.scale_factor*idata+ncvar.add_offset).astype('f')
                    else:
                        tmpdata = idata
                    if hasattr(ncvar,'missing_value'):
                        tmpdata = Numeric.where(idata >= ncvar.missing_value, 1.e30, tmpdata)
                    var.append(tmpdata)
            else:
                idata = ncvar[:]
                if dounpackshort:
                    tmpdata = (ncvar.scale_factor*idata+ncvar.add_offset).astype('f')
                else:
                    tmpdata = idata
                if hasattr(ncvar,'missing_value'):
                    tmpdata = Numeric.where(idata >= ncvar.missing_value, 1.e30, tmpdata)
                if ncvar.typecode() == 'c':
                    # numpy string arrays with itemsize=1 used for netCDF char arrays.
                    var[:] = numpy.array(tmpdata.tolist(),
                                         dtype="S1")

                else:
                    var[:] = tmpdata
            # Increment the counters
            nobjects += 1
            nbytes += reduce(lambda x,y:x*y, var._NetCDF_varobj.shape) * var._NetCDF_varobj.atom.itemsize
        # create global attributes.
        for key,val in ncfile.__dict__.iteritems():
            # convert Numeric rank-0 array to a python float/int/string
            if isinstance(val,type(Numeric.array([1]))) and len(val)==1:
                val = val[0]
            # if attribute is a Numeric array, convert to python list.
            if isinstance(val,type(Numeric.array([1]))) and len(val)>1:
                val = val.tolist()
            setattr(self,key,val)
        # close file.
        ncfile.close()
        self.sync()
        return nobjects, nbytes

class NetCDFVariable:
    """Variable in a netCDF file

    NetCDFVariable objects are constructed by calling the method
    'createVariable' on the NetCDFFile object.

    NetCDFVariable objects behave much like array objects defined in
    module Numeric, except that their data resides in a file.  Data is
    read by indexing and written by assigning to an indexed subset;
    the entire array can be accessed by the index '[:]'.

    Variables with an unlimited dimension are can be compressed on
    disk (by default, zlib compression (level=6) and the HDF5
    'shuffle' filter are used). The default can be changed by passing
    a tables.Filters instance to createVariable via the filters
    keyword argument.  Truncating the data to a precision specified by
    the least_significant_digit optional keyword argument to
    createVariable will signficantly improve compression.

    A list of attributes corresponding to variable attributes defined
    in the netCDF file can be obtained with the ncattrs method.
    """

    def __init__(self, varname, NetCDFFile, datatype, dimensions, least_significant_digit=None,expectedsize=1000,filters=None):
        if datatype not in _netcdftype_dict.keys():
            raise ValueError, 'datatype must be one of %s'%_netcdftype_dict.keys()
        self._NetCDF_parent = NetCDFFile
        _NetCDF_FillValue = _fillvalue_dict[datatype]
        vardimsizes = []
        for d in dimensions:
            vardimsizes.append(NetCDFFile.dimensions[d])
        extdim = -1; ndim = 0
        for vardim in vardimsizes:
            if vardim == None:
                extdim = ndim
                break
            ndim += 1
        if extdim >= 0:
            # set shape to 0 for extdim.
            vardimsizes[extdim] = 0
        if datatype == 'c':
        # Special case for Numeric character objects
        # (on which base Scientific.IO.NetCDF works)
            atom = tables.StringAtom(itemsize=1)
        else:
            type_ = _rev_typecode_dict[datatype]
            atom = tables.Atom.from_type(type_)
        if filters is None:
            # default filters instance.
            filters = tables.Filters(complevel=6,complib='zlib',shuffle=1)
        if extdim >= 0:
            # check that unlimited dimension is first (extdim=0).
            #if extdim != 0:
            #    raise ValueError,'unlimited or enlargeable dimension must be most significant (slowest changing, or first) one in order to convert to a true netCDF file'
            # enlargeable dimension, use EArray
            self._NetCDF_varobj = NetCDFFile._NetCDF_h5file.createEArray(
                           where=NetCDFFile._NetCDF_h5file.root,
                           name=varname,atom=atom,shape=tuple(vardimsizes),
                           title=varname,filters=filters,
                           expectedrows=expectedsize)
        else:
            # no enlargeable dimension, use CArray
            self._NetCDF_varobj = NetCDFFile._NetCDF_h5file.createCArray(
                           where=NetCDFFile._NetCDF_h5file.root,
                           name=varname,atom=atom,shape=tuple(vardimsizes),
                           title=varname,filters=filters)
            # fill with _FillValue
            if datatype == 'c':
                # numpy string arrays with itemsize=1 used for char arrays.
                deflen = numpy.prod(vardimsizes, dtype='int64')
                self[:] = numpy.ndarray(buffer=_NetCDF_FillValue*deflen,
                                        shape=tuple(vardimsizes), dtype="S1")
            else:
                dtype = _rev_typecode_dict[datatype]
                self[:] = _NetCDF_FillValue*numpy.ones(tuple(vardimsizes),
                                                       dtype=dtype)
        if least_significant_digit != None:
            setattr(self._NetCDF_varobj.attrs, 'least_significant_digit',
                    least_significant_digit)
        setattr(self._NetCDF_varobj.attrs,'dimensions',dimensions)
        self._NetCDF_FillValue = _NetCDF_FillValue

    def __setitem__(self,key,data):
        if hasattr(self,'least_significant_digit'):
            self._NetCDF_varobj[key] = quantize(data,self.least_significant_digit)
        else:
            self._NetCDF_varobj[key] = data

    def __getitem__(self,key):
        return self._NetCDF_varobj[key]

    def __len__(self):
        return int(self._NetCDF_varobj.shape[0])

    def __setattr__(self,name,value):
        # if name begins with '_NetCDF_', it is a temporary at the python level
        # (not stored in the hdf5 file).
        # dimensions is a read only attribute
        if name in ['dimensions']:
            raise KeyError, '"dimensions" is a  read-only attribute - cannot modify'
        if not name.startswith('_NetCDF_'):
            setattr(self._NetCDF_varobj.attrs,name,value)
        elif not name.endswith('__'):
            self.__dict__[name]=value

    def __getattr__(self,name):
        if name.startswith('__') and name.endswith('__'):
            raise AttributeError
        elif name.startswith('_NetCDF_'):
            return self.__dict__[name]
        else:
            if self._NetCDF_varobj.__dict__.has_key(name):
                return self._NetCDF_varobj.__dict__[name]
            else:
                return self._NetCDF_varobj.attrs.__dict__[name]

    def typecode(self):
        """
 return a single character Numeric typecode.
 Allowed values are
 'd' == float64, 'f' == float32, 'l' == int32,
 'i' == int32, 's' == int16, '1' == int8,
 'c' == string (length 1), 'F' == complex64 and 'D' == complex128.
 The corresponding NetCDF data types are
 'double', 'float', 'int', 'int', 'short', 'byte' and 'character'.
 ('D' and 'F' have no corresponding netCDF data types).
        """
        return _typecode_dict[self._NetCDF_varobj.atom.type]

    def ncattrs(self):
        """return attributes corresponding to netCDF variable attributes"""
        return [attr for attr in self._NetCDF_varobj.attrs._v_attrnamesuser if attr != 'dimensions']

    def append(self,data):
        """
 Append data along unlimited dimension of a NetCDFVariable.

 The data must have either the same number of dimensions as the NetCDFVariable
 instance that it is being append to, or one less. If it has one less
 dimension, it assumed that the missing dimension is a singleton dimension
 corresponding to the unlimited dimension of the NetCDFVariable.

 If the NetCDFVariable has a least_significant_digit attribute,
 the data is truncated (quantized) to improve compression.
        """
        if self._NetCDF_parent._NetCDF_mode == 'r':
            raise IOError, 'file is read only'
        # if data is not an array, try to make it so.
        try:
            datashp = data.shape
        except:
            data = numpy.array(data, _rev_typecode_dict[self.typecode()])
        # check to make sure there is an unlimited dimension.
        # (i.e. data is in an EArray).
        extdim = self._NetCDF_varobj.extdim
        if extdim < 0:
            raise IndexError, 'variable has no unlimited dimension'
        # name of unlimited dimension.
        extdim_name = self.dimensions[extdim]
        # special case that data array is same
        # shape as EArray, minus the enlargeable dimension.
        # if so, add an extra singleton dimension.
        if len(data.shape) != len(self._NetCDF_varobj.shape):
            shapem1 = ()
            for n,dim in enumerate(self._NetCDF_varobj.shape):
                if n != extdim:
                    shapem1 = shapem1+(dim,)
            if data.shape == shapem1:
                shapenew = list(self._NetCDF_varobj.shape)
                shapenew[extdim]=1
                data = numpy.reshape(data, shapenew)
            else:
                raise IndexError,'data must either have same number of dimensions as variable, or one less (excluding unlimited dimension)'
        # append the data to the variable object.
        if hasattr(self,'least_significant_digit'):
            self._NetCDF_varobj.append(quantize(data,self.least_significant_digit))
        else:
            self._NetCDF_varobj.append(data)


    def assignValue(self,value):
        """
 Assigns value to the variable.
        """
        if self._NetCDF_varobj.extdim >=0:
            self.append(value)
        else:
            self[:] = value

    def getValue(self):
        """
 Returns the value of the variable.
        """
        return self[:]

# only used internally to create netCDF variable objects
# from Array objects read in from an hdf5 file.
class _NetCDFVariable(NetCDFVariable):
    def __init__(self, var, NetCDFFile):
        self._NetCDF_parent = NetCDFFile
        self._NetCDF_varobj = var
        self._NetCDF_FillValue = _fillvalue_dict[self.typecode()]
