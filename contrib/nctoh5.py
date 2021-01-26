#!/usr/bin/env python

"""
convert netCDF file to HDF5 using Scientific.IO.NetCDF and PyTables.
Jeff Whitaker <jeffrey.s.whitaker@noaa.gov>

This requires Scientific from 
http://starship.python.net/~hinsen/ScientificPython

"""
import sys
from Scientific.IO import NetCDF
import tables as tb
# open netCDF file
ncfile = NetCDF.NetCDFFile(sys.argv[1], mode = "r")
# open h5 file.
h5file = tb.openFile(sys.argv[2], mode = "w")
# loop over variables in netCDF file.
for varname in ncfile.variables.keys():
    var = ncfile.variables[varname]
    vardims = list(var.dimensions)
    vardimsizes = [ncfile.dimensions[vardim] for vardim in vardims]
    # use long_name for title.
    if hasattr(var, 'long_name'):
       title = var.long_name
    else: # or, just use some bogus title.
       title = varname + ' array'
    # if variable has unlimited dimension or has rank>1,
    # make it enlargeable (with zlib compression).
    if vardimsizes[0] == None or len(vardimsizes) > 1:
        vardimsizes[0] = 0
        vardata = h5file.createEArray(h5file.root, varname,
        tb.Atom(shape=tuple(vardimsizes), dtype=var.typecode(),),
        title, filters=tb.Filters(complevel=6, complib='zlib'))
    # write data to enlargeable array on record at a time.
    # (so the whole array doesn't have to be kept in memory).
        for n in range(var.shape[0]):
            vardata.append(var[n:n+1])
    # or else, create regular array write data to it all at once.
    else:
        vardata=h5file.createArray(h5file.root, varname, var[:], title)
    # set variable attributes.
    for key, val in var.__dict__.iteritems():
        setattr(vardata.attrs, key, val)
    setattr(vardata.attrs, 'dimensions', tuple(vardims))
# set global (file) attributes.
for key, val in ncfile.__dict__.iteritems():
    setattr(h5file.root._v_attrs, key, val)
# Close the file
h5file.close()

