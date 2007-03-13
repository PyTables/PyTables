### bogus example to illustrate the use of tables.netcdf3
### Author: Jeff Whitaker

import tables.netcdf3 as NetCDF
import time

history = 'Created ' + time.ctime(time.time())
file = NetCDF.NetCDFFile('test.h5', 'w', history=history)
file.createDimension('level', 12)
file.createDimension('time', None)
file.createDimension('lat', 90)
print '**dimensions**'
print file.dimensions

times = file.createVariable('time','d',('time',))
levels = file.createVariable('level','i',('level',))
latitudes = file.createVariable('latitude','f',('lat',))
temp = file.createVariable('temp','f',('time','level','lat',))
# try this to see how much smaller the file gets.
#temp = file.createVariable('temp','f',('time','level','lat',),least_significant_digit=1)
pressure = file.createVariable('pressure','i',('level','lat',))
print '**variables**'
print file.variables

file.description = 'bogus example to illustrate the use of tables.netcdf3'
file.source = 'PyTables Users Guide'
latitudes.units = 'degrees north'
pressure.units = 'hPa'
temp.units = 'K'
times.units = 'days since January 1, 2005'
times.scale_factor = 1
print '**global attributes**'
for name in file.ncattrs():
    print 'Global attr', name, '=', getattr(file,name)

import numpy
levels[:] = numpy.arange(12)+1
latitudes[:] = numpy.arange(-89,90,2)
for lev in levels[:]:
    pressure[:,:] = 1000.-100.*lev
print 'levels = ',levels[:]
print 'latitudes =\n',latitudes[:]
for n in range(10):
    times.append(n)
print 'times = ',times[:]
print 'temp.shape before sync = ',temp.shape
file.sync()
print 'temp.shape after sync = ',temp.shape
for n in range(10):
    temp[n] = 10.*numpy.random.random_sample(pressure.shape)
    print 'time, min/max temp, temp[n,0,0] = ',times[n],min(temp[n].flat),max(temp[n].flat),temp[n,0,0]
# print a summary of the file contents
print file

# Check conversions between netCDF <--> HDF5 formats
if NetCDF.ScientificIONetCDF_imported:
    scale_factor = {'temp': 1.75e-4}
    add_offset = {'temp': 5.}
    file.h5tonc('test.nc',packshort=True,scale_factor=scale_factor,add_offset=add_offset)
    file.close()
    history = 'Convert from netCDF ' + time.ctime(time.time())
    file = NetCDF.NetCDFFile('test2.h5', 'w', history=history)
    nobjects, nbytes = file.nctoh5('test.nc',unpackshort=True)
    print nobjects,' objects converted from netCDF, totaling',nbytes,'bytes'
    temp = file.variables['temp']
    times = file.variables['time']
    print 'temp.shape after h5 --> netCDF --> h5 conversion = ',temp.shape
    for n in range(10):
        print 'time, min/max temp, temp[n,0,0] = ',times[n],min(temp[n].flat),max(temp[n].flat),temp[n,0,0]
    file.close()
else:
    print 'skipping netCDF <--> h5 conversion since Scientific.IO.NetCDF not available'
    file.close()
