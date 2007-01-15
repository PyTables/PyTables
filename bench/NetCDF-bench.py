import RandomArray, time, sys
from tables import Filters
import tables.netcdf3 as NetCDF
import Scientific.IO.NetCDF
# create an n1dim by n2dim random array.
n1dim = 1000
n2dim = 10000
print 'reading and writing a %s by %s random array ..'%(n1dim,n2dim)
array = RandomArray.random((n1dim,n2dim))
filters = Filters(complevel=0,complib='zlib',shuffle=0)
# create a file, put a random array in it.
# no compression is used.
# first, use Scientific.IO.NetCDF
t1 = time.time()
file = Scientific.IO.NetCDF.NetCDFFile('test.nc','w')
file.createDimension('n1', None)
file.createDimension('n2', n2dim)
foo = file.createVariable('data', 'd', ('n1','n2',))
for n in range(n1dim):
    foo[n] = array[n]
file.close()
print 'Scientific.IO.NetCDF took',time.time()-t1,'seconds'
# now use pytables NetCDF emulation layer.
t1 = time.time()
file = NetCDF.NetCDFFile('test.h5','w')
file.createDimension('n1', None)
file.createDimension('n2', n2dim)
# no compression (override default filters instance).
foo = file.createVariable('data', 'd', ('n1','n2',),filters=filters)
# this is faster
foo.append(array)
file.close()
print 'pytables NetCDF (1) took',time.time()-t1,'seconds'
t1 = time.time()
file = NetCDF.NetCDFFile('test.h5','w')
file.createDimension('n1', None)
file.createDimension('n2', n2dim)
foo = file.createVariable('data', 'd', ('n1','n2',),filters=filters)
# this is slower
for n in range(n1dim):
    foo.append(array[n])
file.close()
print 'pytables NetCDF (2) took',time.time()-t1,'seconds'
# test reading.
t1 = time.time()
file = Scientific.IO.NetCDF.NetCDFFile('test.nc')
data = file.variables['data'][:]
file.close()
print 'Scientific.IO.NetCDF took',time.time()-t1,'seconds to read'
t1 = time.time()
file = NetCDF.NetCDFFile('test.h5')
data = file.variables['data'][:]
file.close()
print 'pytables NetCDF took',time.time()-t1,'seconds to read'
